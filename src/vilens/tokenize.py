"""Joint prompt-target tokenization helpers that preserve released text exactly."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any, MutableMapping, Sequence


class PrefixViolationError(ValueError):
    """Raised when prompt ids are not a prefix of the jointly encoded sequence."""


@dataclass(frozen=True)
class JointEncoding:
    """Token ids for a prompt and target encoded together."""

    prompt_ids: tuple[int, ...]
    joint_ids: tuple[int, ...]
    prefix_ok: bool

    @property
    def target_ids(self) -> tuple[int, ...]:
        """Return the jointly encoded target suffix when the prefix property holds."""
        return self.joint_ids[len(self.prompt_ids):] if self.prefix_ok else ()


def token_text(tokenizer: Any, token_id: int) -> str:
    """Decode one token without dropping special tokens or cleaning spaces."""
    return tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


class TokenClassifier:
    """Cache decoded token properties for content and byte-fragment checks."""

    def __init__(self, tokenizer: Any) -> None:
        """Initialize the classifier for one tokenizer."""
        self.tokenizer = tokenizer
        self._text_cache: dict[int, str] = {}
        self._content_cache: dict[int, bool] = {}
        self._byte_cache: dict[int, bool] = {}

    def is_content(self, token_id: int) -> bool:
        """Return false for tokens that decode to whitespace only."""
        if token_id not in self._content_cache:
            self._content_cache[token_id] = self._token_text(token_id).strip() != ""
        return self._content_cache[token_id]

    def is_byte_fragment(self, token_id: int) -> bool:
        """Return whether the token decodes with the Unicode replacement marker."""
        if token_id not in self._byte_cache:
            self._byte_cache[token_id] = "\ufffd" in self._token_text(token_id)
        return self._byte_cache[token_id]

    def _token_text(self, token_id: int) -> str:
        """Return and cache the decoded text for a token id."""
        if token_id not in self._text_cache:
            self._text_cache[token_id] = token_text(self.tokenizer, token_id)
        return self._text_cache[token_id]


def _validate_pair(prompt: str, target: str) -> None:
    """Validate exact prompt and non-empty target strings without modifying them."""
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    if not isinstance(target, str):
        raise ValueError("target must be a string")
    if prompt != prompt.rstrip():
        raise ValueError("prompt ends in whitespace")
    if not target or target.isspace():
        raise ValueError("target must not be empty or whitespace-only")


def _ids(encoded: Any) -> tuple[int, ...]:
    """Convert one tokenizer result row to an immutable integer tuple."""
    return tuple(int(token_id) for token_id in encoded)


def encode_joint(tokenizer: Any, prompt: str, target: str, *, strict: bool = True) -> JointEncoding:
    """Encode prompt and target jointly, optionally raising on a prefix violation."""
    _validate_pair(prompt, target)
    prompt_ids = _ids(tokenizer(prompt, add_special_tokens=True)["input_ids"])
    joint_ids = _ids(tokenizer(prompt + target, add_special_tokens=True)["input_ids"])
    prefix_ok = len(joint_ids) > len(prompt_ids) and joint_ids[:len(prompt_ids)] == prompt_ids
    result = JointEncoding(prompt_ids, joint_ids, prefix_ok)
    if strict and not prefix_ok:
        raise PrefixViolationError("prompt token ids are not a prefix of jointly encoded prompt-target ids")
    return result


def encode_joint_batch(
    tokenizer: Any,
    prompts: Sequence[str],
    targets: Sequence[str],
    *,
    strict: bool = True,
    prompt_cache: MutableMapping[str, tuple[int, ...]] | None = None,
) -> list[JointEncoding]:
    """Batch-encode prompt-target pairs with cached prompt-only encodings."""
    if len(prompts) != len(targets):
        raise ValueError(f"prompt and target lengths differ: {len(prompts)} != {len(targets)}")
    for prompt, target in zip(prompts, targets, strict=True):
        _validate_pair(prompt, target)
    cache = prompt_cache if prompt_cache is not None else {}
    missing_prompts = list(dict.fromkeys(prompt for prompt in prompts if prompt not in cache))
    if missing_prompts:
        encoded_prompts = tokenizer(missing_prompts, add_special_tokens=True)["input_ids"]
        for prompt, ids in zip(missing_prompts, encoded_prompts, strict=True):
            cache[prompt] = _ids(ids)
    if not prompts:
        return []
    joint_rows = tokenizer(
        [prompt + target for prompt, target in zip(prompts, targets, strict=True)],
        add_special_tokens=True,
    )["input_ids"]
    results: list[JointEncoding] = []
    for prompt, ids in zip(prompts, joint_rows, strict=True):
        prompt_ids = tuple(cache[prompt])
        joint_ids = _ids(ids)
        prefix_ok = len(joint_ids) > len(prompt_ids) and joint_ids[:len(prompt_ids)] == prompt_ids
        result = JointEncoding(prompt_ids, joint_ids, prefix_ok)
        if strict and not prefix_ok:
            raise PrefixViolationError("prompt token ids are not a prefix of jointly encoded prompt-target ids")
        results.append(result)
    return results


def ntok_build_rule(tokenizer: Any, form: str, prefix: str) -> int:
    """Return prep-pipeline contextual token-count increment for a canonical form."""
    # Ported from data/build/common.py contextual_tokenization at commit 873def8.
    prefix = unicodedata.normalize("NFC", prefix)
    form = unicodedata.normalize("NFC", form)
    if not prefix.strip():
        raise ValueError("Token context prefix must be non-empty")
    if not form.strip():
        raise ValueError("Cannot tokenize an empty canonical form")
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    contextual_ids = tokenizer.encode(f"{prefix} {form}", add_special_tokens=False)
    if len(contextual_ids) < len(prefix_ids):
        raise ValueError("Context encoding is shorter than prefix encoding")
    return len(contextual_ids) - len(prefix_ids)


def first_content_index(target_ids: Sequence[int], classifier: TokenClassifier) -> int | None:
    """Return the target-relative index of its first content token, if present."""
    for index, token_id in enumerate(target_ids):
        if classifier.is_content(int(token_id)):
            return index
    return None
