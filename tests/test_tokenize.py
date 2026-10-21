"""Tests for exact joint tokenization and token classifiers."""

from __future__ import annotations

from typing import Any

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from vilens.tokenize import (
    PrefixViolationError,
    TokenClassifier,
    encode_joint,
    encode_joint_batch,
    first_content_index,
    ntok_build_rule,
)


def byte_level_tokenizer() -> PreTrainedTokenizerFast:
    """Build the planner's byte-level test tokenizer entirely in memory."""
    backend = Tokenizer(models.BPE(
        vocab={char: index for index, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))},
        merges=[],
    ))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=backend)


def prefix_violation_tokenizer() -> PreTrainedTokenizerFast:
    """Build a tokenizer whose target merges with the prompt boundary."""
    backend = Tokenizer(models.BPE(
        vocab={"a": 0, "b": 1, ":": 2, "a:": 3, " ": 4},
        merges=[("a", ":")],
    ))
    return PreTrainedTokenizerFast(tokenizer_object=backend)


def test_content_classifier_and_byte_fragments() -> None:
    """Whitespace tokens are non-content and isolated UTF-8 bytes are flagged."""
    tokenizer = byte_level_tokenizer()
    encoding = encode_joint(tokenizer, "ab:", " cd")
    classifier = TokenClassifier(tokenizer)
    content = [classifier.is_content(token_id) for token_id in encoding.target_ids]
    assert content == [False, True, True]
    assert first_content_index(encoding.target_ids, classifier) == 1
    chinese_ids = tokenizer("千", add_special_tokens=True)["input_ids"]
    assert len(chinese_ids) == 3
    assert all(classifier.is_byte_fragment(token_id) for token_id in chinese_ids)


def test_joint_prefix_violation_strict_and_permissive() -> None:
    """A boundary merge raises in strict mode and is returned as a failed prefix."""
    tokenizer = prefix_violation_tokenizer()
    with pytest.raises(PrefixViolationError):
        encode_joint(tokenizer, "a", ":", strict=True)
    permissive = encode_joint(tokenizer, "a", ":", strict=False)
    assert permissive.prefix_ok is False
    assert permissive.target_ids == ()


@pytest.mark.parametrize("prompt", ["a ", "a\n"])
def test_prompt_trailing_whitespace_fails(prompt: str) -> None:
    """Reject prompts ending in any whitespace."""
    with pytest.raises(ValueError, match="ends in whitespace"):
        encode_joint(byte_level_tokenizer(), prompt, "b")


@pytest.mark.parametrize("target", ["", "  ", "\n"])
def test_empty_or_whitespace_target_fails(target: str) -> None:
    """Reject targets without content."""
    with pytest.raises(ValueError, match="target must not"):
        encode_joint(byte_level_tokenizer(), "a", target)


def test_joint_batch_matches_individual_encodings() -> None:
    """Batch output matches individual encodings and caches repeated prompts."""
    tokenizer = byte_level_tokenizer()
    prompts = ["ab:", "ab:", "x:", "c:", "d:"]
    targets = [" cd", " xy", " ab", " de", " fg"]
    cache: dict[str, tuple[int, ...]] = {}
    batch = encode_joint_batch(tokenizer, prompts, targets, prompt_cache=cache)
    individual = [encode_joint(tokenizer, prompt, target) for prompt, target in zip(prompts, targets, strict=True)]
    assert batch == individual
    assert set(cache) == set(prompts)
    assert len(cache) == len(set(prompts))


def test_ntok_build_rule_contextual_count() -> None:
    """Match the prep contextual-tokenization count on a byte-level tokenizer."""
    assert ntok_build_rule(byte_level_tokenizer(), "ab", prefix="x:") == 3


def test_first_content_index_returns_none_for_whitespace() -> None:
    """Return None if every decoded target token is whitespace."""
    tokenizer = byte_level_tokenizer()
    whitespace_ids = tokenizer(" ", add_special_tokens=False)["input_ids"]
    assert first_content_index(whitespace_ids, TokenClassifier(tokenizer)) is None
