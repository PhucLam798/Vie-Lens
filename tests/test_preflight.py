"""CPU-only tests for the shared job preflight sequence."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from vilens.config import load_config
from vilens.preflight import preflight_ok, run_preflight
from conftest import make_tiny_model

ROOT = Path(__file__).resolve().parents[1]
REVISION = "0123456789abcdef0123456789abcdef01234567"


def _git(repo: Path, *args: str) -> None:
    """Run a Git command with a local test identity."""
    env = os.environ.copy()
    env.update({"GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@example.invalid"})
    env.update({"GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "test@example.invalid"})
    subprocess.run(["git", *args], cwd=repo, env=env, check=True, capture_output=True, text=True)


def _clean_repo(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """Create a clean small repository with a valid Vie-Lens config and empty manifest."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "configs").mkdir()
    (repo / "src").mkdir()
    (repo / "derived").mkdir()
    (repo / "data").mkdir()
    (repo / "configs/experiments.yaml").write_text(
        (ROOT / "configs/experiments.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (repo / "data/MANIFEST.sha256").write_text("", encoding="utf-8")
    _git(repo, "add", "configs", "data")
    _git(repo, "commit", "-qm", "test repo")
    cfg = load_config(repo / "configs/experiments.yaml")
    return repo, cfg


def _fake_hf_cache(cache: Path) -> tuple[str, str]:
    """Create a complete local tiny tokenizer snapshot for preflight-only loading."""
    hf_id = "org/tiny"
    repo = cache / "models--org--tiny"
    blobs = repo / "blobs"
    snapshot = repo / "snapshots" / REVISION
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    tokenizer_source = cache / "tokenizer-source"
    tokenizer_source.mkdir()
    tiny_model = make_tiny_model("llama")
    tiny_model.save_pretrained(tokenizer_source)
    backend = Tokenizer(
        models.BPE(vocab={c: i for i, c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}, merges=[])
    )
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    tokenizer.save_pretrained(tokenizer_source)
    del tiny_model
    files = [path for path in tokenizer_source.iterdir() if path.is_file()]
    for path in files:
        contents = path.read_bytes()
        blob_name = hashlib.sha256(contents).hexdigest()
        (blobs / blob_name).write_bytes(contents)
        (snapshot / path.name).symlink_to(Path("..") / ".." / "blobs" / blob_name)
    (repo / "refs").mkdir()
    (repo / "refs/main").write_text(REVISION, encoding="utf-8")
    return hf_id, REVISION


def _preflight(repo: Path, cfg: dict[str, Any], **kwargs: Any):
    """Call the shared preflight with compact CPU test defaults."""
    return run_preflight(
        cfg,
        repo_root=repo,
        step=kwargs.pop("step", "E0"),
        gpu=False,
        inputs=[],
        families=kwargs.pop("families", ["llama"]),
        records=kwargs.pop("records", None),
        output_dirs=kwargs.pop("output_dirs", []),
        entrypoint=None,
    )


def test_login_preflight_skips_gpu_and_blocks_undecided_models(tmp_path: Path) -> None:
    """Skip GPU checks on CPU and block snapshot loading while D10 is null."""
    repo, cfg = _clean_repo(tmp_path)
    cfg["models"]["study_set"] = None
    checks = _preflight(repo, cfg)
    assert [check.status for check in checks[:2]] == ["SKIPPED", "SKIPPED"]
    snapshot = next(check for check in checks if check.name == "snapshots")
    assert snapshot.status == "BLOCKED"
    assert "D10 undecided" in snapshot.detail
    assert preflight_ok(checks) is False


def test_preflight_fake_local_snapshot_passes(tmp_path: Path) -> None:
    """Load only a fast tokenizer and HF config from a complete local fake cache."""
    repo, cfg = _clean_repo(tmp_path)
    cache = tmp_path / "model-cache"
    hf_id, revision = _fake_hf_cache(cache)
    cfg["models"]["study_set"] = "base"
    cfg["models"]["base"]["llama"] = {"hf_id": hf_id, "revision": revision}
    cfg["paths"]["model_cache"] = str(cache)
    checks = _preflight(repo, cfg)
    snapshot = next(check for check in checks if check.name == "snapshots")
    assert snapshot.status == "PASS", snapshot.detail


def test_preflight_unwritable_output_fails(tmp_path: Path) -> None:
    """Report failure when the output directory cannot be probed for writes."""
    repo, cfg = _clean_repo(tmp_path)
    output = repo / "locked-output"
    output.mkdir()
    output.chmod(0o500)
    try:
        checks = _preflight(repo, cfg, output_dirs=[Path("locked-output")])
    finally:
        output.chmod(0o700)
    assert next(check for check in checks if check.name == "output_dirs_writable").status == "FAIL"


def test_preflight_dirty_repo_fails_git_clean(tmp_path: Path) -> None:
    """Detect dirty source paths in the configured Git clean set."""
    repo, cfg = _clean_repo(tmp_path)
    (repo / "src/new.py").write_text("x = 1\n", encoding="utf-8")
    checks = _preflight(repo, cfg)
    assert next(check for check in checks if check.name == "git_clean").status == "FAIL"


def test_preflight_model_lock_handles_test_and_directions_records(tmp_path: Path) -> None:
    """Use explicit lock values to check test and directions access across lock states."""
    repo, cfg = _clean_repo(tmp_path)
    test_records = [{"split": "test"}]
    directions_records = [{"split": "directions"}]

    cfg["prereg"] = {"analysis_tag": "prereg-v1.4", "test_items_unlocked": False}
    test_checks = _preflight(repo, cfg, records=test_records)
    direction_checks = _preflight(repo, cfg, records=directions_records)
    assert next(check for check in test_checks if check.name == "integrity_locks").status == "FAIL"
    assert next(check for check in direction_checks if check.name == "integrity_locks").status == "PASS"

    cfg["prereg"] = {"analysis_tag": "prereg-v1.4", "test_items_unlocked": True}
    test_checks = _preflight(repo, cfg, records=test_records)
    direction_checks = _preflight(repo, cfg, records=directions_records)
    assert next(check for check in test_checks if check.name == "integrity_locks").status == "PASS"
    assert next(check for check in direction_checks if check.name == "integrity_locks").status == "PASS"

    cfg["prereg"] = {"analysis_tag": None, "test_items_unlocked": True}
    test_checks = _preflight(repo, cfg, records=test_records)
    assert next(check for check in test_checks if check.name == "integrity_locks").status == "FAIL"


def test_preflight_e8_requires_committed_adjudication(tmp_path: Path) -> None:
    """Require the adjudication report before E8 preflight can pass."""
    repo, cfg = _clean_repo(tmp_path)
    checks = _preflight(repo, cfg, step="E8")
    lock = next(check for check in checks if check.name == "integrity_locks")
    assert lock.status == "FAIL"
    assert "adjudication.md" in lock.detail
