from pathlib import Path


def test_analyze_e7_wrapper_resolves_paths_from_repo_root(monkeypatch, tmp_path):
    import job_analyze_e7

    project_root = tmp_path / "project"
    current_directory = tmp_path / "outside"
    current_directory.mkdir()
    monkeypatch.chdir(current_directory)

    resolved = job_analyze_e7.resolve_repo_path(
        "configs/experiments.yaml", repo_root=project_root
    )

    assert resolved == (project_root / "configs/experiments.yaml").resolve()
    assert Path.cwd() == current_directory
