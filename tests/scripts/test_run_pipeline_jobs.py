"""Tests for run_pipeline.py's config handling, command building and job graph.

The Slurm submission path is covered in test_run_pipeline_slurm.py. This file
covers everything upstream of it: path resolution, the config-section → argv
builders, and `_build_jobs`, which carries the transcriptome fan-out rules.

That fan-out validation is the part worth protecting. Its error messages exist
because the failure modes are silent: two groups sharing an `accessibility_dir`
collide on one directory and off-targets then falls back to energy-only
probabilities, producing plausible numbers computed without accessibility.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_PIPELINE = REPO_ROOT / "scripts" / "run_pipeline.py"


def _load():
    spec = importlib.util.spec_from_file_location("run_pipeline_jobs", RUN_PIPELINE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_pipeline_jobs"] = module
    spec.loader.exec_module(module)
    return module


rp = _load()


def _flag(argv: list[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv else None


class TestResolve:
    def test_relative_paths_resolve_against_the_config_directory(self, tmp_path):
        assert rp._resolve("data/x.fa", tmp_path) == str(tmp_path / "data" / "x.fa")

    def test_absolute_paths_are_left_alone(self, tmp_path):
        absolute = tmp_path / "already" / "absolute.fa"
        assert rp._resolve(str(absolute), Path("/some/other/base")) == str(absolute)

    def test_none_stays_none(self, tmp_path):
        assert rp._resolve(None, tmp_path) is None


class TestSlug:
    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("off-targets", "off_targets"),
            ("off-targets:human", "off_targets_human"),
            ("index", "index"),
        ],
    )
    def test_keys_become_filesystem_and_slurm_safe(self, key, expected):
        assert rp._slug(key) == expected


class TestBuildIndex:
    def test_target_is_required(self, tmp_path):
        with pytest.raises(ValueError, match="target"):
            rp._build_index({}, tmp_path)

    def test_builds_the_expected_command(self, tmp_path):
        cmd = rp._build_index({"target": "g.fa", "output": "g.idx"}, tmp_path)

        assert cmd[:2] == ["sioff", "index"]
        assert cmd[2] == str(tmp_path / "g.fa")
        assert _flag(cmd, "--output") == str(tmp_path / "g.idx")

    def test_verbose_is_opt_in(self, tmp_path):
        assert "--verbose" not in rp._build_index({"target": "g.fa"}, tmp_path)
        assert "--verbose" in rp._build_index(
            {"target": "g.fa", "verbose": True}, tmp_path
        )


class TestBuildAccessibility:
    @pytest.mark.parametrize(
        ("cfg", "missing"),
        [({}, "fasta"), ({"fasta": "g.fa"}, "output")],
    )
    def test_required_fields(self, tmp_path, cfg, missing):
        with pytest.raises(ValueError, match=missing):
            rp._build_accessibility(cfg, tmp_path)

    def test_optional_scalars_are_only_emitted_when_set(self, tmp_path):
        bare = rp._build_accessibility({"fasta": "g.fa", "output": "acc/"}, tmp_path)
        assert "--window" not in bare

        full = rp._build_accessibility(
            {
                "fasta": "g.fa",
                "output": "acc/",
                "window": 80,
                "temperature": 37.0,
                "strand": "+",
            },
            tmp_path,
        )
        assert _flag(full, "--window") == "80"
        assert _flag(full, "--temperature") == "37.0"
        assert _flag(full, "--strand") == "+"


class TestBuildOffTargets:
    def test_paths_resolve_and_scalars_stringify(self, tmp_path):
        cmd = rp._build_off_targets(
            {"risearch_file": "p.out", "transcriptome": "a.gtf", "alpha": 1.0},
            tmp_path,
        )

        assert _flag(cmd, "--risearch-file") == str(tmp_path / "p.out")
        assert _flag(cmd, "--transcriptome") == str(tmp_path / "a.gtf")
        assert _flag(cmd, "--alpha") == "1.0"

    def test_accessibility_cache_dir_is_forwarded_as_a_path(self, tmp_path):
        """The read-only -a error tells users to pass this flag; YAML must reach it."""
        assert "--accessibility-cache-dir" not in rp._build_off_targets({}, tmp_path)

        cmd = rp._build_off_targets(
            {"accessibility_dir": "acc/", "accessibility_cache_dir": "scratch/acc"},
            tmp_path,
        )
        assert _flag(cmd, "--accessibility-dir") == str(tmp_path / "acc")
        assert _flag(cmd, "--accessibility-cache-dir") == str(
            tmp_path / "scratch" / "acc"
        )

    def test_removed_off_targets_key_is_an_error(self, tmp_path):
        with pytest.raises(ValueError, match="'legacy_format' was removed"):
            rp._build_off_targets({"legacy_format": True}, tmp_path)

    def test_sirna_table_is_forwarded_and_on_target_ids_was_renamed(self, tmp_path):
        cmd = rp._build_off_targets({"sirna_table": "t.tsv"}, tmp_path)
        assert _flag(cmd, "--sirna-table") == str(tmp_path / "t.tsv")
        with pytest.raises(
            ValueError, match="'on_target_ids' was removed.*sirna_table"
        ):
            rp._build_off_targets({"on_target_ids": "m.tsv"}, tmp_path)

    def test_matrix_is_forwarded(self, tmp_path):
        """E_min's perfect-match energy uses -z; YAML must reach it."""
        assert "--matrix" not in rp._build_off_targets({}, tmp_path)
        # A bundled name passes through.
        cmd = rp._build_off_targets({"matrix": "slh04"}, tmp_path)
        assert _flag(cmd, "--matrix") == "slh04"
        # A table next to the config resolves against it, as sioff -c does.
        (tmp_path / "dsm").mkdir()
        (tmp_path / "dsm" / "custom.tsv").write_text("q1\tq2\tt1\tt2\tdg\n")
        cmd = rp._build_off_targets({"matrix": "dsm/custom.tsv"}, tmp_path)
        assert _flag(cmd, "--matrix") == str(tmp_path / "dsm" / "custom.tsv")
        cmd = rp._build_off_targets({"matrix": "/abs/custom.tsv"}, tmp_path)
        assert _flag(cmd, "--matrix") == "/abs/custom.tsv"

    def test_boolean_flags_are_bare_and_opt_in(self, tmp_path):
        assert "--sense-only" not in rp._build_off_targets({}, tmp_path)

        cmd = rp._build_off_targets({"sense_only": True}, tmp_path)
        assert "--sense-only" in cmd
        assert cmd[cmd.index("--sense-only") + 1 :] == []

    def test_a_false_boolean_is_not_emitted(self, tmp_path):
        """`sense_only: false` in YAML must not turn the flag on."""
        assert "--sense-only" not in rp._build_off_targets(
            {"sense_only": False}, tmp_path
        )


class TestBuildJobsSingleRun:
    def test_one_job_per_step_in_canonical_order(self, tmp_path):
        cfg = {
            "accessibility": {"fasta": "g.fa", "output": "acc/"},
            "off_targets": {"risearch_file": "p.out"},
        }

        jobs = rp._build_jobs(cfg, ["accessibility", "off-targets"], tmp_path)

        assert [j.key for j in jobs] == ["accessibility", "off-targets"]

    def test_off_targets_waits_for_accessibility(self, tmp_path):
        cfg = {
            "accessibility": {"fasta": "g.fa", "output": "acc/"},
            "off_targets": {"risearch_file": "p.out"},
        }

        jobs = rp._build_jobs(cfg, ["accessibility", "off-targets"], tmp_path)

        assert jobs[0].dep_keys == []
        assert jobs[1].dep_keys == ["accessibility"]

    def test_dependencies_on_steps_not_in_the_run_are_dropped(self, tmp_path):
        """Running off-targets alone must not wait on a job that was never submitted."""
        jobs = rp._build_jobs(
            {"off_targets": {"risearch_file": "p.out"}}, ["off-targets"], tmp_path
        )

        assert jobs[0].dep_keys == []


class TestBuildJobsFanOut:
    @staticmethod
    def _cfg(**overrides):
        base = {
            "off_targets": {"transcriptome": "a.gtf"},
            "transcriptomes": [
                {
                    "name": "human",
                    "risearch_file": "h.out",
                    "output": "h.tsv",
                },
                {
                    "name": "mouse",
                    "risearch_file": "m.out",
                    "output": "m.tsv",
                },
            ],
        }
        base.update(overrides)
        return base

    def test_one_job_per_group_and_step(self, tmp_path):
        jobs = rp._build_jobs(self._cfg(), ["off-targets"], tmp_path)

        assert [j.key for j in jobs] == ["off-targets:human", "off-targets:mouse"]

    def test_dependencies_stay_inside_their_group(self, tmp_path):
        cfg = self._cfg(accessibility={})
        for group in cfg["transcriptomes"]:
            group["fasta"] = f"{group['name']}.fa"
            group["accessibility_dir"] = f"acc/{group['name']}"

        jobs = rp._build_jobs(cfg, ["accessibility", "off-targets"], tmp_path)
        deps = {j.key: j.dep_keys for j in jobs}

        assert deps["off-targets:human"] == ["accessibility:human"]
        assert deps["off-targets:mouse"] == ["accessibility:mouse"]

    def test_index_is_not_fanned_out(self, tmp_path):
        cfg = self._cfg(index={"target": "g.fa"})

        jobs = rp._build_jobs(cfg, ["index", "off-targets"], tmp_path)

        assert [j.key for j in jobs if j.step == "index"] == ["index"]

    def test_groups_must_be_a_list(self, tmp_path):
        with pytest.raises(ValueError, match="must be a list"):
            rp._build_jobs(
                self._cfg(transcriptomes={"name": "human"}), ["off-targets"], tmp_path
            )

    def test_every_group_needs_a_name(self, tmp_path):
        cfg = self._cfg(transcriptomes=[{"risearch_file": "h.out"}])

        with pytest.raises(ValueError, match="needs a 'name'"):
            rp._build_jobs(cfg, ["off-targets"], tmp_path)

    def test_group_names_must_be_unique(self, tmp_path):
        cfg = self._cfg(
            transcriptomes=[
                {"name": "human", "risearch_file": "a.out", "output": "a.tsv"},
                {"name": "human", "risearch_file": "b.out", "output": "b.tsv"},
            ]
        )

        with pytest.raises(ValueError, match="duplicate transcriptome names"):
            rp._build_jobs(cfg, ["off-targets"], tmp_path)

    def test_every_group_needs_its_own_predictions(self, tmp_path):
        cfg = self._cfg(transcriptomes=[{"name": "human", "output": "h.tsv"}])

        with pytest.raises(ValueError, match="risearch_file"):
            rp._build_jobs(cfg, ["off-targets"], tmp_path)

    def test_outputs_must_be_unique_so_groups_cannot_overwrite_each_other(
        self, tmp_path
    ):
        cfg = self._cfg(
            transcriptomes=[
                {"name": "human", "risearch_file": "h.out", "output": "same.tsv"},
                {"name": "mouse", "risearch_file": "m.out", "output": "same.tsv"},
            ]
        )

        with pytest.raises(ValueError, match="must be unique"):
            rp._build_jobs(cfg, ["off-targets"], tmp_path)

    def test_default_output_is_per_group_so_the_default_never_collides(self, tmp_path):
        cfg = self._cfg(
            transcriptomes=[
                {"name": "human", "risearch_file": "h.out"},
                {"name": "mouse", "risearch_file": "m.out"},
            ]
        )

        jobs = rp._build_jobs(cfg, ["off-targets"], tmp_path)

        outputs = [_flag(j.cmd, "--output") for j in jobs]
        assert outputs == [
            str(tmp_path / "results" / "human.tsv"),
            str(tmp_path / "results" / "mouse.tsv"),
        ]

    @pytest.mark.parametrize("omitted", ["fasta", "accessibility_dir"])
    def test_computing_accessibility_requires_per_group_fasta_and_dir(
        self, tmp_path, omitted
    ):
        """Without a group-unique accessibility_dir the groups share one directory.

        off-targets would then silently fall back to energy-only probabilities —
        numbers that look fine but were computed without accessibility.
        """
        group = {
            "name": "human",
            "risearch_file": "h.out",
            "output": "h.tsv",
            "fasta": "h.fa",
            "accessibility_dir": "acc/human",
        }
        del group[omitted]
        cfg = self._cfg(accessibility={}, transcriptomes=[group])

        with pytest.raises(ValueError, match=omitted):
            rp._build_jobs(cfg, ["accessibility", "off-targets"], tmp_path)


class TestGroupStepConfig:
    def test_group_values_override_the_shared_defaults(self):
        merged = rp._group_step_cfg(
            "off-targets",
            {"transcriptome": "shared.gtf", "alpha": 1.0},
            {"name": "human", "transcriptome": "human.gtf"},
        )

        assert merged["transcriptome"] == "human.gtf"
        assert merged["alpha"] == 1.0

    def test_accessibility_only_keys_do_not_leak_into_off_targets(self):
        """`fasta` configures accessibility; passing it through would change the run."""
        merged = rp._group_step_cfg(
            "off-targets", {}, {"name": "human", "fasta": "human.fa"}
        )

        assert "fasta" not in merged
        assert "name" not in merged

    def test_accessibility_takes_its_output_from_the_groups_handoff_path(self):
        merged = rp._group_step_cfg(
            "accessibility",
            {"output": "shared/"},
            {"name": "human", "fasta": "human.fa", "accessibility_dir": "acc/human"},
        )

        assert merged["fasta"] == "human.fa"
        assert merged["output"] == "acc/human"


class TestLoadConfig:
    def test_empty_file_yields_an_empty_dict(self, tmp_path):
        config = tmp_path / "empty.yaml"
        config.write_text("")

        assert rp._load_config(config) == {}
