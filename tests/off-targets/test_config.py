"""Tests for config module."""

import inspect

import pytest
from omegaconf import MissingMandatoryValue
from sioff.commands.accessibility import run as accessibility_run
from sioff.commands.off_targets import run as off_targets_run
from sioff.config import load_config, config_to_kwargs


class TestLoadConfig:
    def test_removed_accessibility_strategy_key_is_rejected(self, tmp_path):
        from omegaconf.errors import ConfigKeyError

        cfg = tmp_path / "c.yaml"
        cfg.write_text(
            "command: off-targets\noff_targets:\n  risearch_file: x.out\n"
            "  accessibility_strategy: memmap\n"
        )
        with pytest.raises(ConfigKeyError, match="accessibility_strategy"):
            load_config(cfg)

    def test_removed_legacy_format_key_names_the_key(self, tmp_path):
        cfg = tmp_path / "c.yaml"
        cfg.write_text(
            "command: off-targets\noff_targets:\n  risearch_file: x.out\n"
            "  legacy_format: true\n"
        )
        with pytest.raises(ValueError, match="'legacy_format' was removed"):
            load_config(cfg)

    @pytest.mark.parametrize(
        "key",
        [
            "on_target",
            "on_target_risearch_file",
            "query",
            "on_target_expression",
            "on_target_accessibility",
        ],
    )
    def test_removed_on_target_keys_name_the_key(self, tmp_path, key):
        cfg = tmp_path / "c.yaml"
        cfg.write_text(
            f"command: off-targets\noff_targets:\n  risearch_file: x.out\n  {key}: x\n"
        )
        with pytest.raises(ValueError, match=f"'{key}' was removed"):
            load_config(cfg)

    def test_load_off_targets_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        # Create dummy path for risearch_file
        (tmp_path / "test.tsv").touch()

        config_file.write_text("""
command: off-targets
off_targets:
  risearch_file: test.tsv
  output: results.tsv
""")
        cfg = load_config(config_file)
        assert cfg.command == "off-targets"
        # Path resolution check
        assert cfg.off_targets.risearch_file == str(tmp_path / "test.tsv")
        assert cfg.off_targets.output == str(tmp_path / "results.tsv")

    def test_load_accessibility_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
command: accessibility
accessibility:
  fasta: genome.fa
  output: output_dir
""")
        cfg = load_config(config_file)
        assert cfg.command == "accessibility"
        assert cfg.accessibility.fasta == str(tmp_path / "genome.fa")

    def test_missing_required_field(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
command: off-targets
off_targets: {}
""")
        # load_config triggers validation when resolving paths, so error is raised there
        with pytest.raises(MissingMandatoryValue):
            load_config(config_file)

    def test_invalid_command(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("command: invalid")
        with pytest.raises(ValueError, match="Unknown command"):
            load_config(config_file)

    def test_config_to_kwargs_mappings(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
command: off-targets
off_targets:
  risearch_file: test.tsv
  transcriptome: ann.gtf
  feature: CDS
""")
        cfg = load_config(config_file)
        kwargs = config_to_kwargs(cfg, "off-targets")

        assert "gtf_file" in kwargs
        assert kwargs["gtf_file"] == tmp_path / "ann.gtf"
        assert kwargs["feature_type"] == "CDS"
        assert "transcriptome" not in kwargs

    def test_config_to_kwargs_accessibility(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
command: accessibility
accessibility:
  fasta: genome.fa
  output: out
""")
        cfg = load_config(config_file)
        kwargs = config_to_kwargs(cfg, "accessibility")

        assert "genome" in kwargs
        assert kwargs["genome"] == tmp_path / "genome.fa"
        assert "output" in kwargs
        assert "fasta" not in kwargs

    @pytest.mark.parametrize(
        ("command", "yaml_body", "run_fn"),
        [
            (
                "accessibility",
                "accessibility:\n  fasta: genome.fa\n  output: out\n",
                accessibility_run,
            ),
            (
                "off-targets",
                "off_targets:\n  risearch_file: test.tsv\n  transcriptome: ann.gtf\n",
                off_targets_run,
            ),
        ],
    )
    def test_config_to_kwargs_binds_to_command_signature(
        self, tmp_path, command, yaml_body, run_fn
    ):
        """Every key config_to_kwargs emits must be a real parameter of the command.

        Regression guard: this previously asserted on the returned dict only, so a
        mapping of `output` -> `output_dir` passed the test while making
        `sioff -c <accessibility config>` fail with TypeError on every run. Binding
        against the real signature is what actually catches that.
        """
        config_file = tmp_path / "config.yaml"
        config_file.write_text(f"command: {command}\n{yaml_body}")

        kwargs = config_to_kwargs(load_config(config_file), command)

        # Raises TypeError if any key is not a parameter of the target function.
        inspect.signature(run_fn).bind_partial(**kwargs)
