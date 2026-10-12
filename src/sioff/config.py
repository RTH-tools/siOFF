"""Configuration management using OmegaConf."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast

from omegaconf import MISSING, OmegaConf, DictConfig

from sioff.models import PredictionsMode


@dataclass
class OffTargetsConfig:
    """Schema for off-targets command configuration."""

    risearch_file: str = MISSING  # Required
    transcriptome: Optional[str] = None
    transcriptome_format: str = "auto"
    feature: str = "exon"
    expression_metric: str = "RPKM"
    accessibility_dir: Optional[str] = None
    accessibility_cache_dir: Optional[str] = None
    output: Optional[str] = None
    fasta: Optional[str] = None
    window: int = 80
    span: int = 40
    unpaired: int = 30
    on_target: Optional[str] = None
    on_target_risearch_file: Optional[str] = None
    query: Optional[str] = None
    on_target_expression: float = 1000.0
    on_target_accessibility: Optional[str] = None
    temperature: float = 37.0
    alpha: str = "1.0"
    gamma: str = "1.0"
    theta: str = ""
    legacy_format: bool = False
    detailed_report: bool = False
    sense_only: bool = False
    type: str = PredictionsMode.GW
    verbose: bool = False


@dataclass
class AccessibilityConfig:
    """Schema for accessibility command configuration."""

    fasta: str = MISSING  # Required
    output: str = MISSING  # Required
    window: int = 80
    span: int = 40
    unpaired: int = 30
    temperature: float = 37.0
    verbose: bool = False


@dataclass
class PipelineConfig:
    """Top-level config schema."""

    command: str = MISSING  # "off-targets" or "accessibility"
    verbose: bool = False
    off_targets: Optional[OffTargetsConfig] = None
    accessibility: Optional[AccessibilityConfig] = None


def load_config(config_path: Path) -> DictConfig:
    """Load and validate a YAML config file, resolving relative paths."""
    schema = OmegaConf.structured(PipelineConfig)
    # merge() is typed DictConfig | ListConfig; merging into a structured
    # (dict) schema always yields a DictConfig.
    cfg = cast(DictConfig, OmegaConf.merge(schema, OmegaConf.load(config_path)))

    if cfg.command not in ("off-targets", "accessibility"):
        raise ValueError(
            f"Unknown command: {cfg.command}. Must be 'off-targets' or 'accessibility'."
        )

    return _resolve_paths(cfg, config_path.parent.resolve())


def _resolve_paths(cfg: DictConfig, base_dir: Path) -> DictConfig:
    """Resolve relative path fields in the config relative to base_dir."""
    path_fields = {
        "off_targets": [
            "risearch_file",
            "transcriptome",
            "accessibility_dir",
            "accessibility_cache_dir",
            "output",
            "fasta",
            "on_target",
            "on_target_risearch_file",
            "query",
            "on_target_accessibility",
        ],
        "accessibility": ["fasta", "output"],
    }

    for section, fields in path_fields.items():
        section_cfg = getattr(cfg, section, None)
        if section_cfg is None:
            continue
        for field_name in fields:
            value = getattr(section_cfg, field_name, None)
            if value is not None and isinstance(value, str):
                p = Path(value)
                if not p.is_absolute():
                    resolved = (base_dir / p).resolve()
                    OmegaConf.update(cfg, f"{section}.{field_name}", str(resolved))

    return cfg


def config_to_kwargs(cfg: DictConfig, command: str) -> dict:
    """Convert OmegaConf section to kwargs dict for the given command function."""
    section = getattr(cfg, command.replace("-", "_"))
    # section is a DictConfig, so to_container() returns a plain dict here.
    kwargs = cast(dict[str, Any], OmegaConf.to_container(section, resolve=True))

    for key in [
        "risearch_file",
        "transcriptome",
        "accessibility_dir",
        "accessibility_cache_dir",
        "output",
        "fasta",
        "on_target",
        "on_target_risearch_file",
        "query",
        "on_target_accessibility",
    ]:
        if key in kwargs and kwargs[key] is not None:
            kwargs[key] = Path(kwargs[key])

    if "type" in kwargs:
        kwargs["predictions_type"] = kwargs.pop("type")

    key_mapping = {
        "window": "window_size",
        "span": "max_span",
        "unpaired": "unpaired_prob",
    }

    if command == "off-targets":
        key_mapping.update(
            {
                "transcriptome": "gtf_file",
                "feature": "feature_type",
                "output": "output_file",
                "fasta": "genome_file",
                "on_target": "on_target_file",
                "query": "query_file",
            }
        )
        # Typer OptionInfo objects become defaults when calling the function directly,
        # not via CLI — set these to None so callers get clean None defaults.
        for arg in (
            "sirna_fasta",
            "target_fasta",
            "target_index",
            "workers",
            "on_target_ids_file",
        ):
            kwargs.setdefault(arg, None)

    elif command == "accessibility":
        # NOTE: keys here must match the *parameter names* of the target command
        # function, not the CLI flag names. `output` is deliberately absent —
        # accessibility.run() takes `output`, so remapping it to `output_dir`
        # raised TypeError on every `sioff -c <accessibility config>` invocation.
        # test_config_to_kwargs_binds_to_command_signature guards this.
        key_mapping.update(
            {
                "fasta": "genome",
                "temperature": "temperature",
            }
        )

    for old_key, new_key in key_mapping.items():
        if old_key in kwargs:
            kwargs[new_key] = kwargs.pop(old_key)

    return kwargs
