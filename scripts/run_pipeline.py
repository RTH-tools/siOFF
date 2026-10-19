#!/usr/bin/env python3
"""
Orchestrator for the siOFF siRNA off-target discovery pipeline.

Runs pipeline stages in dependency order:
  1. (optional) index         -- sioff index
  2.            accessibility -- sioff accessibility
  3.            off-targets   -- sioff off-targets

Local mode (default): steps run sequentially; logs go to logs/<timestamp>/<step>.log.
Slurm mode (--slurm): each step submitted with sbatch, chained via --dependency=afterok.
Dry-run (--dry-run): prints commands without executing; Slurm dry-run shows full sbatch
  commands with <JOBID_stepname> placeholders so the dependency chain is visible.

Config format (YAML):
  steps: [accessibility, off-targets]   # default; add index to enable it

  # Per-step Slurm resource overrides (merged with built-in defaults)
  slurm:
    partition: batch
    account: myaccount
    accessibility:
      time: "08:00:00"
      mem: 32G
      cpus_per_task: 8
    off_targets:
      time: "04:00:00"
      mem: 64G
      cpus_per_task: 16

  index:
    target: data/genome.fa          # required
    output: data/genome.idx         # optional

  accessibility:
    fasta: data/genome.fa           # required
    output: data/accessibility/     # required
    window: 80
    span: 40
    unpaired: 30
    temperature: 37.0
    workers: 8

  off_targets:
    risearch_file: data/parquet/    # required (file or parquet directory)
    transcriptome: data/anno.gtf    # optional
    accessibility_dir: data/accessibility/
    output: results/off_targets.tsv
    alpha: "0.8;1.0"
    gamma: "1.0"
    type: gw

Multi-transcriptome fan-out (Slurm-native, one transcriptome per node):
  Add a top-level `transcriptomes:` list to analyze several genomes at once.
  Each entry is an independent run: its own risearch_file/transcriptome/output,
  its own Slurm job(s). The top-level off_targets/accessibility blocks
  become shared defaults; each group overrides the per-group fields. `index` is
  never fanned out. Off-targets math (Z_s) is unchanged — groups never mix.

  off_targets:                      # shared defaults for every group
    alpha: "0.8;1.0"
    type: gw
  transcriptomes:
    - name: human
      risearch_file: data/human.out
      transcriptome: data/human.gtf
      accessibility_dir: data/human_acc/   # precomputed
      output: results/human.tsv
    - name: mouse
      risearch_file: data/mouse.out
      transcriptome: data/mouse.gtf
      accessibility_dir: data/mouse_acc/
      output: results/mouse.tsv
"""

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml


STEP_ORDER = ["index", "accessibility", "off-targets"]
DEFAULT_STEPS = ["accessibility", "off-targets"]

# Steps that fan out to one job per transcriptome group. index is genome-index
# construction and stays a single global job even in fan-out mode.
FANOUT_STEPS = {"accessibility", "off-targets"}


@dataclass
class Job:
    """A single unit of work: one CLI invocation submitted as one Slurm job.

    In single-run mode ``key == step``. In transcriptome fan-out mode
    ``key == f"{step}:{group_name}"`` and ``dep_keys`` holds the keys of the
    upstream jobs (same group) this one waits on via ``--dependency=afterok``.
    """

    key: str
    step: str
    cmd: list[str]
    dep_keys: list[str] = field(default_factory=list)


# Built-in per-step Slurm resource defaults.
_DEFAULT_RESOURCES: dict[str, dict] = {
    "index": {"time": "01:00:00", "mem": "8G", "cpus_per_task": 4},
    "accessibility": {"time": "08:00:00", "mem": "32G", "cpus_per_task": 8},
    "off-targets": {"time": "04:00:00", "mem": "64G", "cpus_per_task": 16},
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def _resolve(value: str | None, base: Path) -> str | None:
    if value is None:
        return None
    p = Path(value)
    return str((base / p).resolve() if not p.is_absolute() else p)


# ---------------------------------------------------------------------------
# Command builders: config section dict → CLI argv list
# ---------------------------------------------------------------------------


def _build_index(cfg: dict, base: Path) -> list[str]:
    target = _resolve(cfg.get("target"), base)
    if not target:
        raise ValueError("'target' is required")
    cmd = ["sioff", "index", target]
    output = _resolve(cfg.get("output"), base)
    if output:
        cmd += ["--output", output]
    if cfg.get("verbose"):
        cmd.append("--verbose")
    return cmd


def _build_accessibility(cfg: dict, base: Path) -> list[str]:
    fasta = _resolve(cfg.get("fasta"), base)
    if not fasta:
        raise ValueError("'fasta' is required")
    output = _resolve(cfg.get("output"), base)
    if not output:
        raise ValueError("'output' is required")
    cmd = ["sioff", "accessibility", "--fasta", fasta, "--output", output]
    for key, flag in [
        ("window", "--window"),
        ("span", "--span"),
        ("unpaired", "--unpaired"),
        ("temperature", "--temperature"),
        ("workers", "--workers"),
        ("output_format", "--output-format"),
    ]:
        if key in cfg and cfg[key] is not None:
            cmd += [flag, str(cfg[key])]
    if cfg.get("verbose"):
        cmd.append("--verbose")
    return cmd


def _build_off_targets(cfg: dict, base: Path) -> list[str]:
    cmd = ["sioff", "off-targets"]

    # Path options
    for key, flag in [
        ("risearch_file", "--risearch-file"),
        ("sirna_fasta", "--sirna-fasta"),
        ("target_fasta", "--target-fasta"),
        ("target_index", "--target-index"),
        ("transcriptome", "--transcriptome"),
        ("accessibility_dir", "--accessibility-dir"),
        ("output", "--output"),
        ("fasta", "--fasta"),
        ("on_target", "--on-target"),
        ("on_target_risearch_file", "--on-target-risearch-file"),
        ("query", "--query"),
        ("on_target_accessibility", "--on-target-accessibility"),
        ("on_target_ids", "--on-target-ids"),
    ]:
        val = _resolve(cfg.get(key), base)
        if val is not None:
            cmd += [flag, val]

    # Scalar options
    for key, flag in [
        ("workers", "--workers"),
        ("feature", "--feature"),
        ("expression_metric", "--expression-metric"),
        ("transcriptome_format", "--transcriptome-format"),
        ("window", "--window"),
        ("span", "--span"),
        ("unpaired", "--unpaired"),
        ("temperature", "--temperature"),
        ("on_target_expression", "--on-target-expression"),
        ("alpha", "--alpha"),
        ("gamma", "--gamma"),
        ("theta", "--theta"),
        ("type", "--type"),
        ("output_format", "--output-format"),
    ]:
        if key in cfg and cfg[key] is not None:
            cmd += [flag, str(cfg[key])]

    # Boolean flags
    for key, flag in [
        ("legacy_format", "--legacy-format"),
        ("detailed_report", "--detailed-report"),
        ("sense_only", "--sense-only"),
        ("verbose", "--verbose"),
        ("profile", "--profile"),
        ("summary_only", "--summary-only"),
    ]:
        if cfg.get(key):
            cmd.append(flag)

    return cmd


_BUILDERS = {
    "index": _build_index,
    "accessibility": _build_accessibility,
    "off-targets": _build_off_targets,
}

_CFG_KEYS = {
    "index": "index",
    "accessibility": "accessibility",
    "off-targets": "off_targets",
}

# Dependency graph: index and accessibility are independent of each other.
# off-targets waits for accessibility.
_DEPS: dict[str, list[str]] = {
    "index": [],
    "accessibility": [],
    "off-targets": ["accessibility"],
}


# ---------------------------------------------------------------------------
# Job assembly: config → list[Job]  (single-run or transcriptome fan-out)
# ---------------------------------------------------------------------------

# Group keys that configure the accessibility step rather than off-targets;
# excluded when a group entry is merged into the off_targets block.
_ACCESSIBILITY_ONLY_GROUP_KEYS = {"name", "fasta"}


def _group_step_cfg(step: str, global_cfg: dict, group: dict) -> dict:
    """Assemble the per-group config dict for one step.

    Top-level ``off_targets:`` / ``accessibility:`` blocks are
    shared defaults; the group entry overrides them. ``accessibility_dir`` is
    the handoff path: it is the accessibility step's ``output`` (where profiles
    are written) and the off-targets step's ``accessibility_dir`` (where they
    are read), so a computed profile feeds straight into that group's analysis.
    """
    merged = dict(global_cfg or {})
    if step == "off-targets":
        for k, v in group.items():
            if k in _ACCESSIBILITY_ONLY_GROUP_KEYS:
                continue
            merged[k] = v
    elif step == "accessibility":
        if group.get("fasta"):
            merged["fasta"] = group["fasta"]
        if group.get("accessibility_dir"):
            merged["output"] = group["accessibility_dir"]
    return merged


def _build_jobs(cfg: dict, steps: list[str], base: Path) -> list[Job]:
    """Build the job list, fanning out per transcriptome group when configured."""
    groups = cfg.get("transcriptomes")

    if not groups:
        # Single-run mode: one job per step, unchanged from the original design.
        jobs: list[Job] = []
        for step in steps:
            step_cfg = cfg.get(_CFG_KEYS[step]) or {}
            cmd = _BUILDERS[step](step_cfg, base)
            dep_keys = [d for d in _DEPS.get(step, []) if d in steps]
            jobs.append(Job(key=step, step=step, cmd=cmd, dep_keys=dep_keys))
        return jobs

    # Fan-out mode: validate groups, then emit one job per (group, step).
    if not isinstance(groups, list):
        raise ValueError("'transcriptomes' must be a list of group entries")
    names: list[str] = []
    for i, group in enumerate(groups):
        if not isinstance(group, dict) or not group.get("name"):
            raise ValueError(f"transcriptomes[{i}] needs a 'name'")
        names.append(group["name"])
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate transcriptome names: {names}")

    # Assign a default per-group output before validating uniqueness.
    global_off = cfg.get("off_targets") or {}
    compute_acc = "accessibility" in steps
    for group in groups:
        merged_off = {
            **global_off,
            **{
                k: v
                for k, v in group.items()
                if k not in _ACCESSIBILITY_ONLY_GROUP_KEYS
            },
        }
        if not merged_off.get("risearch_file"):
            raise ValueError(f"transcriptome '{group['name']}' needs a 'risearch_file'")
        # When accessibility is computed per group, each group must name its own
        # fasta AND a group-unique accessibility_dir (the handoff path). Without
        # the latter, groups would collide on the shared global output dir and
        # off-targets would silently fall back to energy-only probabilities.
        if compute_acc:
            if not group.get("fasta"):
                raise ValueError(
                    f"transcriptome '{group['name']}': 'fasta' is required when "
                    f"'accessibility' is in steps"
                )
            if not group.get("accessibility_dir"):
                raise ValueError(
                    f"transcriptome '{group['name']}': 'accessibility_dir' is required "
                    f"when 'accessibility' is in steps (where profiles are written and read)"
                )
        group.setdefault("output", f"results/{group['name']}.tsv")

    outputs = [_resolve(g["output"], base) for g in groups]
    if len(set(outputs)) != len(outputs):
        raise ValueError(f"transcriptome 'output' paths must be unique, got: {outputs}")

    jobs = []
    for step in steps:
        if step not in FANOUT_STEPS:
            # index stays a single global job.
            step_cfg = cfg.get(_CFG_KEYS[step]) or {}
            cmd = _BUILDERS[step](step_cfg, base)
            jobs.append(Job(key=step, step=step, cmd=cmd, dep_keys=[]))
            continue
        global_cfg = cfg.get(_CFG_KEYS[step]) or {}
        for group in groups:
            name = group["name"]
            step_cfg = _group_step_cfg(step, global_cfg, group)
            cmd = _BUILDERS[step](step_cfg, base)
            dep_keys = []
            for d in _DEPS.get(step, []):
                if d not in steps:
                    continue
                dep_keys.append(d if d not in FANOUT_STEPS else f"{d}:{name}")
            jobs.append(
                Job(key=f"{step}:{name}", step=step, cmd=cmd, dep_keys=dep_keys)
            )
    return jobs


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _slug(key: str) -> str:
    """Filesystem-/Slurm-safe token from a job key (e.g. 'off-targets:human')."""
    return key.replace("-", "_").replace(":", "_")


def _run_local(jobs: list[Job], log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        log_file = log_dir / f"{_slug(job.key)}.log"
        print(f"\n[{job.key}] {shlex.join(job.cmd)}")
        print(f"[{job.key}] log → {log_file}")
        with log_file.open("w") as fh:
            fh.write(f"CMD: {shlex.join(job.cmd)}\n\n")
            fh.flush()
            rc = subprocess.run(job.cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            print(f"[{job.key}] FAILED (exit {rc}) — see {log_file}", file=sys.stderr)
            sys.exit(rc)
        print(f"[{job.key}] OK")


def _run_slurm(
    jobs: list[Job],
    slurm_cfg: dict,
    resources: dict[str, dict],
    log_dir: Path,
    dry_run: bool,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    job_ids: dict[str, str] = {}  # job key → job ID (real or placeholder)

    for job in jobs:
        slug = _slug(job.key)
        res = resources[job.step]
        sbatch = [
            "sbatch",
            "--parsable",
            "--time",
            res.get("time", "04:00:00"),
            "--mem",
            res.get("mem", "16G"),
            "--cpus-per-task",
            str(res.get("cpus_per_task", 4)),
            "--job-name",
            f"rip_{slug}",
            "--output",
            str(log_dir / f"{slug}_%j.out"),
            "--error",
            str(log_dir / f"{slug}_%j.err"),
        ]
        partition = slurm_cfg.get("partition")
        if partition:
            sbatch += ["--partition", partition]
        account = slurm_cfg.get("account")
        if account:
            sbatch += ["--account", account]

        if job.dep_keys:
            dep_str = ":".join(job_ids[d] for d in job.dep_keys)
            sbatch += ["--dependency", f"afterok:{dep_str}"]

        sbatch += ["--wrap", shlex.join(job.cmd)]

        if dry_run:
            print(shlex.join(sbatch))
            print()
            job_ids[job.key] = f"<JOBID_{slug}>"
        else:
            result = subprocess.run(sbatch, capture_output=True, text=True)
            if result.returncode != 0:
                print(
                    f"[{job.key}] sbatch failed: {result.stderr.strip()}",
                    file=sys.stderr,
                )
                sys.exit(result.returncode)
            job_ids[job.key] = result.stdout.strip().split(";")[0]
            print(f"[{job.key}] submitted → job {job_ids[job.key]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="scripts/run_pipeline.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        type=Path,
        metavar="YAML",
        help="Pipeline orchestrator config file.",
    )
    parser.add_argument(
        "--slurm", action="store_true", help="Submit steps as Slurm jobs."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing."
    )
    parser.add_argument(
        "--steps",
        metavar="STEP[,STEP...]",
        help=f"Steps to run (default: {','.join(DEFAULT_STEPS)}). "
        f"Available: {','.join(STEP_ORDER)}.",
    )
    # Global Slurm resource overrides (applied to every step)
    parser.add_argument(
        "--partition", metavar="NAME", help="Slurm partition (overrides config)."
    )
    parser.add_argument(
        "--account", metavar="NAME", help="Slurm account (overrides config)."
    )
    parser.add_argument(
        "--time",
        metavar="HH:MM:SS",
        help="Wall time for every step (overrides per-step defaults).",
    )
    parser.add_argument(
        "--mem",
        metavar="SIZE",
        help="Memory for every step, e.g. 32G (overrides per-step defaults).",
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        metavar="N",
        help="CPUs per step (overrides per-step defaults).",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")

    cfg = _load_config(config_path)
    base = config_path.parent

    # Resolve steps
    if args.steps:
        steps = [s.strip() for s in args.steps.split(",")]
    else:
        steps = cfg.get("steps", DEFAULT_STEPS)

    unknown = [s for s in steps if s not in STEP_ORDER]
    if unknown:
        sys.exit(f"Unknown step(s): {unknown}. Available: {STEP_ORDER}")

    # Preserve canonical dependency order
    steps = [s for s in STEP_ORDER if s in steps]

    # Build jobs (one per step, or one per (group, step) with fan-out)
    try:
        jobs = _build_jobs(cfg, steps, base)
    except ValueError as exc:
        sys.exit(f"Config error: {exc}")

    # Log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(cfg.get("log_dir", "logs")) / timestamp

    # Resolve Slurm config: file → CLI overrides
    slurm_cfg: dict = dict(cfg.get("slurm") or {})
    if args.partition:
        slurm_cfg["partition"] = args.partition
    if args.account:
        slurm_cfg["account"] = args.account

    # Per-step resources: built-in defaults → per-step config block → global CLI overrides
    resources: dict[str, dict] = {}
    for step in steps:
        res = dict(_DEFAULT_RESOURCES.get(step, {}))
        step_key = step.replace("-", "_")
        step_slurm = slurm_cfg.get(step_key) or {}
        res.update(step_slurm)
        if args.time:
            res["time"] = args.time
        if args.mem:
            res["mem"] = args.mem
        if args.cpus_per_task:
            res["cpus_per_task"] = args.cpus_per_task
        resources[step] = res

    # Execute
    if args.dry_run and not args.slurm:
        for job in jobs:
            print(f"# {job.key}")
            print(shlex.join(job.cmd))
            print()
        return

    if args.slurm:
        _run_slurm(jobs, slurm_cfg, resources, log_dir, args.dry_run)
    else:
        _run_local(jobs, log_dir)


if __name__ == "__main__":
    main()
