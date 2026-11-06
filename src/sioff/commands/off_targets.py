import contextlib
import os
import re
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

console = Console()

# Engine argument names in error messages, and the flags a CLI user typed
# instead (only the arguments that have a flag).
_FLAG_NAMES = {
    "genome_file": "-f/--fasta",
    "accessibility_dir": "-a/--accessibility-dir",
    "accessibility_cache_dir": "--accessibility-cache-dir",
    "sirna_fasta": "-s/--sirna-fasta",
    "risearch_file": "-r/--risearch-file",
    "target_fasta": "--target-fasta",
    "target_index": "-idx/--target-index",
    "gtf_file": "-t/--transcriptome",
    "sirna_table": "--sirna-table",
}
# A whole word, not part of a path or file name.
_ARGUMENT_NAME = re.compile(
    r"(?<![\w/.-])(" + "|".join(_FLAG_NAMES) + r")(?![\w/-])(?!\.\w)"
)
_TABLE_SUFFIXES = (".tsv", ".csv", ".parquet")


def _with_flag_names(message: str) -> str:
    """Name the CLI flags where an engine message names its arguments."""
    return _ARGUMENT_NAME.sub(lambda m: _FLAG_NAMES[m.group(1)], message)


def _table_path(output_file: Path, output_format: str) -> Path:
    """Where ``-o`` puts the per-interaction table in ``output_format``.

    A table suffix (``.tsv``, ``.csv``, ``.parquet``) is replaced by the
    format's; any other name keeps its suffix and gets the format's appended,
    so ``results.txt`` writes ``results.txt.tsv`` and ``run.v1`` and
    ``run.v2`` do not collide.
    """
    ext = {"parquet": ".parquet", "tsv": ".tsv"}.get(output_format, ".csv")
    if output_file.suffix.lower() in _TABLE_SUFFIXES:
        return output_file.with_suffix(ext)
    return output_file.with_name(output_file.name + ext)


def _input_at(
    path: Path, inputs: dict[str, Optional[Path]]
) -> Optional[tuple[str, Path]]:
    """The input file ``path`` is, if any: ``(its flag, the input as given)``.

    ``samefile`` catches symlinks and other spellings of one file. Only files
    count: an ``-r`` folder may hold the output.
    """
    if not path.exists():
        return None
    for name, given in inputs.items():
        if given is not None and given.is_file() and path.samefile(given):
            return _FLAG_NAMES.get(name, name), given
    return None


def _as_path(value: Optional[Path | str]) -> Optional[Path]:
    """Coerce a str (e.g. from the Python API) to Path; pass Path/None through."""
    return Path(value) if isinstance(value, str) else value


def run(
    risearch_file: Annotated[
        Optional[Path],
        typer.Option(
            "-r",
            "--risearch-file",
            help="Pre-computed predictions: a file, or a directory of files (RIsearch2 output, 'sioff search' TSV or Parquet; each may hold one siRNA or many).",
            exists=True,
            readable=True,
            dir_okay=True,
        ),
    ] = None,
    sirna_fasta: Annotated[
        Optional[Path],
        typer.Option(
            "-s",
            "--sirna-fasta",
            help="siRNA FASTA (one or more sequences) to search in-process against --target-fasta. Only the search query: it cannot be combined with --risearch-file; E_min's perfect-match energy comes from --sirna-table.",
            exists=True,
            readable=True,
        ),
    ] = None,
    target_fasta: Annotated[
        Optional[Path],
        typer.Option(
            "--target-fasta",
            "--genome",
            help="Path to target FASTA (genome or transcriptome) for RIsearch.",
            exists=True,
            readable=True,
        ),
    ] = None,
    target_index: Annotated[
        Optional[Path],
        typer.Option(
            "-idx",
            "--target-index",
            help="Pre-built RIsearch index (optional, speeds up repeated runs).",
            exists=True,
            readable=True,
        ),
    ] = None,
    workers: Annotated[
        Optional[int],
        typer.Option(
            "-j",
            "--workers",
            help="Worker processes that split the inputs and score the siRNAs, one siRNA per task (default: CPU count).",
        ),
    ] = None,
    gtf_file: Annotated[
        Optional[Path],
        typer.Option(
            "-t",
            "--transcriptome",
            help="Path to transcriptome annotation file (GTF or GFF3, optionally gzipped).",
            exists=True,
            readable=True,
        ),
    ] = None,
    feature_type: Annotated[
        str,
        typer.Option(
            "--feature",
            help="Feature type to select from GTF (default: exon).",
        ),
    ] = "exon",
    expression_metric: Annotated[
        str,
        typer.Option(
            "--expression-metric",
            help="Attribute to use for expression score (default: RPKM).",
        ),
    ] = "RPKM",
    transcriptome_format: Annotated[
        str,
        typer.Option(
            "--transcriptome-format",
            help="Transcriptome file format: auto, gtf or gff3 (default: auto-detect from the attribute column). BED is not supported: it cannot tell exons of one transcript from different transcripts.",
        ),
    ] = "auto",
    accessibility_dir: Annotated[
        Optional[Path],
        typer.Option(
            "-a",
            "--accessibility-dir",
            help="Accessibility profiles from 'sioff accessibility': its packed accessibility.{plus,minus}.packed.npy cache, or per-record Parquet files (packed on the first run).",
            exists=True,
            file_okay=False,
        ),
    ] = None,
    accessibility_cache_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--accessibility-cache-dir",
            help=(
                "Where the packed accessibility cache of --accessibility-dir is "
                "read from and built (default: the accessibility dir itself); "
                "only with --accessibility-dir. Use it when the accessibility dir "
                "is read-only or to stage the cache on local disk; it may also "
                "hold a complete cache without any Parquet."
            ),
            file_okay=False,
        ),
    ] = None,
    output_file: Annotated[
        Optional[Path],
        typer.Option(
            "-o",
            "--output",
            help="Where to write the per-interaction table, in --output-format: a .tsv, .csv or .parquet suffix is replaced by the format's, any other name gets it appended (results.txt -> results.txt.tsv, out -> out.tsv). The .summary files go next to it, or into it when it has no suffix.",
        ),
    ] = None,
    genome_file: Annotated[
        Optional[Path],
        typer.Option(
            "-f",
            "--fasta",
            help="Genome FASTA to fold during the run (cannot be combined with --accessibility-dir).",
            exists=True,
            readable=True,
        ),
    ] = None,
    window_size: Annotated[
        int, typer.Option("--window", "-W", help="Window size (W)")
    ] = 80,
    max_span: Annotated[
        int, typer.Option("--span", "-L", help="Max base pair span (L)")
    ] = 40,
    unpaired_prob: Annotated[
        int, typer.Option("--unpaired", "-u", help="Unpaired probability length (u)")
    ] = 30,
    temperature: Annotated[
        float,
        typer.Option(
            "--temperature",
            "-T",
            help="Folding temperature in °C (default 37.0). Affects both accessibility and partition function.",
        ),
    ] = 37.0,
    matrix: Annotated[
        str,
        typer.Option(
            "--matrix",
            "-z",
            help=(
                "Energy parameter set for in-process searches and the "
                "perfect-match energies of --sirna-table: 't04', 'slh04', "
                "'s95-rna-dna', 's95-dna-rna', or the path of a DSM TSV table. Use "
                "the set the predictions were searched with: E_min is the reference "
                "energy of the alpha/gamma clamp."
            ),
        ),
    ] = "t04",
    sirna_table: Annotated[
        Optional[Path],
        typer.Option(
            "--sirna-table",
            help="TSV with a header: sirna_id, on_target_id, sequence. Gives each siRNA's on-target and the perfect-match energy E_min needs; required when --alpha/--gamma set a pair other than 1,1.",
            exists=True,
            readable=True,
        ),
    ] = None,
    alpha: Annotated[
        str,
        typer.Option(
            "--alpha",
            help="Alpha clamping parameter(s). Separate multiple values with ';' (e.g., '0.8;1.0').",
        ),
    ] = "1.0",
    gamma: Annotated[
        str,
        typer.Option(
            "--gamma",
            help="Gamma clamping parameter(s). Separate multiple values with ';' (e.g., '0.8;1.0').",
        ),
    ] = "1.0",
    theta: Annotated[
        str,
        typer.Option(
            "--theta",
            help="Theta scaling parameter(s). Separate multiple values with ';' (e.g., '0.5;0.7').",
        ),
    ] = "",
    detailed_report: Annotated[
        bool,
        typer.Option(
            "--detailed-report",
            help=(
                "Keep the raw Boltzmann weights, partition functions and E_min "
                "for every interaction in the output table."
            ),
        ),
    ] = False,
    sense_only: Annotated[
        bool,
        typer.Option(
            "--sense-only",
            help="Limit RIsearch2 predictions to sense strand (+), ignore antisense.",
        ),
    ] = False,
    predictions_type: Annotated[
        str,
        typer.Option(
            "--type",
            help="Type of RIsearch2 predictions. gw for genome-wide and tw for transcriptome-wide.",
        ),
    ] = "gw",
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Print the traceback when the run fails.",
        ),
    ] = False,
    profile: Annotated[
        bool,
        typer.Option(
            "--profile",
            help="Print a per-stage timing and memory profile table at the end of the run.",
        ),
    ] = False,
    output_format: Annotated[
        str,
        typer.Option(
            "--output-format",
            help="Final output format: 'tsv', 'csv', or 'parquet'.",
        ),
    ] = "tsv",
    summary_only: Annotated[
        bool,
        typer.Option(
            "--summary-only",
            help="Write only .summary files (partition-function stats per siRNA); skip the per-prediction output TSV/parquet. Equivalent to the old pipeline's default output.",
        ),
    ] = False,
) -> dict[str, Any]:
    """
    Analyze siRNA off-target predictions.

    Integrates RIsearch2 predictions with transcriptome data and calculates
    off-target probabilities, one siRNA at a time, for a file, a directory of
    files or an in-process search alike.

    Returns a dict with the output path (``output``), the number of scored
    rows (``n_rows``), the .summary directory (``summary_dir``) and the number
    of .summary files written (``summary_files``).
    """
    # Heavy imports live here, not at module top, so `sioff --help` does not
    # load polars/pyarrow/numpy and the core (see TestCLIStartup).
    import pyarrow.parquet as pq

    from sioff.core.off_targets import (
        WorkerFailureError,
        _build_alpha_gamma_pairs,
        _downcast_schema,
        _parse_theta,
        compute_off_targets,
    )
    from sioff.services.probability import ProbabilityService
    from sioff.services.profiling import PipelineProfiler

    # Accept str paths (Python API) as well as Path objects (CLI coerces these itself).
    risearch_file = _as_path(risearch_file)
    sirna_fasta = _as_path(sirna_fasta)
    target_fasta = _as_path(target_fasta)
    target_index = _as_path(target_index)
    gtf_file = _as_path(gtf_file)
    accessibility_dir = _as_path(accessibility_dir)
    accessibility_cache_dir = _as_path(accessibility_cache_dir)
    output_file = _as_path(output_file)
    genome_file = _as_path(genome_file)
    sirna_table = _as_path(sirna_table)

    console.print(Panel("siOFF", style="bold cyan"))

    profiler = PipelineProfiler(enabled=profile)
    n_workers: int = workers if workers is not None else os.cpu_count() or 1

    try:
        # Friendly CLI pre-checks (the core re-validates and raises ValueError).
        if detailed_report and summary_only:
            console.print(
                "[bold red]Error:[/bold red] --detailed-report needs the "
                "per-interaction table; drop --summary-only"
            )
            raise typer.Exit(code=1)
        if risearch_file is None and sirna_fasta is None:
            console.print(
                "[bold red]Error:[/bold red] Must provide either --risearch-file (file or directory) OR --sirna-fasta"
            )
            raise typer.Exit(code=1)
        if risearch_file is None and target_fasta is None:
            console.print(
                "[bold red]Error:[/bold red] --sirna-fasta requires --target-fasta when running RIsearch dynamically"
            )
            raise typer.Exit(code=1)

        alpha_gamma_pairs = _build_alpha_gamma_pairs(alpha, gamma)
        theta_vals = _parse_theta(theta)
        # .summary formatting needs no accessibility service.
        fmt_service = ProbabilityService(None, temperature=temperature)

        # Resolve the per-prediction output path (skipped under --summary-only).
        out_path = None
        _csv_first_batch = True
        _pq_writer = None
        if output_file and not summary_only:
            out_path = _table_path(output_file, output_format)
            if out_path.exists() and not out_path.is_file():
                console.print(
                    f"[bold red]Error:[/bold red] {escape(str(out_path))} exists "
                    "and is not a regular file; pass another -o/--output"
                )
                raise typer.Exit(code=1)
            clobbered = _input_at(
                out_path,
                {
                    "risearch_file": risearch_file,
                    "sirna_table": sirna_table,
                    "sirna_fasta": sirna_fasta,
                    "gtf_file": gtf_file,
                    "genome_file": genome_file,
                    "target_fasta": target_fasta,
                    "target_index": target_index,
                    "-z/--matrix": Path(matrix),
                },
            )
            if clobbered is not None:
                flag, given = clobbered
                console.print(
                    f"[bold red]Error:[/bold red] -o/--output resolves to the {flag} "
                    f"input {escape(str(given))}; choose another -o/--output"
                )
                raise typer.Exit(code=1)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # A run that scores nothing writes no table, so an earlier run's
            # must not be left in its place.
            out_path.unlink(missing_ok=True)

        # Resolve the .summary output dir. If --output is a directory path
        # (no suffix; pathlib drops a trailing slash), write there; else
        # alongside it.
        summary_out_dir = None
        if output_file is not None:
            if output_file.suffix == "" or output_file.is_dir():
                summary_out_dir = output_file
            else:
                summary_out_dir = output_file.parent
            summary_out_dir.mkdir(parents=True, exist_ok=True)
        summary_count = 0
        total_rows = 0

        _drain_stage_totals: dict[str, float] = {}
        _drain_n = 0
        _worker_error: Optional[WorkerFailureError] = None
        # One stage for the whole run; the per-siRNA breakdown follows.
        with profiler.stage("Split, score and write") as _stage:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                acc_cb = None
                if genome_file is not None:
                    _acc_task = progress.add_task(
                        "Calculating Accessibility...", total=None
                    )

                    def acc_cb(advance=0, description=""):
                        progress.update(
                            _acc_task, advance=advance, description=description
                        )

                # The total is known once the inputs are split into siRNAs.
                task = progress.add_task("Splitting predictions...", total=None)

                def on_split(n_sirnas: int) -> None:
                    progress.update(
                        task, total=n_sirnas, description="Processing siRNAs..."
                    )

                gen = compute_off_targets(
                    risearch_file=risearch_file,
                    sirna_fasta=sirna_fasta,
                    target_fasta=target_fasta,
                    target_index=target_index,
                    gtf_file=gtf_file,
                    feature_type=feature_type,
                    expression_metric=expression_metric,
                    transcriptome_format=transcriptome_format,
                    accessibility_dir=accessibility_dir,
                    accessibility_cache_dir=accessibility_cache_dir,
                    genome_file=genome_file,
                    window_size=window_size,
                    max_span=max_span,
                    unpaired_prob=unpaired_prob,
                    temperature=temperature,
                    matrix=matrix,
                    sirna_table=sirna_table,
                    alpha=alpha,
                    gamma=gamma,
                    theta=theta,
                    sense_only=sense_only,
                    predictions_type=predictions_type,
                    n_workers=n_workers,
                    # --summary-only writes the .summary files from the per-siRNA
                    # metadata alone; workers then send back no per-site table.
                    summary_only=summary_only,
                    detailed_report=detailed_report,
                    on_split=on_split,
                    accessibility_progress_callback=acc_cb,
                )

                # contextlib.closing guarantees the generator's pool and temporary
                # files are torn down even if a write below raises mid-stream.
                with contextlib.closing(gen):
                    try:
                        for df_chunk, batch_metadata in gen:
                            progress.update(task, advance=1)
                            total_rows += batch_metadata.get("n_rows", df_chunk.height)

                            # Stream .summary files as each siRNA completes.
                            if summary_out_dir is not None:
                                z_per_sirna = batch_metadata.get("z_per_sirna", {})
                                w_per_sirna = batch_metadata.get(
                                    "on_target_weights", {}
                                )
                                for sid, z_dict in z_per_sirna.items():
                                    text = fmt_service.format_legacy_summary(
                                        sid,
                                        z_dict,
                                        w_per_sirna.get(sid, {}),
                                        alpha_gamma_pairs,
                                        theta_vals,
                                    )
                                    (summary_out_dir / f"{sid}.summary").write_text(
                                        text
                                    )
                                    summary_count += 1

                            # Stream per-prediction output: one block per siRNA,
                            # its rows by P_off_target, highest first.
                            if out_path is not None:
                                if "P_off_target" in df_chunk.columns:
                                    df_chunk = df_chunk.sort(
                                        "P_off_target", descending=True
                                    )
                                if output_format == "parquet":
                                    arrow_tbl = _downcast_schema(df_chunk).to_arrow()
                                    if _pq_writer is None:
                                        _pq_writer = pq.ParquetWriter(
                                            out_path, arrow_tbl.schema
                                        )
                                    _pq_writer.write_table(arrow_tbl)
                                else:
                                    sep = "\t" if output_format == "tsv" else ","
                                    with open(
                                        out_path, "w" if _csv_first_batch else "a"
                                    ) as _f:
                                        df_chunk.write_csv(
                                            _f,
                                            separator=sep,
                                            include_header=_csv_first_batch,
                                        )
                                    _csv_first_batch = False

                            for k, v in batch_metadata.get("_timings", {}).items():
                                _drain_stage_totals[k] = (
                                    _drain_stage_totals.get(k, 0.0) + v
                                )
                            _drain_n += 1
                    except WorkerFailureError as exc:
                        # Raised after the last successful siRNA was yielded; keep
                        # what was written and report below.
                        _worker_error = exc
                    finally:
                        # Flush the Parquet footer (CLI owns the writer).
                        if _pq_writer is not None:
                            _pq_writer.close()
            _stage.rows_out = total_rows

        if profile and _drain_n > 0:
            _perf_lines = [
                f"[perf] per-siRNA stage breakdown (N={_drain_n}, sums over all siRNAs):"
            ]
            for k, v in sorted(_drain_stage_totals.items()):
                _perf_lines.append(
                    f"  {k}: {v:.3f}s  (avg {v / _drain_n * 1000:.1f}ms/siRNA)"
                )
            console.print("\n".join(_perf_lines), markup=False, highlight=False)

        if total_rows > 0 and out_path is not None:
            console.print(
                f"[green]✓[/green] Processed [bold]{total_rows}[/bold] predictions → [bold]{out_path}[/bold]"
            )
        elif total_rows > 0:
            console.print(
                f"[green]✓[/green] Processed [bold]{total_rows}[/bold] predictions (no output file specified)"
            )
        elif _worker_error is None:
            console.print("[yellow]Warning:[/yellow] No predictions to process")
        if summary_count > 0:
            console.print(
                f"[green]✓[/green] Wrote {summary_count} .summary files to [bold]{summary_out_dir}[/bold]"
            )

        profiler.print_summary(console)
        if _worker_error is not None:
            console.print(f"[bold red]Error:[/bold red] {escape(str(_worker_error))}")
            for name, err in _worker_error.failures[:10]:
                console.print(f"  {name}: {err}", markup=False, highlight=False)
            if _worker_error.n_failed > 10:
                console.print(f"  ... and {_worker_error.n_failed - 10} more")
            raise typer.Exit(code=1)
        return {
            "output": out_path,
            "n_rows": total_rows,
            "summary_dir": summary_out_dir,
            "summary_files": summary_count,
        }

    except typer.Exit:
        raise
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {escape(_with_flag_names(str(e)))}")
        raise typer.Exit(code=1)
    except Exception as e:
        # The engine's argument checks name its parameters; say the flags.
        message = _with_flag_names(str(e)) if isinstance(e, ValueError) else str(e)
        console.print(f"[bold red]Failed:[/bold red] {escape(message)}")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)
