import contextlib
import os
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
from rich.table import Table

console = Console()


def _as_path(value: Optional[Path | str]) -> Optional[Path]:
    """Coerce a str (e.g. from the Python API) to Path; pass Path/None through."""
    return Path(value) if isinstance(value, str) else value


def run(
    risearch_file: Annotated[
        Optional[Path],
        typer.Option(
            "-r",
            "--risearch-file",
            help="Path to pre-computed RIsearch output file or directory of files.",
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
            help="Path to siRNA FASTA file (one or more sequences). Runs RIsearch in-process via PyO3 bindings; also used to compute self-hybridisation E_min for directory inputs.",
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
            help="Number of parallel threads for intersection (default: CPU count).",
        ),
    ] = None,
    gtf_file: Annotated[
        Optional[Path],
        typer.Option(
            "-t",
            "--transcriptome",
            help="Path to transcriptome annotation file (.gtf) or .bed file.",
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
            help="Transcriptome file format: auto, gtf, gff3, bed6, or bed7 (default: auto-detect; gtf vs gff3 is detected from the attribute column).",
        ),
    ] = "auto",
    accessibility_dir: Annotated[
        Optional[Path],
        typer.Option(
            "-a",
            "--accessibility-dir",
            help="Directory of per-chromosome accessibility Parquet files (from 'sioff accessibility'), or of the .access.npy cache files built from them.",
            exists=True,
            file_okay=False,
        ),
    ] = None,
    accessibility_strategy: Annotated[
        str,
        typer.Option(
            "--accessibility-strategy",
            help=(
                "How opening energies are looked up from --accessibility-dir: "
                "'chrom-major' (directory mode) streams each chromosome's Parquet "
                "once per run for all siRNAs, memory per worker ~ one row group, "
                "writes nothing; 'memmap' builds a uint8 .access.npy cache per "
                "chromosome and strand (in --accessibility-cache-dir, default the "
                "accessibility dir) that all workers memory-map, and is the only "
                "strategy that writes cache files; 'eager' loads whole-chromosome "
                "profiles in every worker (small genomes only). 'auto' (default) "
                "is chrom-major in directory mode and eager for a single file."
            ),
        ),
    ] = "auto",
    accessibility_cache_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--accessibility-cache-dir",
            help=(
                "With --accessibility-strategy memmap: directory of the "
                "<chrom>_<plus|minus>.access.npy cache, built there from the Parquet "
                "when missing and created if absent (default: the accessibility dir "
                "itself). Use it when the accessibility dir is read-only or to stage "
                "the cache on local disk; it may also hold a complete cache without "
                "any Parquet. Ignored by the other strategies."
            ),
            file_okay=False,
        ),
    ] = None,
    output_file: Annotated[
        Optional[Path],
        typer.Option(
            "-o",
            "--output",
            help="Path to save the analysis results (TSV).",
        ),
    ] = None,
    genome_file: Annotated[
        Optional[Path],
        typer.Option(
            "-f",
            "--fasta",
            help="Path to genome FASTA file (computes accessibility on-the-fly).",
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
    on_target_file: Annotated[
        Optional[Path],
        typer.Option(
            "-on",
            "--on-target",
            help="Path to On-Target sequence FASTA (for Partition Function).",
            exists=True,
            readable=True,
        ),
    ] = None,
    on_target_risearch_file: Annotated[
        Optional[Path],
        typer.Option(
            "--on-target-risearch-file",
            "-on-ris",
            help="Path to pre-computed RIsearch output file for On-Target (skips on-the-fly calculation).",
            exists=True,
            readable=True,
        ),
    ] = None,
    query_file: Annotated[
        Optional[Path],
        typer.Option(
            "-q",
            "--query",
            help="Path to siRNA query FASTA (required if --on-target is used).",
            exists=True,
            readable=True,
        ),
    ] = None,
    on_target_expression: Annotated[
        float,
        typer.Option(
            "--on-target-expression",
            "-oexp",
            help="Expression level for On-Target (default: 1000.0).",
        ),
    ] = 1000.0,
    on_target_accessibility: Annotated[
        Optional[Path],
        typer.Option(
            "--on-target-accessibility",
            help="Accessibility Parquet for the on-target (same schema as accessibility command output). Falls back to on-the-fly computation if omitted.",
        ),
    ] = None,
    on_target_ids_file: Annotated[
        Optional[Path],
        typer.Option(
            "-oi",
            "--on-target-ids",
            help="TSV file mapping siRNA IDs to on-target transcript IDs (sirna_id \\t transcript_id).",
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
    legacy_format: Annotated[
        bool,
        typer.Option(
            "--legacy-format",
            help="Output in legacy format (gw.results style, aggregated by transcript).",
        ),
    ] = False,
    detailed_report: Annotated[
        bool,
        typer.Option(
            "--detailed-report",
            help="Report off-target probabilities for individual transcripts (Legacy Format).",
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
            help="Show detailed output (tables, stats).",
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
) -> Any:
    """
    Analyze siRNA off-target predictions.

    Integrates RIsearch2 predictions with transcriptome data and optionally
    calculates off-target probabilities.

    Returns a dict (directory mode), a polars DataFrame (single-file mode) or
    None. Spelled ``Any`` because Typer evaluates every annotation while
    building ``--help``, and polars is imported lazily below to keep that fast.
    """
    # Heavy imports live here, not at module top, so `sioff --help` does not
    # load polars/pyarrow/numpy and the core (see TestCLIStartup).
    import pyarrow.parquet as pq
    from loguru import logger

    from sioff.core.off_targets import (
        WorkerFailureError,
        _build_alpha_gamma_pairs,
        _downcast_schema,
        _parse_theta,
        compute_off_targets_directory,
        compute_off_targets_single,
    )
    from sioff.services.probability import ProbabilityService
    from sioff.services.profiling import PipelineProfiler
    from sioff.services.risearch_parser import RIsearchParser

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
    on_target_file = _as_path(on_target_file)
    on_target_risearch_file = _as_path(on_target_risearch_file)
    query_file = _as_path(query_file)
    on_target_accessibility = _as_path(on_target_accessibility)
    on_target_ids_file = _as_path(on_target_ids_file)

    console.print(Panel("siOFF", style="bold cyan"))

    profiler = PipelineProfiler(enabled=profile)
    n_workers: int = workers if workers is not None else os.cpu_count() or 1

    try:
        # A directory passed to --risearch-file triggers per-siRNA parallel mode.
        input_dir: Optional[Path] = None
        if risearch_file is not None and risearch_file.is_dir():
            input_dir = risearch_file
            risearch_file = None

        # Friendly CLI pre-checks (the core re-validates and raises ValueError).
        if risearch_file is None and sirna_fasta is None and input_dir is None:
            console.print(
                "[bold red]Error:[/bold red] Must provide either --risearch-file (file or directory) OR --sirna-fasta"
            )
            raise typer.Exit(code=1)
        is_running_risearch = (
            (sirna_fasta is not None)
            and (risearch_file is None)
            and (input_dir is None)
        )
        if is_running_risearch and target_fasta is None:
            console.print(
                "[bold red]Error:[/bold red] --sirna-fasta requires --target-fasta when running RIsearch dynamically"
            )
            raise typer.Exit(code=1)

        # -------------------------------------------------------------------
        # Directory mode: stream per-siRNA DataFrames from the core generator
        # and write the same outputs (per-prediction TSV/CSV/parquet + .summary).
        # -------------------------------------------------------------------
        if input_dir is not None:
            console.print(
                f"[bold cyan]Directory mode[/bold cyan] - loading files from {input_dir}"
            )
            risearch_parser = RIsearchParser()
            all_files = risearch_parser.list_directory_files(input_dir)
            if not all_files:
                console.print(
                    f"[bold red]Error:[/bold red] No RIsearch files found in {input_dir}"
                )
                raise typer.Exit(code=1)
            console.print(
                f"[green]✓[/green] Found [bold]{len(all_files)}[/bold] files to process"
            )

            alpha_gamma_pairs = _build_alpha_gamma_pairs(alpha, gamma)
            theta_vals = _parse_theta(theta)
            # .summary formatting needs no accessibility service.
            fmt_service = ProbabilityService(None, temperature=temperature)

            # Resolve the per-prediction output path (skipped under --summary-only).
            out_path = None
            _csv_first_batch = True
            _pq_writer = None
            if output_file and not summary_only:
                if output_format == "parquet":
                    out_path = output_file.with_suffix(".parquet")
                elif output_format == "tsv":
                    out_path = output_file.with_suffix(".tsv")
                else:
                    out_path = output_file.with_suffix(".csv")
                out_path.parent.mkdir(parents=True, exist_ok=True)

            # Resolve the .summary output dir. If --output is a directory path
            # (no extension / trailing slash), write there; else alongside it.
            summary_out_dir = None
            if output_file is not None:
                if (
                    output_file.suffix == ""
                    or str(output_file).endswith("/")
                    or output_file.is_dir()
                ):
                    summary_out_dir = output_file
                else:
                    summary_out_dir = output_file.parent
                summary_out_dir.mkdir(parents=True, exist_ok=True)
            summary_count = 0
            total_rows = 0

            gen = compute_off_targets_directory(
                input_dir=input_dir,
                sirna_fasta=sirna_fasta,
                gtf_file=gtf_file,
                feature_type=feature_type,
                expression_metric=expression_metric,
                transcriptome_format=transcriptome_format,
                accessibility_dir=accessibility_dir,
                accessibility_cache_dir=accessibility_cache_dir,
                temperature=temperature,
                on_target_ids_file=on_target_ids_file,
                on_target_expression=on_target_expression,
                alpha=alpha,
                gamma=gamma,
                theta=theta,
                sense_only=sense_only,
                predictions_type=predictions_type,
                n_workers=n_workers,
                accessibility_strategy=accessibility_strategy,
            )

            _drain_stage_totals: dict[str, float] = {}
            _drain_n = 0
            _worker_error: Optional[WorkerFailureError] = None
            # contextlib.closing guarantees the generator's pool + temp-IPC are
            # torn down even if a write below raises mid-stream.
            with contextlib.closing(gen):
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.completed}/{task.total}"),
                    TimeElapsedColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task(
                        "Processing siRNAs...", total=len(all_files)
                    )
                    completed = 0
                    try:
                        for df_chunk, batch_metadata in gen:
                            completed += 1
                            progress.update(
                                task,
                                advance=1,
                                description=f"Batch {completed}/{len(all_files)}",
                            )
                            total_rows += df_chunk.height

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

                            # Stream per-prediction output.
                            if out_path is not None:
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
                        # Raised after the last successful siRNA was yielded
                        # (or at once for a failed accessibility contig); keep
                        # what was written and report below.
                        _worker_error = exc
                    finally:
                        # Flush the Parquet footer (CLI owns the writer).
                        if _pq_writer is not None:
                            _pq_writer.close()

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
                console.print(
                    f"[bold red]Error:[/bold red] {escape(str(_worker_error))}"
                )
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

        # -------------------------------------------------------------------
        # Single-file / inline-RIsearch mode: the core returns (df, meta);
        # this wrapper renders the same files and console output as before.
        # -------------------------------------------------------------------
        console.print("[bold green]Calculating probabilities...[/bold green]")

        # A progress bar is shown only while accessibility is folded on-the-fly;
        # otherwise a null context yields ``progress = None``. Either way there is a
        # single, explicitly-keyworded core call (keeping the type checker happy —
        # a **kwargs dict would widen every value to one union type).
        if genome_file and not accessibility_dir:
            _progress_cm = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console,
            )
        else:
            _progress_cm = contextlib.nullcontext()

        with _progress_cm as progress:
            acc_cb = None
            if progress is not None:
                _task = progress.add_task("Calculating Accessibility...", total=None)

                def acc_cb(advance=0, description=""):
                    progress.update(_task, advance=advance, description=description)

            df, meta = compute_off_targets_single(
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
                accessibility_strategy=accessibility_strategy,
                genome_file=genome_file,
                window_size=window_size,
                max_span=max_span,
                unpaired_prob=unpaired_prob,
                temperature=temperature,
                on_target_file=on_target_file,
                on_target_risearch_file=on_target_risearch_file,
                query_file=query_file,
                on_target_expression=on_target_expression,
                on_target_accessibility=on_target_accessibility,
                on_target_ids_file=on_target_ids_file,
                alpha=alpha,
                gamma=gamma,
                theta=theta,
                sense_only=sense_only,
                predictions_type=predictions_type,
                legacy_format=legacy_format,
                detailed_report=detailed_report,
                n_workers=n_workers,
                profiler=profiler,
                accessibility_progress_callback=acc_cb,
            )

        # Diagnostics (loaded count / energy range / intersection counts).
        rep = meta.get("_report", {})
        source_name = (
            sirna_fasta.name
            if sirna_fasta
            else (risearch_file.name if risearch_file else "input")
        )
        if rep:
            console.print(
                f"[green]✓[/green] Loaded [bold]{rep['n_loaded']}[/bold] predictions from {source_name}"
            )
            console.print(
                f"  └─ Energy range: {rep['energy_min']:.2f} to {rep['energy_max']:.2f} kcal/mol"
            )
            if rep.get("sense_only"):
                console.print("  └─ (Filtered to sense strand only)")
            if rep.get("n_features") is not None:
                _gtf_name = gtf_file.name if gtf_file else ""
                console.print(
                    f"[green]✓[/green] Loaded [bold]{rep['n_features']}[/bold] features from {_gtf_name}"
                )
                console.print(
                    f"[green]✓[/green] Found [bold]{rep['n_intersected']}[/bold] intersecting off-target candidates"
                )
        if verbose and "preview" in rep:
            console.print("[dim]--- RIsearch Predictions (First 5 rows) ---[/dim]")
            console.print(rep["preview"])

        # Multi-siRNA runs (discriminated by the per-siRNA partition-function key)
        # write one .summary per siRNA next to the output file.
        if output_file and "z_per_sirna" in meta:
            out_dir = output_file.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            fmt_service = ProbabilityService(None, temperature=temperature)
            alpha_gamma_pairs = _build_alpha_gamma_pairs(alpha, gamma)
            theta_vals = _parse_theta(theta)
            on_weights = meta.get("on_target_weights", {})
            for sid in meta["z_per_sirna"]:
                summary_text = fmt_service.format_legacy_summary(
                    sid,
                    meta["z_per_sirna"][sid],
                    on_weights.get(sid, {}),
                    alpha_gamma_pairs,
                    theta_vals,
                )
                (out_dir / f"{sid}.summary").write_text(summary_text)

        # On-target details (single-siRNA mode only).
        if meta.get("has_on_target", False):
            console.print(
                f"  └─ On-Target: ΔG_total={meta['dG_on_target']:.2f} kcal/mol, "
                f"P(on)={meta.get('p_on_target', 0.0):.10f}"
            )

        # Legacy format: write .results + detailed_results.tsv, then return early
        # (skips the preview table and the standard TSV write).
        if legacy_format:
            legacy_output = meta.get("legacy_text", "")
            if output_file:
                legacy_path = output_file.with_suffix(".results")
                with open(legacy_path, "w") as f:
                    f.write(legacy_output)
                console.print(
                    f"[green]✓[/green] Legacy format results saved to {legacy_path}"
                )
                detailed_path = output_file.parent / "detailed_results.tsv"
                df.write_csv(detailed_path, separator="\t")
                console.print(
                    f"[green]✓[/green] Detailed results saved to {detailed_path}"
                )
            else:
                console.print(legacy_output)
            return df

        # Results preview table (sorts df by P_off_target — preserved so the saved
        # TSV is ordered identically to the pre-refactor output).
        if df.height > 0:
            if "P_off_target" in df.columns:
                df = df.sort("P_off_target", descending=True)
                title = "Top 10 Candidates (By P_off_target)"
            else:
                title = "Predictions Preview"

            table = Table(title=title)
            potential_cols = [
                "sirna_id",
                "chrom",
                "gene_id",
                "transcript_id",
                "energy",
                "dG_total",
                "P_off_target",
                "exp_value",
            ]
            display_cols = [c for c in potential_cols if c in df.columns]
            for col in display_cols:
                table.add_column(col)
            for row in df.select(display_cols).head(10).iter_rows():
                formatted_row = [
                    f"{val:.4g}" if isinstance(val, float) else str(val) for val in row
                ]
                table.add_row(*formatted_row)
            console.print(table)

        # Save to output file.
        if output_file:
            logger.info(f"Writing {df.height:,} rows to {output_file}")
            with profiler.stage("Write output", rows_in=df.height):
                df.write_csv(output_file, separator="\t")
            console.print(
                f"\n[green]✓[/green] Results saved to [bold]{output_file}[/bold]"
            )

        profiler.print_summary(console)
        return df

    except typer.Exit:
        raise
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {escape(str(e))}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Failed:[/bold red] {escape(str(e))}")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)
