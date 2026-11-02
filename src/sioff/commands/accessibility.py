from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def _make_progress(console):
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def run(
    genome: Annotated[
        Optional[Path],
        typer.Option(
            "--fasta", "-f", help="Path to genome or transcriptome FASTA file"
        ),
    ] = None,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output directory — the packed memory-map cache, accessibility.{plus,minus}.packed.{npy,npz} for all records (default; the plus pair only with --strand +), or one {record}.accessibility.parquet per record (--output-format parquet) is written here; a re-run replaces the packed files",
        ),
    ] = ...,  # ty: ignore[invalid-parameter-default]  # pyrefly: ignore[bad-function-definition]  # Typer required-option idiom (Ellipsis); kept required, must follow defaulted `genome`
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
            "--temperature", "-T", help="Folding temperature in °C (default 37.0)."
        ),
    ] = 37.0,
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            "-j",
            help="Number of parallel folding processes.",
        ),
    ] = 1,
    output_format: Annotated[
        str,
        typer.Option(
            "--output-format",
            help="Profile format: 'npy' (the uint8 cache off-targets memory-maps) or 'parquet'.",
        ),
    ] = "npy",
    strand: Annotated[
        str,
        typer.Option(
            "--strand",
            help="Strands to fold: 'both' or '+' (the sequences as given only, for a transcriptome FASTA; pair with off-targets --sense-only).",
        ),
    ] = "both",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed progress information."),
    ] = False,
):
    """
    Pre-compute accessibility profiles for every record of a FASTA.

    Folds every record in the FASTA (chromosomes or transcripts; IDs, the
    first word of each header, must be unique) with ViennaRNA RNAplfold and
    writes its opening energies to the output directory. The default npy
    format is the packed uint8 cache that off-targets memory-maps, one pair
    of files per strand whatever the number of records:
    accessibility.{plus,minus}.packed.npy, a code matrix [position, u] with
    every record's rows back to back in FASTA order, and a .packed.npz index
    of record names, offsets and lengths plus the values a code cannot hold
    exactly. A re-run into the same directory replaces the pair.
    --output-format parquet writes one {record}.accessibility.parquet per
    record instead, with columns [position (int32), strand (utf8),
    u1..u{u} (float32)].

    Each sequence is folded as given (+) and as its reverse complement (-).
    --strand + skips the reverse complement, halving the work for a
    transcriptome FASTA, whose records are already in sense orientation; the
    profiles then hold no '-' rows (no minus pair in npy form).

    Profiles can later be consumed by the off-targets command via
    --accessibility-dir.
    """
    # Heavy imports live here, not at module top, so `sioff --help` does not
    # load polars/numpy/Biopython through the service (see TestCLIStartup).
    from loguru import logger

    from sioff.services.accessibility import GenomeAccessibilityService, packed_paths

    console = Console(stderr=True)

    # Accept str paths (Python API) as well as Path objects.
    genome = Path(genome) if isinstance(genome, str) else genome
    output = Path(output) if isinstance(output, str) else output

    if genome is None:
        console.print("[red]Error: --fasta is required[/red]")
        raise typer.Exit(code=1)
    if output_format not in ("npy", "parquet"):
        console.print(
            f"[red]Error: --output-format must be npy or parquet, got {output_format!r}[/red]"
        )
        raise typer.Exit(code=1)
    if strand not in ("both", "+"):
        console.print(f"[red]Error: --strand must be both or +, got {strand!r}[/red]")
        raise typer.Exit(code=1)

    output.mkdir(parents=True, exist_ok=True)
    service = GenomeAccessibilityService(output)

    console.print("\n[bold]Full-genome accessibility computation[/bold]")
    console.print(f"  Genome:       {genome}")
    console.print(f"  Output dir:   {output} ({output_format})")
    console.print(f"  Strands:      {'+ only' if strand == '+' else 'both'}")
    console.print(
        f"  Parameters:   W={window_size}, L={max_span}, u={unpaired_prob}, T={temperature}°C"
    )
    console.print(f"  Workers:      {workers}\n")

    try:
        with _make_progress(console) as prog:
            results = service.compute_genome_accessibility(
                genome,
                window_size=window_size,
                max_span=max_span,
                unpaired_prob=unpaired_prob,
                workers=workers,
                progress=prog,
                temperature=temperature,
                output_format=output_format,
                strand=strand,
            )

        console.print(
            f"\n[green bold]✓[/green bold] Processed {len(results)} record(s)"
        )
        if output_format == "npy":
            # One packed pair per strand holds every record.
            packed = sorted(
                (p for s in ("+", "-") for p in packed_paths(output, s) if p.is_file()),
                key=lambda p: p.name,
            )
            for path in packed:
                console.print(f"  [cyan]{escape(str(path))}[/cyan]", soft_wrap=True)
        else:
            for chrom, path in results.items():
                console.print(f"  {chrom}: [cyan]{path}[/cyan]")

        return results

    except Exception as e:
        logger.exception("Failed to compute accessibility")
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)
