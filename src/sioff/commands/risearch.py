"""CLI commands for running RIsearch index and search operations."""

from pathlib import Path
from typing import Annotated, Any, Optional

import typer

from sioff._logging import setup_logging


def index(
    target: Annotated[Path, typer.Argument(help="Target FASTA file to index.")],
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output index path. Defaults to <target>.idx next to the target file.",
        ),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose logging.")
    ] = False,
) -> Path:
    """Build a RIsearch index from a target FASTA file."""
    from sioff.core import risearch as core

    setup_logging(verbose)
    index_path = core.build_index(target, output)
    typer.echo(f"Index written to: {index_path}")
    return index_path


def search(
    query: Annotated[Path, typer.Argument(help="Query siRNA FASTA file.")],
    index: Annotated[Path, typer.Argument(help="Pre-built RIsearch index (.idx).")],
    target: Annotated[
        Optional[Path],
        typer.Option(
            "--target",
            "-t",
            help="Target FASTA used to build the index (required if index was built externally).",
        ),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output TSV file. Defaults to stdout.",
        ),
    ] = None,
    seed: Annotated[
        str,
        typer.Option(
            "--seed",
            "-s",
            help=(
                "Seed spec, RIsearch2 syntax: 'l' (length only), 'n:m' or 'n:m/l' "
                "(seed must lie in 1-based query positions n..m and be l nt). "
                "E.g. '7', '2:8/7'."
            ),
        ),
    ] = "6",
    no_gu_seed: Annotated[
        bool,
        typer.Option(
            "--no-gu-seed",
            help=(
                "Forbid G:U wobble pairs inside the seed (RIsearch2 --noGUseed). "
                "Affects seed location only, not the energy model, so this is a "
                "candidate-generation knob. Removes ~85% of hits on fly 3'UTRs."
            ),
        ),
    ] = False,
    matrix: Annotated[
        str,
        typer.Option(
            "--matrix",
            "-z",
            help=(
                "Nearest-neighbour energy parameter set: 't04' (Turner 2004, "
                "RNA-RNA), 'slh04' "
                "(SantaLucia-Hicks 2004, DNA-DNA), 's95-rna-dna' or "
                "'s95-dna-rna' (Sugimoto 1995, RNA/DNA hybrids), or the path "
                "of a custom DSM TSV table (columns q1 q2 t1 t2 "
                "delta_g_kcal_per_mol)."
            ),
        ),
    ] = "t04",
    max_extension: Annotated[
        int,
        typer.Option(
            "--max-extension", "-e", help="Max extension length on each side."
        ),
    ] = 20,
    energy_threshold: Annotated[
        float,
        typer.Option(
            "--energy",
            "-E",
            help="Energy threshold in kcal/mol (only hits below this value are kept).",
        ),
    ] = -10.0,
    mismatches: Annotated[
        int,
        typer.Option(
            "--mismatches",
            help=(
                "Mismatches allowed inside the seed (RIsearch --mismatch-max). A "
                "mismatched seed holding an uninterrupted run of the full seed "
                "length is dropped: the perfect seed covers it."
            ),
        ),
    ] = 0,
    mismatch_prefix: Annotated[
        int,
        typer.Option(
            "--mismatch-prefix",
            help="Matches required at the seed's 5' end before any mismatch.",
        ),
    ] = 1,
    mismatch_suffix: Annotated[
        int,
        typer.Option(
            "--mismatch-suffix",
            help="Matches required at the seed's 3' end after any mismatch.",
        ),
    ] = 0,
    temperature: Annotated[
        int,
        typer.Option(
            "--temperature",
            "-T",
            help="Energy-model temperature in °C, 0-50 (a custom DSM table ignores it).",
        ),
    ] = 37,
    penalty: Annotated[
        float,
        typer.Option(
            "--penalty",
            "-d",
            help=(
                "kcal/mol added per extended nucleotide, included in the reported "
                "energy. RIsearch2's C -d took dacal/mol: its recommended 30 is 0.3 here."
            ),
        ),
    ] = 0.0,
    seed_energy: Annotated[
        float,
        typer.Option(
            "--seed-energy",
            help="Accepted by risearch for compatibility; no effect in risearch 3.0.0a4/a5.",
        ),
    ] = 0.0,
    no_max_prune: Annotated[
        bool,
        typer.Option(
            "--no-max-prune",
            help="Also extend non-maximal seeds (gives redundant hits).",
        ),
    ] = False,
    no_dedup: Annotated[
        bool,
        typer.Option(
            "--no-dedup",
            help="Report every seed's hit instead of one best hit per duplex.",
        ),
    ] = False,
    alignment: Annotated[
        bool,
        typer.Option(
            "--alignment",
            help=(
                "Add the columns q_start, q_end and alignment (pairing string, query "
                "5'->3': P pair, W G:U, U mismatch, T/Q extra target/query base)."
            ),
        ),
    ] = False,
    threads: Annotated[
        Optional[int],
        typer.Option(
            "--threads",
            "-j",
            help="Worker threads (default: RAYON_NUM_THREADS, else every core).",
        ),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose logging.")
    ] = False,
) -> Any:
    """Run a RIsearch search and output hits as TSV.

    Returns the hits as a polars DataFrame. Spelled ``Any`` because Typer
    evaluates every annotation while building ``--help``, and polars is
    imported lazily (via the core) to keep that fast.
    """
    from sioff.core import risearch as core
    from sioff.services.risearch_service import RIsearchError

    setup_logging(verbose)
    # Seed geometry and matrix are validated before the search starts, so a bad
    # spec is a one-line CLI error rather than a traceback out of the bindings.
    try:
        seed_start, seed_end, seed_length = core.parse_seed_spec(seed)
        df = core.run_search(
            query=query,
            index=index,
            target=target,
            seed_length=seed_length,
            seed_start=seed_start,
            seed_end=seed_end,
            seed_wobble=not no_gu_seed,
            matrix=matrix,
            max_extension=max_extension,
            energy_threshold=energy_threshold,
            mismatches=mismatches,
            mismatch_prefix=mismatch_prefix,
            mismatch_suffix=mismatch_suffix,
            temperature=temperature,
            penalty=penalty,
            seed_energy=seed_energy,
            no_max_prune=no_max_prune,
            no_dedup=no_dedup,
            alignment=alignment,
            threads=threads,
        )
    except RIsearchError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    output = Path(output) if isinstance(output, str) else output
    if output is not None:
        df.write_csv(output, separator="\t")
        typer.echo(f"Results written to: {output} ({df.height} hits)")
    else:
        typer.echo(df.write_csv(separator="\t"), nl=False)

    return df
