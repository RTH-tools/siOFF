"""Pure core for RIsearch index/search.

Wraps `sioff.services.risearch_service.RIsearchService` without any CLI
side-effects (no stdout, no file writes beyond the index binary itself) and
returns in-memory objects. [`parse_seed_spec`][sioff.core.risearch.parse_seed_spec] turns the RIsearch2 ``-s``
seed string into the ``seed_start``/``seed_end``/``seed_length`` arguments of
[`run_search`][sioff.core.risearch.run_search].

The public wrappers are [`sioff.index`][sioff.index] and [`sioff.search`][sioff.search].
"""

import re
from pathlib import Path
from typing import Optional, Tuple, Union

import polars as pl

from sioff.services.risearch_service import RIsearchError, RIsearchService

_SEED_SPEC = re.compile(r"^\s*(?:(\d+):(\d+))?(?:/?(\d+))?\s*$")


def parse_seed_spec(spec: str) -> Tuple[Optional[int], Optional[int], int]:
    """Parse a RIsearch2 ``-s`` seed specification into ``(start, end, length)``.

    Accepts the forms the RIsearch2 CLI accepts, so a command line can be
    copied between the two without translation (CLI ``-s/--seed``):

    - ``"7"``: length only, no positional constraint -> ``(None, None, 7)``
    - ``"2:8/7"``: seed must lie in query positions 2..8 and be 7 nt
      -> ``(2, 8, 7)``
    - ``"2:8"``: positions given, length defaults to the window width
      -> ``(2, 8, 7)``

    Positions are 1-based and inclusive, matching RIsearch2. Surrounding
    whitespace is ignored.

    Parameters
    ----------
    spec : str
        Seed specification: ``"l"``, ``"n:m"`` or ``"n:m/l"``.

    Returns
    -------
    tuple[int or None, int or None, int]
        ``(seed_start, seed_end, seed_length)`` ready to pass to
        [`run_search`][sioff.core.risearch.run_search]; ``seed_start`` and ``seed_end`` are ``None`` for the
        length-only form.

    Raises
    ------
    RIsearchError
        When ``spec`` is empty, does not match any of the three forms, or gives
        neither positions nor a length.

    See Also
    --------
    run_search : Consumes the returned triple.
    sioff.search : Public wrapper around [`run_search`][sioff.core.risearch.run_search].
    """
    m = _SEED_SPEC.match(spec)
    if not m or spec.strip() == "":
        raise RIsearchError(
            f"Could not parse seed spec {spec!r}. Expected 'l', 'n:m' or 'n:m/l' "
            f"(e.g. '7', '2:8', '2:8/7')."
        )
    start, end, length = m.group(1), m.group(2), m.group(3)
    if start is None and length is None:
        raise RIsearchError(f"Could not parse seed spec {spec!r}: no length given.")
    if start is None:
        return None, None, int(length)
    s, e = int(start), int(end)
    return s, e, int(length) if length is not None else e - s + 1


def build_index(target: Path, output: Optional[Path] = None) -> Path:
    """Build (or reuse) a RIsearch index for a target FASTA.

    The index is a binary on-disk artifact consumed by [`run_search`][sioff.core.risearch.run_search], so
    this returns its `Path` rather than in-memory data. An
    existing index at the output path is reused when it is newer than the
    target and rebuilt otherwise. The target is registered against the index
    so that a later [`run_search`][sioff.core.risearch.run_search] in the same process can resolve target
    names without ``target``.

    Exceptions from the underlying
    `index_target`
    propagate unchanged: `FileNotFoundError` when ``target`` does not
    exist, and `RIsearchError` when
    the risearch bindings fail to build the index.

    Parameters
    ----------
    target : pathlib.Path
        Target FASTA to index (CLI positional ``target``).
    output : pathlib.Path, optional
        Index path (CLI ``-o/--output``). Defaults to ``<target>.idx`` next to
        the target file.

    Returns
    -------
    pathlib.Path
        Path of the (new or reused) index file.

    See Also
    --------
    sioff.index : Public wrapper accepting ``str`` paths.
    run_search : Searches against the index built here.
    """
    target = Path(target)
    output = Path(output) if output is not None else None
    return RIsearchService().index_target(target, output)


def run_search(
    query: Path,
    index: Path,
    target: Optional[Path] = None,
    seed_length: int = 6,
    max_extension: int = 20,
    energy_threshold: float = -10.0,
    seed_start: Optional[int] = None,
    seed_end: Optional[int] = None,
    seed_wobble: bool = True,
    matrix: Union[str, Path] = "t04",
    mismatches: int = 0,
    mismatch_prefix: int = 1,
    mismatch_suffix: int = 0,
    temperature: int = 37,
    penalty: float = 0.0,
    seed_energy: float = 0.0,
    no_max_prune: bool = False,
    no_dedup: bool = False,
    alignment: bool = False,
    threads: Optional[int] = None,
    sense_only: bool = False,
) -> pl.DataFrame:
    """Run a RIsearch search and return hits as a Polars DataFrame.

    ``target`` is required when the index was built outside this process (it
    is needed to resolve RIsearch's integer target indices to sequence names);
    after [`build_index`][sioff.core.risearch.build_index] in the same process it may be omitted.

    Every ``risearch.search`` option is passed through, each with siOFF's own
    default, so an upstream default change cannot move the results; the
    defaults reproduce the searches siOFF ran before these options existed.

    Seed geometry and energy-model parameters mirror the RIsearch2 CLI.
    ``seed_start``/``seed_end``/``seed_length`` are the ``-s n:m/l`` seed spec:
    the seed must fall within query positions ``n..m`` and be ``l`` nt long;
    leaving ``seed_start``/``seed_end`` as ``None`` constrains length only,
    which is RIsearch's default behaviour. ``seed_wobble=False`` is
    ``--noGUseed`` and forbids G:U wobble pairs inside the seed; this affects
    seed location only, not the energy model, so it is a candidate-generation
    knob rather than a rescoring one. Measured on 3,826 fly S2 3'UTRs x 10
    miRNAs it removes 84.7% of hits (783,836 -> 119,928). ``matrix`` is ``-z``,
    the nearest-neighbour energy parameter set: a bundled id or the path of a
    custom DSM TSV table.

    Exceptions from the underlying
    `run_search`
    propagate unchanged: `FileNotFoundError` when ``query`` or ``index``
    does not exist, and `RIsearchError`
    when ``target`` is needed but missing, the seed geometry is invalid,
    ``matrix`` is neither a bundled id nor an existing file, a mismatch count
    is negative, ``temperature`` is outside 0-50 °C, ``threads`` is below 1,
    or the risearch bindings fail.

    Parameters
    ----------
    query : pathlib.Path
        Query siRNA FASTA (CLI positional ``query``).
    index : pathlib.Path
        Pre-built RIsearch index, typically from [`build_index`][sioff.core.risearch.build_index] (CLI
        positional ``index``).
    target : pathlib.Path, optional
        Target FASTA the index was built from (CLI ``-t/--target``). Required
        when the index was built externally or in another process.
    seed_length : int, default 6
        Seed length ``l`` in nt (the ``l`` of CLI ``-s/--seed``).
    max_extension : int, default 20
        Maximum extension length on each side of the seed (CLI
        ``-e/--max-extension``).
    energy_threshold : float, default -10.0
        Energy threshold in kcal/mol; only hits below this value are kept (CLI
        ``-E/--energy``).
    seed_start : int, optional
        First allowed 1-based query position of the seed (the ``n`` of ``-s
        n:m/l``). Must be given together with ``seed_end``.
    seed_end : int, optional
        Last allowed 1-based query position of the seed, inclusive (the ``m``
        of ``-s n:m/l``). Must be given together with ``seed_start``.
    seed_wobble : bool, default True
        Allow G:U wobble pairs inside the seed. ``False`` is RIsearch2's
        ``--noGUseed`` (CLI ``--no-gu-seed``).
    matrix : str or pathlib.Path, default "t04"
        Nearest-neighbour energy parameter set (CLI ``-z/--matrix``): one of
        the bundled ids ``"t04"`` (Turner 2004, RNA-RNA), ``"slh04"``
        (SantaLucia-Hicks 2004, DNA-DNA), ``"s95-rna-dna"`` or
        ``"s95-dna-rna"`` (Sugimoto 1995, RNA/DNA hybrids), or the path of a
        custom long-form DSM TSV table with columns
        ``q1 q2 t1 t2 delta_g_kcal_per_mol``.
    mismatches : int, default 0
        Mismatches allowed inside a seed (CLI ``--mismatches``; RIsearch's
        ``--mismatch-max``). A mismatched seed that holds an uninterrupted run
        of the full seed length is dropped, as the perfect seed covers it.
    mismatch_prefix : int, default 1
        Matches required at the seed's 5' end before any mismatch (CLI
        ``--mismatch-prefix``).
    mismatch_suffix : int, default 0
        Matches required at the seed's 3' end after any mismatch (CLI
        ``--mismatch-suffix``).
    temperature : int, default 37
        Energy-model temperature in °C, 0 to 50 (CLI ``-T/--temperature``). A
        custom DSM table ignores it.
    penalty : float, default 0.0
        kcal/mol added per extended nucleotide, included in the reported
        energy (CLI ``-d/--penalty``). RIsearch2's C ``-d`` took dacal/mol: its
        recommended 30 is 0.3 here.
    seed_energy : float, default 0.0
        Accepted by risearch for compatibility; it has no effect in risearch
        3.0.0a4/a5 (CLI ``--seed-energy``).
    no_max_prune : bool, default False
        Also extend non-maximal seeds, which gives redundant hits (CLI
        ``--no-max-prune``).
    no_dedup : bool, default False
        Report every seed's hit instead of one best hit per duplex (CLI
        ``--no-dedup``).
    alignment : bool, default False
        Add the columns ``q_start``, ``q_end`` (first and last paired query
        nucleotide, counted from 1 like ``start``/``end``) and ``alignment``
        (pairing string,
        query 5' to 3': ``P`` pair, ``W`` G:U, ``U`` mismatch, ``T`` / ``Q``
        extra target / query base) (CLI ``--alignment``).
    threads : int, optional
        Worker threads; ``None`` uses ``RAYON_NUM_THREADS``, else every core
        (CLI ``-j/--threads``).
    sense_only : bool, default False
        Keep only ``+``-strand hits, for a target whose records are already in
        sense orientation, such as a transcriptome (CLI ``--sense-only``). A
        siOFF filter, not a risearch option: risearch still searches both
        strands, so the search costs the same time and memory.

    Returns
    -------
    polars.DataFrame
        One row per hit with columns ``sirna_id, chrom, start, end, strand,
        energy`` (the schema [`sioff.off_targets`][sioff.off_targets] accepts as
        ``predictions``), followed by ``q_start, q_end, alignment`` when
        ``alignment`` is set. ``start`` and ``end`` are the first and last
        target nucleotide of the site, counted from 1 along the target as
        written (the RIsearch2 convention; the ``risearch`` package counts
        from 0).

    See Also
    --------
    sioff.search : Public wrapper accepting ``str`` paths.
    parse_seed_spec : Builds ``seed_start``/``seed_end``/``seed_length`` from
        a ``-s`` string.
    build_index : Builds the ``index`` argument.
    """
    return RIsearchService().run_search(
        query_path=Path(query),
        index_path=Path(index),
        target_fasta=Path(target) if target is not None else None,
        seed_length=seed_length,
        max_extension=max_extension,
        energy_threshold=energy_threshold,
        seed_start=seed_start,
        seed_end=seed_end,
        seed_wobble=seed_wobble,
        matrix=matrix,
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
        sense_only=sense_only,
    )
