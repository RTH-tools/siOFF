"""Public Python API for siOFF.

Importable functions that return results **in memory** and raise ordinary Python
exceptions: no output files are written, nothing is printed, and no
``typer.Exit`` / Click exceptions leak out. These wrap the pure `sioff.core` layer; the
``sioff`` CLI is a separate, file-writing wrapper over the same core.

The four public functions are [`off_targets`][sioff.off_targets], [`accessibility`][sioff.accessibility],
[`index`][sioff.index] and [`search`][sioff.search]. They are re-exported at package level, so
``import sioff`` is all a script needs.

Notes
-----
``index`` returns a `Path`: a RIsearch index is a binary
on-disk artifact, so the path (not in-memory data) is the natural result.

``sioff.index`` / ``sioff.search`` require the external ``risearch`` package,
the same dependency the CLI's index/search commands need. ``off_targets``
imports it only for the in-process search and for the perfect-match energies
of E_min (an alpha/gamma pair other than (1, 1), or ``detailed_report``);
``accessibility`` never does.

Examples
--------
>>> import sioff
>>> import polars as pl

A predictions file, or a directory of them -> one DataFrame.

>>> df = sioff.off_targets(risearch_file="predictions.tsv", gtf_file="ann.gtf")
>>> df = sioff.off_targets(risearch_file="preds_dir/", gtf_file="ann.gtf")

The same, one DataFrame per siRNA as each is scored.

>>> for sirna_df in sioff.off_targets(risearch_file="preds_dir/", stream=True):
...     ...

RIsearch index / search, and straight into the off-target analysis with no
intermediate file.

>>> idx = sioff.index("target.fa")  # Path (binary artifact)
>>> hits = sioff.search("query.fa", idx, target="target.fa")  # pl.DataFrame
>>> df = sioff.off_targets(predictions=hits, gtf_file="ann.gtf")

Accessibility profiles in memory, keyed by chromosome.

>>> profiles = sioff.accessibility(genome="genome.fa")  # dict[str, pl.DataFrame]
>>> df = sioff.off_targets(predictions=hits, accessibility=profiles)
"""

from pathlib import Path
from typing import Generator, Iterator, Mapping, Optional, Union, cast

import polars as pl

from sioff.core import accessibility as _accessibility
from sioff.core import off_targets as _off_targets
from sioff.core import risearch as _risearch

__all__ = ["off_targets", "accessibility", "index", "search"]


def _p(value: Optional[Union[str, Path]]) -> Optional[Path]:
    """Coerce a str path (scripting style) to Path; pass Path/None through.

    Parameters
    ----------
    value : str or pathlib.Path, optional
        Path-like value from the caller, or ``None``.

    Returns
    -------
    pathlib.Path or None
        ``Path(value)`` for a ``str``; *value* unchanged otherwise.
    """
    return Path(value) if isinstance(value, str) else value


def off_targets(
    risearch_file: Optional[Union[str, Path]] = None,
    predictions: Optional[pl.DataFrame] = None,
    sirna_fasta: Optional[Union[str, Path]] = None,
    target_fasta: Optional[Union[str, Path]] = None,
    target_index: Optional[Union[str, Path]] = None,
    gtf_file: Optional[Union[str, Path]] = None,
    feature_type: str = "exon",
    expression_metric: str = "RPKM",
    transcriptome_format: str = "auto",
    accessibility: Optional[Mapping[str, pl.DataFrame]] = None,
    accessibility_dir: Optional[Union[str, Path]] = None,
    accessibility_cache_dir: Optional[Union[str, Path]] = None,
    genome_file: Optional[Union[str, Path]] = None,
    window_size: int = 80,
    max_span: int = 40,
    unpaired_prob: int = 30,
    temperature: float = 37.0,
    matrix: Union[str, Path] = "t04",
    sirna_table: Optional[Union[str, Path]] = None,
    alpha: str = "1.0",
    gamma: str = "1.0",
    theta: str = "",
    sense_only: bool = False,
    predictions_type: str = "gw",
    n_workers: int = 1,
    detailed_report: bool = False,
    stream: bool = False,
) -> Union[pl.DataFrame, Iterator[pl.DataFrame]]:
    """Analyse siRNA off-target predictions, returning results in memory.

    Intersects RIsearch predictions with a transcriptome annotation, adds an
    accessibility (opening-energy) penalty, and computes partition-function
    off-target probabilities. This is the in-memory counterpart of the
    ``sioff off-targets`` command; the defaults reproduce its behaviour, and each
    parameter below names its CLI flag.

    Predictions come from exactly one of:

    - ``risearch_file``: a predictions file, or a directory of them. Each file
      may hold one siRNA or many.
    - ``predictions``: an in-memory `polars.DataFrame` in the
      [`search`][sioff.search] schema (``sirna_id, chrom, start, end, strand, energy``),
      typically the value returned by [`search`][sioff.search].
    - ``sirna_fasta`` + ``target_fasta``: run RIsearch in-process first
      (requires the ``risearch`` package).

    Every form takes the same path: the predictions are split into one part
    per siRNA, and each siRNA is scored on its own, in ``n_workers``
    processes. The result is one DataFrame, or with ``stream=True`` one
    DataFrame per siRNA as each is scored.

    **The siRNA table** (``sirna_table``) gives each siRNA its on-target and
    its guide sequence. **E_min**, the reference energy of the alpha/gamma
    clamp, is the lower of the siRNA's perfect-match energy (its sequence
    searched against itself with ``matrix``) and its strongest hit left after
    ``sense_only`` and the intersection. Every alpha/gamma pair other than
    (1, 1) needs it, so every siRNA with predictions then needs a table row.

    Accessibility profiles come from at most one of ``accessibility`` (the
    ``dict[chrom -> DataFrame]`` returned by [`accessibility`][sioff.accessibility]),
    ``accessibility_dir`` (the packed cache or per-chromosome Parquet files)
    or ``genome_file`` (fold during the call). Without any profile source the
    probabilities are computed from hybridisation energy alone.

    Writes no output files and prints nothing. The in-process search builds
    the index ``<target>.idx`` next to ``target_fasta`` when ``target_index``
    is omitted, and ``accessibility_dir`` gets its packed cache; temporary
    files are removed before the call returns (with ``stream=True``, when the
    stream is exhausted or closed).

    Parameters
    ----------
    risearch_file : str or pathlib.Path, optional
        Pre-computed predictions: a file, or a directory of files. Read are
        RIsearch2 output (plain or ``.gz``), ``sioff search -o`` TSV and
        Parquet; from a directory every ``*.gz``, ``*.tsv``, ``*.out`` and
        ``*.parquet``. A ``sirna_id`` found in two files is an error. A
        target name is cut at its first whitespace: the ``risearch`` command
        line writes the whole FASTA header, while the annotation and the
        accessibility profiles are keyed by its first word. CLI:
        ``-r/--risearch-file``.
    predictions : polars.DataFrame, optional
        In-memory predictions in the [`search`][sioff.search] schema. The columns
        ``sirna_id, chrom, start, end, strand, energy`` are required; extra
        columns are dropped and dtypes are cast to the schema (``Utf8`` ids and
        strand, ``Int32`` coordinates, ``Float32`` energy), so a frame that went
        through Parquet or your own manipulation still works. ``chrom`` is
        cut at its first whitespace, as for ``risearch_file``. Mutually
        exclusive with ``risearch_file``. No CLI equivalent.
    sirna_fasta : str or pathlib.Path, optional
        siRNA FASTA with one or more sequences and unique IDs, searched
        in-process against ``target_fasta``. It is only the search query: it
        cannot be combined with ``risearch_file`` or ``predictions``, and the
        perfect-match energies come from ``sirna_table``. CLI:
        ``-s/--sirna-fasta``.
    target_fasta : str or pathlib.Path, optional
        Target FASTA (genome or transcriptome) for the in-process RIsearch
        run; required together with ``sirna_fasta`` and refused with
        ``risearch_file`` or ``predictions``. CLI: ``--target-fasta`` (alias
        ``--genome``).
    target_index : str or pathlib.Path, optional
        Pre-built RIsearch index of ``target_fasta``, to skip indexing on
        repeated runs; refused with ``risearch_file`` or ``predictions``. When
        omitted the index is built (or reused) next to ``target_fasta`` with an
        ``.idx`` suffix. CLI: ``-idx/--target-index``.
    gtf_file : str or pathlib.Path, optional
        Transcriptome annotation (GTF or GFF3). Predictions are
        intersected with its features, which adds ``transcript_id``,
        ``gene_id`` and the expression column ``exp_value``; features with
        expression <= 0 are dropped first. Without it no intersection is
        performed and every prediction is kept with unit expression. CLI:
        ``-t/--transcriptome``.
    feature_type : str, default "exon"
        Feature type to select from a GTF/GFF3 annotation; a type that no
        row of the file has is an error. CLI: ``--feature``.
    expression_metric : str, default "RPKM"
        Annotation attribute used as the expression score (``exp_value``).
        CLI: ``--expression-metric``.
    transcriptome_format : str, default "auto"
        One of ``"auto"``, ``"gtf"`` or ``"gff3"``; BED is rejected.
        ``"auto"`` detects the format from the file, telling GTF from GFF3 by
        the attribute column. CLI: ``--transcriptome-format``.
    accessibility : dict[str, polars.DataFrame], optional
        In-memory accessibility profiles keyed by chromosome, each frame with
        columns ``position, strand, u1..uN`` and both strands stacked, as
        returned by [`accessibility`][sioff.accessibility]. They are written to a
        temporary directory and packed into the same cache
        ``accessibility_dir`` uses, so the opening energies match that path.
        Each key becomes a file name, so it cannot be empty, ``.`` or ``..``,
        or hold ``/``; nor whitespace, since a target is matched by the
        first word of its name. Mutually exclusive with
        ``accessibility_dir``. No CLI equivalent.
    accessibility_dir : str or pathlib.Path, optional
        The packed npy cache from ``sioff accessibility`` or per-chromosome
        Parquet files. Parquet is packed into the cache in
        ``accessibility_cache_dir`` before scoring starts, and rebuilt when
        the Parquet changes. CLI: ``-a/--accessibility-dir``.
    accessibility_cache_dir : str or pathlib.Path, optional
        Where the cache of ``accessibility_dir`` is read from and built;
        default ``accessibility_dir``. Created if absent; use it when
        ``accessibility_dir`` is read-only or to stage the cache on local
        disk. It may also hold a complete cache without any Parquet. Refused
        without ``accessibility_dir``. CLI: ``--accessibility-cache-dir``.
    genome_file : str or pathlib.Path, optional
        Genome FASTA to fold during the call; the profiles go to a temporary
        directory that is removed when the run ends. Refused with
        ``accessibility`` or ``accessibility_dir``. CLI: ``-f/--fasta``.
    window_size : int, default 80
        Local-folding window size ``W`` for ``genome_file``. CLI:
        ``-W/--window``.
    max_span : int, default 40
        Maximum base-pair span ``L`` for ``genome_file``. CLI:
        ``-L/--span``.
    unpaired_prob : int, default 30
        Length ``u`` of the unpaired stretch for which opening energies are
        computed (profile columns ``u1..u{unpaired_prob}``). CLI:
        ``-u/--unpaired``.
    temperature : float, default 37.0
        Folding temperature in degrees Celsius. Affects both the
        ``genome_file`` fold and the ``RT`` term of the partition function.
        CLI: ``-T/--temperature``.
    matrix : str or pathlib.Path, default "t04"
        Energy parameter set for the in-process search and for the
        perfect-match energies of E_min: a bundled id or the path of a DSM
        TSV table. Use the set the predictions were searched with; E_min is
        the reference energy of the alpha/gamma clamp. CLI: ``-z/--matrix``.
    sirna_table : str or pathlib.Path, optional
        Tab-separated siRNA table with a header row: ``sirna_id``,
        ``on_target_id`` (the transcript or gene the siRNA is designed
        against; may be empty) and ``sequence`` (the guide strand, ACGU or
        ACGT). Predictions of a siRNA that fall on its on-target are tagged
        ``is_on_target`` and contribute the on-target weight instead of
        counting as off-targets; this needs ``gtf_file``, so that
        ``transcript_id`` exists. The sequence gives the perfect-match energy
        of E_min. Required when an alpha/gamma pair other than (1, 1) is
        computed; optional otherwise. CLI: ``--sirna-table``.
    alpha : str, default "1.0"
        Alpha clamping parameter(s) as semicolon-separated floats, e.g.
        ``"0.8;1.0"``. For every ``(alpha, gamma)`` pair with
        ``alpha <= gamma``, predictions whose ``energy < alpha * E_min`` have
        their energy replaced by ``gamma * E_min``. The baseline ``(1.0, 1.0)``
        is always computed and every other pair adds a
        ``P_off_target:alpha=X,gamma=Y`` column. CLI: ``--alpha``.
    gamma : str, default "1.0"
        Gamma clamping parameter(s), same format as ``alpha``; the two lists
        are combined as a Cartesian product. CLI: ``--gamma``.
    theta : str, default ""
        Theta scaling parameter(s) as semicolon-separated floats, e.g.
        ``"0.5;0.7"``; the empty string disables the sweep. Each value rescales
        energies as ``theta * (energy + 10) - 10`` and adds a
        ``P_off_target:theta=T`` column. CLI: ``--theta``.
    sense_only : bool, default False
        Keep only sense-strand (``+``) predictions and drop antisense ones.
        CLI: ``--sense-only``.
    predictions_type : str, default "gw"
        ``"gw"`` for genome-wide predictions (interval overlap with annotation
        features) or ``"tw"`` for transcriptome-wide predictions, where
        ``chrom`` holds a transcript ID that is joined on ``transcript_id``.
        CLI: ``--type``.
    n_workers : int, default 1
        Worker processes that split the inputs and score the siRNAs, one
        siRNA per task, for every input form. A pool is started only when
        more than one task can run at once. CLI: ``-j/--workers``, whose
        default is the CPU count.
    detailed_report : bool, default False
        Keep the raw ``boltzmann_weight*``, ``Z_sirna*`` and ``E_min`` columns
        for every interaction in the output; by default they are dropped.
        ``E_min`` follows the rule above for a siRNA with a ``sirna_table``
        row; for one without (no table, or a baseline run's missing row) it
        is the strongest hit left. CLI: ``--detailed-report``.
    stream : bool, default False
        Return a generator of one DataFrame per siRNA, in the order they are
        scored, instead of one DataFrame. The worker pool and the temporary
        files live as long as the generator: consume it fully, or close it
        (``contextlib.closing``), for prompt cleanup. The generator is lazy,
        so the argument errors under Raises come from the first ``next()``,
        not from this call. No CLI equivalent; the CLI always streams to its
        output file.

    Returns
    -------
    polars.DataFrame or Iterator[polars.DataFrame]
        One DataFrame holding every siRNA's rows (an empty DataFrame with no
        columns when nothing was scored), or with ``stream=True`` a generator
        yielding one DataFrame per siRNA. Each row is one prediction with the input columns
        ``sirna_id, chrom, start, end, strand, energy``; the annotation
        columns ``trans_start, trans_end, gene_id, transcript_id`` when
        ``gtf_file`` is given; ``exp_value`` (1.0 without ``gtf_file``);
        ``opening_energy`` (the accessibility penalty, kcal/mol; 0.0 without
        a profile source); ``dG_total`` (``energy + opening_energy``);
        ``is_on_target``; ``P_off_target`` (baseline alpha=gamma=theta=1);
        and ``energy``, ``dG_total`` and ``P_off_target`` again with a
        ``:alpha=X,gamma=Y`` or ``:theta=T`` suffix per extra parameter
        value. With ``detailed_report`` the ``E_min``, ``boltzmann_weight*``
        and ``Z_sirna*`` intermediates are kept too. The weights and Z are
        Float64; every ``P_off_target*`` column is Float32.

    Raises
    ------
    ValueError
        When no predictions source is given; when an argument would go
        unused: ``predictions`` with ``risearch_file``; ``sirna_fasta``,
        ``target_fasta`` or ``target_index`` with ``risearch_file`` or
        ``predictions``; ``accessibility`` with ``accessibility_dir``;
        ``genome_file`` with ``accessibility`` or ``accessibility_dir``;
        ``accessibility_cache_dir`` without ``accessibility_dir``. Also when
        ``accessibility`` is empty or has a key that cannot be a file name;
        when ``sirna_fasta`` is given without ``target_fasta``; when
        ``predictions`` lacks one of its required columns; when
        ``sirna_fasta`` contains duplicate IDs; when ``sirna_table`` cannot
        be read; when a ``sirna_id`` occurs in two input files; and when an
        alpha/gamma pair other than (1, 1) is computed and a siRNA with
        predictions has no row in ``sirna_table`` (or there is no table).
    FileNotFoundError
        When ``risearch_file``, ``gtf_file``, ``genome_file``, ``sirna_fasta``,
        ``sirna_table``, ``target_fasta`` or ``target_index`` does not exist,
        or a directory ``risearch_file`` contains no prediction files.
    RIsearchError
        `sioff.services.risearch_service.RIsearchError`, when the
        in-process RIsearch run (``sirna_fasta`` form) fails, including when the
        ``risearch`` package is not installed.
    AccessibilityError
        `sioff.services.accessibility.AccessibilityError`, when folding
        ``genome_file`` fails, an accessibility profile is missing or
        malformed, ``accessibility_dir`` holds no profile, or the cache
        cannot be written.
    WorkerFailureError
        `sioff.core.off_targets.WorkerFailureError`, when an input file
        could not be split or a siRNA could not be scored. With
        ``stream=True`` it is raised after the last siRNA that succeeded; with
        ``stream=False`` nothing is returned, so the frames already scored are
        discarded.

    See Also
    --------
    search : Produce a ``predictions`` frame in the schema this function accepts.
    accessibility : Produce the in-memory ``accessibility`` profiles.

    Examples
    --------
    A predictions file with annotation, pre-computed accessibility, a siRNA
    table and an alpha sweep:

    >>> import sioff
    >>> df = sioff.off_targets(
    ...     risearch_file="predictions.tsv",
    ...     gtf_file="annotation.gtf",
    ...     accessibility_dir="accessibility/",
    ...     sirna_table="sirna_table.tsv",
    ...     alpha="0.8;1.0",
    ...     gamma="1.0",
    ... )
    >>> df.select("sirna_id", "transcript_id", "P_off_target").head()

    A directory of per-siRNA files, consumed as a stream and torn down promptly:

    >>> import contextlib
    >>> gen = sioff.off_targets(risearch_file="predictions/", n_workers=8, stream=True)
    >>> with contextlib.closing(gen) as frames:
    ...     for frame in frames:
    ...         print(frame["sirna_id"][0], frame.height)
    """
    core_gen = _off_targets.compute_off_targets(
        predictions=predictions,
        risearch_file=_p(risearch_file),
        sirna_fasta=_p(sirna_fasta),
        target_fasta=_p(target_fasta),
        target_index=_p(target_index),
        gtf_file=_p(gtf_file),
        feature_type=feature_type,
        expression_metric=expression_metric,
        transcriptome_format=transcriptome_format,
        accessibility=accessibility,
        accessibility_dir=_p(accessibility_dir),
        accessibility_cache_dir=_p(accessibility_cache_dir),
        genome_file=_p(genome_file),
        window_size=window_size,
        max_span=max_span,
        unpaired_prob=unpaired_prob,
        temperature=temperature,
        matrix=matrix,
        sirna_table=_p(sirna_table),
        alpha=alpha,
        gamma=gamma,
        theta=theta,
        sense_only=sense_only,
        predictions_type=predictions_type,
        n_workers=n_workers,
        detailed_report=detailed_report,
    )
    if stream:
        return _frames(core_gen)
    frames = [frame for frame, _meta in core_gen]
    return pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()


def _frames(
    core_gen: Generator[tuple[pl.DataFrame, dict], None, None],
) -> Iterator[pl.DataFrame]:
    """Yield the frames of ``core_gen`` and close it when this generator closes.

    Parameters
    ----------
    core_gen : Generator[tuple[polars.DataFrame, dict], None, None]
        The engine's ``(df, meta)`` generator.

    Yields
    ------
    polars.DataFrame
        One siRNA's frame.
    """
    try:
        for frame, _meta in core_gen:
            yield frame
    finally:
        # A for loop does not close what it iterates: without this, closing
        # the stream early would leave the pool and the temporary files to
        # garbage collection.
        core_gen.close()


def accessibility(
    genome: Union[str, Path],
    window_size: int = 80,
    max_span: int = 40,
    unpaired_prob: int = 30,
    temperature: float = 37.0,
    strand: str = "both",
) -> dict[str, pl.DataFrame]:
    """Compute per-chromosome accessibility profiles in memory.

    Folds every sequence in *genome* with ViennaRNA (local folding, both
    strands unless ``strand="+"``) and returns the opening-energy profiles in the same schema the
    ``sioff accessibility`` command streams to ``{chrom}.accessibility.parquet``.
    Writes no files. The result is the ``accessibility`` argument of
    [`off_targets`][sioff.off_targets]. Every profile stays resident at once, so for
    genome-scale inputs the CLI's streaming path (then ``accessibility_dir``) is
    the cheaper route.

    Parameters
    ----------
    genome : str or pathlib.Path
        Genome or transcriptome FASTA; one profile is produced per record.
        CLI: ``-f/--fasta``.
    window_size : int, default 80
        Local-folding window size ``W``. CLI: ``-W/--window``.
    max_span : int, default 40
        Maximum base-pair span ``L``. CLI: ``-L/--span``.
    unpaired_prob : int, default 30
        Length ``u`` of the unpaired stretch for which opening energies are
        computed; sets the number of ``u`` columns. CLI: ``-u/--unpaired``.
    temperature : float, default 37.0
        Folding temperature in degrees Celsius. CLI: ``-T/--temperature``.
    strand : {"both", "+"}, default "both"
        ``"+"`` folds each sequence as given only, skipping its reverse
        complement: for a transcriptome FASTA, whose records are already
        sense. Hits on ``-`` then have no profile, so pair it with
        ``off_targets(sense_only=True)``. CLI: ``--strand``.

    Returns
    -------
    dict[str, polars.DataFrame]
        Chromosome (FASTA record ID) to profile, in file order. Each frame has
        the columns ``position`` (1-based), ``strand`` (``"+"`` or ``"-"``) and
        ``u1..u{unpaired_prob}`` (opening energies in kcal/mol), with all
        ``+`` rows followed by all ``-`` rows (none with ``strand="+"``).

    Raises
    ------
    FileNotFoundError
        When *genome* does not exist.
    AccessibilityError
        `sioff.services.accessibility.AccessibilityError`, when
        ViennaRNA fails to fold a sequence or ``strand`` is not ``"both"``
        or ``"+"``.

    See Also
    --------
    off_targets : Consumes the returned dict through its ``accessibility`` argument.

    Examples
    --------
    >>> import sioff
    >>> profiles = sioff.accessibility("genome.fa", unpaired_prob=30)
    >>> profiles["chr1"].columns[:4]
    ['position', 'strand', 'u1', 'u2']
    >>> df = sioff.off_targets(risearch_file="predictions.tsv", accessibility=profiles)
    """
    return _accessibility.compute_accessibility(
        cast(Path, _p(genome)),  # genome is required, so _p never returns None
        window_size=window_size,
        max_span=max_span,
        unpaired_prob=unpaired_prob,
        temperature=temperature,
        strand=strand,
    )


def index(
    target: Union[str, Path],
    output: Optional[Union[str, Path]] = None,
) -> Path:
    """Build (or reuse) a RIsearch index for a target FASTA.

    Runs the in-process ``risearch`` indexer and returns the index location. An
    existing index that is newer than *target* is reused as is; an outdated one
    is rebuilt. The function returns a `Path` rather than data
    because a RIsearch index is a binary on-disk artifact consumed by
    [`search`][sioff.search]. Mirrors the ``sioff index`` command and, like it, requires
    the external ``risearch`` package.

    Parameters
    ----------
    target : str or pathlib.Path
        Target FASTA file to index. CLI: the ``TARGET`` argument.
    output : str or pathlib.Path, optional
        Index path. Defaults to ``<target>.idx`` next to the target file. CLI:
        ``-o/--output``.

    Returns
    -------
    pathlib.Path
        Path of the (new or reused) index file.

    Raises
    ------
    FileNotFoundError
        When *target* does not exist.
    RIsearchError
        `sioff.services.risearch_service.RIsearchError`, when indexing
        fails, including when the ``risearch`` package is not installed.

    See Also
    --------
    search : Query the returned index.

    Examples
    --------
    >>> import sioff
    >>> idx = sioff.index("target.fa")
    >>> idx.name
    'target.idx'
    >>> hits = sioff.search("sirnas.fa", idx, target="target.fa")
    """
    return _risearch.build_index(cast(Path, _p(target)), _p(output))


def search(
    query: Union[str, Path],
    index: Union[str, Path],
    target: Optional[Union[str, Path]] = None,
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
    """Run a RIsearch search and return the hits as a DataFrame.

    Searches every sequence in *query* against a RIsearch *index* in-process
    (no subprocess, no intermediate file) and returns the hits in the schema
    [`off_targets`][sioff.off_targets] accepts through its ``predictions`` argument. Mirrors
    the ``sioff search`` command; the defaults reproduce its behaviour, and the
    function requires the external ``risearch`` package.

    ``seed_start`` / ``seed_end`` / ``seed_length`` are the RIsearch2
    ``-s n:m/l`` seed specification and ``seed_wobble=False`` is
    ``--noGUseed``; both affect where seeds may be placed, not the energy
    model. ``matrix`` (``-z``) selects the nearest-neighbour parameter set: a
    bundled id (see `sioff.services.risearch_service.VALID_DSM_IDS` for
    the models risearch ships) or, given a path, a custom long-form DSM TSV
    table. Every other ``risearch.search`` option is passed through too, each
    with siOFF's own default, so an upstream default change cannot move the
    results.

    Parameters
    ----------
    query : str or pathlib.Path
        Query siRNA FASTA. CLI: the ``QUERY`` argument.
    index : str or pathlib.Path
        RIsearch index built by [`index`][sioff.index] or ``sioff index``. CLI: the
        ``INDEX`` argument.
    target : str or pathlib.Path, optional
        FASTA the index was built from, used to map RIsearch's integer target
        indices back to sequence names. Optional in the signature, but required
        in practice through this function: the name registry that could resolve
        it lives in a service instance this call does not keep, so omitting it
        raises ``RIsearchError``. CLI: ``-t/--target``.
    seed_length : int, default 6
        Seed length ``l`` in nucleotides. With ``seed_start`` / ``seed_end`` it
        must not exceed the window ``seed_end - seed_start + 1``. CLI: the
        ``l`` part of ``-s/--seed``.
    max_extension : int, default 20
        Maximum extension length on each side of the seed. CLI:
        ``-e/--max-extension``.
    energy_threshold : float, default -10.0
        Energy threshold in kcal/mol; only hits with energy below this value
        are kept. CLI: ``-E/--energy``.
    seed_start : int, optional
        First 1-based query position the seed may occupy (``n`` in
        ``-s n:m/l``). Must be given together with ``seed_end``; both ``None``
        constrains the seed by length only, RIsearch's default. CLI: the ``n``
        part of ``-s/--seed``.
    seed_end : int, optional
        Last 1-based query position the seed may occupy (``m`` in
        ``-s n:m/l``), inclusive and not below ``seed_start``. CLI: the ``m``
        part of ``-s/--seed``.
    seed_wobble : bool, default True
        Allow G:U wobble pairs inside the seed. ``False`` is RIsearch2's
        ``--noGUseed``: a candidate-generation filter that changes seed
        placement, not scoring. CLI: ``--no-gu-seed`` sets it to ``False``.
    matrix : str or pathlib.Path, default "t04"
        Nearest-neighbour energy parameter set: one of the bundled ids
        ``"t04"`` (Turner 2004, RNA-RNA), ``"slh04"`` (SantaLucia-Hicks 2004,
        DNA-DNA), ``"s95-rna-dna"`` or ``"s95-dna-rna"`` (Sugimoto 1995,
        RNA/DNA hybrids), or the path of a custom long-form DSM TSV table with
        columns ``q1 q2 t1 t2 delta_g_kcal_per_mol``. CLI: ``-z/--matrix``.
    mismatches : int, default 0
        Mismatches allowed inside a seed (RIsearch's ``--mismatch-max``). A
        mismatched seed that holds an uninterrupted run of the full seed length
        is dropped, as the perfect seed covers it. CLI: ``--mismatches``.
    mismatch_prefix : int, default 1
        Matches required at the seed's 5' end before any mismatch. CLI:
        ``--mismatch-prefix``.
    mismatch_suffix : int, default 0
        Matches required at the seed's 3' end after any mismatch. CLI:
        ``--mismatch-suffix``.
    temperature : int, default 37
        Energy-model temperature in °C, 0 to 50; a custom DSM table ignores it.
        CLI: ``-T/--temperature``.
    penalty : float, default 0.0
        kcal/mol added per extended nucleotide, included in the reported
        energy. RIsearch2's C ``-d`` took dacal/mol: its recommended 30 is 0.3
        here. CLI: ``-d/--penalty``.
    seed_energy : float, default 0.0
        Accepted by risearch for compatibility; it has no effect in risearch
        3.0.0a4/a5. CLI: ``--seed-energy``.
    no_max_prune : bool, default False
        Also extend non-maximal seeds, which gives redundant hits. CLI:
        ``--no-max-prune``.
    no_dedup : bool, default False
        Report every seed's hit instead of one best hit per duplex. CLI:
        ``--no-dedup``.
    alignment : bool, default False
        Add the columns ``q_start``, ``q_end`` (query coordinates, same
        convention as ``start``/``end``) and ``alignment`` (pairing string,
        query 5' to 3': ``P`` pair, ``W`` G:U, ``U`` mismatch, ``T`` / ``Q``
        extra target / query base). CLI: ``--alignment``.
    threads : int, optional
        Worker threads; ``None`` uses ``RAYON_NUM_THREADS``, else every core.
        CLI: ``-j/--threads``.
    sense_only : bool, default False
        Keep only ``+``-strand hits, for a target whose records are already in
        sense orientation, such as a transcriptome. A siOFF filter, not a
        risearch option: risearch indexes and searches both strands, so this
        shrinks the returned frame but not the search's time or memory. CLI:
        ``--sense-only``.

    Returns
    -------
    polars.DataFrame
        One row per hit with the columns ``sirna_id`` (query record ID),
        ``chrom`` (target sequence name), ``start``, ``end``, ``strand`` and
        ``energy`` (hybridisation energy, kcal/mol), followed by ``q_start``,
        ``q_end`` and ``alignment`` when *alignment* is set.

    Raises
    ------
    FileNotFoundError
        When *query* or *index* does not exist.
    RIsearchError
        `sioff.services.risearch_service.RIsearchError`, when *target*
        cannot be resolved, *matrix* is neither a bundled id nor an existing
        file, the seed specification is inconsistent (``seed_start`` without
        ``seed_end``, ``seed_start < 1``, ``seed_end < seed_start``, or a seed
        longer than its window), a mismatch count is negative, *temperature*
        is outside 0-50 °C, *threads* is below 1, or the search itself fails,
        including when the ``risearch`` package is not installed.

    See Also
    --------
    index : Build the index this function queries.
    off_targets : Accepts the returned frame as ``predictions``.

    Examples
    --------
    >>> import sioff
    >>> idx = sioff.index("target.fa")
    >>> hits = sioff.search(
    ...     "sirnas.fa",
    ...     idx,
    ...     target="target.fa",
    ...     seed_start=2,
    ...     seed_end=8,
    ...     seed_length=7,
    ...     seed_wobble=False,
    ...     energy_threshold=-15.0,
    ... )
    >>> hits.columns
    ['sirna_id', 'chrom', 'start', 'end', 'strand', 'energy']
    >>> df = sioff.off_targets(predictions=hits, gtf_file="annotation.gtf")
    """
    return _risearch.run_search(
        query=cast(Path, _p(query)),
        index=cast(Path, _p(index)),
        target=_p(target),
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
