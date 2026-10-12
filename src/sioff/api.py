"""Public Python API for siOFF.

Importable functions that return results **in memory** and raise ordinary Python
exceptions: no files are written, nothing is printed, and no ``typer.Exit`` /
Click exceptions leak out. These wrap the pure `sioff.core` layer; the
``sioff`` CLI is a separate, file-writing wrapper over the same core.

The four public functions are [`off_targets`][sioff.off_targets], [`accessibility`][sioff.accessibility],
[`index`][sioff.index] and [`search`][sioff.search]. They are re-exported at package level, so
``import sioff`` is all a script needs.

Notes
-----
``index`` returns a `Path`: a RIsearch index is a binary
on-disk artifact, so the path (not in-memory data) is the natural result.

``sioff.index`` / ``sioff.search`` require the external ``risearch`` package,
the same dependency the CLI's index/search commands need. ``off_targets`` on
pre-computed predictions and ``accessibility`` never import it.

Examples
--------
>>> import sioff
>>> import polars as pl

Single predictions file -> one DataFrame.

>>> df = sioff.off_targets(risearch_file="predictions.tsv", gtf_file="ann.gtf")

A *directory* of per-siRNA files -> a generator of per-siRNA DataFrames.

>>> for sirna_df in sioff.off_targets(risearch_file="preds_dir/"):
...     ...
>>> everything = pl.concat(list(sioff.off_targets(risearch_file="preds_dir/")))

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
from typing import Iterator, Mapping, Optional, Union, cast

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
    genome_file: Optional[Union[str, Path]] = None,
    window_size: int = 80,
    max_span: int = 40,
    unpaired_prob: int = 30,
    temperature: float = 37.0,
    on_target_file: Optional[Union[str, Path]] = None,
    on_target_risearch_file: Optional[Union[str, Path]] = None,
    query_file: Optional[Union[str, Path]] = None,
    on_target_expression: float = 1000.0,
    on_target_accessibility: Optional[Union[str, Path]] = None,
    on_target_ids_file: Optional[Union[str, Path]] = None,
    alpha: str = "1.0",
    gamma: str = "1.0",
    theta: str = "",
    sense_only: bool = False,
    predictions_type: str = "gw",
    n_workers: int = 1,
    accessibility_strategy: str = "auto",
) -> Union[pl.DataFrame, Iterator[pl.DataFrame]]:
    """Analyse siRNA off-target predictions, returning results in memory.

    Intersects RIsearch predictions with a transcriptome annotation, adds an
    accessibility (opening-energy) penalty, and computes partition-function
    off-target probabilities. This is the in-memory counterpart of the
    ``sioff off-targets`` command; the defaults reproduce its behaviour, and each
    parameter below names its CLI flag.

    Predictions come from exactly one of:

    - ``predictions``: an in-memory `polars.DataFrame` in the
      [`search`][sioff.search] schema (``sirna_id, chrom, start, end, strand, energy``),
      typically the value returned by [`search`][sioff.search]; returns one DataFrame.
    - ``risearch_file``: a RIsearch2 output file; returns one DataFrame. A
      *directory* of per-siRNA files instead returns a **generator** yielding
      one DataFrame per siRNA. The worker pool and a temporary Arrow IPC copy
      of the transcriptome live for the lifetime of the generator, so consume
      it fully, or wrap it in `contextlib.closing`, for prompt cleanup.
    - ``sirna_fasta`` + ``target_fasta``: run RIsearch in-process first
      (requires the ``risearch`` package).

    Accessibility profiles come from at most one of ``accessibility`` (the
    ``dict[chrom -> DataFrame]`` returned by [`accessibility`][sioff.accessibility]),
    ``accessibility_dir`` (per-chromosome Parquet files) or ``genome_file``
    (fold on the fly). In-memory ``accessibility`` is for the single-frame
    forms; the directory form streams through worker processes that read
    profiles from disk, so it takes ``accessibility_dir`` only. Without any
    profile source the probabilities are computed from hybridisation energy
    alone.

    Writes no files and prints nothing.

    Parameters
    ----------
    risearch_file : str or pathlib.Path, optional
        Pre-computed RIsearch2 output: a single file, or a directory of
        per-siRNA files. A file returns one DataFrame; a directory switches to
        the streaming per-siRNA generator described above. CLI:
        ``-r/--risearch-file``.
    predictions : polars.DataFrame, optional
        In-memory predictions in the [`search`][sioff.search] schema. The columns
        ``sirna_id, chrom, start, end, strand, energy`` are required; extra
        columns are dropped and dtypes are cast to the schema (``Utf8`` ids and
        strand, ``Int32`` coordinates, ``Float32`` energy), so a frame that went
        through Parquet or your own manipulation still works. Mutually
        exclusive with ``risearch_file``. No CLI equivalent.
    sirna_fasta : str or pathlib.Path, optional
        siRNA FASTA with one or more sequences and unique IDs. Without
        ``risearch_file`` / ``predictions`` it triggers an in-process RIsearch
        run against ``target_fasta``. With a directory ``risearch_file`` it is
        instead used to compute each siRNA's self-hybridisation ``E_min`` for
        the alpha/gamma clamp. CLI: ``-s/--sirna-fasta``.
    target_fasta : str or pathlib.Path, optional
        Target FASTA (genome or transcriptome) for the in-process RIsearch
        run; required together with ``sirna_fasta``. CLI: ``--target-fasta``
        (alias ``--genome``).
    target_index : str or pathlib.Path, optional
        Pre-built RIsearch index of ``target_fasta``, to skip indexing on
        repeated runs. When omitted the index is built (or reused) next to
        ``target_fasta`` with an ``.idx`` suffix. CLI: ``-idx/--target-index``.
    gtf_file : str or pathlib.Path, optional
        Transcriptome annotation (GTF, GFF3, BED6 or BED7). Predictions are
        intersected with its features, which adds ``transcript_id``,
        ``gene_id`` and the expression column ``exp_value``. Without it no
        intersection is performed and every prediction is kept with unit
        expression. CLI: ``-t/--transcriptome``.
    feature_type : str, default "exon"
        Feature type to select from a GTF/GFF3 annotation. CLI: ``--feature``.
    expression_metric : str, default "RPKM"
        Annotation attribute used as the expression score (``exp_value``).
        CLI: ``--expression-metric``.
    transcriptome_format : str, default "auto"
        One of ``"auto"``, ``"gtf"``, ``"gff3"``, ``"bed6"`` or ``"bed7"``.
        ``"auto"`` detects the format from the file, telling GTF from GFF3 by
        the attribute column. CLI: ``--transcriptome-format``.
    accessibility : dict[str, polars.DataFrame], optional
        In-memory accessibility profiles keyed by chromosome, each frame with
        columns ``position, strand, u1..uN`` and both strands stacked, as
        returned by [`accessibility`][sioff.accessibility]. Accepted by the single-frame forms
        only; mutually exclusive with ``accessibility_dir``. No CLI equivalent.
    accessibility_dir : str or pathlib.Path, optional
        Directory of per-chromosome ``{chrom}.accessibility.parquet`` files
        written by ``sioff accessibility``. The only accessibility source the
        directory form accepts. CLI: ``-a/--accessibility-dir``.
    genome_file : str or pathlib.Path, optional
        Genome FASTA to fold on the fly; the profiles go to a temporary
        directory that is removed when the call returns. Ignored when
        ``accessibility`` or ``accessibility_dir`` is given, and not available
        for the directory form. CLI: ``-f/--fasta``.
    window_size : int, default 80
        Local-folding window size ``W`` for on-the-fly accessibility
        (``genome_file`` only). CLI: ``-W/--window``.
    max_span : int, default 40
        Maximum base-pair span ``L`` for on-the-fly accessibility. CLI:
        ``-L/--span``.
    unpaired_prob : int, default 30
        Length ``u`` of the unpaired stretch for which opening energies are
        computed (profile columns ``u1..u{unpaired_prob}``). CLI:
        ``-u/--unpaired``.
    temperature : float, default 37.0
        Folding temperature in degrees Celsius. Affects both on-the-fly
        accessibility and the ``RT`` term of the partition function. CLI:
        ``-T/--temperature``.
    on_target_file : str or pathlib.Path, optional
        On-target sequence FASTA for the single-siRNA partition function; its
        hybridisation energy against ``query_file`` is computed in-process with
        RIsearch and enters the partition function as the on-target term.
        Requires ``query_file``. CLI: ``-on/--on-target``.
    on_target_risearch_file : str or pathlib.Path, optional
        Pre-computed RIsearch output for the on-target, skipping the
        on-the-fly on-target search. CLI: ``-on-ris/--on-target-risearch-file``.
    query_file : str or pathlib.Path, optional
        siRNA query FASTA used together with ``on_target_file``. CLI:
        ``-q/--query``.
    on_target_expression : float, default 1000.0
        Expression level assigned to the on-target transcript in the partition
        function. CLI: ``-oexp/--on-target-expression``.
    on_target_accessibility : str or pathlib.Path, optional
        Accessibility Parquet for the on-target sequence (same schema as the
        ``sioff accessibility`` output). Computed on the fly when omitted.
        CLI: ``--on-target-accessibility``.
    on_target_ids_file : str or pathlib.Path, optional
        Two-column tab-separated file mapping ``sirna_id`` to the ID of its
        on-target transcript or gene (multi-siRNA form). Predictions of a siRNA
        that fall on its mapped ID are tagged ``is_on_target`` and contribute
        the on-target weight instead of counting as off-targets. Needs
        ``gtf_file`` so that ``transcript_id`` exists. CLI:
        ``-oi/--on-target-ids``.
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
        Parallelism: the number of threads for the intersection in the
        single-frame forms, and the number of spawned worker processes (one
        siRNA file at a time each) for the directory form. CLI:
        ``-j/--workers``, whose default is the CPU count.
    accessibility_strategy : {"auto", "eager", "chrom-major"}, default "auto"
        Directory form only: how opening energies are looked up from
        ``accessibility_dir``. ``"chrom-major"`` streams each chromosome's
        Parquet once per run for the hits of every siRNA, so memory per worker
        stays about one Parquet row group; ``"eager"`` loads whole-chromosome
        profiles in every worker (30 GB per chr1 strand on hg38); ``"auto"``
        is chromosome-major with more than one worker. Results are identical.
        CLI: ``--accessibility-strategy``.

    Returns
    -------
    polars.DataFrame or Iterator[polars.DataFrame]
        One DataFrame for ``predictions``, a single ``risearch_file`` or
        ``sirna_fasta``; for a directory ``risearch_file``, a generator that
        yields one DataFrame per siRNA. Each row is one prediction with the
        input columns ``sirna_id, chrom, start, end, strand, energy``; the
        annotation columns ``transcript_id, gene_id, exp_value`` when
        ``gtf_file`` is given; ``opening_energy`` (the accessibility penalty,
        kcal/mol) when a profile source is given; ``dG_total``
        (``energy + opening_energy``); ``P_off_target`` (baseline
        alpha=gamma=theta=1); and one ``P_off_target:alpha=X,gamma=Y`` or
        ``P_off_target:theta=T`` column per extra parameter value. Multi-siRNA
        and swept runs also carry ``is_on_target`` and the per-parameter
        ``dG_total:...`` columns; the single-frame forms additionally keep the
        ``E_min``, Boltzmann-weight and ``Z_sirna`` intermediates, which the
        directory form drops before yielding.

    Raises
    ------
    ValueError
        When no predictions source is given; when ``predictions`` is combined
        with ``risearch_file``; when ``accessibility`` is combined with
        ``accessibility_dir`` or with a directory of predictions; when
        ``sirna_fasta`` is given without ``target_fasta``; when ``predictions``
        lacks one of its required columns; when ``sirna_fasta`` contains
        duplicate IDs; or when ``on_target_ids_file`` cannot be parsed.
    FileNotFoundError
        When ``risearch_file``, ``gtf_file``, ``genome_file``, ``sirna_fasta``
        or ``target_fasta`` does not exist, or a directory ``risearch_file``
        contains no prediction files.
    RIsearchError
        `sioff.services.risearch_service.RIsearchError`, when the
        in-process RIsearch run (``sirna_fasta`` form or on-target search)
        fails, including when the ``risearch`` package is not installed.
    AccessibilityError
        `sioff.services.accessibility.AccessibilityError`, when folding
        ``genome_file`` fails or an accessibility profile is missing or
        malformed.

    See Also
    --------
    search : Produce a ``predictions`` frame in the schema this function accepts.
    accessibility : Produce the in-memory ``accessibility`` profiles.

    Examples
    --------
    Single predictions file with annotation, pre-computed accessibility and an
    alpha sweep:

    >>> import sioff
    >>> df = sioff.off_targets(
    ...     risearch_file="predictions.tsv",
    ...     gtf_file="annotation.gtf",
    ...     accessibility_dir="accessibility/",
    ...     alpha="0.8;1.0",
    ...     gamma="1.0",
    ... )
    >>> df.select("sirna_id", "transcript_id", "P_off_target").head()

    A directory of per-siRNA files, consumed as a stream and torn down promptly:

    >>> import contextlib
    >>> gen = sioff.off_targets(risearch_file="predictions/", n_workers=8)
    >>> with contextlib.closing(gen) as frames:
    ...     for frame in frames:
    ...         print(frame["sirna_id"][0], frame.height)
    """
    rf = _p(risearch_file)

    if rf is not None and rf.is_dir():
        if accessibility is not None:
            raise ValueError(
                "a directory of predictions cannot take in-memory accessibility "
                "profiles; write them to Parquet and pass accessibility_dir"
            )
        core_gen = _off_targets.compute_off_targets_directory(
            input_dir=rf,
            sirna_fasta=_p(sirna_fasta),
            gtf_file=_p(gtf_file),
            feature_type=feature_type,
            expression_metric=expression_metric,
            transcriptome_format=transcriptome_format,
            accessibility_dir=_p(accessibility_dir),
            temperature=temperature,
            on_target_ids_file=_p(on_target_ids_file),
            on_target_expression=on_target_expression,
            alpha=alpha,
            gamma=gamma,
            theta=theta,
            sense_only=sense_only,
            predictions_type=predictions_type,
            n_workers=n_workers,
            accessibility_strategy=accessibility_strategy,
        )
        # Yield bare per-siRNA DataFrames (the core yields (df, meta) tuples).
        # Closing this generator propagates GeneratorExit to core_gen, tearing
        # down its worker pool + temp-IPC dir.
        return (frame for frame, _meta in core_gen)

    df, _meta = _off_targets.compute_off_targets_single(
        predictions=predictions,
        risearch_file=rf,
        sirna_fasta=_p(sirna_fasta),
        target_fasta=_p(target_fasta),
        target_index=_p(target_index),
        gtf_file=_p(gtf_file),
        feature_type=feature_type,
        expression_metric=expression_metric,
        transcriptome_format=transcriptome_format,
        accessibility=accessibility,
        accessibility_dir=_p(accessibility_dir),
        genome_file=_p(genome_file),
        window_size=window_size,
        max_span=max_span,
        unpaired_prob=unpaired_prob,
        temperature=temperature,
        on_target_file=_p(on_target_file),
        on_target_risearch_file=_p(on_target_risearch_file),
        query_file=_p(query_file),
        on_target_expression=on_target_expression,
        on_target_accessibility=_p(on_target_accessibility),
        on_target_ids_file=_p(on_target_ids_file),
        alpha=alpha,
        gamma=gamma,
        theta=theta,
        sense_only=sense_only,
        predictions_type=predictions_type,
        n_workers=n_workers,
    )
    return df


def accessibility(
    genome: Union[str, Path],
    window_size: int = 80,
    max_span: int = 40,
    unpaired_prob: int = 30,
    temperature: float = 37.0,
) -> dict[str, pl.DataFrame]:
    """Compute per-chromosome accessibility profiles in memory.

    Folds every sequence in *genome* with ViennaRNA (local folding, both
    strands) and returns the opening-energy profiles in the same schema the
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

    Returns
    -------
    dict[str, polars.DataFrame]
        Chromosome (FASTA record ID) to profile, in file order. Each frame has
        the columns ``position`` (1-based), ``strand`` (``"+"`` or ``"-"``) and
        ``u1..u{unpaired_prob}`` (opening energies in kcal/mol), with all
        ``+`` rows followed by all ``-`` rows.

    Raises
    ------
    FileNotFoundError
        When *genome* does not exist.
    AccessibilityError
        `sioff.services.accessibility.AccessibilityError`, when
        ViennaRNA fails to fold a sequence.

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
    table.

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

    Returns
    -------
    polars.DataFrame
        One row per hit with the columns ``sirna_id`` (query record ID),
        ``chrom`` (target sequence name), ``start``, ``end``, ``strand`` and
        ``energy`` (hybridisation energy, kcal/mol).

    Raises
    ------
    FileNotFoundError
        When *query* or *index* does not exist.
    RIsearchError
        `sioff.services.risearch_service.RIsearchError`, when *target*
        cannot be resolved, *matrix* is neither a bundled id nor an existing
        file, the seed specification is inconsistent (``seed_start`` without
        ``seed_end``, ``seed_start < 1``, ``seed_end < seed_start``, or a seed
        longer than its window), or the search itself fails, including when
        the ``risearch`` package is not installed.

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
    )
