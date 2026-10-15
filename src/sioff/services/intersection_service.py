"""Service for intersecting RIsearch predictions with transcriptome annotations."""

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Optional

import numpy as np
import polars as pl
from loguru import logger
from ncls import NCLS

from sioff.models import PredictionsMode

_BATCH_SIZE = 50_000


_SITE_TRANSCRIPT_KEY = [
    "sirna_id",
    "chrom",
    "start",
    "end",
    "strand",
    "energy",
    "transcript_id",
]


def _one_row_per_site_and_feature(df: pl.DataFrame) -> pl.DataFrame:
    """Keep one overlap row per binding site and transcript.

    A site can touch several exon rows of one transcript (e.g. across a short
    intron); the transcript is one molecule and counts once. Rows of different
    transcripts are kept, so every transcript containing the site contributes its
    own expression. Exon rows of one transcript carry the same expression, so
    which of them is kept does not change the result.
    """
    return df.unique(subset=_SITE_TRANSCRIPT_KEY, keep="first", maintain_order=True)


class IntersectionService:
    """Service to intersect off-target predictions with genomic features."""

    def __init__(self) -> None:
        # Populated by preload_transcriptome() from a worker init; avoids
        # rebuilding NCLS on every intersect() call (~89 ms/siRNA saved).
        self._trans_partition: dict[tuple, pl.DataFrame] | None = None
        self._trans_ncls_index: dict[tuple, dict] | None = None

    def preload_transcriptome(self, transcriptome_df: pl.DataFrame) -> None:
        """Pre-partition transcriptome and build NCLS indices for all (chrom,strand) pairs."""
        self._trans_partition = {
            (k, v): sub_df
            for (k, v), sub_df in transcriptome_df.partition_by(
                ["chrom", "strand"], as_dict=True
            ).items()
        }
        self._trans_ncls_index = {}
        for key, sub_df in self._trans_partition.items():
            if sub_df.height > 0:
                self._trans_ncls_index[key] = self._build_transcript_index(
                    sub_df.sort("start")
                )

    def intersect(
        self,
        risearch_df: pl.DataFrame,
        transcriptome_df: pl.DataFrame,
        mode: str = PredictionsMode.GW,
        workers: int = 1,
        _timings: Optional[dict] = None,
    ) -> pl.DataFrame:
        """Filter predictions to those overlapping transcriptome features."""
        if mode == PredictionsMode.TW:
            return self._transcriptome_wide_join(risearch_df, transcriptome_df)
        else:
            return self._genome_wide_streaming(
                risearch_df, transcriptome_df, workers=workers, _timings=_timings
            )

    def _transcriptome_wide_join(
        self,
        risearch_df: pl.DataFrame,
        transcriptome_df: pl.DataFrame,
    ) -> pl.DataFrame:
        joined = risearch_df.join(
            transcriptome_df,
            left_on="chrom",
            right_on="transcript_id",
            how="inner",
            suffix="_trans",
            maintain_order="left",
        )

        if "transcript_id" not in joined.columns:
            joined = joined.with_columns(pl.col("chrom").alias("transcript_id"))

        return joined

    def _genome_wide_streaming(
        self,
        risearch_df: pl.DataFrame,
        transcriptome_df: pl.DataFrame,
        workers: int = 1,
        _timings: Optional[dict] = None,
    ) -> pl.DataFrame:
        # maintain_order: the pairs decide the order of the concatenated result
        # and, through it, the float32 summation order of the partition
        # functions downstream. Without it, polars returns the pairs in a
        # hash order that varies between runs, so P_off_target drifted by a
        # few ULP from run to run on identical input.
        chrom_strand_pairs = (
            risearch_df.select(["chrom", "strand"])
            .unique(maintain_order=True)
            .to_dicts()
        )

        _t_part_preds = time.perf_counter()
        preds_by_pair: dict[tuple, pl.DataFrame] = {
            (key, val): sub_df
            for (key, val), sub_df in risearch_df.partition_by(
                ["chrom", "strand"], as_dict=True
            ).items()
        }
        _t_part_trans = time.perf_counter()
        if self._trans_partition is not None:
            trans_by_pair = self._trans_partition
        else:
            trans_by_pair = {
                (key, val): sub_df
                for (key, val), sub_df in transcriptome_df.partition_by(
                    ["chrom", "strand"], as_dict=True
                ).items()
            }
        _t_after_partition = time.perf_counter()

        _ncls_time = [0.0]
        _match_time = [0.0]

        def _process_pair(pair: dict) -> list[pl.DataFrame]:
            chrom = pair["chrom"]
            strand = pair["strand"]

            preds = preds_by_pair.get((chrom, strand))
            trans = trans_by_pair.get((chrom, strand))

            if preds is None or preds.height == 0 or trans is None or trans.height == 0:
                return []

            _tb = time.perf_counter()
            if self._trans_ncls_index is not None:
                trans_index = self._trans_ncls_index.get((chrom, strand))
                if trans_index is None:
                    return []
            else:
                trans_index = self._build_transcript_index(trans.sort("start"))
            _ncls_time[0] += time.perf_counter() - _tb
            logger.debug(
                f"Intersecting chrom={chrom} strand={strand}: {preds.height} preds × {trans.height} transcripts"
            )

            results = []
            _tm = time.perf_counter()
            for batch_start in range(0, preds.height, _BATCH_SIZE):
                batch = preds.slice(batch_start, _BATCH_SIZE)
                batch_result = self._process_batch(batch, trans_index)
                if batch_result is not None and batch_result.height > 0:
                    results.append(batch_result)
            _match_time[0] += time.perf_counter() - _tm

            if results:
                return [
                    _one_row_per_site_and_feature(pl.concat(results, how="diagonal"))
                ]
            return []

        all_results: list[pl.DataFrame] = []

        if workers > 1 and len(chrom_strand_pairs) > 1:
            # Cap at 8: beyond that, peak memory from concurrent chromosome-sized DataFrames outweighs speedup.
            safe_workers = min(workers, 8)
            with ThreadPoolExecutor(max_workers=safe_workers) as pool:
                futures = [
                    pool.submit(_process_pair, pair) for pair in chrom_strand_pairs
                ]
                # Collect in submission order (not completion order) so the
                # result's row order does not depend on thread scheduling.
                for future in futures:
                    all_results.extend(future.result())
        else:
            for pair in chrom_strand_pairs:
                all_results.extend(_process_pair(pair))

        if _timings is not None:
            _timings["intersect_partition_preds_s"] = _t_part_trans - _t_part_preds
            _timings["intersect_partition_trans_s"] = _t_after_partition - _t_part_trans
            _timings["intersect_ncls_build_s"] = _ncls_time[0]
            _timings["intersect_match_s"] = _match_time[0]

        if not all_results:
            return self._empty_result_schema(risearch_df)

        result = pl.concat(all_results, how="diagonal")
        logger.debug(f"GW intersection: {result.height:,} rows after per-batch dedup")
        return result

    def _build_transcript_index(self, trans_sorted: pl.DataFrame) -> dict:
        starts = trans_sorted["start"].to_numpy().astype(np.int64, copy=False)
        ends = trans_sorted["end"].to_numpy().astype(np.int64, copy=False)

        interval_ids = np.arange(len(starts), dtype=np.int64)
        index = {
            "gene_ids": trans_sorted["gene_id"].to_list(),
            "transcript_ids": trans_sorted["transcript_id"].to_list(),
            "exp_values": trans_sorted["exp_value"].to_numpy(),
            "trans_df": trans_sorted,
            "ncls": NCLS(starts, ends, interval_ids),
        }

        return index

    def _process_batch(
        self,
        preds: pl.DataFrame,
        trans_index: dict,
    ) -> Optional[pl.DataFrame]:
        if preds.height == 0:
            return None

        trans_df = trans_index["trans_df"]
        ncls = trans_index["ncls"]

        pred_starts = preds["start"].to_numpy().astype(np.int64, copy=False)
        pred_ends = preds["end"].to_numpy().astype(np.int64, copy=False)
        pred_ids = np.arange(preds.height, dtype=np.int64)

        pred_idx, trans_idx = ncls.all_overlaps_both(pred_starts, pred_ends, pred_ids)

        if len(pred_idx) == 0:
            return None

        preds_sel = preds[pred_idx]
        trans_sel = trans_df[trans_idx].select(
            [
                pl.col("start").alias("trans_start"),
                pl.col("end").alias("trans_end"),
                pl.col("gene_id"),
                pl.col("transcript_id"),
                pl.col("exp_value"),
            ]
        )

        return preds_sel.hstack(trans_sel)

    def intersect_streaming(
        self,
        risearch_df: pl.DataFrame,
        transcriptome_df: pl.DataFrame,
        mode: str = PredictionsMode.GW,
        workers: int = 1,
    ) -> Iterator[pl.DataFrame]:
        """Streaming intersection that yields batches instead of accumulating."""
        if mode == PredictionsMode.TW:
            yield self._transcriptome_wide_join(risearch_df, transcriptome_df)
            return

        if workers > 1:
            result = self._genome_wide_streaming(
                risearch_df, transcriptome_df, workers=workers
            )
            for offset in range(0, result.height, _BATCH_SIZE):
                yield result.slice(offset, _BATCH_SIZE)
            return

        chrom_strand_pairs = (
            risearch_df.select(["chrom", "strand"])
            .unique(maintain_order=True)
            .to_dicts()
        )

        for pair in chrom_strand_pairs:
            chrom = pair["chrom"]
            strand = pair["strand"]

            preds = risearch_df.filter(
                (pl.col("chrom") == chrom) & (pl.col("strand") == strand)
            )
            trans = transcriptome_df.filter(
                (pl.col("chrom") == chrom) & (pl.col("strand") == strand)
            )

            if preds.height == 0 or trans.height == 0:
                continue

            trans_index = self._build_transcript_index(trans.sort("start"))

            for batch_start in range(0, preds.height, _BATCH_SIZE):
                batch = preds.slice(batch_start, _BATCH_SIZE)
                batch_result = self._process_batch(batch, trans_index)
                if batch_result is not None and batch_result.height > 0:
                    yield batch_result

    def _empty_result_schema(self, risearch_df: pl.DataFrame) -> pl.DataFrame:
        return risearch_df.head(0).with_columns(
            [
                pl.lit(None).cast(pl.Int64).alias("trans_start"),
                pl.lit(None).cast(pl.Int64).alias("trans_end"),
                pl.lit(None).cast(pl.Utf8).alias("gene_id"),
                pl.lit(None).cast(pl.Utf8).alias("transcript_id"),
                pl.lit(None).cast(pl.Float32).alias("exp_value"),
            ]
        )
