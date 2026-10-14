"""Service for parsing annotation files (GTF/GFF/BED) into genomic feature DataFrames."""

import gzip
import re
from pathlib import Path

import polars as pl

from sioff.models import GTF_SCHEMA


class AnnotationParser:
    """Parser for annotation files (GTF/GFF/BED) — extracts gene/transcript locations and expression."""

    def load_gtf(
        self,
        path: Path,
        feature: str = "exon",
        score_col: str = "RPKM",
        format: str = "auto",
    ) -> pl.DataFrame:
        """Load GTF/GFF or BED file; return DataFrame with chrom/start/end/strand/gene_id/transcript_id/exp_value."""
        if not path.exists():
            raise FileNotFoundError(f"Transcriptome file not found: {path}")

        fmt = format.lower()
        if fmt == "auto":
            if path.suffix.lower() in [".bed", ".bed.gz"]:
                return self._load_bed(path)
            else:
                return self._load_gtf_impl(path, feature, score_col, self._sniff(path))
        elif fmt == "bed6":
            return self._load_bed(path, force_bed6=True)
        elif fmt == "bed7":
            return self._load_bed(path, force_bed7=True)
        elif fmt in ("gff", "gff3"):
            return self._load_gtf_impl(path, feature, score_col, "gff3")
        elif fmt == "gtf":
            return self._load_gtf_impl(path, feature, score_col, "gtf")
        else:
            return self._load_gtf_impl(path, feature, score_col, self._sniff(path))

    @staticmethod
    def _sniff(path: Path) -> str:
        """Detect gtf vs gff3 from the attribute column, not the file extension.

        `auto` is the default and annotation files are routinely misnamed, so the
        attribute syntax is the only trustworthy signal: GTF writes `key "value";`
        while GFF3 writes `key=value;`.
        """
        opener = gzip.open if str(path).endswith(".gz") else open
        try:
            with opener(path, "rt") as fh:
                for line in fh:
                    if line.startswith("##gff-version"):
                        return "gff3"
                    if line.startswith("#") or not line.strip():
                        continue
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) < 9:
                        continue
                    attrs = fields[8]
                    if re.search(r'\w+\s+"', attrs):
                        return "gtf"
                    if re.search(r"\w+=", attrs):
                        return "gff3"
        except OSError:
            pass
        return "gtf"

    def _load_bed(
        self, path: Path, force_bed6: bool = False, force_bed7: bool = False
    ) -> pl.DataFrame:
        if force_bed6:
            num_cols = 6
        elif force_bed7:
            num_cols = 7
        else:
            opener = gzip.open if str(path).endswith(".gz") else open
            try:
                with opener(path, "rt") as f:
                    num_cols = len(f.readline().strip().split("\t"))
            except Exception:
                num_cols = 6

        if num_cols >= 7:
            headers = [
                "chrom",
                "start",
                "end",
                "transcript_id",
                "score_placeholder",
                "strand",
                "exp_value",
            ]
            df = pl.read_csv(
                path,
                separator="\t",
                has_header=False,
                columns=range(7),
                truncate_ragged_lines=True,
            )
            df = df.rename(dict(zip(df.columns, headers, strict=True)))
            return df.select(
                [
                    pl.col("chrom"),
                    pl.col("start"),
                    pl.col("end"),
                    pl.col("strand"),
                    pl.col("transcript_id").alias("gene_id"),
                    pl.col("transcript_id"),
                    pl.col("exp_value").cast(pl.Float32, strict=False).fill_null(0.0),
                ]
            )
        else:
            headers = ["chrom", "start", "end", "transcript_id", "score", "strand"]
            df = pl.read_csv(
                path,
                separator="\t",
                has_header=False,
                columns=range(6),
                infer_schema=False,
                truncate_ragged_lines=True,
            )
            df = df.rename(dict(zip(df.columns, headers, strict=True)))
            return df.select(
                [
                    pl.col("chrom"),
                    pl.col("start").cast(pl.Int64),
                    pl.col("end").cast(pl.Int64),
                    pl.col("strand"),
                    pl.col("transcript_id").alias("gene_id"),
                    pl.col("transcript_id"),
                    pl.col("score")
                    .cast(pl.Float32, strict=False)
                    .fill_null(0.0)
                    .alias("exp_value"),
                ]
            )

    def _load_gtf_impl(
        self, path: Path, feature: str, score_col: str, fmt: str = "gtf"
    ) -> pl.DataFrame:
        df = pl.read_csv(
            path,
            separator="\t",
            has_header=False,
            comment_prefix="#",
            new_columns=list(GTF_SCHEMA.keys()),
            schema_overrides=GTF_SCHEMA,
            truncate_ragged_lines=True,
        )

        if feature:
            df = df.filter(pl.col("feature") == feature)

        def extract_attr(key: str) -> pl.Expr:
            if fmt == "gff3":
                # GFF3: `key=value;`. Values are URL-escaped per the spec.
                return self._url_decode(
                    pl.col("attributes").str.extract(rf"(?:^|;)\s*{key}=([^;]*)", 1)
                )
            # GTF: `key "value";`
            return pl.col("attributes").str.extract(rf'{key}\s+"?([^";]+)"?', 1)

        if fmt == "gff3":
            # GENCODE carries literal gene_id/transcript_id; Ensembl carries only
            # Parent=transcript:ENST... on exon rows. Prefer the explicit keys and
            # fall back so both dialects yield usable identifiers.
            parent = extract_attr("Parent").str.replace(r"^\w+:", "")
            own_id = extract_attr("ID").str.replace(r"^\w+:", "")
            transcript_id = pl.coalesce(
                extract_attr("transcript_id"), parent, own_id
            ).alias("transcript_id")
            gene_id = pl.coalesce(extract_attr("gene_id"), parent, own_id).alias(
                "gene_id"
            )
        else:
            gene_id = extract_attr("gene_id").alias("gene_id")
            transcript_id = extract_attr("transcript_id").alias("transcript_id")

        out = df.select(
            [
                pl.col("chrom"),
                pl.col("start"),
                pl.col("end"),
                pl.col("strand"),
                gene_id,
                transcript_id,
                extract_attr(score_col).alias("_exp_raw"),
            ]
        )

        self._reject_total_extraction_failure(out, path, fmt, score_col)

        return out.select(
            [
                pl.col("chrom"),
                pl.col("start"),
                pl.col("end"),
                pl.col("strand"),
                pl.col("gene_id"),
                pl.col("transcript_id"),
                pl.col("_exp_raw")
                .cast(pl.Float32, strict=False)
                .fill_null(0.0)
                .alias("exp_value"),
            ]
        )

    @staticmethod
    def _url_decode(expr: pl.Expr) -> pl.Expr:
        """Decode the percent-escapes GFF3 requires in column 9 values.

        `%25` is decoded last so an escaped percent (`%2525`) does not cascade
        into a second round of decoding.
        """
        for escape, char in (
            ("%09", "\t"),
            ("%0A", "\n"),
            ("%0D", "\r"),
            ("%26", "&"),
            ("%2C", ","),
            ("%3B", ";"),
            ("%3D", "="),
            ("%3E", ">"),
            ("%25", "%"),
        ):
            expr = expr.str.replace_all(escape, char, literal=True)
        return expr

    @staticmethod
    def _reject_total_extraction_failure(
        df: pl.DataFrame, path: Path, fmt: str, score_col: str
    ) -> None:
        """Fail loudly when an attribute matched nothing at all.

        A wholly unmatched attribute means the file was parsed with the wrong
        dialect or the wrong key. Left unchecked it yields null ids and
        ``exp_value=0.0``, which silently zeroes ``W_i = Expression_i *
        exp(-dG/RT)`` instead of failing — a wrong answer that looks like a
        successful run. Partial nulls stay allowed: sparse annotation is normal.
        """
        if df.height == 0:
            return

        for column, label in (
            ("transcript_id", "transcript_id"),
            ("gene_id", "gene_id"),
            ("_exp_raw", score_col),
        ):
            if df[column].null_count() == df.height:
                raise ValueError(
                    f"No '{label}' attribute found in any of the {df.height} rows of "
                    f"{path.name} (detected format: {fmt}). The file may be in a "
                    f"different format than detected, or use different attribute "
                    f"names — check --transcriptome-format and --feature."
                )

    def summary(self, df: pl.DataFrame) -> dict:
        return {
            "row_count": df.height,
            "genes": df["gene_id"].n_unique(),
            "transcripts": df["transcript_id"].n_unique(),
            "chromosomes": df["chrom"].unique().to_list(),
            "score_mean": df["exp_value"].mean(),
        }
