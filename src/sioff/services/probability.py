import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, cast

import numpy as np
import polars as pl
from loguru import logger

from sioff.services.accessibility import (
    GenomeAccessibilityService,
    _reverse_complement,
    site_cells,
)
from sioff.services.helpers import read_fasta
from sioff.services.risearch_service import RIsearchService

# Gas constant in kcal/mol*K
R = 0.001987
# Default temperature in Kelvin (37 C) — kept for backward compatibility
T = 310.15
RT = R * T


def _require_opening_energy(df: pl.DataFrame) -> None:
    if "opening_energy" not in df.columns:
        raise ValueError(
            "precomputed_opening_energy=True but the frame has no 'opening_energy' "
            "column"
        )


class ProbabilityService:
    """Calculate off-target probabilities by integrating hybridization energy and accessibility."""

    def __init__(
        self,
        accessibility_service: Optional[GenomeAccessibilityService] = None,
        precomputed_accessibility: Optional[pl.DataFrame] = None,
        temperature: float = 37.0,
    ):
        self.accessibility_service = accessibility_service
        self.precomputed_accessibility = precomputed_accessibility
        self._temperature = temperature
        self._RT = R * (temperature + 273.15)

    def calculate_probabilities(
        self,
        df: pl.DataFrame,
        on_target_path: Optional[Path] = None,
        query_path: Optional[Path] = None,
        on_target_expression: float = 1000.0,
        on_target_accessibility_path: Optional[Path] = None,
        on_target_risearch_path: Optional[Path] = None,
        precomputed_opening_energy: bool = False,
    ) -> Tuple[pl.DataFrame, Dict]:
        """Z = Σ(Expr_i × exp(-dG_total_i / RT)); P(t) = W_t / Z.

        With ``precomputed_opening_energy`` the frame must already carry an
        ``opening_energy`` column and no accessibility lookup is performed.
        """
        if "energy" not in df.columns:
            logger.warning("No 'energy' column found. Cannot calculate probabilities.")
            return df, {}

        if precomputed_opening_energy:
            _require_opening_energy(df)
        elif self.accessibility_service:
            df = self._annotate_opening_energy(df)

        if "opening_energy" in df.columns:
            expr_dG_total = pl.col("energy") + pl.col("opening_energy")
        else:
            expr_dG_total = pl.col("energy")

        df = df.with_columns(expr_dG_total.alias("dG_total"))

        if "exp_value" in df.columns:
            expr_val = pl.col("exp_value").fill_null(0.0)
        else:
            logger.info(
                "No expression value found; assuming uniform expression=1.0 for Off-Targets"
            )
            expr_val = pl.lit(1.0)

        df = df.with_columns(
            (expr_val * ((-pl.col("dG_total") / self._RT).exp())).alias(
                "boltzmann_weight"
            )
        )
        z_partial = cast(float, df["boltzmann_weight"].sum())

        w_on = 0.0
        dG_total_on = 0.0
        dG_open_on = 0.0
        dG_hyb_on = 0.0

        if on_target_path and query_path:
            logger.info("Calculating On-Target weight...")
            dG_hyb_on, dG_open_on = self._calculate_on_target_components(
                on_target_path,
                query_path,
                on_target_accessibility_path,
                on_target_risearch_path,
            )
            dG_total_on = dG_hyb_on + dG_open_on
            w_on = on_target_expression * cast(float, np.exp(-dG_total_on / self._RT))
            logger.info(f"On-Target dG_total={dG_total_on:.2f}, Weight={w_on:.2e}")

        z_total = z_partial + w_on
        logger.info(
            f"Partition Function Z = {z_total:.2e} (Off-Target={z_partial:.2e}, On-Target={w_on:.2e})"
        )

        if z_total > 0:
            df = df.with_columns(
                (pl.col("boltzmann_weight") / z_total).alias("P_off_target")
            )
        else:
            df = df.with_columns(pl.lit(0.0).alias("P_off_target"))

        if on_target_path and query_path:
            if on_target_risearch_path:
                _hyb, _start, _end, _strand = self._parse_risearch_file(
                    on_target_risearch_path
                )
            else:
                _hyb, _start, _end, _strand = self._run_risearch_binary(
                    query_path, on_target_path
                )

            p_on_val = w_on / z_total if z_total > 0 else 0.0

            row_dict = {
                "chrom": "onTarget",
                "start": _start,
                "end": _end,
                "strand": _strand,
                "transcript_id": "onTarget",
                "gene_id": "onTarget",
                "energy": _hyb,
                "opening_energy": dG_open_on,
                "dG_total": dG_total_on,
                "exp_value": on_target_expression,
                "P_off_target": p_on_val,
            }

            ont_df = pl.DataFrame([row_dict])

            for col in df.columns:
                if col not in ont_df.columns:
                    ont_df = ont_df.with_columns(
                        pl.lit(None, dtype=df.schema[col]).alias(col)
                    )
                else:
                    target_dtype = df.schema[col]
                    if target_dtype != ont_df.schema[col]:
                        ont_df = ont_df.with_columns(pl.col(col).cast(target_dtype))

            df = pl.concat([df, ont_df.select(df.columns)], how="vertical")

        metadata = {
            "z_total": z_total,
            "z_off_target": z_partial,
            "w_on_target": w_on,
            "dG_on_target": dG_total_on,
            "has_on_target": bool(on_target_path and query_path),
        }

        if on_target_path and query_path:
            metadata["p_on_target"] = w_on / z_total if z_total > 0 else 0.0

        return df, metadata

    def calculate_probabilities_per_sirna(
        self,
        df: pl.DataFrame,
        alpha_gamma_pairs: Optional[list[tuple[float, float]]] = None,
        theta_values: Optional[list[float]] = None,
        on_target_map: Optional[dict[str, str]] = None,
        on_target_expression: float = 1000.0,
        on_target_fasta_data: Optional[dict[str, dict]] = None,
        precomputed_opening_energy: bool = False,
        aggregate_only: bool = False,
    ) -> Tuple[pl.DataFrame, Dict]:
        """Per-siRNA partition-function probabilities with optional alpha/gamma/theta sweeps.

        With ``aggregate_only`` only the per-siRNA sums are computed (the
        metadata ``z_per_sirna``, ``on_target_weights`` and ``n_rows``) and an
        empty frame is returned: the per-row energy, dG, weight and probability
        columns of every alpha/gamma/theta set are never materialised. That is
        all ``--summary-only`` needs, and for a large grid it is most of the
        memory (385 columns on millions of rows for the RIsearch3 benchmark).

        Mode A (in_predictions): on-target rows already in df, tagged by ID match.
        Mode B (fasta): on-target energy comes from on_target_fasta_data; rows excluded from df.

        With ``precomputed_opening_energy`` the frame must already carry an
        ``opening_energy`` column (e.g. gathered chromosome-major by the
        directory core) and the accessibility service is not consulted.
        """
        if "energy" not in df.columns:
            logger.warning("No 'energy' column found. Cannot calculate probabilities.")
            return df, {}

        if "sirna_id" not in df.columns:
            logger.info("No sirna_id column; falling back to single partition function")
            return self.calculate_probabilities(
                df, precomputed_opening_energy=precomputed_opening_energy
            )

        unique_sirnas = df["sirna_id"].unique()
        if (
            len(unique_sirnas) == 1
            and not alpha_gamma_pairs
            and not theta_values
            and not on_target_map
            and not on_target_fasta_data
        ):
            logger.info("Single siRNA detected; using standard calculation")
            return self.calculate_probabilities(
                df, precomputed_opening_energy=precomputed_opening_energy
            )

        logger.info(
            f"Processing {len(unique_sirnas)} siRNAs with per-siRNA partition functions"
        )

        if precomputed_opening_energy:
            _require_opening_energy(df)
        elif self.accessibility_service:
            df = self._annotate_opening_energy(df)

        if "opening_energy" in df.columns:
            df = df.with_columns(
                (pl.col("energy") + pl.col("opening_energy")).alias("dG_total")
            )
        else:
            df = df.with_columns(
                pl.lit(0.0).alias("opening_energy"),
                pl.col("energy").alias("dG_total"),
            )

        if "exp_value" not in df.columns:
            df = df.with_columns(pl.lit(1.0).alias("exp_value"))

        use_fasta_mode = on_target_fasta_data is not None
        in_predictions_mode = (
            on_target_map is not None
            and not use_fasta_mode
            and "transcript_id" in df.columns
        )

        on_target_data_fasta: dict[str, dict] = {}

        if in_predictions_mode:
            # Filter to siRNAs present here — avoids O(N_corpus) OR chain per worker call
            sirna_set = set(unique_sirnas.to_list())
            local_map = {k: v for k, v in on_target_map.items() if k in sirna_set}

            tag_conditions = []
            for sirna_id, target_id in local_map.items():
                cond = pl.col("sirna_id") == sirna_id
                id_match = pl.col("transcript_id") == target_id
                if "gene_id" in df.columns:
                    id_match = id_match | (pl.col("gene_id") == target_id)
                tag_conditions.append(cond & id_match)

            if tag_conditions:
                combined = tag_conditions[0]
                for c in tag_conditions[1:]:
                    combined = combined | c
                df = df.with_columns(combined.alias("is_on_target"))
            else:
                df = df.with_columns(pl.lit(False).alias("is_on_target"))

            n_on = df.filter(pl.col("is_on_target")).height
            logger.info(
                f"In-predictions on-target mode: tagged {n_on} rows as on-target (from {len(local_map)} mappings)"
            )

        elif (
            on_target_fasta_data is not None
            and on_target_map
            and "transcript_id" in df.columns
        ):
            on_target_data_fasta = on_target_fasta_data

            exclude_conditions = [
                (pl.col("sirna_id") == sid)
                & (
                    (pl.col("transcript_id") == on_target_map[sid])
                    | (
                        pl.col("gene_id") == on_target_map[sid]
                        if "gene_id" in df.columns
                        else pl.lit(False)
                    )
                )
                for sid in on_target_fasta_data
                if sid in on_target_map
            ]
            if exclude_conditions:
                exclude_expr = exclude_conditions[0]
                for cond in exclude_conditions[1:]:
                    exclude_expr = exclude_expr | cond
                df = df.filter(~exclude_expr)
                logger.info(
                    f"FASTA mode: excluded on-target rows for {len(on_target_fasta_data)} siRNAs"
                )

            df = df.with_columns(pl.lit(False).alias("is_on_target"))
        else:
            df = df.with_columns(pl.lit(False).alias("is_on_target"))

        # Use raw_e_min (from full genome scan) when available; fall back to
        # post-intersection minimum. raw_e_min is set by the parser when loading
        # per-siRNA directory files.
        if "raw_e_min" in df.columns:
            min_energies = df.group_by("sirna_id").agg(
                pl.col("raw_e_min").min().alias("E_min")
            )
        else:
            min_energies = df.group_by("sirna_id").agg(
                pl.col("energy").min().alias("E_min")
            )
        df = df.join(min_energies, on="sirna_id", how="left")

        calc_configs = [("", "dG_total", "energy")]
        all_dG_exprs = []
        # suffix -> (dG, dG_noacc) expressions, so aggregate_only can sum the
        # weights without first adding the dG columns to the frame.
        dG_exprs: dict[str, tuple[pl.Expr, pl.Expr]] = {
            "": (pl.col("dG_total"), pl.col("energy"))
        }

        if alpha_gamma_pairs:
            for alpha, gamma in alpha_gamma_pairs:
                if alpha == 1.0 and gamma == 1.0:
                    continue  # base case already in calc_configs

                suffix = f":alpha={alpha},gamma={gamma}"
                col_name = f"dG_total{suffix}"
                col_name_noacc = f"energy{suffix}"

                # Clamp: energy < α·E_min → use γ·E_min; otherwise keep energy
                dG = (
                    pl.when(pl.col("energy") < alpha * pl.col("E_min"))
                    .then(gamma * pl.col("E_min") + pl.col("opening_energy"))
                    .otherwise(pl.col("energy") + pl.col("opening_energy"))
                )
                dG_noacc = (
                    pl.when(pl.col("energy") < alpha * pl.col("E_min"))
                    .then(gamma * pl.col("E_min"))
                    .otherwise(pl.col("energy"))
                )
                all_dG_exprs.extend(
                    [dG.alias(col_name), dG_noacc.alias(col_name_noacc)]
                )
                calc_configs.append((suffix, col_name, col_name_noacc))
                dG_exprs[suffix] = (dG, dG_noacc)

        if theta_values:
            for theta in theta_values:
                suffix = f":theta={theta}"
                col_name = f"dG_total{suffix}"
                col_name_noacc = f"energy{suffix}"

                # Theta scaling: ((θ·(E + 10)) - 10) + open_energy
                dG_noacc = (theta * (pl.col("energy") + 10.0)) - 10.0
                dG = dG_noacc + pl.col("opening_energy")
                all_dG_exprs.extend(
                    [dG.alias(col_name), dG_noacc.alias(col_name_noacc)]
                )
                calc_configs.append((suffix, col_name, col_name_noacc))
                dG_exprs[suffix] = (dG, dG_noacc)

        if all_dG_exprs and not aggregate_only:
            df = df.with_columns(all_dG_exprs)

        all_weight_exprs = []
        z_aggs = []
        on_aggs = []
        z_col_names = []

        for suffix, dG_col, dG_noacc_col in calc_configs:
            weight_col = f"boltzmann_weight{suffix}"
            weight_noacc_col = f"boltzmann_weight_noacc{suffix}"
            all_weight_exprs.extend(
                [
                    (pl.col("exp_value") * ((-pl.col(dG_col) / self._RT).exp())).alias(
                        weight_col
                    ),
                    (
                        pl.col("exp_value") * ((-pl.col(dG_noacc_col) / self._RT).exp())
                    ).alias(weight_noacc_col),
                ]
            )
            if aggregate_only:
                dG, dG_noacc = dG_exprs[suffix]
                w = pl.col("exp_value") * ((-dG / self._RT).exp())
                w_noacc = pl.col("exp_value") * ((-dG_noacc / self._RT).exp())
            else:
                w, w_noacc = pl.col(weight_col), pl.col(weight_noacc_col)
            z_col = f"Z_sirna{suffix}"
            z_noacc_col = f"Z_sirna_noacc{suffix}"
            z_aggs.extend([w.sum().alias(z_col), w_noacc.sum().alias(z_noacc_col)])
            z_col_names.extend([z_col, z_noacc_col])

            if in_predictions_mode:
                on_aggs.extend(
                    [
                        w.filter(pl.col("is_on_target")).sum().alias(f"W_on{suffix}"),
                        w_noacc.filter(pl.col("is_on_target"))
                        .sum()
                        .alias(f"W_on_noacc{suffix}"),
                    ]
                )

        if not aggregate_only:
            df = df.with_columns(all_weight_exprs)

        agg_df = df.group_by("sirna_id").agg(z_aggs + on_aggs)
        z_df = agg_df.select(["sirna_id"] + z_col_names)

        on_target_weight_metadata = {}

        if in_predictions_mode:
            on_df = agg_df.drop(z_col_names)
            for row in on_df.to_dicts():
                sid = row["sirna_id"]
                on_target_weight_metadata[sid] = {
                    k: v for k, v in row.items() if k != "sirna_id"
                }
            logger.info(
                f"In-predictions mode: computed on-target weights for {on_df.height} siRNAs"
            )

        elif on_target_data_fasta:
            min_e_dict = {
                row["sirna_id"]: cast(float, row["E_min"])
                for row in min_energies.to_dicts()
            }
            z_modifications = []

            for suffix, dG_col, dG_noacc_col in calc_configs:
                on_target_w = {}
                on_target_w_noacc = {}

                for sirna_id, data in on_target_data_fasta.items():
                    orig_energy = data["energy"]
                    orig_open = data.get("opening_energy", 0.0)
                    e_min = min(min_e_dict.get(sirna_id, orig_energy), orig_energy)

                    if suffix.startswith(":alpha="):
                        parts = suffix.split(",")
                        alpha = float(parts[0].split("=")[1])
                        gamma = float(parts[1].split("=")[1])
                        if orig_energy < alpha * e_min:
                            assigned_e = gamma * e_min + orig_open
                            assigned_e_noacc = gamma * e_min
                        else:
                            assigned_e = orig_energy + orig_open
                            assigned_e_noacc = orig_energy
                    elif suffix.startswith(":theta="):
                        theta = float(suffix.split("=")[1])
                        assigned_e = ((theta * (orig_energy + 10.0)) - 10.0) + orig_open
                        assigned_e_noacc = (theta * (orig_energy + 10.0)) - 10.0
                    else:
                        assigned_e = orig_energy + orig_open
                        assigned_e_noacc = orig_energy

                    w_on = on_target_expression * np.exp(-assigned_e / self._RT)
                    w_on_noacc = on_target_expression * np.exp(
                        -assigned_e_noacc / self._RT
                    )
                    on_target_w[sirna_id] = w_on
                    on_target_w_noacc[sirna_id] = w_on_noacc

                    if sirna_id not in on_target_weight_metadata:
                        on_target_weight_metadata[sirna_id] = {}
                    on_target_weight_metadata[sirna_id][f"W_on{suffix}"] = w_on
                    on_target_weight_metadata[sirna_id][f"W_on_noacc{suffix}"] = (
                        w_on_noacc
                    )

                z_col = f"Z_sirna{suffix}"
                z_noacc_col = f"Z_sirna_noacc{suffix}"
                z_modifications.extend(
                    [
                        (
                            pl.col(z_col)
                            + pl.col("sirna_id").replace_strict(
                                on_target_w, default=0.0
                            )
                        ).alias(z_col),
                        (
                            pl.col(z_noacc_col)
                            + pl.col("sirna_id").replace_strict(
                                on_target_w_noacc, default=0.0
                            )
                        ).alias(z_noacc_col),
                    ]
                )

            z_df = z_df.with_columns(z_modifications)
            logger.info(
                f"FASTA mode: added on-target weights for {len(on_target_data_fasta)} siRNAs to partition functions"
            )

        z_stats_dict = {row["sirna_id"]: row for row in z_df.to_dicts()}
        metadata = {
            "n_sirnas": len(unique_sirnas),
            "n_rows": df.height,
            "z_per_sirna": z_stats_dict,
            "on_target_weights": on_target_weight_metadata,
            "on_target_count": len(on_target_weight_metadata),
        }
        if aggregate_only:
            return df.head(0), metadata

        df = df.join(z_df, on="sirna_id", how="left")

        probs_exprs = []
        for suffix, _, _ in calc_configs:
            weight_col = f"boltzmann_weight{suffix}"
            z_col = f"Z_sirna{suffix}"
            p_col = f"P_off_target{suffix}"
            probs_exprs.append(
                pl.when(pl.col(z_col) > 0)
                .then(pl.col(weight_col) / pl.col(z_col))
                .otherwise(0.0)
                .alias(p_col)
            )

        df = df.with_columns(probs_exprs)

        return df, metadata

    def format_legacy_summary(
        self,
        sirna_id: str,
        z_per_config: dict[str, float],
        on_target_weights: dict[str, float],
        alpha_gamma_pairs: list[tuple[float, float]],
        theta_values: list[float],
    ) -> str:
        """Format the aggregated metadata into the exact legacy .off header block."""
        lines = []
        lines.append(f"# On-target info for {sirna_id} #")

        # third element (gamma) is None in theta mode, which has no gamma
        suffixes: list[tuple[str, float, float | None]] = [("", 1.0, 1.0)]
        for theta in theta_values or []:
            suffixes.append((f":theta={theta}", theta, None))
        for alpha, gamma in alpha_gamma_pairs or []:
            if alpha == 1.0 and gamma == 1.0:
                continue
            suffixes.append((f":alpha={alpha},gamma={gamma}", alpha, gamma))

        for suffix_info in suffixes:
            suffix = suffix_info[0]

            if suffix == "":
                preamble = "# For alpha=1.0 and gamma=1.0;"
            elif suffix.startswith(":theta="):
                preamble = f"# For theta={suffix_info[1]};"
            else:
                preamble = f"# For alpha={suffix_info[1]} and gamma={suffix_info[2]};"

            z = z_per_config.get(f"Z_sirna{suffix}", 0.0)
            z_noacc = z_per_config.get(f"Z_sirna_noacc{suffix}", 0.0)
            w_on = on_target_weights.get(f"W_on{suffix}", 0.0)
            w_on_noacc = on_target_weights.get(f"W_on_noacc{suffix}", 0.0)

            z_off = z - w_on
            z_off_noacc = z_noacc - w_on_noacc

            p_on = w_on / z if z > 0 else 0.0
            p_on_noacc = w_on_noacc / z_noacc if z_noacc > 0 else 0.0

            lines.append(
                f"{preamble} P: {p_on!s}; P_noacc: {p_on_noacc!s}; "
                f"Z: {z!s}; Z_noacc: {z_noacc!s}; "
                f"Zoff: {z_off!s}; Zoff_noacc: {z_off_noacc!s}"
            )

        lines.append("## End of on-target info ##\n")
        return "\n".join(lines)

    def calculate_legacy_format(
        self,
        df: pl.DataFrame,
        sirna_id: str = "siRNA",
        on_target_path: Optional[Path] = None,
        query_path: Optional[Path] = None,
        on_target_expression: float = 1000.0,
        on_target_accessibility_path: Optional[Path] = None,
        on_target_risearch_path: Optional[Path] = None,
        alpha_gamma_pairs: list[tuple[float, float]] = [(1.0, 1.0), (0.8, 0.8)],
        verbose: bool = False,
    ) -> str:
        """Return formatted string matching legacy gw.results format."""
        if "energy" not in df.columns:
            return "# Error: No energy column found\n"

        if self.accessibility_service:
            df = self._annotate_opening_energy(df)

        if "opening_energy" not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias("opening_energy"))

        if "exp_value" not in df.columns:
            df = df.with_columns(pl.lit(1.0).alias("exp_value"))

        dG_hyb_on = 0.0
        dG_open_on = 0.0
        if on_target_path and query_path:
            dG_hyb_on, dG_open_on = self._calculate_on_target_components(
                on_target_path,
                query_path,
                on_target_accessibility_path,
                on_target_risearch_path,
            )

        results_by_ag = {}
        on_target_stats = {}

        # Min energy anchors clamping; includes on-target energy per old pipeline behavior
        current_min = dG_hyb_on if (on_target_path and query_path) else 0.0
        if df.height > 0:
            off_min = cast(float, df["energy"].min())
            if off_min < current_min:
                current_min = off_min
        min_energy = current_min

        df = df.filter(pl.col("transcript_id") != "onTarget")

        for alpha, gamma in alpha_gamma_pairs:
            df_scaled = df.with_columns(
                [
                    pl.when(pl.col("energy") < alpha * min_energy)
                    .then(gamma * min_energy + pl.col("opening_energy"))
                    .otherwise(pl.col("energy") + pl.col("opening_energy"))
                    .alias("dG_scaled"),
                    pl.when(pl.col("energy") < alpha * min_energy)
                    .then(gamma * min_energy)
                    .otherwise(pl.col("energy"))
                    .alias("dG_scaled_noacc"),
                ]
            )

            df_scaled = df_scaled.with_columns(
                [
                    (
                        pl.col("exp_value") * ((-pl.col("dG_scaled") / self._RT).exp())
                    ).alias("W"),
                    (
                        pl.col("exp_value")
                        * ((-pl.col("dG_scaled_noacc") / self._RT).exp())
                    ).alias("W_noacc"),
                ]
            )

            df_agg = df_scaled.group_by(["chrom", "transcript_id"]).agg(
                [
                    pl.col("W").sum().alias("W_transcript"),
                    pl.col("W_noacc").sum().alias("W_transcript_noacc"),
                ]
            )

            z_off = cast(float, df_agg["W_transcript"].sum())
            z_off_noacc = cast(float, df_agg["W_transcript_noacc"].sum())

            if dG_hyb_on < alpha * min_energy:
                dG_total_on = gamma * min_energy + dG_open_on
                dG_on_noacc = gamma * min_energy
            else:
                dG_total_on = dG_hyb_on + dG_open_on
                dG_on_noacc = dG_hyb_on

            w_on = on_target_expression * cast(float, np.exp(-dG_total_on / self._RT))
            w_on_noacc = on_target_expression * cast(
                float, np.exp(-dG_on_noacc / self._RT)
            )

            z_total = z_off + w_on
            z_total_noacc = z_off_noacc + w_on_noacc

            p_on = w_on / z_total if z_total > 0 else 0.0
            p_off_total = z_off / z_total if z_total > 0 else 0.0
            p_on_noacc = w_on_noacc / z_total_noacc if z_total_noacc > 0 else 0.0
            p_off_total_noacc = (
                z_off_noacc / z_total_noacc if z_total_noacc > 0 else 0.0
            )

            on_target_stats[(alpha, gamma)] = {
                "Pon": p_on,
                "Pon_noacc": p_on_noacc,
                "Poff": p_off_total,
                "Poff_noacc": p_off_total_noacc,
                "Z": w_on,
                "Z_noacc": w_on_noacc,
                "Zoff": z_off,
                "Zoff_noacc": z_off_noacc,
            }

            df_agg = df_agg.with_columns(
                [
                    (pl.col("W_transcript") / z_total).alias("P_transcript")
                    if z_total > 0
                    else pl.lit(0.0).alias("P_transcript"),
                    (pl.col("W_transcript_noacc") / z_total_noacc).alias(
                        "P_transcript_noacc"
                    )
                    if z_total_noacc > 0
                    else pl.lit(0.0).alias("P_transcript_noacc"),
                ]
            )

            results_by_ag[(alpha, gamma)] = df_agg

        lines = [f"# On-target info for {sirna_id} #"]

        for (alpha, gamma), stats in on_target_stats.items():
            line = (
                f"# For alpha={alpha} and gamma={gamma}; "
                f"Pon: {stats['Pon']:.12g}; Pon_noacc: {stats['Pon_noacc']:.12g}; "
                f"Poff: {stats['Poff']:.12g}; Poff_noacc: {stats['Poff_noacc']:.12g}; "
                f"Z: {stats['Z']:.12g}; Z_noacc: {stats['Z_noacc']:.12g}; "
                f"Zoff: {stats['Zoff']:.12g}; Zoff_noacc: {stats['Zoff_noacc']:.12g}"
            )
            lines.append(line)
        lines.append("## End of on-target info ##")

        if verbose:
            lines.append("# Off-target info#")

            header_parts = ["# ID1", "ID2"]
            for alpha, gamma in alpha_gamma_pairs:
                header_parts.append(f"alpha:{alpha},gamma:{gamma}")
                header_parts.append(f"alpha:{alpha},gamma:{gamma}(NoAcc)")
            lines.append("\t".join(header_parts))

            first_ag = alpha_gamma_pairs[0]
            merged = (
                results_by_ag[first_ag]
                .select(
                    ["chrom", "transcript_id", "P_transcript", "P_transcript_noacc"]
                )
                .rename(
                    {
                        "P_transcript": f"P_{first_ag[0]}_{first_ag[1]}",
                        "P_transcript_noacc": f"P_{first_ag[0]}_{first_ag[1]}_noacc",
                    }
                )
            )

            for ag in alpha_gamma_pairs[1:]:
                other = (
                    results_by_ag[ag]
                    .select(
                        ["chrom", "transcript_id", "P_transcript", "P_transcript_noacc"]
                    )
                    .rename(
                        {
                            "P_transcript": f"P_{ag[0]}_{ag[1]}",
                            "P_transcript_noacc": f"P_{ag[0]}_{ag[1]}_noacc",
                        }
                    )
                )
                merged = merged.join(
                    other, on=["chrom", "transcript_id"], how="full", coalesce=True
                )

            merged = merged.sort(["chrom", "transcript_id"])

            for row in merged.iter_rows(named=True):
                parts = [row["chrom"], row["transcript_id"]]
                for ag in alpha_gamma_pairs:
                    parts.append(f"{row.get(f'P_{ag[0]}_{ag[1]}', 0.0) or 0.0:.12g}")
                    parts.append(
                        f"{row.get(f'P_{ag[0]}_{ag[1]}_noacc', 0.0) or 0.0:.12g}"
                    )
                lines.append("\t".join(parts))

            lines.append("## End of off-target info")

        lines.append("")
        return "\n".join(lines)

    def _annotate_opening_energy(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.height == 0:
            return df

        if not all(c in df.columns for c in ["chrom", "start", "end", "strand"]):
            logger.warning(
                "Missing coordinate or strand columns. Cannot lookup accessibility."
            )
            return df

        if self.precomputed_accessibility is not None:
            join_keys = ["chrom", "start", "end", "strand"]
            result = df.join(
                self.precomputed_accessibility.select(join_keys + ["opening_energy"]),
                on=join_keys,
                how="left",
            )
            if "opening_energy" in result.columns:
                result = result.with_columns(pl.col("opening_energy").fill_null(10.0))
            else:
                result = result.with_columns(pl.lit(10.0).alias("opening_energy"))
            logger.info(
                f"Annotated {result.height} rows with precomputed accessibility"
            )
            return result

        if not self.accessibility_service:
            return df

        return self.accessibility_service.annotate_opening_energy_vectorized(df)

    def _calculate_on_target_dg(
        self,
        on_target_path: Path,
        query_path: Path,
        accessibility_path: Optional[Path] = None,
        risearch_path: Optional[Path] = None,
    ) -> float:
        dG_hyb, dG_open = self._calculate_on_target_components(
            on_target_path, query_path, accessibility_path, risearch_path
        )
        return dG_hyb + dG_open

    def _calculate_on_target_components(
        self,
        on_target_path: Path,
        query_path: Path,
        accessibility_path: Optional[Path] = None,
        risearch_path: Optional[Path] = None,
    ) -> tuple[float, float]:
        """Return (dG_hyb, dG_open) for on-target; computes accessibility on-the-fly if not provided."""
        if risearch_path:
            dG_hyb, t_start, t_end, strand = self._parse_risearch_file(risearch_path)
        else:
            dG_hyb, t_start, t_end, strand = self._run_risearch_binary(
                query_path, on_target_path
            )

        # 2. Accessibility Energy
        dG_open = 0.0

        if accessibility_path and accessibility_path.exists() and t_end > 0:
            # Use pre-computed accessibility Parquet file
            try:
                df_acc = (
                    pl.read_parquet(accessibility_path)
                    .filter(pl.col("strand") == strand)
                    .sort("position")
                )
                u_cols = sorted(
                    [
                        c
                        for c in df_acc.columns
                        if c.startswith("u") and c[1:].isdigit()
                    ],
                    key=lambda x: int(x[1:]),
                )
                max_pos = int(cast(int, df_acc["position"].max()))
                n_u = len(u_cols)
                profile = np.full((max_pos, n_u), 25.5, dtype=np.float32)
                pos_0 = df_acc["position"].to_numpy() - 1
                for col_idx, col_name in enumerate(u_cols):
                    profile[pos_0, col_idx] = df_acc[col_name].to_numpy()

                rows, cols, in_bounds = site_cells(
                    strand,
                    np.array([t_start]),
                    np.array([t_end]),
                    profile.shape[0],
                    profile.shape[1],
                )
                if in_bounds[0]:
                    dG_open = int(round(float(profile[rows[0], cols[0]]) * 10.0)) / 10.0
                else:
                    dG_open = 10.0

            except Exception as e:
                logger.error(f"Failed to read on-target accessibility parquet: {e}")
                dG_open = 0.0

        elif t_end > 0:
            # Compute accessibility on-the-fly from the on-target FASTA
            try:
                sequence = None
                for seq_id, seq in read_fasta(on_target_path):
                    sequence = seq
                    break

                if sequence:
                    logger.info(
                        f"Computing on-target accessibility on-the-fly (len={len(sequence)})"
                    )

                    # A '-' hit sits on the reverse complement of the on-target
                    # sequence: fold that molecule, whose rows are RC positions,
                    # exactly as the genome-wide profiles store the '-' strand.
                    folded = (
                        _reverse_complement(sequence) if strand == "-" else sequence
                    )
                    with tempfile.TemporaryDirectory() as tmpdir:
                        temp_service = GenomeAccessibilityService(Path(tmpdir))
                        profile = temp_service.compute_sequence_accessibility(
                            folded, temperature=self._temperature
                        )

                        if profile.size > 0 and profile.ndim == 2:
                            rows, cols, in_bounds = site_cells(
                                strand,
                                np.array([t_start]),
                                np.array([t_end]),
                                profile.shape[0],
                                profile.shape[1],
                            )
                            target_idx = int(rows[0])
                            col_idx = int(cols[0])

                            if in_bounds[0]:
                                dG_open = float(profile[target_idx, col_idx])
                                logger.info(
                                    f"On-target opening energy: {dG_open:.2f} kcal/mol"
                                )
                            else:
                                logger.warning(
                                    f"Binding site index {target_idx} out of bounds (len={profile.shape[0]})"
                                )
                                dG_open = 10.0
                        else:
                            logger.warning("Empty or 1D profile from ViennaRNA")
                            dG_open = 10.0
                else:
                    logger.warning(
                        f"No sequence found in on-target FASTA: {on_target_path}"
                    )

            except Exception as e:
                logger.error(f"Failed to compute on-target accessibility: {e}")
                dG_open = 0.0

        return dG_hyb, dG_open

    def _run_risearch_binary(
        self, query_path: Path, target_path: Path
    ) -> tuple[float, int, int, str]:
        return RIsearchService().search_single_sirna(query_path, target_path)

    def _parse_risearch_file(self, path: Path) -> tuple[float, int, int, str]:
        """Parse pre-computed RIsearch output; return (best_energy, start, end, strand)."""
        if not path.exists():
            logger.error(f"Provided RIsearch file not found: {path}")
            return 0.0, 0, 0, "+"

        energies = []
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 8:
                        try:
                            energies.append(
                                (
                                    float(parts[7]),
                                    int(parts[4]),
                                    int(parts[5]),
                                    parts[6],
                                )
                            )
                        except ValueError:
                            pass
        except Exception as e:
            logger.error(f"Error reading RIsearch file {path}: {e}")
            return 0.0, 0, 0, "+"

        if energies:
            return min(energies, key=lambda x: x[0])
        logger.warning(f"No valid predictions found in {path}")
        return 0.0, 0, 0, "+"
