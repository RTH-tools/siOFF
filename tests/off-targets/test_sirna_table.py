from pathlib import Path

import pytest

from sioff.services.sirna_table import SirnaRecord, read_sirna_table


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "sirnas.tsv"
    path.write_text(text)
    return path


def test_reads_columns_in_any_order_and_ignores_extras(tmp_path):
    path = _write(
        tmp_path, "sequence\textra\tsirna_id\ton_target_id\nugcAU\tx\ts1\tENST1\n"
    )
    assert read_sirna_table(path) == {"s1": SirnaRecord("s1", "ENST1", "TGCAT")}


def test_empty_on_target_id_is_none(tmp_path):
    path = _write(tmp_path, "sirna_id\ton_target_id\tsequence\ns1\t\tACGT\n")
    assert read_sirna_table(path)["s1"].on_target_id is None


def test_comment_lines_are_skipped(tmp_path):
    path = _write(
        tmp_path, "# note\nsirna_id\ton_target_id\tsequence\n# mid\ns1\tE1\tACGT\n"
    )
    assert list(read_sirna_table(path)) == ["s1"]


def test_missing_column_is_named(tmp_path):
    path = _write(tmp_path, "sirna_id\tsequence\ns1\tACGT\n")
    with pytest.raises(ValueError, match="on_target_id"):
        read_sirna_table(path)


def test_duplicate_sirna_id_is_named(tmp_path):
    path = _write(
        tmp_path, "sirna_id\ton_target_id\tsequence\ns1\tE1\tACGT\ns1\tE2\tACGA\n"
    )
    with pytest.raises(ValueError, match="'s1' appears more than once"):
        read_sirna_table(path)


def test_empty_sequence_is_rejected(tmp_path):
    path = _write(tmp_path, "sirna_id\ton_target_id\tsequence\ns1\tE1\t\n")
    with pytest.raises(ValueError, match="'s1' has an empty sequence"):
        read_sirna_table(path)


def test_non_nucleotide_sequence_is_rejected(tmp_path):
    path = _write(tmp_path, "sirna_id\ton_target_id\tsequence\ns1\tE1\tACGN\n")
    with pytest.raises(ValueError, match="'s1'.*ACGTU"):
        read_sirna_table(path)


def test_header_only_table_is_empty(tmp_path):
    path = _write(tmp_path, "sirna_id\ton_target_id\tsequence\n")
    assert read_sirna_table(path) == {}
