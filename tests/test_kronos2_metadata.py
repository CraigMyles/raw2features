"""Canonical validation for user-supplied KRONOS2 marker metadata."""

from __future__ import annotations

import hashlib
import json

import pytest

from raw2features.embedders.kronos2_metadata import (
    KRONOS2_ADDITIONAL_MARKER_COLUMNS,
    KRONOS2_NOVEL_MARKER_REGISTRATION_CONTRACT,
    Kronos2MarkerMetadataError,
    materialize_kronos2_additional_markers_csv,
    parse_kronos2_additional_markers,
)

HEADER = ",".join(KRONOS2_ADDITIONAL_MARKER_COLUMNS)


def _write(tmp_path, text: str, name: str = "markers.csv"):
    path = tmp_path / name
    path.write_bytes(text.encode("utf-8"))
    return path


def _row(
    marker_name: str = "FOXP3",
    marker_full_name: str = "Forkhead box P3",
    compartment: str = "nucleus",
    family: str = "transcription_factor",
    family_desc: str = "transcription factor",
    mean: str = "0.25",
    std: str = "0.5",
) -> str:
    return ",".join(
        [
            marker_name,
            marker_full_name,
            compartment,
            family,
            family_desc,
            mean,
            std,
        ]
    )


def test_parser_returns_canonical_json_safe_ordered_contract(tmp_path):
    source = (
        "\ufeff"
        + HEADER
        + ",ignored\n"
        + _row("  FOXP3  ", mean="-0", std="5e-1")
        + ",one\n"
        + _row("Novel-2", " Novel marker 2 ", mean="1.25", std="2")
        + ",two\n"
    )
    path = _write(tmp_path, source)

    contract = parse_kronos2_additional_markers(path)

    assert contract["contract_version"] == 1
    assert contract["consumed_columns"] == list(KRONOS2_ADDITIONAL_MARKER_COLUMNS)
    assert [row["marker_name"] for row in contract["rows"]] == ["FOXP3", "Novel-2"]
    assert contract["rows"][0]["mean"] == 0.0
    assert contract["rows"][0]["std"] == 0.5
    assert contract["rows"][1]["mean"] == 1.25
    assert contract["rows"][1]["std"] == 2.0
    assert "original_file_sha256" not in contract
    assert contract["registration"] == KRONOS2_NOVEL_MARKER_REGISTRATION_CONTRACT
    json.dumps(contract, allow_nan=False)

    unhashed = {
        "contract_version": contract["contract_version"],
        "consumed_columns": contract["consumed_columns"],
        "registration": contract["registration"],
        "rows": contract["rows"],
    }
    canonical = json.dumps(
        unhashed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    assert (
        contract["content_sha256"]
        == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    )


def test_semantic_digest_ignores_csv_format_column_order_and_extra_columns(tmp_path):
    first = _write(
        tmp_path,
        HEADER + "\r\n" + _row(mean="0.2500", std="0.500") + "\r\n",
        "first.csv",
    )
    second = _write(
        tmp_path,
        "std,unused,marker_name,family_desc,mean,family,compartment,"
        "marker_full_name\n"
        "5e-1,ignored,FOXP3,transcription factor,2.5e-1,"
        "transcription_factor,nucleus,Forkhead box P3\n",
        "second.csv",
    )

    one = parse_kronos2_additional_markers(first)
    two = parse_kronos2_additional_markers(second)

    assert one["rows"] == two["rows"]
    assert one["content_sha256"] == two["content_sha256"]
    assert "original_file_sha256" not in one
    assert "original_file_sha256" not in two


def test_prompt_unicode_is_preserved_as_semantic_content(tmp_path):
    fullwidth = _write(
        tmp_path,
        f"{HEADER}\n{_row('ＦＯＸＰ３', 'Ｆｏｒｋｈｅａｄ box P3')}\n",
        "fullwidth.csv",
    )
    ascii_text = _write(
        tmp_path,
        f"{HEADER}\n{_row('FOXP3', 'Forkhead box P3')}\n",
        "ascii.csv",
    )

    one = parse_kronos2_additional_markers(fullwidth)
    two = parse_kronos2_additional_markers(ascii_text)

    assert one["rows"][0]["marker_name"] == "ＦＯＸＰ３"
    assert one["rows"][0]["marker_full_name"] == "Ｆｏｒｋｈｅａｄ box P3"
    assert one["content_sha256"] != two["content_sha256"]


def test_original_file_digest_is_available_only_as_explicit_non_identity_audit_data(
    tmp_path,
):
    path = _write(tmp_path, f"{HEADER}\n{_row()}\n")

    contract = parse_kronos2_additional_markers(path, include_original_file_sha256=True)

    assert (
        contract["original_file_sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )


def test_row_order_is_part_of_the_semantic_digest(tmp_path):
    one = _row("A", "Marker A")
    two = _row("B", "Marker B")
    forward = _write(tmp_path, f"{HEADER}\n{one}\n{two}\n", "forward.csv")
    reverse = _write(tmp_path, f"{HEADER}\n{two}\n{one}\n", "reverse.csv")

    assert (
        parse_kronos2_additional_markers(forward)["content_sha256"]
        != parse_kronos2_additional_markers(reverse)["content_sha256"]
    )


@pytest.mark.parametrize("missing", KRONOS2_ADDITIONAL_MARKER_COLUMNS)
def test_all_consumed_columns_are_required(tmp_path, missing):
    columns = [
        column for column in KRONOS2_ADDITIONAL_MARKER_COLUMNS if column != missing
    ]
    path = _write(tmp_path, ",".join(columns) + "\n" + ",".join("x" for _ in columns))

    with pytest.raises(Kronos2MarkerMetadataError, match="missing required columns"):
        parse_kronos2_additional_markers(path)


def test_duplicate_headers_and_empty_tables_are_rejected(tmp_path):
    duplicate = _write(
        tmp_path,
        HEADER + ",marker_name\n" + _row() + ",duplicate\n",
        "duplicate.csv",
    )
    empty = _write(tmp_path, HEADER + "\n", "empty.csv")

    with pytest.raises(Kronos2MarkerMetadataError, match="duplicate columns"):
        parse_kronos2_additional_markers(duplicate)
    with pytest.raises(Kronos2MarkerMetadataError, match="at least one marker row"):
        parse_kronos2_additional_markers(empty)


@pytest.mark.parametrize(
    "column",
    ["marker_name", "marker_full_name", "compartment", "family", "family_desc"],
)
def test_identity_fields_must_not_be_blank(tmp_path, column):
    values = dict(
        zip(KRONOS2_ADDITIONAL_MARKER_COLUMNS, _row().split(","), strict=True)
    )
    values[column] = "   "
    path = _write(
        tmp_path,
        HEADER
        + "\n"
        + ",".join(values[name] for name in KRONOS2_ADDITIONAL_MARKER_COLUMNS),
    )

    with pytest.raises(
        Kronos2MarkerMetadataError, match=rf"{column!r} must not be blank"
    ):
        parse_kronos2_additional_markers(path)


def test_control_characters_are_rejected_before_whitespace_is_trimmed(tmp_path):
    path = _write(
        tmp_path,
        f"{HEADER}\n"
        '"FOXP3","Forkhead\nbox P3",nucleus,transcription_factor,'
        "transcription factor,0.25,0.5\n",
    )

    with pytest.raises(Kronos2MarkerMetadataError, match="control character"):
        parse_kronos2_additional_markers(path)


@pytest.mark.parametrize(
    ("mean", "std", "message"),
    [
        ("nan", "1", "finite number"),
        ("inf", "1", "finite number"),
        ("not-a-number", "1", "finite number"),
        ("1", "nan", "finite number"),
        ("1", "0", "greater than zero"),
        ("1", "-0.01", "greater than zero"),
    ],
)
def test_statistics_must_be_finite_and_std_positive(tmp_path, mean, std, message):
    path = _write(tmp_path, f"{HEADER}\n{_row(mean=mean, std=std)}\n")

    with pytest.raises(Kronos2MarkerMetadataError, match=message):
        parse_kronos2_additional_markers(path)


@pytest.mark.parametrize(
    ("first", "second", "message"),
    [
        ("FOXP3", "foxp3", "case-insensitively"),
        ("CD-8", "CD_8", "separator-insensitive"),
        ("TCR-Vα7.2", "TCR_VA7.2", "separator-insensitive"),
    ],
)
def test_duplicate_marker_identities_are_rejected(tmp_path, first, second, message):
    path = _write(
        tmp_path,
        f"{HEADER}\n{_row(first, 'First')}\n{_row(second, 'Second')}\n",
    )

    with pytest.raises(Kronos2MarkerMetadataError, match=message):
        parse_kronos2_additional_markers(path)


def test_materialized_csv_contains_only_consumed_columns_and_round_trips(tmp_path):
    source = _write(
        tmp_path,
        HEADER + ",ignored\n" + _row("Novel A", "Novel A full") + ",value\n",
        "source.csv",
    )
    contract = parse_kronos2_additional_markers(
        source, include_original_file_sha256=False
    )
    destination = tmp_path / "canonical.csv"

    materialize_kronos2_additional_markers_csv(contract, destination)

    assert destination.read_text().splitlines()[0] == HEADER
    reparsed = parse_kronos2_additional_markers(destination)
    assert reparsed["rows"] == contract["rows"]
    assert reparsed["content_sha256"] == contract["content_sha256"]
    assert "original_file_sha256" not in contract
