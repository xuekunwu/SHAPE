import csv
import gzip
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

MODULE = Path(__file__).resolve().parents[1] / "metadata.py"
IMPORT = importlib.util.spec_from_file_location("study_metadata", MODULE)
metadata = importlib.util.module_from_spec(IMPORT)
IMPORT.loader.exec_module(metadata)
STUDIES = MODULE.parents[1] / "cardiomyocytes"


def fixture_rows():
    return [dict(image_id=f"im{i}", plate_id="plate1", well_id="A01",
                 site_id=str(i), relative_path=f"plate1/im{i}.TIF", metadata_status="partial",
                 donor_id="donor1", flow_sample_id="assay1", ctnt_positive_percent="35.5", dpd="")
            for i in range(2)]


def encoded(rows):
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    raw = text.getvalue().encode()
    spec = dict(study="ctnt_flowcytometry", repo_id="fixture", revision="a" * 40,
                metadata_path="metadata.csv", metadata_sha256=metadata.sha256(raw),
                expected_images=2, expected_wells=1)
    return raw, spec


def test_partial_covariates_do_not_drop_labeled_views():
    raw, spec = encoded(fixture_rows())
    result = metadata.audit(raw, spec)
    assert result["images"] == 2 and result["wells"] == 1
    assert result["missing_values"]["dpd"] == 2
    assert result["labels_available_in_metadata"] is True
    assert result["end_to_end_ready"] is False
    assert "35.5" not in json.dumps(result) and "donor1" not in json.dumps(result)


def test_wrong_metadata_version_rejected():
    raw, spec = encoded(fixture_rows())
    with pytest.raises(ValueError, match="SHA256"):
        metadata.audit(raw + b"\n", spec)


@pytest.mark.parametrize("column,value,pattern", [
    ("image_id", "im0", "duplicate"),
    ("site_id", "0", "duplicate"),
    ("relative_path", "../outside.TIF", "unsafe"),
    ("relative_path", "C:/outside.TIF", "unsafe"),
    ("relative_path", "plate1/../outside.TIF", "unsafe"),
    ("flow_sample_id", "assay2", "conflicting"),
    ("donor_id", "donor2", "conflicting"),
    ("ctnt_positive_percent", "nan", "invalid"),
    ("ctnt_positive_percent", "101", "invalid"),
    ("ctnt_positive_percent", "-1", "invalid"),
    ("ctnt_positive_percent", "", "missing"),
    ("ctnt_positive_percent", "35.6", "conflicting"),
])
def test_bad_identity_path_or_label_rejected(column, value, pattern):
    rows = fixture_rows()
    rows[1][column] = value
    raw, spec = encoded(rows)
    with pytest.raises(ValueError, match=pattern):
        metadata.audit(raw, spec)


def test_shared_assay_cannot_count_as_two_wells():
    rows = fixture_rows()
    rows[1]["well_id"] = "B01"
    raw, spec = encoded(rows)
    spec["expected_wells"] = 2
    with pytest.raises(ValueError, match="multiple imaging wells"):
        metadata.audit(raw, spec)


def test_staining_does_not_invent_cell_labels():
    rows = fixture_rows()
    for row in rows:
        del row["ctnt_positive_percent"]
        del row["flow_sample_id"]
        row["donor_id"] = ""
        row["metadata_status"] = "layout_excluded"
    raw, spec = encoded(rows)
    spec["study"] = "ctnt_staining"
    result = metadata.audit(raw, spec)
    assert result["labels_available_in_metadata"] is False
    assert result["metadata_status_counts"] == {"layout_excluded": 2}
    assert result["images"] == 2


def test_archive_member_read_without_extraction(tmp_path):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        member = tarfile.TarInfo("root/metadata.csv")
        raw = b"column\nvalue\n"
        member.size = len(raw)
        archive.addfile(member, io.BytesIO(raw))
    compressed = gzip.compress(stream.getvalue())
    assert metadata.archive_metadata(compressed, "root/metadata.csv") == raw
    with pytest.raises(ValueError, match="absent"):
        metadata.archive_metadata(compressed, "root/missing.csv")
    assert list(tmp_path.iterdir()) == []


def test_linked_archive_metadata_rejected():
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        member = tarfile.TarInfo("metadata.csv")
        member.type = tarfile.SYMTYPE
        member.linkname = "../secret"
        archive.addfile(member)
    with pytest.raises(ValueError, match="regular"):
        metadata.archive_metadata(gzip.compress(stream.getvalue()), "metadata.csv")


def test_report_never_overwrites_existing_output(tmp_path):
    output = tmp_path / "report.json"
    metadata.write_json(output, {"first": True})
    assert b"\r" not in output.read_bytes()
    with pytest.raises(FileExistsError):
        metadata.write_json(output, {"first": False})
    assert json.loads(output.read_text()) == {"first": True}


@pytest.mark.parametrize("study", ["ctnt_staining", "ctnt_flowcytometry"])
def test_entrypoint_works_from_other_directory_with_spaces(tmp_path, study):
    script = STUDIES / study / "runners/run.py"
    proc = subprocess.run([sys.executable, str(script), "--help"], cwd=tmp_path,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "fetch-metadata" in proc.stdout and study in proc.stdout
