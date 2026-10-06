"""Extract supplied DOCX tables and embedded images verbatim; no result synthesis."""

import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--paper", required=True)
    parser.add_argument("--out", default="results/reported")
    parser.add_argument("--figures", default="figures/paper")
    args = parser.parse_args()
    source = Path(args.paper)
    out, figures = Path(args.out), Path(args.figures)
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    records = []
    with zipfile.ZipFile(source) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))
        for i, table in enumerate(document.findall(".//w:tbl", namespace), 1):
            rows = [
                [
                    " ".join(node.text or "" for node in cell.findall(".//w:t", namespace))
                    for cell in row.findall("w:tc", namespace)
                ]
                for row in table.findall("w:tr", namespace)
            ]
            path = out / f"table_{i:02d}.csv"
            # Keep source cells and row lengths; merged cells may yield ragged rows.
            with path.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream, lineterminator="\n").writerows(rows)
            records.append(
                {
                    "file": path.name,
                    "rows": len(rows),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        images = []
        for member in archive.namelist():
            if member.startswith("word/media/"):
                path = figures / Path(member).name
                path.write_bytes(archive.read(member))
                images.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    provenance = {
        "source_filename": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "status": "Reported manuscript values; not independently reproduced",
        "tables": records,
        "figures": images,
        "extraction": "OOXML text per table cell; whitespace joined; row/cell order retained; images copied unchanged",
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
