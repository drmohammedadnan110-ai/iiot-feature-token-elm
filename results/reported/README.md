# Reported manuscript tables — not reproduced experiments

`table_01.csv` through `table_20.csv` preserve the supplied DOCX's table order and cell text. A merged source cell can produce a shorter CSV row; preserve that source structure when interpreting the file. Values and discrepancies have not been silently corrected. [Provenance](provenance.json) records the source document and extracted-file checksums.

These are **paper-reported** values. They are separate from [executed synthetic results](../executed/synthetic_demo/). Neither a manuscript table nor an extracted figure is evidence that this implementation achieved its stated score. The [paper audit](../../docs/paper_audit.md) identifies arithmetic checks and unresolved experimental details.

Re-extract from a lawfully obtained copy using `python scripts/extract_paper.py --paper "path/to/paper.docx"`. The original DOCX is not distributed here.
