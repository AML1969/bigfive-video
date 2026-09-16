"""Build the PDF report for an existing web job folder and rasterise page 1 for a visual check.
Usage: pdf_test.py JOB_DIR [OUT_PNG]
"""
import sys
from pathlib import Path

from bs_bigfive.webapp import export_pdf

job = Path(sys.argv[1])
pdf = export_pdf(job)
print("pdf:", pdf, Path(pdf).stat().st_size, "bytes")
try:
    import fitz  # PyMuPDF
    doc = fitz.open(pdf)
    print("pages:", len(doc))
    out = sys.argv[2] if len(sys.argv) > 2 else str(job / "report_page1.png")
    for i in range(min(2, len(doc))):
        pix = doc[i].get_pixmap(dpi=70)
        p = out.replace(".png", f"_{i + 1}.png")
        pix.save(p)
        print("saved", p)
except ImportError:
    print("pymupdf not installed; skip rasterisation")
