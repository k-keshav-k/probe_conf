"""Build report/PROJECT_SUMMARY.pdf from PROJECT_SUMMARY.md (reportlab).

    .venv/bin/python report/make_project_summary_pdf.py

Markdown stays the content source; this renders headings, paragraphs, tables,
lists, quotes, rules, and images.  Target length: two to three pages.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
SOURCE = REPORT_DIR / "PROJECT_SUMMARY.md"
OUTPUT = REPORT_DIR / "PROJECT_SUMMARY.pdf"
FRAME_WIDTH = 16.8 * cm

_spec = importlib.util.spec_from_file_location(
    "section33_parser", REPORT_DIR / "make_section33_confirmatory_pdf.py"
)
s33 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s33)

TITLE = ParagraphStyle(
    "title", fontName="Times-Bold", fontSize=19, leading=23,
    spaceBefore=2, spaceAfter=2,
)
SUBTITLE = ParagraphStyle(
    "subtitle", fontName="Times-Roman", fontSize=10.5, leading=13.5,
    textColor=colors.HexColor("#333333"), spaceBefore=0, spaceAfter=6,
)
HEADING = ParagraphStyle(
    "heading", fontName="Times-Bold", fontSize=13, leading=16,
    spaceBefore=12, spaceAfter=4,
)
SUBHEADING = ParagraphStyle(
    "subheading", fontName="Times-Bold", fontSize=10.5, leading=13,
    spaceBefore=8, spaceAfter=2,
)
BODY = ParagraphStyle(
    "body", fontName="Times-Roman", fontSize=9.6, leading=12.6,
    alignment=4, spaceAfter=4,
)
BULLET = ParagraphStyle("bullet", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=2)
NUMBERED = ParagraphStyle("numbered", parent=BODY, leftIndent=16, bulletIndent=4, spaceAfter=2)
CELL = ParagraphStyle("cell", fontName="Times-Roman", fontSize=8, leading=10)
CELL_HEAD = ParagraphStyle("cellhead", fontName="Times-Bold", fontSize=8, leading=10)
CAPTION = ParagraphStyle(
    "caption", fontName="Times-Italic", fontSize=8, leading=10.5,
    alignment=1, textColor=colors.HexColor("#444444"), spaceBefore=2, spaceAfter=8,
)


def table_widths(ncols: int) -> list[float]:
    if ncols == 2:
        return [5.0 * cm, 11.8 * cm]
    if ncols == 3:
        return [5.6 * cm, 5.6 * cm, 5.6 * cm]
    if ncols == 4:
        return [4.2 * cm, 4.2 * cm, 4.2 * cm, 4.2 * cm]
    if ncols == 5:
        return [3.36 * cm] * 5
    return [FRAME_WIDTH / ncols] * ncols


def make_table(rows: list[str]) -> Table:
    parsed = []
    for row in rows:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        parsed.append(cells)
    header, body = parsed[0], parsed[1:]
    data = [[Paragraph(md(cell), CELL_HEAD) for cell in header]]
    for row in body:
        data.append([Paragraph(md(cell), CELL) for cell in row])
    table = Table(data, colWidths=table_widths(len(header)), repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f6")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#22394C")),
                ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#bbbbbb")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return table


def md(text: str) -> str:
    """Markdown inline -> reportlab markup, with links reduced to their label."""

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    return s33.inline(text)


def image_flowable(caption: str, relative: str) -> KeepTogether:
    from reportlab.lib.utils import ImageReader

    path = REPORT_DIR / relative
    reader = ImageReader(str(path))
    width_px, height_px = reader.getSize()
    width = FRAME_WIDTH
    height = width * height_px / width_px
    max_height = 20.0 * cm
    if height > max_height:
        height = max_height
        width = height * width_px / height_px
    image = Image(str(path), width=width, height=height)
    image.hAlign = "CENTER"
    parts = [image]
    if caption:
        parts.append(Paragraph(s33.inline(caption), CAPTION))
    else:
        parts.append(Spacer(1, 6))
    return KeepTogether(parts)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 8.5)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"{doc.page}")
    canvas.restoreState()


def build() -> None:
    blocks = s33.parse_markdown(SOURCE.read_text())
    story = []
    first_heading = True
    for kind, payload in blocks:
        if kind == "h1":
            story.append(Paragraph(md(str(payload)), TITLE))
        elif kind == "h2":
            story.append(Paragraph(md(str(payload)), HEADING))
        elif kind == "h3":
            story.append(Paragraph(md(str(payload)), SUBHEADING))
        elif kind == "para":
            text = str(payload)
            if first_heading and text.startswith("**Keshav"):
                story.append(Paragraph(md(text), SUBTITLE))
                first_heading = False
            else:
                story.append(Paragraph(md(text), BODY))
        elif kind == "bullets":
            for item in payload:
                story.append(Paragraph(s33.inline(item), BULLET, bulletText="\u2022"))
            story.append(Spacer(1, 3))
        elif kind == "numbers":
            for index, item in enumerate(payload, start=1):
                story.append(Paragraph(md(item), NUMBERED, bulletText=f"{index}."))
            story.append(Spacer(1, 3))
        elif kind == "table":
            story.append(Spacer(1, 2))
            story.append(make_table(payload))
            story.append(Spacer(1, 5))
        elif kind == "quote":
            story.append(Paragraph(md(str(payload)), s33.QUOTE))
        elif kind == "rule":
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#999999")))
        elif kind == "image":
            caption, relative = payload
            story.append(image_flowable(caption, relative))
        else:
            raise ValueError(f"unsupported block: {kind}")

    document = BaseDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="Confidence reporting: summary of everything done so far",
        author="Keshav Krishna",
    )
    frame = Frame(
        document.leftMargin,
        document.bottomMargin,
        A4[0] - document.leftMargin - document.rightMargin,
        A4[1] - document.topMargin - document.bottomMargin,
        id="body",
    )
    document.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=footer)])
    document.build(story)

    import subprocess

    info = subprocess.check_output(["pdfinfo", str(OUTPUT)], text=True)
    pages = int(re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE).group(1))
    if not 1 <= pages <= 4:
        raise RuntimeError(f"Unexpected page count {pages}; inspect before publishing.")
    print(f"Wrote {OUTPUT.name} ({pages} pages).")


if __name__ == "__main__":
    build()
