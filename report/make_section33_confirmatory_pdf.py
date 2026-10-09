"""Build the section 3.3 confirmatory report as a technical-report PDF.

Uses reportlab only (no LaTeX).  Reads report/SECTION_3_3_CONFIRMATORY_REPORT.md
and writes report/SECTION_3_3_CONFIRMATORY_REPORT.pdf.

    .venv/bin/python report/make_section33_confirmatory_pdf.py
"""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
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
SOURCE = REPORT_DIR / "SECTION_3_3_CONFIRMATORY_REPORT.md"
OUTPUT = REPORT_DIR / "SECTION_3_3_CONFIRMATORY_REPORT.pdf"
FRAME_WIDTH = 16.6 * cm

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
BODY = ParagraphStyle(
    "body", fontName="Times-Roman", fontSize=10.5, leading=14.2,
    alignment=TA_JUSTIFY, spaceAfter=6,
)
HEADING = ParagraphStyle(
    "heading", fontName="Times-Bold", fontSize=14, leading=17,
    spaceBefore=16, spaceAfter=6,
)
SUBHEADING = ParagraphStyle(
    "subheading", fontName="Times-Bold", fontSize=11.5, leading=14,
    spaceBefore=10, spaceAfter=4,
)
TITLE = ParagraphStyle(
    "title", fontName="Times-Bold", fontSize=22, leading=26, alignment=TA_CENTER,
)
SUBTITLE = ParagraphStyle(
    "subtitle", fontName="Times-Roman", fontSize=14, leading=17.5,
    alignment=TA_CENTER, textColor=colors.HexColor("#333333"),
)
META = ParagraphStyle(
    "meta", fontName="Times-Roman", fontSize=10.5, leading=13.5,
    alignment=TA_CENTER, textColor=colors.HexColor("#333333"),
)
ABSTRACT = ParagraphStyle(
    "abstract", fontName="Times-Roman", fontSize=10, leading=13.6,
    alignment=TA_JUSTIFY, leftIndent=16, rightIndent=16,
)
ABSTRACT_HEAD = ParagraphStyle(
    "abstracthead", fontName="Times-Bold", fontSize=11.5, leading=14,
    alignment=TA_CENTER, spaceBefore=2, spaceAfter=4,
)
CAPTION = ParagraphStyle(
    "caption", fontName="Times-Italic", fontSize=8.5, leading=11,
    alignment=TA_CENTER, textColor=colors.HexColor("#444444"),
    spaceBefore=4, spaceAfter=10,
)
TABLE_LABEL = ParagraphStyle(
    "tablelabel", fontName="Times-Bold", fontSize=8.5, leading=11,
    spaceBefore=8, spaceAfter=2,
)
CELL = ParagraphStyle("cell", fontName="Times-Roman", fontSize=8.5, leading=11)
CELL_HEAD = ParagraphStyle("cellhead", fontName="Times-Bold", fontSize=8.5, leading=11)
QUOTE = ParagraphStyle(
    "quote", fontName="Times-Italic", fontSize=10.2, leading=13.6,
    alignment=TA_JUSTIFY, leftIndent=22, rightIndent=22, spaceBefore=2, spaceAfter=8,
)
BULLET = ParagraphStyle(
    "bullet", parent=BODY, leftIndent=16, bulletIndent=5, spaceAfter=3,
)
NUMBERED = ParagraphStyle(
    "numbered", parent=BODY, leftIndent=18, bulletIndent=5, spaceAfter=3,
)
VERDICT = ParagraphStyle(
    "verdict", fontName="Times-Bold", fontSize=11, leading=14, alignment=TA_CENTER,
)

TABLE_WIDTHS = {
    2: [5.5 * cm, 11.1 * cm],
    3: [6.0 * cm, 5.3 * cm, 5.3 * cm],
    4: [4.0 * cm, 4.2 * cm, 4.2 * cm, 4.2 * cm],
    6: [2.2 * cm, 2.6 * cm, 3.6 * cm, 2.6 * cm, 3.6 * cm, 2.0 * cm],
}

UNICODE_FIXES = {"−": "-", "–": "-", "—": "-", "≈": "~", "≥": ">=", "≤": "<=", "×": "x", "→": "->"}


def sanitize(text: str) -> str:
    for source, target in UNICODE_FIXES.items():
        text = text.replace(source, target)
    return text


def inline(text: str) -> str:
    text = sanitize(text)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Courier" size="9">\1</font>', text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    return text


def is_block_start(line: str) -> bool:
    stripped = line.strip()
    return bool(
        stripped.startswith(("#", "-", ">", "|", "!["))
        or re.match(r"^\d+\.\s", stripped)
        or stripped == "---"
    )


def parse_list(lines: list[str], index: int, marker) -> tuple[list[str], int]:
    """Collect a list block, joining continuation lines and skipping blank lines."""

    items: list[str] = []
    while index < len(lines):
        stripped = lines[index].strip()
        indented = lines[index][:1] in (" ", "\t")
        if not stripped:
            look = index + 1
            while look < len(lines) and not lines[look].strip():
                look += 1
            if look < len(lines) and (
                marker.match(lines[look].strip()) or lines[look][:1] in (" ", "\t")
            ):
                index = look
                continue
            break
        if marker.match(stripped):
            items.append(marker.sub("", stripped, count=1).strip())
            index += 1
        elif items and indented and not is_block_start(lines[index]):
            items[-1] += " " + stripped
            index += 1
        else:
            break
    return items, index


def parse_markdown(text: str) -> list[tuple[str, object]]:
    blocks: list[tuple[str, object]] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            continue
        if stripped.startswith("!["):
            match = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
            blocks.append(("image", (match.group(1), match.group(2))))
            index += 1
        elif stripped.startswith("### "):
            blocks.append(("h3", stripped[4:].strip()))
            index += 1
        elif stripped.startswith("## "):
            blocks.append(("h2", stripped[3:].strip()))
            index += 1
        elif stripped.startswith("# "):
            blocks.append(("h1", stripped[2:].strip()))
            index += 1
        elif stripped == "---":
            blocks.append(("rule", None))
            index += 1
        elif stripped.startswith(">"):
            quote = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote.append(lines[index].strip().lstrip(">").strip())
                index += 1
            blocks.append(("quote", " ".join(quote)))
        elif stripped.startswith("- "):
            items, index = parse_list(lines, index, re.compile(r"^-\s"))
            blocks.append(("bullets", items))
        elif re.match(r"^\d+\.\s", stripped):
            items, index = parse_list(lines, index, re.compile(r"^\d+\.\s"))
            blocks.append(("numbers", items))
        elif stripped.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(lines[index].strip())
                index += 1
            blocks.append(("table", rows))
        else:
            paragraph = [stripped]
            index += 1
            while index < len(lines) and lines[index].strip() and not is_block_start(lines[index]):
                paragraph.append(lines[index].strip())
                index += 1
            blocks.append(("para", " ".join(paragraph)))
    return blocks


def make_table(rows: list[str]) -> Table:
    parsed = []
    for row in rows:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        parsed.append(cells)
    header, body = parsed[0], parsed[1:]
    data = [[Paragraph(inline(cell), CELL_HEAD) for cell in header]]
    for row in body:
        data.append([Paragraph(inline(cell), CELL) for cell in row])
    widths = TABLE_WIDTHS.get(len(header))
    table = Table(data, colWidths=widths, hAlign="CENTER", repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def build() -> None:
    text = SOURCE.read_text()
    # Skip the markdown front matter; the PDF builds its own title block.
    body_text = text.split("\n---\n", 1)[1]
    blocks = parse_markdown(body_text)

    story: list = []
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Section 3.3 Confirmatory Report", TITLE))
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph("Do MLP Interaction Features Improve Model Confidence?", SUBTITLE))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("Keshav Krishna", META))
    story.append(Paragraph("New York University", META))
    story.append(Paragraph("9 October 2026", META))
    story.append(Spacer(1, 0.4 * cm))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#999999")))
    story.append(Paragraph("Abstract", ABSTRACT_HEAD))
    story.append(
        Paragraph(
            "We test whether probes over MLP interaction features add a useful confidence signal "
            "beyond internal signals. The confirmatory experiment uses Qwen2.5-7B-Instruct on a "
            "fresh Wikidata cohort of 7,600 subjects with a frozen 4,000-subject test set. A useful "
            "improvement is ruled out on both the before-answer and after-answer tasks: the AUROC "
            "upper bounds are +0.010 and +0.002 against the registered margin of +0.020. Probes with "
            "MLP features only are practically equivalent to probes with hidden states only, and the "
            "MLP-augmented probe remains at least as sensitive to question difficulty as the baseline. "
            "The dot-product signal is real but redundant with hidden states.",
            ABSTRACT,
        )
    )
    story.append(Spacer(1, 0.35 * cm))
    verdict = Table(
        [[Paragraph("Verdict: a useful improvement from MLP interaction features is ruled out.", VERDICT)]],
        colWidths=[FRAME_WIDTH],
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f2f2f2")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#777777")),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        ),
    )
    story.append(verdict)
    story.append(Spacer(1, 0.5 * cm))

    figure_number = 0
    table_number = 0
    for kind, payload in blocks:
        if kind == "rule":
            story.append(HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#cccccc"),
                                    spaceBefore=8, spaceAfter=8))
        elif kind == "h1":
            story.append(Paragraph(inline(str(payload)), HEADING))
        elif kind == "h2":
            story.append(Paragraph(inline(str(payload)), HEADING))
        elif kind == "h3":
            story.append(Paragraph(inline(str(payload)), SUBHEADING))
        elif kind == "para":
            story.append(Paragraph(inline(str(payload)), BODY))
        elif kind == "quote":
            story.append(Paragraph(inline(str(payload)), QUOTE))
        elif kind == "bullets":
            for item in payload:
                story.append(Paragraph(inline(item), BULLET, bulletText="\u2022"))
        elif kind == "numbers":
            for number, item in enumerate(payload, start=1):
                story.append(Paragraph(inline(item), NUMBERED, bulletText=f"{number}."))
        elif kind == "table":
            table_number += 1
            story.append(Paragraph(f"Table {table_number}.", TABLE_LABEL))
            story.append(make_table(payload))
        elif kind == "image":
            caption, path = payload
            figure_number += 1
            image_path = REPORT_DIR / path
            reader = ImageReader(str(image_path))
            width_px, height_px = reader.getSize()
            draw_width = 15.5 * cm
            draw_height = draw_width * height_px / width_px
            image = Image(str(image_path), width=draw_width, height=draw_height)
            story.append(
                KeepTogether(
                    [
                        Spacer(1, 4),
                        image,
                        Paragraph(f"Figure {figure_number}: {inline(caption)}", CAPTION),
                    ]
                )
            )

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setFont("Times-Roman", 8.5)
        canvas.setFillGray(0.4)
        if doc.page > 1:
            canvas.drawString(2.2 * cm, A4[1] - 1.5 * cm, "Section 3.3 Confirmatory Report")
            canvas.drawRightString(A4[0] - 2.2 * cm, A4[1] - 1.5 * cm, "Qwen2.5-7B-Instruct")
        canvas.drawCentredString(A4[0] / 2, 1.3 * cm, str(doc.page))
        canvas.restoreState()

    document = BaseDocTemplate(
        str(OUTPUT), pagesize=A4,
        leftMargin=2.2 * cm, rightMargin=2.2 * cm,
        topMargin=2.2 * cm, bottomMargin=2.0 * cm,
        title="Section 3.3 Confirmatory Report",
        author="Keshav Krishna",
    )
    frame = Frame(document.leftMargin, document.bottomMargin, document.width, document.height, id="body")
    document.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=decorate)])
    document.build(story)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    build()
