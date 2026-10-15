"""Build the selective-metacognition toy report as a PDF.

Reuses the markdown parser and paragraph styles from
``make_section33_confirmatory_pdf.py``.

    .venv/bin/python report/make_selective_metacognition_pdf.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
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
SOURCE = REPORT_DIR / "SELECTIVE_METACOGNITION_TOY_REPORT.md"
OUTPUT = REPORT_DIR / "SELECTIVE_METACOGNITION_TOY_REPORT.pdf"
FRAME_WIDTH = 16.6 * cm

_spec = importlib.util.spec_from_file_location(
    "section33_pdf", REPORT_DIR / "make_section33_confirmatory_pdf.py"
)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

base.TABLE_WIDTHS[5] = [2.6 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm]

# Compact styles for a two-page report.
base.BODY.fontSize = 8.9
base.BODY.leading = 10.9
base.BODY.spaceAfter = 3
base.HEADING.fontSize = 11.0
base.HEADING.leading = 13.2
base.HEADING.spaceBefore = 6
base.HEADING.spaceAfter = 2
base.SUBHEADING.fontSize = 9.5
base.SUBHEADING.leading = 11.6
base.CELL.fontSize = 7.0
base.CELL.leading = 8.8
base.CELL_HEAD.fontSize = 7.0
base.CELL_HEAD.leading = 8.8
base.TABLE_LABEL.fontSize = 7.0
base.CAPTION.fontSize = 7.0
base.CAPTION.leading = 8.8
base.CAPTION.spaceBefore = 2
base.CAPTION.spaceAfter = 5
base.ABSTRACT.fontSize = 8.5
base.ABSTRACT.leading = 11.0
base.META.fontSize = 9.0
base.QUOTE.fontSize = 9.0

TITLE_TEXT = "Selective MLP Retrieval for Metacognitive Confidence"
SUBTITLE_TEXT = "Toy experiments E1-E3: can training selection improve self-report?"
ABSTRACT_TEXT = (
    "We test whether training a functionally selective retrieval computation makes a model's "
    "confidence report more useful. In a controlled toy world with exact ground truth, selectivity "
    "training transfers to fresh bindings and its report is consistently closer to the true success "
    "probability, but it adds no information beyond output probabilities plus the intervention "
    "description, and it does not beat a matched-supervision control. A probe diagnostic shows why: "
    "no linear or MLP readout of the clean internal state approaches the baseline, even when trained "
    "directly on the true post-intervention probability. The added-information claim is not supported "
    "in the toy; the failure is representational for the tested readouts."
)
VERDICT_TEXT = (
    "Verdict: selectivity training improves retrieval and report accuracy, "
    "but adds no reporting information beyond output probabilities."
)


def make_table(rows: list[str]) -> Table:
    parsed = []
    for row in rows:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        parsed.append(cells)
    header, body = parsed[0], parsed[1:]
    data = [[Paragraph(base.inline(cell), base.CELL_HEAD) for cell in header]]
    for row in body:
        data.append([Paragraph(base.inline(cell), base.CELL) for cell in row])
    table = Table(data, colWidths=base.TABLE_WIDTHS.get(len(header)), hAlign="CENTER", repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def build() -> None:
    text = SOURCE.read_text()
    body_text = text.split("\n---\n", 1)[1]
    blocks = base.parse_markdown(body_text)

    story: list = []
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(TITLE_TEXT, base.TITLE))
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph(SUBTITLE_TEXT, base.SUBTITLE))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Keshav Krishna", base.META))
    story.append(Paragraph("New York University", base.META))
    story.append(Paragraph("15 October 2026", base.META))
    story.append(Spacer(1, 0.35 * cm))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#999999")))
    story.append(Paragraph("Summary", base.ABSTRACT_HEAD))
    story.append(Paragraph(base.inline(ABSTRACT_TEXT), base.ABSTRACT))
    story.append(Spacer(1, 0.3 * cm))
    verdict = Table(
        [[Paragraph(VERDICT_TEXT, base.VERDICT)]],
        colWidths=[FRAME_WIDTH],
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f2f2f2")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#777777")),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        ),
    )
    story.append(verdict)
    story.append(Spacer(1, 0.45 * cm))

    figure_number = 0
    table_number = 0
    for kind, payload in blocks:
        if kind == "rule":
            story.append(
                HRFlowable(
                    width="100%",
                    thickness=0.4,
                    color=colors.HexColor("#cccccc"),
                    spaceBefore=8,
                    spaceAfter=8,
                )
            )
        elif kind in ("h1", "h2"):
            story.append(Paragraph(base.inline(str(payload)), base.HEADING))
        elif kind == "h3":
            story.append(Paragraph(base.inline(str(payload)), base.SUBHEADING))
        elif kind == "para":
            story.append(Paragraph(base.inline(str(payload)), base.BODY))
        elif kind == "quote":
            story.append(Paragraph(base.inline(str(payload)), base.QUOTE))
        elif kind == "bullets":
            for item in payload:
                story.append(Paragraph(base.inline(item), base.BULLET, bulletText="\u2022"))
        elif kind == "numbers":
            for number, item in enumerate(payload, start=1):
                story.append(Paragraph(base.inline(item), base.NUMBERED, bulletText=f"{number}."))
        elif kind == "table":
            table_number += 1
            story.append(Paragraph(f"Table {table_number}.", base.TABLE_LABEL))
            story.append(make_table(payload))
        elif kind == "image":
            caption, path = payload
            figure_number += 1
            image_path = REPORT_DIR / path
            reader = ImageReader(str(image_path))
            width_px, height_px = reader.getSize()
            draw_width = 7.8 * cm
            draw_height = draw_width * height_px / width_px
            image = Image(str(image_path), width=draw_width, height=draw_height)
            story.append(
                KeepTogether(
                    [
                        Spacer(1, 4),
                        image,
                        Paragraph(f"Figure {figure_number}: {base.inline(caption)}", base.CAPTION),
                    ]
                )
            )

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setFont("Times-Roman", 8.5)
        canvas.setFillGray(0.4)
        if doc.page > 1:
            canvas.drawString(2.2 * cm, A4[1] - 1.5 * cm, "Selective Metacognition Toy Report")
            canvas.drawRightString(A4[0] - 2.2 * cm, A4[1] - 1.5 * cm, "E1-E3")
        canvas.drawCentredString(A4[0] / 2, 1.3 * cm, str(doc.page))
        canvas.restoreState()

    document = BaseDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.6 * cm,
        title="Selective MLP Retrieval for Metacognitive Confidence: Toy Experiments",
        author="Keshav Krishna",
    )
    frame = Frame(document.leftMargin, document.bottomMargin, document.width, document.height, id="body")
    document.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=decorate)])
    document.build(story)
    print(f"wrote {OUTPUT} ({document.page} pages)")


if __name__ == "__main__":
    build()
