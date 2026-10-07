"""Build the report PDF directly from Markdown, without a browser.

Requires markdown, Pillow, and reportlab. Run with the project Python.
"""
from __future__ import annotations

import html
import xml.etree.ElementTree as ET
from pathlib import Path

import markdown
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                               Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"
WIDTH = letter[0] - 88


def inline(node):
    text = html.escape(node.text or "")
    for child in node:
        content = inline(child)
        if child.tag in ("strong", "b"):
            text += f"<b>{content}</b>"
        elif child.tag in ("em", "i"):
            text += f"<i>{content}</i>"
        elif child.tag == "code":
            text += f'<font name="ReportMono" size="8">{content}</font>'
        elif child.tag == "a":
            target = html.escape(child.get("href", ""), quote=True)
            text += f'<a href="{target}" color="#1d5289">{content}</a>'
        elif child.tag == "br":
            text += "<br/>"
        else:
            text += content
        text += html.escape(child.tail or "")
    return text


def main(source_path=None, output_path=None, footer_text=None, generic_tables=False):
    source_path = Path(source_path) if source_path else REPORT / "REPORT.md"
    output_path = Path(output_path) if output_path else REPORT / "REPORT.pdf"
    footer_text = footer_text or "Offline probe pilot • Evidence review: 3 October 2026"
    font_dir = Path("/usr/share/fonts/truetype/dejavu")
    for name, file in [("ReportSerif", "DejaVuSerif.ttf"), ("ReportBold", "DejaVuSerif-Bold.ttf"),
                       ("ReportMono", "DejaVuSansMono.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(font_dir / file)))
    pdfmetrics.registerFontFamily("ReportSerif", normal="ReportSerif", bold="ReportBold",
                                  italic="ReportSerif", boldItalic="ReportBold")
    base = ParagraphStyle("body", fontName="ReportSerif", fontSize=9.3, leading=14,
                          spaceAfter=8, alignment=TA_LEFT)
    styles = {"p": base}
    for tag, size in [("h1", 16), ("h2", 12), ("h3", 10.4)]:
        styles[tag] = ParagraphStyle(tag, parent=base, fontName="ReportBold", fontSize=size,
            leading=size*1.35, spaceBefore=14, spaceAfter=8, keepWithNext=True,
            alignment=TA_CENTER if tag == "h1" else TA_LEFT)
    caption = ParagraphStyle("caption", parent=base, fontSize=8.5, leading=12,
                             textColor=colors.HexColor("#384b5c"), alignment=TA_CENTER)
    cell = ParagraphStyle("cell", parent=base, fontSize=8, leading=11, spaceAfter=0)
    code = ParagraphStyle("code", parent=base, fontName="ReportMono", fontSize=7.7, leading=10.8)
    source = source_path.read_text()
    body = markdown.markdown(source, extensions=["tables", "fenced_code", "attr_list", "sane_lists"])
    root = ET.fromstring("<root>" + body + "</root>")
    story = []
    pending_image = None
    for node in root:
        if node.tag == "p" and node.find("img") is not None:
            img = node.find("img")
            image = Image(str(REPORT / img.get("src")))
            max_height = 602 if "flow_" in img.get("src") else 310
            scale = min(WIDTH/image.imageWidth, max_height/image.imageHeight)
            image.drawWidth = image.imageWidth * scale
            image.drawHeight = image.imageHeight * scale
            pending_image = image
            if "flow_" in img.get("src"):
                story.append(PageBreak())
            continue
        if pending_image is not None:
            block = [pending_image, Spacer(1, 8)]
            if node.tag == "p" and "caption" in node.get("class", ""):
                block += [Paragraph(inline(node), caption), Spacer(1, 10)]
                story.append(KeepTogether(block))
                pending_image = None
                continue
            story.append(KeepTogether(block))
            pending_image = None
        if node.tag in styles:
            style = styles[node.tag]
            if node.get("class") in ("subtitle", "author", "meta", "keywords"):
                style = ParagraphStyle(node.get("class"), parent=base, alignment=TA_CENTER)
            story.append(Paragraph(inline(node), style))
        elif node.tag == "table":
            rows = [list(tr) for tr in node.findall(".//tr")]
            ncols = len(rows[0])
            if generic_tables:
                widths = [WIDTH / ncols] * ncols
            elif ncols >= 6:
                widths = [WIDTH*.26] + [WIDTH*.74/(ncols-1)]*(ncols-1)
            elif ncols == 3:
                if "Feature" == "".join(rows[0][0].itertext()):
                    widths = [WIDTH*.24, WIDTH*.55, WIDTH*.21]
                else:
                    widths = [WIDTH*.46, WIDTH*.20, WIDTH*.34]
            elif ncols == 4:
                widths = [WIDTH*.39, WIDTH*.16, WIDTH*.23, WIDTH*.22]
            else:
                widths = [WIDTH*.43, WIDTH*.57]
            data = [[Paragraph(inline(c), cell) for c in row] for row in rows]
            table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0,0),(-1,0), colors.HexColor("#eaf1f7")),
                ("LINEBELOW",(0,0),(-1,0),.8,colors.HexColor("#45596b")),
                ("LINEBELOW",(0,1),(-1,-1),.25,colors.HexColor("#cbd5df")),
                ("VALIGN",(0,0),(-1,-1),"TOP"),
                ("LEFTPADDING",(0,0),(-1,-1),5), ("RIGHTPADDING",(0,0),(-1,-1),5),
                ("TOPPADDING",(0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5),
            ]))
            story += [table, Spacer(1,12)]
        elif node.tag in ("ul", "ol"):
            for i, li in enumerate(node, 1):
                prefix = f"{i}. " if node.tag == "ol" else "• "
                story.append(Paragraph(prefix + inline(li), base))
        elif node.tag == "pre":
            story += [KeepTogether([Preformatted("".join(node.itertext()), code), Spacer(1,12)])]
        elif node.tag == "blockquote":
            for child in node:
                story.append(Paragraph(inline(child), base))
        elif node.tag == "hr":
            story.append(Spacer(1,8))
    if pending_image is not None:
        story.append(pending_image)
    doc = SimpleDocTemplate(str(output_path), pagesize=letter,
                            rightMargin=44, leftMargin=44, topMargin=42, bottomMargin=42,
                            title=source.splitlines()[0].removeprefix("# "), author="Keshav Krishna")
    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("ReportSerif", 7)
        canvas.setFillColor(colors.HexColor("#45596b"))
        canvas.drawString(44, 22, footer_text)
        canvas.drawRightString(letter[0]-44, 22, str(document.page))
        canvas.restoreState()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("Wrote", output_path)


if __name__ == "__main__":
    main()
