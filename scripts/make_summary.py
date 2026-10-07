"""Build the short summary page (report/SUMMARY.html) and its PDF.

Reads report/SUMMARY.md, writes a self-contained HTML file with embedded
images, and builds the PDF with ReportLab. Explicit page breaks keep
the summary on pages 1 and 2 and put the three claim-focused plots on pages 3–4.

Run: .venv/bin/python -B scripts/make_summary.py
"""

from __future__ import annotations

import base64
import html as html_lib
import mimetypes
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import markdown
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle,
)

REPORT_DIR = Path(__file__).resolve().parent.parent / "report"
MD_PATH = REPORT_DIR / "SUMMARY.md"
HTML_PATH = REPORT_DIR / "SUMMARY.html"
PDF_PATH = REPORT_DIR / "SUMMARY.pdf"

CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
html { font-size: 17px; }
body {
  font-family: "DejaVu Sans", Arial, sans-serif;
  max-width: 860px; margin: 0 auto; padding: 48px 28px 80px;
  color: #111; background: #fff; line-height: 1.5;
  text-align: left; hyphens: none;
}
h1 { font-size: 1.45em; text-align: center; line-height: 1.3; margin: 0 0 8px; font-weight: 700; }
h2 { font-size: 1.12em; margin-top: 1.5em; margin-bottom: 0.5em; break-after: avoid; }
p { margin: 0.5em 0; }
p.subtitle { text-align: center; font-size: 0.98em; margin: 0 0 8px; }
p.author { text-align: center; font-size: 1.02em; margin: 8px 0 2px; }
p.meta { text-align: center; font-size: 0.84em; color: #444; margin: 0 0 14px; }
blockquote {
  margin: 14px 2%; padding: 10px 16px; border-left: 4px solid #4C72B0;
  background: #f4f7fb; color: #111;
}
blockquote p { margin: 0; }
img { max-width: 100%; height: auto; display: block; margin: 14px auto 4px; break-inside: avoid; break-after: avoid; }
p.caption { text-align: center; font-size: 0.86em; color: #333; margin: 0 auto 18px; max-width: 94%; break-before: avoid; break-inside: avoid; }
table { width: 100%; border-collapse: collapse; font-size: 0.88em; margin: 6px 0 14px; text-align: left; break-inside: avoid; }
th, td { padding: 4px 8px; vertical-align: top; }
th { border-top: 1.6px solid #111; border-bottom: 0.9px solid #111; }
td { border-bottom: 0.6px solid #ddd; }
table tr:last-child td { border-bottom: 1.6px solid #111; }
code { font-family: "SFMono-Regular", Consolas, monospace; background: #f2f2f2; padding: 1px 4px; border-radius: 3px; font-size: 0.88em; }
ol, ul { padding-left: 1.35em; margin: 0.4em 0; }
ol li, ul li { margin-bottom: 3px; }
a { color: #1a3d7c; text-decoration: none; }
a:hover { text-decoration: underline; }
.sheet + .sheet { margin-top: 36px; border-top: 1px solid #ddd; padding-top: 18px; }
.footer { margin-top: 40px; padding-top: 10px; border-top: 1px solid #ddd; color: #666; font-size: 0.8em; text-align: center; }
@media print {
  html { font-size: 10.5pt; }
  body { max-width: none; padding: 0; line-height: 1.30; }
  @page {
    size: A4;
    margin: 14mm 17mm 16mm;
    @bottom-left { content: "PROBE_CONF · PROJECT SUMMARY"; font-size: 8pt; color: #667; }
    @bottom-right { content: counter(page); font-size: 8pt; color: #667; }
  }
  h1 { font-size: 17pt; text-align: left; }
  h2 { font-size: 11.3pt; margin: 11px 0 5px; color: #173a58; }
  p { margin: 5px 0; }
  p.subtitle, p.meta { text-align: left; }
  p.subtitle { font-size: 9pt; }
  p.meta { font-size: 8.5pt; margin-bottom: 9px; }
  blockquote { margin: 8px 0; padding: 7px 11px; }
  table { font-size: 9.3pt; margin: 5px 0 8px; }
  th, td { padding: 3px 7px; }
  .sheet + .sheet { margin: 0; padding: 0; border: 0; break-before: page; }
  .sheet > h2:first-child { margin-top: 0; }
  .plots img { max-height: 92mm; width: 100%; object-fit: contain; margin: 10px auto 4px; }
  .plots p.caption { font-size: 9.1pt; line-height: 1.35; text-align: left; max-width: none; margin-bottom: 17px; }
  .plots p { font-size: 9.5pt; }
  h2, h3, th { break-after: avoid; }
  img, table, blockquote, li { break-inside: avoid; }
  a { color: #111; }
}
"""


def embed_images(html: str) -> str:
    def repl(match: re.Match) -> str:
        alt, src = match.group(1), match.group(2)
        path = REPORT_DIR / src
        if not path.exists():
            raise FileNotFoundError(f"summary image does not exist: {path}")
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return f'<img alt="{alt}" src="data:{mime};base64,{b64}" />'

    return re.sub(r'<img alt="([^"]*)" src="([^"]+)"\s*/?>', repl, html)


def build_pdf(body: str, title: str) -> None:
    """Render the same Markdown content without a browser dependency."""
    fonts = Path("/usr/share/fonts/truetype/dejavu")
    for name, filename in (("Summary", "DejaVuSans.ttf"), ("SummaryBold", "DejaVuSans-Bold.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))
    pdfmetrics.registerFontFamily(
        "Summary", normal="Summary", bold="SummaryBold", italic="Summary", boldItalic="SummaryBold"
    )
    base = ParagraphStyle("body", fontName="Summary", fontSize=10.2, leading=13.2, spaceAfter=5)
    styles = {
        "p": base,
        "h1": ParagraphStyle("title", parent=base, fontName="SummaryBold", fontSize=17,
                             leading=20, spaceAfter=8, textColor=colors.HexColor("#173a58")),
        "h2": ParagraphStyle("heading", parent=base, fontName="SummaryBold", fontSize=11.3,
                             leading=14, spaceBefore=8, spaceAfter=5, keepWithNext=True,
                             textColor=colors.HexColor("#173a58")),
        "subtitle": ParagraphStyle("subtitle", parent=base, fontSize=9, leading=12),
        "meta": ParagraphStyle("meta", parent=base, fontSize=8.5, leading=11, spaceAfter=9,
                               textColor=colors.HexColor("#555555")),
        "caption": ParagraphStyle("caption", parent=base, fontSize=9.1, leading=12,
                                  spaceAfter=17),
        "quote": ParagraphStyle("quote", parent=base, backColor=colors.HexColor("#f0f5fa"),
                                borderPadding=7, spaceBefore=7, spaceAfter=12),
        "cell": ParagraphStyle("cell", parent=base, fontSize=9.3, leading=11.5, spaceAfter=0),
        "list": ParagraphStyle("list", parent=base, leftIndent=14, firstLineIndent=-14, spaceAfter=3),
    }
    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=A4, leftMargin=17*mm, rightMargin=17*mm,
                            topMargin=14*mm, bottomMargin=16*mm, title=title, author="Keshav Krishna")

    def inline(node):
        value = (html_lib.escape(node.text or "")
                 + "".join(ET.tostring(child, encoding="unicode", method="xml") for child in node))
        return value.replace("<code>", '<font size="8.5">').replace("</code>", "</font>")

    story = []
    sections = ET.fromstring(f"<root>{body}</root>")
    for page, section in enumerate(sections):
        if page:
            story.append(PageBreak())
        for node in section:
            if node.tag in ("h1", "h2", "p"):
                img = node.find("img")
                if img is not None:
                    item = Image(str(REPORT_DIR / img.attrib["src"]))
                    factor = min((doc.width-12)/item.imageWidth, 92*mm/item.imageHeight)
                    item.drawWidth = item.imageWidth * factor
                    item.drawHeight = item.imageHeight * factor
                    item.keepWithNext = True
                    item.spaceBefore = 9
                    item.spaceAfter = 5
                    story.append(item)
                else:
                    style = styles.get(node.attrib.get("class", node.tag), base)
                    story.append(Paragraph(inline(node), style))
            elif node.tag == "blockquote":
                for child in node:
                    story.append(Paragraph(inline(child), styles["quote"]))
            elif node.tag == "table":
                rows = [[Paragraph(inline(cell), styles["cell"]) for cell in row]
                        for row in node.findall(".//tr")]
                table = Table(rows, colWidths=[doc.width-70, 58], hAlign="LEFT")
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf3f8")),
                    ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.HexColor("#173a58")),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]))
                story.append(table)
            elif node.tag == "ol":
                for i, child in enumerate(node, 1):
                    story.append(Paragraph(f"{i}. {inline(child)}", styles["list"]))
            else:
                raise ValueError(f"Unsupported summary element: {node.tag}")

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Summary", 8)
        canvas.setFillColor(colors.HexColor("#666677"))
        canvas.drawString(17*mm, 9*mm, "PROBE_CONF · PROJECT SUMMARY")
        canvas.drawRightString(A4[0]-17*mm, 9*mm, str(document.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    text = MD_PATH.read_text()
    pages = text.split('<div class="pagebreak"></div>')
    body = "\n".join(
        f'<section class="sheet {"summary" if i < 2 else "plots"}">'
        + markdown.markdown(
            page,
            extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
        )
        + "</section>"
        for i, page in enumerate(pages)
    )
    pdf_body = body
    body = embed_images(body)
    title = html_lib.escape(text.splitlines()[0].removeprefix("# "))
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""
    HTML_PATH.write_text(html)
    n_imgs = body.count("data:image/png;base64")
    print(f"wrote {HTML_PATH} ({HTML_PATH.stat().st_size/1e6:.2f} MB, {n_imgs} embedded images)")

    build_pdf(pdf_body, html_lib.unescape(title))
    print(f"wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
