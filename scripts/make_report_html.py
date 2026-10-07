"""Convert report/REPORT.md into a paper-styled, self-contained HTML file.

- Serif body, centered title block, abstract, numbered sections, captions.
- Images referenced from figures/ are embedded as base64 data URIs, so the HTML
  file can be shared or moved on its own.
- Print rules give a clean PDF via headless Chrome.

Run: .venv/bin/python scripts/make_report_html.py
Then: google-chrome --headless --no-sandbox --disable-gpu --no-pdf-header-footer \
        --print-to-pdf=report/REPORT.pdf file://$PWD/report/REPORT.html
"""

from __future__ import annotations

import base64
import html as html_lib
import mimetypes
import re
from pathlib import Path

import markdown

REPORT_DIR = Path(__file__).resolve().parent.parent / "report"
MD_PATH = REPORT_DIR / "REPORT.md"
OUT_PATH = REPORT_DIR / "REPORT.html"

CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
html { font-size: 17px; }
body {
  font-family: Georgia, "Times New Roman", "Nimbus Roman", serif;
  max-width: 860px; margin: 0 auto; padding: 56px 28px 100px;
  color: #111; background: #fff; line-height: 1.62;
  text-align: left; hyphens: auto;
}
h1 { font-size: 1.6em; text-align: center; line-height: 1.3; margin: 0 0 10px; font-weight: 700; }
h2 { font-size: 1.18em; margin-top: 2.1em; margin-bottom: 0.6em; break-after: avoid; }
h3 { font-size: 1.03em; margin-top: 1.5em; margin-bottom: 0.5em; font-style: italic; font-weight: 700; break-after: avoid; }
p { margin: 0.55em 0; }
p.subtitle { text-align: center; font-size: 1.02em; margin: 0 0 10px; }
p.subtitle strong { font-weight: 400; font-style: italic; }
p.author { text-align: center; font-size: 1.05em; margin: 10px 0 2px; }
p.meta { text-align: center; font-size: 0.86em; color: #444; margin: 0 0 6px; }
p.keywords { text-align: center; font-size: 0.88em; color: #333; margin: 2px 10% 18px; padding-bottom: 14px; border-bottom: 1px solid #ccc; }
p.abstract {
  font-size: 0.94em; text-align: justify; margin: 22px 3% 6px; padding: 12px 18px;
  background: #f8f8f6; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;
}
blockquote {
  margin: 18px 4%; padding: 4px 16px; border-left: 3px solid #999;
  background: #fafafa; color: #222; font-style: italic;
}
img { max-width: 100%; height: auto; display: block; margin: 24px auto 6px; break-inside: avoid; }
p.caption { text-align: center; font-size: 0.88em; color: #333; margin: 0 auto 24px; max-width: 94%; break-before: avoid; }
p.caption em { font-style: normal; }
p.tablecaption { font-size: 0.88em; margin: 22px 0 4px; text-align: left; break-after: avoid; }
table { width: 100%; border-collapse: collapse; font-size: 0.9em; margin: 0 0 18px; text-align: left; break-inside: avoid; }
th, td { padding: 5px 8px; vertical-align: top; }
th { border-top: 1.6px solid #111; border-bottom: 0.9px solid #111; }
td { border-bottom: 0.6px solid #ddd; }
table tr:last-child td { border-bottom: 1.6px solid #111; }
code { font-family: "SFMono-Regular", Consolas, monospace; background: #f2f2f2; padding: 1px 4px; border-radius: 3px; font-size: 0.88em; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; font-size: 0.83em; }
td code { overflow-wrap: anywhere; }
.flow-panel { break-before: page; }
hr { border: none; border-top: 1px solid #ddd; margin: 26px 0; }
ol, ul { padding-left: 1.4em; }
ol li, ul li { margin-bottom: 4px; }
ol.references li { margin-bottom: 7px; text-align: left; }
a { color: #1a3d7c; text-decoration: none; }
a:hover { text-decoration: underline; }
.footer { margin-top: 56px; padding-top: 12px; border-top: 1px solid #ddd; color: #666; font-size: 0.82em; text-align: center; }
@media print {
  html { font-size: 10.6pt; }
  body { max-width: none; padding: 0; }
  @page { margin: 0.95in; }
  h2, h3, p.tablecaption { break-after: avoid; }
  img, table, p.abstract, blockquote { break-inside: avoid; }
  img.flow-panel { width: 100%; max-height: 8.4in; object-fit: contain; margin-top: 0; }
  a { color: #111; }
}
"""


def embed_images(html: str) -> str:
    def repl(match: re.Match) -> str:
        alt, src = match.group(1), match.group(2)
        path = REPORT_DIR / src
        if not path.exists():
            raise FileNotFoundError(f"report image does not exist: {path}")
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        css_class = ' class="flow-panel"' if path.name.startswith("flow_") else ""
        return f'<img alt="{alt}"{css_class} src="data:{mime};base64,{b64}" />'

    return re.sub(r'<img alt="([^"]*)" src="([^"]+)"\s*/?>', repl, html)


def main() -> None:
    text = MD_PATH.read_text()
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "toc", "sane_lists", "attr_list"],
    )
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
<div class="footer">Working paper generated from report/REPORT.md. Figures embedded.
Pilot run: Qwen2.5-7B-Instruct on 1,087 PopQA person questions, 28–29 September 2026.</div>
</body>
</html>
"""
    OUT_PATH.write_text(html)
    n_imgs = body.count("data:image/png;base64")
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.2f} MB, {n_imgs} embedded images)")


if __name__ == "__main__":
    main()
