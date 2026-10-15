"""Generate LaTeX from the two-page Markdown report and compile with Tectonic.

    .venv/bin/python report/make_selective_metacognition_pdf.py

Markdown is the content source. The generated .tex is retained for review.
Compilation uses cached TeX resources and disables shell escape.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
SOURCE = REPORT_DIR / "SELECTIVE_METACOGNITION_TOY_REPORT.md"
TEX = SOURCE.with_suffix(".tex")
OUTPUT = SOURCE.with_suffix(".pdf")

_spec = importlib.util.spec_from_file_location(
    "section33_parser", REPORT_DIR / "make_section33_confirmatory_pdf.py"
)
parser = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parser)

PREAMBLE = r"""
\documentclass[10pt,a4paper]{article}
\usepackage[margin=17mm,top=15mm,bottom=15mm]{geometry}
\usepackage{amsmath,amssymb,booktabs,tabularx,xcolor,hyperref,titlesec,fancyhdr}
\DeclareFontShape{TU}{lmr}{bx}{n}{<-> "[lmroman10-bold]:mapping=tex-text;"}{}
\definecolor{ink}{HTML}{22394C}
\definecolor{linkblue}{HTML}{244E70}
\hypersetup{colorlinks=true,urlcolor=linkblue,
 pdftitle={Can better retrieval make confidence more useful?},
 pdfauthor={Keshav Krishna}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.12}
\titleformat{\section}{\large\bfseries\color{ink}}{}{0pt}{}
\titlespacing*{\section}{0pt}{7pt}{3pt}
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\fancyfoot[R]{\small\thepage}
\setlength{\footskip}{9mm}
\setlength{\emergencystretch}{1em}
\begin{document}
\fontsize{10}{11.7}\selectfont
\setlength{\abovedisplayskip}{4pt}
\setlength{\belowdisplayskip}{4pt}
"""


def escape(text: str) -> str:
    chars = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
        "·": r"\textperiodcentered{}", "—": "---", "–": "--",
    }
    return "".join(chars.get(c, c) for c in text)


def inline(text: str) -> str:
    token = re.compile(
        r"\$([^$]+)\$|\*\*(.+?)\*\*|\x60([^\x60]+)\x60|\[([^\]]+)\]\(([^)]+)\)"
    )
    out, last = [], 0
    for match in token.finditer(text):
        out.append(escape(text[last:match.start()]))
        math, bold, code, label, url = match.groups()
        if math is not None:
            out.append("$" + math + "$")
        elif bold is not None:
            out.append(r"\textbf{" + inline(bold) + "}")
        elif code is not None:
            out.append(r"\texttt{" + escape(code) + "}")
        else:
            out.append(r"\href{\detokenize{" + url + "}}{" + inline(label) + "}")
        last = match.end()
    return "".join(out) + escape(text[last:])


def table(rows: list[str]) -> str:
    data = []
    for row in rows:
        cells = [c.strip() for c in row.strip("|").split("|")]
        if all(set(c) <= {"-", ":"} for c in cells):
            continue
        data.append(cells)
    header, body = data[0], data[1:]
    ncols = len(header)
    if ncols == 2:
        start = r"\begin{tabularx}{\linewidth}{@{}lX@{}}"
        end = r"\end{tabularx}"
    else:
        start = r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l" + "r" * (ncols - 1) + r"@{}}"
        end = r"\end{tabular*}"
    aliases = {
        "Report Brier": r"\shortstack{Report\\Brier}",
        "Baseline Brier": r"\shortstack{Baseline\\Brier}",
        "Probability error": r"\shortstack{Probability\\error}",
        "Added: seen strengths": r"\shortstack{Added value\\seen strengths}",
        "Added: unseen strengths": r"\shortstack{Added value\\unseen strengths}",
    }
    lines = [r"\par\smallskip{\small", start, r"\toprule"]
    lines.append(" & ".join(aliases.get(c, inline(c)) for c in header) + r"\\ \midrule")
    for row in body:
        cells = [inline(c) for c in row]
        if row[0] == "C":
            cells = [r"\textbf{" + c + "}" for c in cells]
        lines.append(" & ".join(cells) + r"\\")
    return "\n".join(lines + [r"\bottomrule", end, r"}\par\smallskip"])


def make_tex() -> str:
    blocks = parser.parse_markdown(SOURCE.read_text())
    output = [PREAMBLE]
    for kind, payload in blocks:
        text = str(payload)
        if kind == "h1":
            output.append(r"{\bfseries\fontsize{18}{21}\selectfont\color{ink} " + inline(text) + r"\par}\vspace{3pt}")
        elif kind in ("h2", "h3"):
            output.append(r"\section*{" + inline(text) + "}")
        elif kind == "para":
            if text == "<!-- pagebreak -->":
                output.append(r"\newpage")
            elif text.startswith("$$") and text.endswith("$$"):
                output.append("\\[\n" + text[2:-2].strip() + "\n\\]")
            elif text.startswith("**Selective MLP"):
                output.append(r"{\small " + inline(text) + r"\par}")
            elif text.startswith("**Evidence:"):
                output.append(r"{\footnotesize " + inline(text) + r"\par}")
            else:
                output.append(inline(text) + "\n\\par")
        elif kind == "table":
            output.append(table(payload))
        elif kind == "rule":
            output.append(r"\smallskip\hrule\smallskip")
        else:
            raise ValueError(f"Unsupported report block: {kind}")
    return "\n\n".join(output) + "\n\\end{document}\n"


def build() -> None:
    TEX.write_text(make_tex())
    compiler = shutil.which("tectonic")
    if compiler is None:
        raise RuntimeError("Tectonic is required to build this LaTeX report.")
    cache = Path(tempfile.gettempdir()) / "probe-conf-tectonic-cache"
    cache.mkdir(exist_ok=True)
    for name in ("tectonic", "Tectonic"):
        source = Path.home() / ".cache" / name
        destination = cache / name
        if source.is_dir() and not destination.exists():
            shutil.copytree(source, destination)
    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = str(cache)
    with tempfile.TemporaryDirectory(prefix="selective-toy-tex-") as build_dir:
        subprocess.run(
            [compiler, "--only-cached", "--untrusted", "--keep-logs",
             "--outdir", build_dir, str(TEX)],
            env=env, check=True,
        )
        result = Path(build_dir) / OUTPUT.name
        info = subprocess.check_output(["pdfinfo", str(result)], text=True)
        pages = int(re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE).group(1))
        if pages != 2:
            raise RuntimeError(f"Expected exactly two pages, got {pages}; PDF not replaced.")
        log = (Path(build_dir) / TEX.with_suffix(".log").name).read_text(errors="replace")
        if "Overfull" in log:
            raise RuntimeError("LaTeX reports overfull content; PDF not replaced.")
        shutil.copyfile(result, OUTPUT)
    print(f"Wrote {TEX.name} and {OUTPUT.name} ({pages} pages, LaTeX).")


if __name__ == "__main__":
    build()
