"""Turn reports/gdoc_week5-7/report_*.md into paste-ready pages for the group's Google Doc ("Week 5-7" tab),
in the style of the "Week 3-4" tab: Arial 11, headings bold #1f4e79, tables with grey borders and a light
blue header row, pictures 6.5 in wide.

Rules learned on the Week 3-4 tab:
* every style inline, longhand margins; no whitespace between tags (Docs turns it into blank lines);
* tables without a margin, <tr style="height:0pt">, notes above a table rather than below;
* pictures as data URIs (Docs uploads them when the HTML is pasted).

Writes, per part (the parts are split at [[pagebreak]]): part<N>.html (open it, Ctrl+A, Ctrl+C: a copy handler puts
the exact HTML on the clipboard) and part<N>.json ({"html", "text"} for a synthetic paste event).

    .venv/Scripts/python.exe -X utf8 scripts/gdoc_report_html.py
"""
from __future__ import annotations

import base64
import html
import json
import re
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports" / "gdoc_week5-7"
FIG = REP / "figures"
OUT = REP / "paste"
OUT.mkdir(parents=True, exist_ok=True)

BODY = "font-family:Arial;color:#000000;line-height:1.15;"
P = BODY + "font-size:11pt;margin-top:0pt;margin-bottom:8pt;"
H1 = BODY + "font-size:20pt;font-weight:700;color:#1f4e79;text-align:center;margin-top:0pt;margin-bottom:12pt;"
H2 = BODY + "font-size:13pt;font-weight:700;color:#1f4e79;margin-top:18pt;margin-bottom:6pt;"
H3 = BODY + "font-size:11pt;font-weight:700;margin-top:10pt;margin-bottom:4pt;"
LI = BODY + "font-size:11pt;margin-top:0pt;margin-bottom:2pt;"
TD = ("border:1pt solid #bfbfbf;padding:4pt 6pt 4pt 6pt;vertical-align:top;")
TDP = BODY + "font-size:10pt;margin-top:0pt;margin-bottom:0pt;"
TEXT_WIDTH_PT = 468.0                                                       # 6.5 in


def inline(s: str) -> str:
    s = s.replace("\\|", "\x00")
    s = html.escape(s, quote=False).replace("\x00", "|")
    s = re.sub(r"\*\*(.+?)\*\*", r'<span style="font-weight:700">\1</span>', s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r'<span style="font-style:italic">\1</span>', s)
    return s


def png_size(path: Path):
    with open(path, "rb") as f:
        head = f.read(24)
    return struct.unpack(">II", head[16:24])


def image(name: str) -> str:
    path = FIG / f"{name}.png"
    w, h = png_size(path)
    W = 624
    H = round(W * h / w)
    data = base64.b64encode(path.read_bytes()).decode()
    return (f'<p style="{P}text-align:center;"><img src="data:image/png;base64,{data}" width="{W}" height="{H}" '
            f'style="width:{W}px;height:{H}px" alt="{name}"></p>')


def table(rows):
    cells = [[c.strip() for c in r.strip().strip("|").replace("\\|", "\x00").split("|")] for r in rows]
    cells = [[c.replace("\x00", "\\|") for c in r] for r in cells]
    n = max(len(r) for r in cells)
    cells = [r + [""] * (n - len(r)) for r in cells]
    lens = [max(len(re.sub(r"\*\*", "", r[j])) for r in cells) for j in range(n)]
    weights = [min(max(L, 6), 70) ** 0.8 for L in lens]
    widths = [TEXT_WIDTH_PT * w / sum(weights) for w in weights]
    out = ['<table style="border-collapse:collapse;border-spacing:0;margin-left:0pt;margin-right:auto;"><tbody>']
    for i, r in enumerate(cells):
        out.append('<tr style="height:0pt">')
        for j, c in enumerate(r):
            bg = "background-color:#e8eef5;" if i == 0 else ""
            txt = inline(c)
            if i == 0:
                txt = f'<span style="font-weight:700">{txt}</span>'
            out.append(f'<td style="{TD}{bg}width:{widths[j]:.0f}pt"><p style="{TDP}">{txt}</p></td>')
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def lists(items):
    """items: list of (level, ordered, text). Builds nested <ul>/<ol> with spacing on every <li>."""
    out, stack = [], []
    for k, (lvl, ordered, text) in enumerate(items):
        tag = "ol" if ordered else "ul"
        while len(stack) > lvl + 1:
            out.append(f"</li></{stack.pop()}>")
        if len(stack) == lvl + 1 and k:
            out.append("</li>")
        if len(stack) < lvl + 1:
            style = "list-style-type:decimal;" if ordered else ("list-style-type:disc;" if lvl == 0 else "list-style-type:circle;")
            out.append(f'<{tag} style="{style}margin-top:0pt;margin-bottom:0pt;padding-left:0pt;">')
            stack.append(tag)
        last = k == len(items) - 1 or items[k + 1][0] < lvl or (items[k + 1][0] == lvl and False)
        mb = "8pt" if (k == len(items) - 1) else "2pt"
        out.append(f'<li style="{LI}margin-left:{18 + 18 * lvl}pt;margin-bottom:{mb};">'
                   f'<p style="{LI}margin-bottom:0pt;">{inline(text)}</p>')
    while stack:
        out.append(f"</li></{stack.pop()}>")
    return "".join(out)


def convert(md: str) -> str:
    lines = md.splitlines()
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            i += 1
            continue
        m = re.match(r"\[\[img:(\w+)\]\]", s)
        if m:
            out.append(image(m.group(1)))
            i += 1
            continue
        if s.startswith("### "):
            out.append(f'<h3 style="{H3}">{inline(s[4:])}</h3>')
        elif s.startswith("## "):
            out.append(f'<h2 style="{H2}">{inline(s[3:])}</h2>')
        elif s.startswith("# "):
            out.append(f'<h1 style="{H1}">{inline(s[2:])}</h1>')
        elif s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            out.append(table(rows))
            continue
        elif re.match(r"^(\s*)(- |\d+\. )", line):
            items = []
            while i < len(lines) and re.match(r"^(\s*)(- |\d+\. )", lines[i]):
                mm = re.match(r"^(\s*)(- |\d+\. )(.*)$", lines[i])
                items.append((len(mm.group(1)) // 2, mm.group(2) != "- ", mm.group(3)))
                i += 1
            out.append(lists(items))
            continue
        else:
            out.append(f'<p style="{P}">{inline(s)}</p>')
        i += 1
    return "".join(out)


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{background:#f3f3f3;margin:0;padding:24px}}#content{{background:#fff;width:624px;margin:0 auto;padding:48px 72px;box-shadow:0 1px 4px rgba(0,0,0,.15)}}</style>
</head><body><div id="content">{body}</div>
<script>
document.addEventListener('copy', function (e) {{
  var c = document.getElementById('content');
  e.clipboardData.setData('text/html', '<meta charset="utf-8">' + c.innerHTML);
  e.clipboardData.setData('text/plain', c.innerText);
  e.preventDefault();
}});
</script></body></html>"""


def main():
    parts = []
    for f in sorted(REP.glob("report_*.md")):
        for chunk in f.read_text(encoding="utf-8").split("[[pagebreak]]"):
            if chunk.strip():
                parts.append(chunk.strip())
    for k, md in enumerate(parts):
        body = convert(md)
        title = re.search(r"^# (.+)$", md, re.M).group(1)
        (OUT / f"part{k}.html").write_text(PAGE.format(title=html.escape(title), body=body), encoding="utf-8")
        text = re.sub(r"<[^>]+>", " ", body)
        (OUT / f"part{k}.json").write_text(json.dumps({"html": '<meta charset="utf-8">' + body, "text": html.unescape(text)}),
                                          encoding="utf-8")
        n_img = body.count("<img")
        print(f"part{k}: {title} | {len(body) / 1e6:.2f} MB html | {n_img} pictures | {body.count('<table')} tables")


if __name__ == "__main__":
    main()
