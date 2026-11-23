"""Minimal Markdown -> LaTeX for docs/technote_a1_round3.md (headings, paragraphs, nested lists, tables, code)."""
import re, sys

UNI = {"ν": r"$\nu$", "Δ": r"$\Delta$", "≥": r"$\geq$", "≤": r"$\leq$", "−": r"$-$", "×": r"$\times$",
       "±": r"$\pm$", "µ": r"$\mu$", "μ": r"$\mu$", "→": r"$\rightarrow$", "≈": r"$\approx$", "…": r"\ldots{}",
       "σ": r"$\sigma$", "²": r"$^2$", "–": "--", "—": "---", "'": "'", "≠": r"$\neq$", "∈": r"$\in$", "·": r"$\cdot$",
       "∪": r"$\cup$", "≲": r"$\lesssim$", "≳": r"$\gtrsim$", "~": r"\textasciitilde{}"}

def esc(s):
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("^", r"\^{}")):
        s = s.replace(a, b)
    s = s.replace(r"\textbackslash\{\}", r"\textbackslash{}")
    return "".join(UNI.get(c, c) for c in s)

def inline(s):
    codes = []
    def keep(m):
        c = esc(m.group(1)).replace("/", r"/\allowbreak{}").replace(r"\_", r"\_\allowbreak{}")
        codes.append(r"\texttt{" + c + "}"); return "\x00%d\x00" % (len(codes) - 1)
    t = re.sub(r"`([^`]*)`", keep, s)
    t = re.sub(r'"([^"]+)"', r"``\1''", t)
    t = esc(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", t)
    return re.sub("\x00(\\d+)\x00", lambda m: codes[int(m.group(1))], t)

def table(rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    head, body = cells[0], cells[2:]
    n = len(head)
    size = r"\small" if n <= 4 else (r"\footnotesize" if n <= 7 else r"\scriptsize")
    L = [r"\begin{center}" + size, r"\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.15}",
         r"\begin{tabularx}{\linewidth}{" + "L" * n + "}", r"\toprule",
         " & ".join(r"\textbf{" + inline(h) + "}" for h in head) + r" \\", r"\midrule"]
    for r in body:
        r = (r + [""] * n)[:n]
        L.append(" & ".join(inline(c) for c in r) + r" \\")
    L += [r"\bottomrule", r"\end{tabularx}", r"\end{center}"]
    return "\n".join(L)

def convert(md):
    lines, out, i = md.split("\n"), [], 0
    stack = []          # list of (indent, env)
    title = None
    def close_to(indent):
        while stack and stack[-1][0] >= indent:
            out.append(r"\end{" + stack.pop()[1] + "}")
    para = []
    def flush():
        if para:
            out.append(inline(" ".join(para))); out.append(""); para.clear()
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            flush(); close_to(0); j = i + 1; code = []
            while j < len(lines) and not lines[j].startswith("```"):
                code.append(lines[j]); j += 1
            out.append(r"\begin{small}\begin{verbatim}" + "\n" + "\n".join(code) + "\n" + r"\end{verbatim}\end{small}")
            i = j + 1; continue
        if ln.lstrip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            flush(); close_to(len(ln) - len(ln.lstrip()) + 1 if stack else 0)
            j = i; rows = []
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                rows.append(lines[j].strip()); j += 1
            out.append(table(rows)); i = j; continue
        m = re.match(r"^!\[(.*)\]\((.+?)\)\s*$", ln)
        if m:
            flush(); close_to(0)
            out.append(r"\begin{figure}[htbp]\centering\includegraphics[width=0.92\linewidth,height=0.42\textheight,keepaspectratio]{"
                       + m.group(2) + r"}\caption{" + inline(m.group(1)) + r"}\end{figure}")
            i += 1; continue
        m = re.match(r"^(#{1,4})\s+(.*)$", ln)
        if m:
            flush(); close_to(0)
            lvl, txt = len(m.group(1)), m.group(2)
            if lvl == 1 and title is None:
                title = txt
            else:
                cmd = {1: "section", 2: "section", 3: "subsection", 4: "subsubsection"}[lvl]
                txt = re.sub(r"^(\d+(\.\d+)*|[A-Z](\.\d+)?)\.?\s+", "", txt) if not txt.startswith("Appendix") else txt
                if txt.startswith("Appendix"):
                    out.append(r"\appendix")
                    out.append(r"\section{" + inline(re.sub(r"^Appendix\s+[A-Z]\.?\s*", "", txt)) + "}")
                else:
                    out.append("\\" + cmd + ("*" if txt == "Summary" else "") + "{" + inline(txt) + "}")
            i += 1; continue
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", ln)
        if m:
            flush()
            ind, mark, txt = len(m.group(1)), m.group(2), m.group(3)
            env = "enumerate" if mark[0].isdigit() else "itemize"
            while stack and (stack[-1][0] > ind or (stack[-1][0] == ind and stack[-1][1] != env)):
                out.append(r"\end{" + stack.pop()[1] + "}")
            if not stack or stack[-1][0] < ind:
                out.append(r"\begin{" + env + "}"); stack.append((ind, env))
            # continuation lines
            j = i + 1
            while j < len(lines) and lines[j].strip() and not re.match(r"^\s*([-*]|\d+\.)\s+", lines[j]) \
                    and not lines[j].lstrip().startswith("|") and not lines[j].startswith("#") \
                    and not lines[j].lstrip().startswith("```"):
                txt += " " + lines[j].strip(); j += 1
            out.append(r"\item " + inline(txt)); i = j; continue
        if not ln.strip():
            flush()
            # keep list open if next non-empty line is indented list/table content
            k = i + 1
            while k < len(lines) and not lines[k].strip():
                k += 1
            if k < len(lines) and stack and (lines[k].startswith(" ") or re.match(r"^\s*([-*]|\d+\.)\s+", lines[k])):
                i += 1; continue
            close_to(0); i += 1; continue
        if stack and ln.startswith(" "):
            out.append(inline(ln.strip())); i += 1; continue
        close_to(0); para.append(ln.strip()); i += 1
    flush(); close_to(0)
    return title, "\n".join(out)

md = open(sys.argv[1]).read()
title, body = convert(md)
pre = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs,array,tabularx}
\usepackage{xcolor}
\usepackage{xurl}
\usepackage[hidelinks]{hyperref}
\usepackage{enumitem}
\usepackage{graphicx,float,caption}
\captionsetup{font=small,labelfont=bf}
\setlist{itemsep=1pt,topsep=3pt}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\setlength{\parindent}{0pt}\setlength{\parskip}{4pt}
\title{\textbf{""" + inline(title) + r"""}}
\author{}
\date{}
\begin{document}
\maketitle
\vspace{-3em}
"""
open(sys.argv[2], "w").write(pre + body + "\n\\end{document}\n")
