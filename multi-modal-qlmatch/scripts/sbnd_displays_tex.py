#!/usr/bin/env python
"""Beamer frames (one event display per slide) for the SBND real-data campaign deck (#14).

Usage: sbnd_displays_tex.py DISPLAY_DIR TEX_DIR
Reads DISPLAY_DIR/displays.json and {sim,on,off}_NN.png from scripts/sbnd_event_displays.py, writes the pages as JPEG
(quality 82, to keep the PDF small) to TEX_DIR/displays/ and the frames to TEX_DIR/displays.tex.
"""
import json
import os
import sys

from PIL import Image

src, tex = sys.argv[1], sys.argv[2]
d = json.load(open(os.path.join(src, "displays.json")))
os.makedirs(os.path.join(tex, "displays"), exist_ok=True)
HEAD = {"sim": "Simulation", "on": "Beam-on data", "off": "Off-beam data"}
lines = []
for kind in ("sim", "on", "off"):
    pages = sorted(k for k in d["pages"] if k.startswith(kind + "_"))
    lines.append(f"\\subsection{{{HEAD[kind]}}}")
    for i, k in enumerate(pages):
        p = d["pages"][k]
        Image.open(os.path.join(src, k + ".png")).convert("RGB").save(
            os.path.join(tex, "displays", k + ".jpg"), quality=82, optimize=True)
        pnu = ", ".join(f"{n} {m['p_nu']:.2f}" for n, m in p["models"].items())
        # full-slide page: the figure carries its own title (sample, event, run/subrun/event) and the model summary
        lines.append(f"% {HEAD[kind]} {i + 1}/{len(pages)}: run {p['run']}, subrun {p['subrun']}, event {p['event']}; "
                     f"event p(nu): {pnu}\n"
                     f"\\begin{{frame}}[plain]\n  \\centering\\vspace*{{-2pt}}\n"
                     f"  \\makebox[\\linewidth][c]{{\\includegraphics[width=\\paperwidth,height=\\paperheight,keepaspectratio]{{displays/{k}.jpg}}}}\n"
                     f"\\end{{frame}}")
open(os.path.join(tex, "displays.tex"), "w").write("\n".join(lines) + "\n")
print(f"wrote {sum(l.count('[plain]') for l in lines)} frames")
