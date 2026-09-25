#!/usr/bin/env python3
"""Regenerate manuscript.txt (plain text, numbers filled in) from main.tex.
Needs pandoc and a compiled main.aux (for citation numbers)."""

import os
import re
import subprocess

root = os.path.dirname(os.path.abspath(__file__))
tex = open(os.path.join(root, "main.tex")).read()
macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", open(os.path.join(root, "numbers.tex")).read()))
for k in sorted(macros, key=len, reverse=True):
    tex = re.sub(r"\\" + k + r"(?![A-Za-z])(\{\})?", lambda m: macros[k].replace("\\", "\\\\"), tex)
cites = dict(re.findall(r"\\bibcite\{([^}]*)\}\{(\d+)\}", open(os.path.join(root, "main.aux")).read()))
tex = re.sub(r"~?\\cite\{([^}]*)\}",
             lambda m: " [" + ", ".join(cites[c.strip()] for c in m.group(1).split(",")) + "]", tex)
labels = dict(re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", open(os.path.join(root, "main.aux")).read()))
tex = re.sub(r"\\(?:eq)?ref\{([^}]*)\}", lambda m: labels.get(m.group(1), "?"), tex)
tex = re.sub(r"\\IEEEPARstart\{(\w)\}\{(\w+)\}", r"\1\2", tex)
tex = re.sub(r"\\begin\{IEEEproof\}", r"\\emph{Proof.} ", tex)
tex = tex.replace("\\end{IEEEproof}", "")

n = {"c": 0}


def fig(m):
    n["c"] += 1
    cap = re.search(r"\\caption\{(.*)\}\s*\\label", m.group(0), re.S).group(1)
    f = re.search(r"\\includegraphics\[[^]]*\]\{([^}]*)\}", m.group(0)).group(1)
    return f"\n\n[FIGURE -- file: figures/{f}]\n\nCaption: {cap}\n\n"


tex = re.sub(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", fig, tex, flags=re.S)
rows = [r for r in open(os.path.join(root, "table_csl.tex")).read().splitlines() if "&" in r]
rows = [re.sub(r"\\\\.*", "", r).replace("$", "").replace("\\ ", " ").replace("\\Delta", "Delta")
        .replace("^*", "*").replace("&", "|") for r in rows]
tex = re.sub(r"\\begin\{table\}.*?\\end\{table\}",
             lambda m: "\n\n[TABLE I] CSL horizons W*(r) and per-class onsets.\n\n"
             + "\n\n".join(rows) + "\n\n", tex, flags=re.S)
title = re.search(r"\\title\{(.*?)\}\n", tex, re.S).group(1).replace("\n", " ")
abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1)
keywords = re.search(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", tex, re.S).group(1)
body = tex[tex.index("\\section{Introduction}"):tex.index("\\bibliographystyle")]
preamble = (r"\newcommand{\R}{\mathbb{R}}\newcommand{\E}{\mathbb{E}}\newcommand{\one}{\mathbf{1}}"
            r"\newcommand{\Hk}{\mathcal{H}_k}\newcommand{\tfrac}[2]{#1/#2}"
            r"\newcommand{\rank}{\operatorname{rank}}")
doc = ("\\documentclass{article}" + preamble + "\\begin{document}\n\\section*{" + title + "}\n"
       "Olivia Kim and Aryan Padarthi\n\n\\section*{Abstract}" + abstract
       + "\n\n\\textbf{Index Terms:} " + keywords + "\n" + body + "\\end{document}")
txt = subprocess.run(["pandoc", "-f", "latex", "-t", "plain", "--columns=78"],
                     input=doc, capture_output=True, text=True).stdout
bbl = open(os.path.join(root, "main.bbl")).read()
refs = []
for i, item in enumerate(re.findall(r"\\bibitem\{[^}]*\}(.*?)(?=\\bibitem|\\end\{thebibliography\})",
                                    bbl, re.S), 1):
    item = re.sub(r"1em plus 0\.5em minus 0\.4em|\\BIBentry\w*\{[^}]*\}|\\newblock|\\emph|\\/", " ", item)
    item = re.sub(r"[{}]", "", item).replace("~", " ").replace("``", '"').replace("''", '"')
    item = item.replace("--", "-").replace("$", "")
    item = re.sub(r"\\['\"`^~=.u]\s*", "", item)
    item = re.sub(r"\\\w+", "", item)
    refs.append(f"[{i}] " + re.sub(r"\s+", " ", item).strip())
with open(os.path.join(root, "manuscript.txt"), "w") as fh:
    fh.write("[Plain-text copy generated from main.tex by make_txt.py; all numbers are\n"
             " filled in. Math that has no plain-text form is left as LaTeX.]\n\n"
             + txt.strip() + "\n\nREFERENCES\n\n" + "\n".join(refs) + "\n")
print(f"wrote manuscript.txt ({len(txt.splitlines())} lines, {len(refs)} references)")
