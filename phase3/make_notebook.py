"""Convert the percent-format script F26-11.py into the notebook F26-11.ipynb."""
import re
from pathlib import Path

import nbformat

src = Path(__file__).with_name("F26-11.py").read_text(encoding="utf-8")
cells = []
for block in re.split(r"^# %%", src, flags=re.M)[1:]:
    header, _, body = block.partition("\n")
    body = body.strip("\n")
    if header.strip() == "[markdown]":
        text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
        cells.append(nbformat.v4.new_markdown_cell(text))
    else:
        cells.append(nbformat.v4.new_code_cell(body))
nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbformat.write(nb, Path(__file__).with_name("F26-11.ipynb"))
print(len(cells), "cells")
