"""Builds FaceAI_v2.ipynb from the cell list in scripts/v2_notebook_text.py (the v2 improvements, step by step, in
the same What / Why / Code / What you should see style as FaceAI_pipeline.ipynb). The notebook only READS saved
results, so Run All takes about a minute; the heavy steps are the scripts listed in its Step 0.

    .venv/Scripts/python.exe scripts/build_v2_notebook.py
    .venv/Scripts/jupyter-execute.exe --inplace FaceAI_v2.ipynb          (runs it and stores outputs and figures)
"""
import sys
from pathlib import Path

import nbformat as nbf

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))


def build():
    from v2_notebook_text import CELLS
    cells = [nbf.v4.new_markdown_cell(src.strip("\n")) if kind == "md" else nbf.v4.new_code_cell(src.strip("\n"))
             for kind, src in CELLS]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {"kernelspec": {"display_name": "Python 3 (.venv)", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}}
    return nb


if __name__ == "__main__":
    out = PROJECT / "FaceAI_v2.ipynb"
    nbf.write(build(), out)
    print("written", out)
