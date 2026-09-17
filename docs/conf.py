# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re
import shutil
from pathlib import Path

_DOCS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_DOCS)
_TEMP = Path(_DOCS) / "src" / "_temp"

# -- Project information -----------------------------------------------------

project = "Hermit"
copyright = "2026, LSDOlab"
author = "LSDOlab"

try:
    from importlib.metadata import version as _v

    release = _v("hermit")
except Exception:
    release = "0.9.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx_rtd_theme",
    "autoapi.extension",
    "numpydoc",
    "sphinx_copybutton",
    "myst_nb",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
]

bibtex_bibfiles = ["src/references.bib"]

myst_title_to_header = True
myst_enable_extensions = ["dollarmath", "amsmath", "tasklist"]
nb_execution_mode = "off"   # notebooks need FEniCSx; ship pre-run outputs

# -- autoapi ---------------------------------------------------------------

autoapi_dirs = ["../hermit"]
autoapi_root = "src/autoapi"
autoapi_type = "python"
autoapi_file_patterns = ["*.py", "*.pyi"]
autoapi_options = [
    "members", "undoc-members", "show-inheritance",
    "show-module-summary", "imported-members",
]
autoapi_ignore = ["*/_ufl_compat.py"]
autoapi_add_toctree_entry = False
autoapi_member_order = "groupwise"
autoapi_python_class_content = "both"

root_doc = "index"
templates_path = ["_templates"]

# autoapi.python_import_resolution: autoapi noise about the package re-exporting itself.
suppress_warnings = ["autoapi.python_import_resolution"]
exclude_patterns = ["README.md", "_build", "Thumbs.db", ".DS_Store", "src/welcome.md"]

# -- HTML output ---------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "style_nav_header_background": "#2980B9",
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": True,
}

# -- Stage examples/tutorials, and turn each example script into a page --------


def _py2md(path):
    """Turn each examples/**/ex_*.py into a .md page: module docstring (first line =
    title, rest = prose) followed by the full source in a python code block."""
    code = path.read_text(encoding="utf-8")
    m = re.match(r'\s*(?:"""|\'\'\')(.*?)(?:"""|\'\'\')', code, re.DOTALL)
    if not m:
        raise SyntaxError(f"{path}: a module docstring (title line) is required")
    lines = m.group(1).strip().splitlines()
    title, body = lines[0].strip(), "\n".join(lines[1:]).strip()
    path.with_suffix(".md").write_text(
        f"# {title}\n\n{body}\n\n```python\n{code}\n```\n", encoding="utf-8"
    )


def _stage_examples_and_tutorials(app, config):
    """Create the ignored, build-local sources required by the documentation."""
    shutil.rmtree(_TEMP, ignore_errors=True)
    for name in ("tutorials", "examples"):
        shutil.copytree(Path(_REPO) / name, _TEMP / name)
    for example in (_TEMP / "examples").glob("**/ex_*.py"):
        _py2md(example)


def setup(app):
    app.connect("config-inited", _stage_examples_and_tutorials)
