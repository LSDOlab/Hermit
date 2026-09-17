# Building the documentation

```sh
pip install -r requirements.txt        # sphinx + extensions (see requirements.txt)
cd docs && make html
```

Open `docs/_build/html/index.html`.

## Layout

- `conf.py` --- Sphinx config. `autoapi_dirs = ["../hermit"]` drives the API
  reference; its standard-library hook copies `examples/` and `tutorials/` into
  `src/_temp/` and turns each `ex_*.py` into a page.
- `src/*.md` --- prose pages (MyST markdown).
- `src/welcome.md` --- the landing page (included by `index.md`).
- `src/references.bib` --- bibliography.

## Read the Docs

The requirements install Hermit's lightweight package metadata and the documentation
extra; autoapi parses the source tree directly, and notebooks are not executed
(`nb_execution_mode = "off"`), so FEniCSx is not required. `.readthedocs.yaml` +
`requirements.txt` at the repo root configure it.
