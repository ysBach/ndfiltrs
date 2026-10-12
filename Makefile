DOCS_PYTHON ?= $(CURDIR)/.venv-docs/bin/python
QUARTO ?= quarto

.PHONY: docs-api docs-render docs-build docs-preview

docs-api:
	"$(DOCS_PYTHON)" docs/quarto/_scripts/stage_guides.py
	cd docs/quarto && "$(DOCS_PYTHON)" -m quartodoc build
	"$(DOCS_PYTHON)" docs/quarto/_scripts/stage_guides.py --navigation-only

docs-render: docs-api
	$(QUARTO) render docs/quarto

docs-build: docs-render

docs-preview: docs-api
	$(QUARTO) preview docs/quarto
