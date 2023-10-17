default:

install:
	pip install -e .[dev]

install-ete:
	pip install -e .[dev,ete]

test:
	pytest
	pytest --nbval notebooks/*

format:
	black .
	docformatter --black --in-place **/*.py

lint:
	flake8 .

docs:
	make -C docs html

.PHONY: install test format lint docs
