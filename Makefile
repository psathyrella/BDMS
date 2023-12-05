default:

install:
	pip install -e .[dev]

install-pyqt:
	pip install -e .[dev,pyqt]

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
