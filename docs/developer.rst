Developer tools
===============

Developer install::

  make install

Install with PyQt5::

  make install-pyqt

.. warning::

  Pip installation of ETE's PyQt5 dependency has been found to fail (with an error like `this <https://stackoverflow.com/questions/70961915/error-while-installing-pytq5-with-pip-preparing-metadata-pyproject-toml-did-n)>`_) on ARM Mac.
  You can instead install PyQt5 with Conda::

    conda install pyqt=5
  

Run tests::

  make test

Test notebooks::

  make notebooks

Format code::

  make format

Lint::

  make lint

Build docs locally (you can then see the generated documentation in ``docs/_build/html/index.html``)::

  make docs

Docs are automatically deployed to github pages via a workflow on push to the main branch.
