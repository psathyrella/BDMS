Developer tools
===============

We use `hatch <https://hatch.pypa.io>`_ to manage install, CI/testing, docs build, and versioning.

Create and enter the dev environment::

  hatch env create
  hatch shell

Run tests::

  hatch run tests

Lint checker::

  hatch run lint

Format code::

  hatch run format

Build docs locally (you can then see the generated documentation in ``docs/_build/html/index.html``)::

  hatch run docs

.. note::

  To render inheritance diagrams in the docs, you'll need to install `Graphviz <https://graphviz.org>`_.
  We use the Conda package::

    conda install -c conda-forge graphviz

Docs are automatically deployed to github pages via a workflow on push to the main branch.
