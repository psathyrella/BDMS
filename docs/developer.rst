Developer tools
===============

We use `hatch <https://hatch.pypa.io>`_ to manage install, CI/testing, docs build, and versioning.

Create the dev environment::

  hatch env create

If you want to activate the dev environment on the terminal,
run the following command (but it's not if for running the subsequence hatch commands)::

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
