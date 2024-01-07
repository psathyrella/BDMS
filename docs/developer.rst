Developer tools
===============

Developer install::

  We use `hatch <https://hatch.pypa.io>`_ to manage install, CI/testing, docs build, and versioning.
  You'll need to install hatch.

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
