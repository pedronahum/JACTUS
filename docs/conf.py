"""Sphinx configuration for jactus documentation."""

import os
import sys
from datetime import datetime

# Add source directory to path
sys.path.insert(0, os.path.abspath("../src"))

# Project information
project = "jactus"
author = "Pedro N. Rodriguez"
copyright = f"{datetime.now().year}, {author}"
version = "0.1.0"
release = "0.1.0"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "myst_parser",
]

# Template paths
templates_path = ["_templates"]

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Master document
master_doc = "index"

# Language
language = "en"

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Pygments style for syntax highlighting
pygments_style = "sphinx"

# HTML output options
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
}

html_static_path = ["_static"]
html_title = f"{project} {version} documentation"
html_short_title = project
html_logo = None
html_favicon = None
html_baseurl = "https://pedronahum.github.io/JACTUS/"

# Autodoc options
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

autodoc_typehints = "description"
autodoc_type_aliases = {}

# Napoleon options (for Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "flax": ("https://flax.readthedocs.io/en/latest/", None),
}

# MathJax configuration
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

# Todo extension options
todo_include_todos = True
