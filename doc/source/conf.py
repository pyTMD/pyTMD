# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os

# import sys
import logging
import datetime
import warnings
import subprocess

# sys.path.insert(0, os.path.abspath('.'))
import importlib.metadata
import importlib.util


# -- Project information -----------------------------------------------------
on_rtd = os.environ.get("READTHEDOCS") == "True"
on_github = os.environ.get("GITHUB_ACTIONS") == "true"

# package metadata
metadata = importlib.metadata.metadata("pyTMD")
project = metadata["Name"]
year = datetime.date.today().year
copyright = f"2017\u2013{year}, Tyler C. Sutterley"
author = "Tyler C. Sutterley"

# software version
version = metadata["version"]
# append "v" before the version
release = f"v{version}"


# filter out numfig warnings when building documentation, see
# https://github.com/sphinx-doc/sphinx/issues/10316
# https://github.com/sphinx-doc/sphinx/pull/14446
class numfig_filter(logging.Filter):
    def filter(self, record):
        warning_type = getattr(record, "type", "")
        warning_subtype = getattr(record, "subtype", "")
        suppress_warning = (
            f"{warning_type}.{warning_subtype}" == "html.numfig_format"
            or record.getMessage().startswith("numfig_format")
        )
        return not suppress_warning


# add filter to logger
logging.getLogger("sphinx").addFilter(numfig_filter())

# suppress warnings in examples and documentation
if on_rtd:
    warnings.filterwarnings("ignore")

# create tables
for module_name in ["model_table", "constituent_table"]:
    spec = importlib.util.spec_from_file_location(
        module_name, f"{module_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

# download a tide model for rendering documentation
if on_rtd or on_github:
    subprocess.run(["fetch_gsfc_got.py", "--tide", "GOT4.10"])

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "matplotlib.sphinxext.plot_directive",
    "myst_nb",
    "numpydoc",
    "sphinxcontrib.bibtex",
    "sphinx.ext.autodoc",
    "sphinx.ext.graphviz",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "sphinxarg.ext",
]

# use myst for notebooks
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
}
# execute notebooks on build
if on_rtd:
    nb_execution_mode = "auto"
    nb_execution_excludepatterns = [
        "Plot-ATLAS-Compact.ipynb",
    ]
    nb_output_stderr = "remove-warn"
else:
    nb_execution_mode = "off"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_assets/paper.md",
    "**Untitled*.ipynb",
    "**.ipynb_checkpoints",
]

# location of master document (by default sphinx looks for contents.rst)
master_doc = "index"

# -- Configuration options ---------------------------------------------------
autosummary_generate = True
autodoc_member_order = "bysource"
numpydoc_show_class_members = False
pygments_style = "native"
bibtex_bibfiles = ["_assets/pytmd-refs.bib"]
bibtex_default_style = "plain"
plot_formats = ["png"]
plot_html_show_formats = False
plot_html_show_source_link = False
numfig = True
numfig_secnum_depth = 1

# -- Options for HTML output -------------------------------------------------

# html_title = metadata["Name"]
html_short_title = metadata["Name"]
html_show_sourcelink = False
html_show_sphinx = True
html_show_copyright = True

numfig_format = {
    "code-block": None,
    "figure": "Figure %s:",
    "table": "Table %s:",
}

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_logo = "_assets/pyTMD_logo.png"
html_static_path = ["_static"]
# fetch the project urls
project_urls = {}
for project_url in metadata.get_all("Project-URL"):
    name, _, url = project_url.partition(", ")
    project_urls[name.lower()] = url
# fetch the repository url
github_url = project_urls.get("repository")
*_, github_user, github_repo = github_url.split("/")
# add html context
html_context = {
    "display_github": True,
    "github_user": github_user,
    "github_repo": github_repo,
    "github_version": "main",
    "conf_py_path": "/doc/source/",
    "menu_links": [
        (
            '<i class="fa fa-github fa-fw"></i> Source Code',
            github_url,
        ),
        (
            '<i class="fa fa-book fa-fw"></i> License',
            f"{github_url}/blob/main/LICENSE",
        ),
        (
            '<i class="fa fa-comment fa-fw"></i> Discussions',
            f"{github_url}/discussions",
        ),
    ],
}


# Load the custom CSS files (needs sphinx >= 1.6 for this to work)
def setup(app):
    app.add_css_file("style.css")
