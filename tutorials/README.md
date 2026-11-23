# Tutorials

| tutorial | what you learn | runtime |
|---|---|---|
| [Selection in head and neck cancer](selection_in_hnsc.md) | a public TCGA cohort to rates, selection coefficients, the selection call and two research projects | about 80 minutes; 16-20 GB of memory at one step |

Each tutorial is a Python script in the
[jupytext](https://jupytext.readthedocs.io) "percent" format: run it
as `python selection_in_hnsc.py`, open it as a notebook in Jupyter
(with `jupytext` installed) or step through its `# %%` cells in VS
Code. The `.md` page is the same file run once, with its outputs and
figures; `scripts/render_tutorial.sh` regenerates it.

The covariates come from
[sigmutselcovs](https://github.com/alfaromurillo/sigmutselcovs), whose
own tutorial covers the covariate matrices and their use in dNdScv.
