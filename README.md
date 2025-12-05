# A Variational Manifold Embedding Framework for Nonlinear Dimensionality Reduction (NeurReps 2025)

<div align="center"><a href="https://openreview.net/forum?id=v1vLWn256x">OpenReview</a></div>
<br/>

This repo contains code that reproduces the figure from "**A Variational Manifold Embedding Framework for Nonlinear Dimensionality Reduction**", a paper accepted to [NeurReps](https://www.neurreps.org) 2025. 

**Abstract:**
> Dimensionality reduction algorithms like principal component analysis (PCA) are workhorses of machine learning and neuroscience, but each has well-known limitations. Variants of PCA are simple and interpretable, but not flexible enough to capture nonlinear data manifold structure. More flexible approaches have other problems: autoencoders are generally difficult to interpret, and graph-embedding-based methods can produce pathological distortions in manifold geometry. Motivated by these shortcomings, we propose a variational framework that casts dimensionality reduction algorithms as solutions to an optimal manifold embedding problem. By construction, this framework permits nonlinear embeddings, allowing its solutions to be more flexible than PCA. Moreover, the variational nature of the framework has useful consequences for interpretability: each solution satisfies a set of partial differential equations, and can be shown to reflect symmetries of the embedding objective. We discuss these features in detail and show that solutions can be analytically characterized in some cases. Interestingly, one special case exactly recovers PCA.

Only standard libraries (NumPy, Matplotlib, and PyTorch) are used. There is one Jupyter notebook, `fig-embed-examples.ipynb`, which contains code for reproducing Figure 1. A file containing functions used by the notebook, `embedding_functions.py`, is in the `functions/` folder. Intended notebook outputs are in the `results/` folder.

## Citation

```bibtex
@inproceedings{
vastola2025_embed,
title={A Variational Manifold Embedding Framework for Nonlinear Dimensionality Reduction},
author={John J. Vastola and Samuel J. Gershman and Kanaka Rajan},
booktitle={NeurIPS 2025 Workshop on Symmetry and Geometry in Neural Representations},
year={2025},
url={https://openreview.net/forum?id=v1vLWn256x}
}
```
