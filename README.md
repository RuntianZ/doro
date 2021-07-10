# DORO: Distributional and Outlier Robust Optimization
**Runtian Zhai\*, Chen Dan\*, J. Zico Kolter, Pradeep Ravikumar**  
In ICML 2021  
Paper: [Link](http://proceedings.mlr.press/v139/zhai21a/zhai21a.pdf)

## Table of Contents
- [Quick Start](#quick-start)
- [Introduction](#introduction)
- [DRO is Sensitive to Outliers](#dro-is-sensitive-to-outliers)
- [DORO](#doro)
  - [CelebA](#celeba)
  - [CivilComments-Wilds](#civilcomments-wilds)
- [Citation and Contact](#citation-and-contact)

## Quick Start
For a demonstration on the sensitivity of DRO to outliers, see [this Jupyter notebook](https://drive.google.com/file/d/1z-ugawAr-2rFYPavMksHohEVN_scZwSR/view?usp=sharing) (you can view it online with Google Colab).

To install the required packages, use
```shell
pip install -r requirements.txt
```
To run experiments on the CivilComments-Wilds dataset, you need to manually install `torch-scatter` and `torch-geometric` (see instructions [here](#civilcomments-wilds)). 

The algorithms we implement are included in `dro.py`. To run these algorithms on CelebA, use
```shell
python celeba.py --data_root [ROOT] --alg [ALG] --alpha [ALPHA] --eps [EPSILON] --seed [SEED] --download
```
Here `[ROOT]` is the path to the dataset. `[ALG]` is the algorithm (`erm`, `cvar`, `cvar_doro`, `chisq` or `chisq_doro`). `[ALPHA]` and `[EPSILON]` are the hyperparameters described in the paper. `[SEED]` is the random seed.


## Introduction
While DRO has been proved to be effective against subpopulation shift, its performance is significantly downgraded by the outliers existing in the dataset. DORO enhances the outlier-robustness of DRO by filtering out a small fraction of instances with high training loss that are potentially outliers. First we show that DRO is sensitive to outliers with some intriguing experimental results on COMPAS. Then we conduct large-scale experiments on COMPAS, CelebA and CivilComments-Wilds. Our strong theoretical and empirical results demonstrate the effectiveness of DORO.


## DRO is Sensitive to Outliers
In Section 3 of our paper, we use experimental results on the COMPAS dataset to demonstrate that the original DRO algorithms are not robust to outliers that widely exist in real datasets. We have prepared a [Jupyter notebook](https://drive.google.com/file/d/1z-ugawAr-2rFYPavMksHohEVN_scZwSR/view?usp=sharing) that includes all experiments in this section, which you can view online with Google Colab.


## DORO

In Section 6, we conduct large-experiments on modern datasets. Here we describe how to run the experiments on CelebA and CivilComments-Wilds.

### CelebA
CelebA official website: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

We run the experiments on one NVIDIA GTX 1080ti. To reproduce the results in the paper, please use the following command with the hyperparameters listed in Appendix B.3 of our paper:
```shell
python celeba.py --data_root [ROOT] --alg [ALG] --alpha [ALPHA] --eps [EPSILON] --seed [SEED]
```
Please use `--download` to download the dataset if you are running for the first time.

### CivilComments-Wilds
We use the CivilComments dataset from the `wilds` package. Please follow the instructions on https://wilds.stanford.edu/get_started/ to use this dataset. Our codes are included in the `wilds-exp` folder, which is based on https://github.com/p-lambda/wilds/tree/main/examples.

We run the experiments on four NVIDIA Tesla V100s. To reproduce the results in the paper, please use the following command with the hyperparameters listed in Appendix B.3 of our paper:
```shell
cd wilds-exp
python run_expt.py --dataset civilcomments --algorithm doro --root_dir [ROOT] --doro_alg [ALG] --alpha [ALPHA] --eps [EPSILON] --batch_size 128 --data_parallel --evaluate_steps 500 --seed [SEED]
```
Please use `--download` to download the dataset if you are running for the first time.

## Citation and Contact
Please use the following BibTex entry to cite this paper:
```

@InProceedings{pmlr-v139-zhai21a,
  title = 	 {DORO: Distributional and Outlier Robust Optimization},
  author =       {Zhai, Runtian and Dan, Chen and Kolter, Zico and Ravikumar, Pradeep},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {12345--12355},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/zhai21a/zhai21a.pdf},
  url = 	 {http://proceedings.mlr.press/v139/zhai21a.html}
}

```

To contact us, please email to the following address:
`Runtian Zhai <rzhai@cmu.edu>`
