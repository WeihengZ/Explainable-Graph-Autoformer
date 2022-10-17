# Explainable-Graph-Autoformer

This repository is the code of the paper: [Explainable Graph Pyramid Autoformer (X-GPA) for Long-Term Traffic Forecasting](https://arxiv.org/abs/2209.13123).

## Getting start
```
pip install -r requirements.txt
```

## data

We use dataset of PEMS-bay and Metr-LA traffic dataset for validation. Please put the data (h5 file) in the folder "data".

## implement

For reproducing the results of short-term prediction:
```
python main_short_term.py
```

For reproducing the results of long-term prediction:
```
python main_long_term.py
```

If you make advantage of the X-GPA in your research, please consider citing our paper in your manuscript:
```
@misc{https://doi.org/10.48550/arxiv.2209.13123,
  doi = {10.48550/ARXIV.2209.13123},
  url = {https://arxiv.org/abs/2209.13123},
  author = {Zhong, Weiheng and Mallick, Tanwi and Meidani, Hadi and Macfarlane, Jane and Balaprakash, Prasanna},
  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Explainable Graph Pyramid Autoformer for Long-Term Traffic Forecasting},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
