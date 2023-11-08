# BaselineGNNs




The --max_iters indicate the number of edges to be deleted or added.

The datasets considered here are available as Pytorch-Geometric datasets - https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html
For node classification, the popular datasets are - Cora, Citeseer, Pubmed (Homophilic datasets), Cornell, Wisconsin, Texas, Chameleon and Squirrel (Heterophilic datasets).

Run the code using -- 

```python
python DirectPlanet.py --dataset 'Cora' --out 'DirectGapCSV/ProxyDeletions/CoraProxyGapDel.csv' --max_iters 500
```

The above heterophilic datasets are small and the performance on these datasets have high variance. Larger heterophilic datasets have been introduced here - https://github.com/yandex-research/heterophilous-graphs/tree/main


