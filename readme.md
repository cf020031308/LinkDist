# Distilling Knowledge From Links to Classify Graph Nodes Without Passing Messages

*LinkDist* distils knowledge from links into a Multi-Layer Perceptron (MLP) without the
need to aggregate messages.

Experiments with 8 real-world datasets show the Learnt MLP can predict the label of a node without knowing its adjacencies but achieve comparable accuracy against GNNs in the context of semi-supervised node classification.

![](arch.png)

## Reproducibility

Install dependencies [torch](https://pytorch.org/) and [DGL](https://github.com/dmlc/dgl):

```bash
pip3 install torch ogb
```

Run the script with arguments `algorithm_name` and `dataset_name`.

The following command experiments **GCN** on **Cora** dataset with the inductive setting:

```bash
python3 main.py gcn cora
```

* available algorithms:
  * mlp
  * transudctive setting: gcn-trans2mlp / linkdist-nomp-trans / gcn-trans / linkdist-trans
  * inductive setting: gcn2mlp / linkdist-nomp / gcn / linkdist
* available datasets: cora / citeseer / pubmed / corafull / amazon-photo / amazon-com / coauthor-cs / coauthor-phy

## Citation

```bibtex
```
