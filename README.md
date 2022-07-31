# Model-Aware-Contrastive-Loss
This repository provides a PyTorch version implementation of MACL loss proposed in [***Model-Aware Contrastive Learning: Towards Escaping Uniformity-Tolerance Dilemma in Training***](https://arxiv.org/abs/2207.07874)

## Usage

```python
from MACL import MACLa

# alpha is the base value of the temperature exponential scaling factor
l = MACLa(tau_init=0.1, alpha=2.0)
# f1 and f2 are output features
loss = l(f1,f2)+l(f2,f1)
```

```python
from MACL import MACLb

# alpha is the temperature multiplication factor
l = MACLb(tau_init=0.1, beta=1.0)
loss = l(f1,f2)+l(f2,f1)
```

## Citation

```
@misc{https://doi.org/10.48550/arxiv.2207.07874,
  doi = {10.48550/ARXIV.2207.07874},
  author = {Huang, Zizheng and Zhang, Chao and Li, Huaxiong and Wang, Bo and Chen, Chunlin},
  title = {Model-Aware Contrastive Learning: Towards Escaping Uniformity-Tolerance Dilemma in Training},
  year = {2022}
}
```

## Citation
Please feel free to contact us if you have any problems.

Email: zizhenghuang@smail.nju.edu.cn


