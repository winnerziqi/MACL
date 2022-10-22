# Model-Aware-Contrastive-Loss
This repository provides a PyTorch version implementation of MACL loss proposed in ***Model-Aware Contrastive Learning: Towards Escaping Uniformity-Tolerance Dilemma in Training***

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

# beta is the temperature multiplication factor
l = MACLb(tau_init=0.1, beta=1.0)
loss = l(f1,f2)+l(f2,f1)
```


