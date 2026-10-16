# KEGD-NP

A dual-branch molecular classification framework integrating molecular graph representations with 208 RDKit molecular descriptors for natural product biosource prediction.

<div align="center">
  <img src="./figure/workflow.jpg" alt="Overview of the KEGD-NP framework" width="750" />
</div>

## Overview

KEGD-NP integrates learned molecular graph representations with global molecular descriptors to capture complementary structural information for biosource prediction.

The framework consists of three main components:

- **KAN-enhanced multihead graph attention encoder:** learns molecular graph representations.
- **Gated descriptor encoder:** processes 208 RDKit molecular descriptors.
- **Adaptive gated fusion module:** integrates graph representations and descriptor features for binary classification.

Each biosource task is modeled independently as a binary classification problem. This repository uses **botanical-source prediction** as the primary example for training, inference, and SHAP analysis.

## Dataset

The data used in this study were obtained from **NPBS Atlas**, a public resource containing natural products and their biological source annotations.

- **Database:** [NPBS Atlas](https://biochemai.cstspace.cn/npbs/)
- **Database publication:** [NPBS Atlas: a comprehensive data resource for exploring the biological sources of natural products](https://doi.org/10.1186/s13321-025-01116-y)




## Installation

### Python

Python **3.11.0** was used in the tested environment.

Download Python from the [official Python website](https://www.python.org/).

### Dependencies

Install the base dependencies using:

```bash
python -m pip install -r requirements.txt
```



## Acknowledgements

We acknowledge NPBS Atlas for providing the natural product biological-source data used in this study, and the developers of RDKit, PyTorch, and SHAP for their open-source tools.


