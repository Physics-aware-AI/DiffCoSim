---

<div align="center">    
 
# Extending Lagrangian and Hamiltonian Neural Networks with Differentiable Contact Models

Yaofeng Desmond Zhong, Biswadip Dey, Amit Chakraborty | 2021

[![Conference](http://img.shields.io/badge/NeurIPS-2021-4b44ce.svg)](https://arxiv.org/abs/2102.06794)
[![Paper](http://img.shields.io/badge/arXiv-2102.06794-B31B1B.svg)](https://arxiv.org/abs/2102.06794)


</div>

This repository is the official implementation of [Extending Lagrangian and Hamiltonian Neural Networks with Differentiable Contact Models](https://arxiv.org/abs/2102.06794). 

__WIP__: We are preparing the release of dataset, pre-trained models and analysis notebooks. In the meantime, you can run the following commands to run the dynamics and parameter tasks in the paper. The commands will generate the dataset first and then start training. (Note that it might take a while to generate the dataset.)

```python
# train CM-CD-CLNN models
# BP5-e
python trainer.py --body-class BouncingPointMasses --body-kwargs-file _BP5-e 
# BP5
python trainer.py --body-class BouncingPointMasses --body-kwargs-file _BP5 
# BD5
python trainer.py --body-class BouncingDisks --body-kwargs-file _BD5 
# CP3-e
python trainer.py --body-class ChainPendulumWithContact --body-kwargs-file _CP3-e 
# CP3
python trainer.py --body-class ChainPendulumWithContact --body-kwargs-file _CP3
# Rope 
python trainer.py --body-class Rope --body-kwargs-file _Rope 
# Gyro-e
python trainer.py --body-class GyroscopeWithWall --body-kwargs-file _Gyro-e 
# Gyro
python trainer.py --body-class GyroscopeWithWall --body-kwargs-file _Gyro 
```
