## Skew-Normal Distributions for Modeling Asymmetric Moving Tendencies in Pedestrian Trajectories (Under Review)

This is an exemplar repository developed based on [GraphTERN](https://github.com/InhwanBae/GraphTERN/tree/AAAI2023) to show the implementation of `Skew-Normal Module` for Trajectory Prediction. The code can be adapted to other methods that model the output distribution as normal distributions or Gaussian mixture distributions.

File structure
- `SN-GraphTERN`: code that adapts GraphTERN to SN-GraphTERN by replacing the Gaussian mixture distribution with skew-normal mixture distribution.
- `skew_normal_class.py`: define key classes and functions on skew-normal distributions.
- `my_mixture_same_family.py`: modify the `MixtureSameFamily` class from torch.distributions for more numerically stable implementation of the log-likelihood of mixture distributions.
- `generaly.py`: provide instrumental functions.
- `README.md`: provide a brief description of the repository.

#### Environment
- Ubuntu 24.04
- Cuda 11.7
- Python 3.8
- PyTorch 1.13.1

#### Scripts for running the experiments

```bash
cd SN-GraphTERN
# train
# available datasets: eth, hotel, univ, zara1, zara2, sdd
CUDA_VISIBLE_DEVICES=0 python train.py --dataset dataset --dist skew
# test
CUDA_VISIBLE_DEVICES=0 python test.py --dataset dataset --dist skew --date "YYYY-mm-dd_HH-MM-SS"
```