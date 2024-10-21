# Revisiting K-mer Profile for Effective and Scalable Genome Representation Learning

### Overview
This project explores effective and scalable genome representation learning approaches relying on the k-mer features for the metagenomics binning task.

### Installation
1. Clone this repository:
```
git clone https://github.com/abdcelikkanat/revisitingkmers.git
cd revisitingkmers
```
2. Install dependencies: Make sure you have Python 3.8 installed. You can install the required Python packages using ```pip```:
```
pip install -r requirements.txt
```
3. Install ```gdown``` (if you don't already have it) for downloading the datasets:
```
pip install gdown
```

### Datasets
To download and prepare the training dataset, run the following commands:
```
gdown 1p59ch_MO-9DXh3LUIvorllPJGLEAwsUp
unzip dnabert-s_train.zip
```

To download the evaluation datasets, use the following commands:
```
gdown 1I44T2alXrtXPZrhkuca6QP3tFHxDW98c
unzip dnabert-s_eval.zip
```

### Usage
To view the detailed usage instructions for each model, you can use the --help flag:

Poisson Model
```
python poisson_model.py --help
```
Nonlinear Model
```
python nonlinear.py --help
```

### Citation
If you find the work useful for your research, please consider citing the following paper:
```
@article{celikkanat2024revisiting,
  title={Revisiting K-mer Profile for Effective and Scalable Genome Representation Learning},
  author={Celikkanat, Abdulkadir and Masegosa, Andres R. and Nielsen, Thomas D.},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  year={2024}
}
```
