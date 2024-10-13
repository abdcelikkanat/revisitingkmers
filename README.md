# Revisiting K-mer Profile for Effective and Scalable Genome Representation Learning
This is the repository for the project entitled "Revisiting K-mer Profile for Effective and Scalable Genome Representation Learning"

+ In order to get the detailed instructions to run the models, please type
```
python poisson_model.py --help
```
```
python nonlinear.py --help
```

+ As their names suggest, poisson_model.py is for the poisson model and nonlinear.py is for the nonlinear model.


+ The training datasets can be obtained by running the following command:
```
gdown 1p59ch_MO-9DXh3LUIvorllPJGLEAwsUp # pip install gdown
unzip dnabert-s_train.zip  # unzip the data
```

+ For the evaluation, please use the following command to get the data:
```
gdown 1I44T2alXrtXPZrhkuca6QP3tFHxDW98c # pip install gdown
unzip dnabert-s_eval.zip  # unzip the data
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