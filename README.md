# Modularized VAE

## Data
Extract data.zip to ```data``` directory

Download links:
[data.zip](https://drive.google.com/file/d/1WmLVm4bRNr6OOJCNqIHlTEnThpT016bX/view?usp=sharing)


## Setup

- python3.5 or higher version

Install packages
```
pip3 install -r requirements.txt
```

## Usage

### Training

```
python3 train.py --data ./data/nottingham ./data/jsb ./data/piano
```

For more details, see
```
python3 train.py --help
```

### Generate a midi file
```
python3 generate.py --model $MODEL_PATH
```

For more details, see
```
python3 generate.py --help
```


## Reference

Main paper to be cited

```
@article{wang2018modeling,
  title={Modeling Melodic Feature Dependency with Modularized Variational Auto-Encoder},
  author={Wang, Yu-An and Huang, Yu-Kai and Lin, Tzu-Chuan and Su, Shang-Yu and Chen, Yun-Nung},
  journal={arXiv preprint arXiv:1811.00162},
  year={2018}
}
```
