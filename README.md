# Variational Label Enhancement for Feature-Dependent Partial Label Learning

## Installation
---
pip install -r requirements.txt 

## Run the Demo
---
### benchmark-random
#### mnist
python -u main.py --gpu 0 --bs 256 --partial_type random --dt benchmark --ds mnist --gamma 10 --beta 0.1
#### kmnist
python -u main.py --gpu 0 --bs 256 --partial_type random --dt benchmark --ds kmnist --gamma 10 --beta 0.1
#### fmnist
python -u main.py --gpu 0 --bs 256 --partial_type random --dt benchmark --ds fmnist --gamma 10 --beta 0.1
#### cifar10
python -u main.py --gpu 0 --bs 256 --partial_type random --dt benchmark --ds cifar10 --lr 5e-2 --wd 1e-3 --gamma 10 --beta 0.1

### benchmark-feature
#### mnist
python -u main.py --gpu 0 --bs 256 --partial_type feature --dt benchmark --ds mnist --warm_up 10 --gamma 5 --beta 0.1
#### kmnist
python -u main.py --gpu 0 --bs 256 --partial_type feature --dt benchmark --ds kmnist --warm_up 100 --gamma 5 --beta 0.1
#### fmnist
python -u main.py --gpu 0 --bs 256 --partial_type feature --dt benchmark --ds fmnist --warm_up 10 --gamma 5 --beta 0.1
#### cifar10
python -u main.py --gpu 0 --bs 256 --partial_type feature --dt benchmark --ds cifar10 --lr 5e-2 --wd 1e-3 --warm_up 10 --gamma 10 --beta 0.1 --correct 0.2


### realword
#### lost
python -u main.py --gpu 0 --bs 100 --dt realworld --ds lost --gamma 20 --beta 0.01
#### MSRCv2
python -u main.py --gpu 0 --bs 100 --dt realworld --ds MSRCv2 --gamma 20 --beta 0.01
#### BirdSong
python -u main.py --gpu 0 --bs 100 --dt realworld --ds birdac --gamma 20 --beta 0.01
#### Soccer Player
python -u main.py --gpu 0 --dt realworld --ds spd --gamma 20 --beta 0.01 --correct 0.2
#### LYN
python -u main.py --gpu 0 --dt realworld --ds LYN --gamma 20 --beta 0.01 --correct 0.2

## Data
---



