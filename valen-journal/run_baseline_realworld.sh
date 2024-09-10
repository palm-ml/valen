python -u main_realworld.py --gpu 0 --bs 100 --dt realworld --ds lost --gamma 20 --beta 0.01
python -u main_realworld.py --gpu 0 --bs 100 --dt realworld --ds MSRCv2 --gamma 20 --beta 0.01
python -u main_realworld.py --gpu 0 --bs 100 --dt realworld --ds birdac --gamma 20 --beta 0.01
python -u main_realworld.py --gpu 0 --dt realworld --ds spd --gamma 20 --beta 0.01 --correct 0.2
python -u main_realworld.py --gpu 0 --dt realworld --ds LYN --gamma 20 --beta 0.01 --correct 0.2
