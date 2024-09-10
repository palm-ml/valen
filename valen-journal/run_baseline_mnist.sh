dt="benchmark"
ds="mnist"
lr=0.01
wd=0.001
bs=256
ep=500
lo="valen"
seed=0
gpu="1"
mo="lenet"
partial_type="feature"
warm_up=10
gamma=5
beta=0.1

seed_array=(0 1 2 3 4)

mkdir results_baseline_2/
mkdir results_baseline_2/${ds}
mkdir results_baseline_2/${ds}/${lo}

for seed in ${seed_array[@]}
do
    nohup python -u main_aug.py --dt ${dt} --ds ${ds} --lr ${lr} --wd ${wd} --bs ${bs} --ep ${ep} --lo ${lo} --gpu ${gpu} --mo ${mo} --seed ${seed} \
            --partial_type ${partial_type} --warm_up ${warm_up} --gamma ${gamma} --beta ${beta} \
            >results_baseline_2/${ds}/${lo}/${ds}_lo=${lo}_seed=${seed}.log 2>&1 
done

dt="benchmark"
ds="fmnist"
lr=0.01
wd=0.00001
bs=256
ep=500
lo="valen"
seed=0
gpu="1"
mo="lenet"
partial_type="feature"
warm_up=10
gamma=5
beta=0.1

seed_array=(0 1 2 3 4)

mkdir results_baseline_2/
mkdir results_baseline_2/${ds}
mkdir results_baseline_2/${ds}/${lo}

for seed in ${seed_array[@]}
do
    nohup python -u main_aug.py --dt ${dt} --ds ${ds} --lr ${lr} --wd ${wd} --bs ${bs} --ep ${ep} --lo ${lo} --gpu ${gpu} --mo ${mo} --seed ${seed} \
            --partial_type ${partial_type} --warm_up ${warm_up} --gamma ${gamma} --beta ${beta} \
            >results_baseline_2/${ds}/${lo}/${ds}_lo=${lo}_seed=${seed}.log 2>&1 
done

dt="benchmark"
ds="kmnist"
lr=0.01
wd=0.0001
bs=256
ep=500
lo="valen"
seed=0
gpu="1"
mo="lenet"
partial_type="feature"
warm_up=10
gamma=5
beta=0.1
correct=0.2

seed_array=(0 1 2 3 4)

mkdir results_baseline_2/
mkdir results_baseline_2/${ds}
mkdir results_baseline_2/${ds}/${lo}

for seed in ${seed_array[@]}
do
    nohup python -u main_aug.py --dt ${dt} --ds ${ds} --lr ${lr} --wd ${wd} --bs ${bs} --ep ${ep} --lo ${lo} --gpu ${gpu} --mo ${mo} --seed ${seed} \
            --partial_type ${partial_type} --warm_up ${warm_up} --gamma ${gamma} --beta ${beta} --correct ${correct} \
            >results_baseline_2/${ds}/${lo}/${ds}_lo=${lo}_seed=${seed}.log 2>&1 
done

