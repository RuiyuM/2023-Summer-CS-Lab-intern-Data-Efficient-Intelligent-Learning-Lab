#! /bin/bash

#methods=("active_query" "test_query")

methods=('test_query' 'uncertainty' "AV_temperature" "active_query" "Core_set" 'certainty' 'OpenMax' 'BGADL' 'random')
structures=('resnet34' 'resnet50' 'vgg16')

# shellcheck disable=SC2068
for method in ${methods[@]};
do
    for structure in ${structures[@]};
    do
        for j in 400
        do
            for i in 10
            do
              CUDA_VISIBLE_DEVICES=0 python NEAT_main.py --gpu 0 --k $i --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 8 --known-class 40 --query-batch $j --seed 1 --model $structure --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset Tiny-Imagenet &
			        CUDA_VISIBLE_DEVICES=1 python NEAT_main.py --gpu 1 --k $i --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 8 --known-class 40 --query-batch $j --seed 2 --model $structure --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset Tiny-Imagenet &
			        CUDA_VISIBLE_DEVICES=2 python NEAT_main.py --gpu 2 --k $i --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 8 --known-class 40 --query-batch $j --seed 3 --model $structure --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset Tiny-Imagenet &
              wait
            done
        done
    done
done



