#!/bin/bash

run_nums=$1
exp_name="TWM"
num_processes=1
datasets="UMLS Kinships" #$4 #delimit by spaces
models="DistMult TransE" #delimit by spaces

start=`date +%s`
seed=None


for dataset in $datasets
do
    for model in $models
    do
        for run_num in $run_nums
        do
            run_name="$dataset-$model-$exp_name-run2.$run_num"
            echo "running $run_name"

            mkdir output/$run_name &> /dev/null
            python pipeline.py \
                output/$run_name/$run_name.grid \
                output/$run_name/ \
                $num_processes \
                $dataset \
                $model \
                $seed \
                1>> output/$run_name/$run_name.res \
                2>> output/$run_name/$run_name.log

            end=`date +%s`
            runtime=$((end-start))
            echo "Experiments took $runtime seconds" 1>> output/$run_name/$run_name.log
        done
    done
done
