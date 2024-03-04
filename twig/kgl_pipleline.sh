#!/bin/bash

datasets=$1 #delimit by -
epochs=$2
npp=$3
lr=$4 #5e-4 works for UMLS; 5e-5 is good for Kinships; 
sampler=$5 #simple or vector
hyp_validation_mode=$6
batch_size=$7
tag=$8

version="0"
normalisation="zscore"
use_train_filter='0' #was 0
use_valid_and_test_filters='1' #was 0

export TWIG_CHANNEL=3

cd kg_learn_1/
out_file="rec_v${version}_${datasets}_norm-${normalisation}_e${epochs}-lr${lr}_bs${batch_size}_npp${npp}_${sampler}-sampler_filter-code${use_train_filter}${use_valid_and_test_filters}_tag-${tag}.log"

start=`date +%s`
for dataset in $datasets
do
    python -u run_exp.py \
        $version \
        $datasets \
        $epochs \
        $lr \
        $normalisation \
        $batch_size \
        $npp \
        $use_train_filter \
        $use_valid_and_test_filters \
        $sampler \
        $hyp_validation_mode \
        &> $out_file

    end=`date +%s`
    runtime=$((end-start))
    echo "Experiments took $runtime seconds" &>> $out_file
done
