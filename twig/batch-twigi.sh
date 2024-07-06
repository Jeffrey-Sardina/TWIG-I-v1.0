#!/bin/bash

# constants for TWIG-I from-scratch baselines
hyp_validation_mode=0
batch_size_test=64
use_2_hop_fts=1
fts_blacklist="None"

for epochs in 20 100
do
    npp=100
    lr=5e-3
    batch_size=64
    dataset="CoDExSmall"
    ./TWIG-I_pipeline.sh \
        $dataset \
        $epochs \
        $npp \
        $lr \
        $hyp_validation_mode \
        $batch_size \
        $batch_size_test \
        $use_2_hop_fts \
        $fts_blacklist \
        "$dataset-e${epochs}"

    npp=30
    lr=5e-3
    batch_size=128
    dataset="DBpedia50"
    ./TWIG-I_pipeline.sh \
        $dataset \
        $epochs \
        $npp \
        $lr \
        $hyp_validation_mode \
        $batch_size \
        $batch_size_test \
        $use_2_hop_fts \
        $fts_blacklist \
        "$dataset-e${epochs}-r1"

    npp=100
    lr=5e-4
    batch_size=128
    dataset="FB15k237"
    ./TWIG-I_pipeline.sh \
        $dataset \
        $epochs \
        $npp \
        $lr \
        $hyp_validation_mode \
        $batch_size \
        $batch_size_test \
        $use_2_hop_fts \
        $fts_blacklist \
        "$dataset-e${epochs}"

    npp=500
    lr=5e-3
    batch_size=128
    dataset="WN18RR"
    ./TWIG-I_pipeline.sh \
        $dataset \
        $epochs \
        $npp \
        $lr \
        $hyp_validation_mode \
        $batch_size \
        $batch_size_test \
        $use_2_hop_fts \
        $fts_blacklist \
        "$dataset-e${epochs}"
done
