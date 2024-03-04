#!/bin/bash

hyp_val_mode='1'
epochs=10
negsamp='simple'
tag='ASC-hyp'

for negs in 5 30 100
do
    for lr in 5e-4 5e-5 5e-6
    do
        for batch_size in 32 256 1024
        do
            ./kgl_pipleline.sh UMLS-Kinships $epochs $negs $lr $negsamp $hyp_val_mode $batch_size $tag
            ./kgl_pipleline.sh UMLS $epochs $negs $lr $negsamp $hyp_val_mode $batch_size $tag
            ./kgl_pipleline.sh Kinships $epochs $negs $lr $negsamp $hyp_val_mode $batch_size $tag
        done
    done
done
