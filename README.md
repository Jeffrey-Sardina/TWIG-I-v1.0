# TWIG-I
TWIG-I (Topologically-Weighted Intelligence Generation for Inference) is an evolution of TWIG (https://github.com/Jeffrey-Sardina/TWIG-release-1.0) that performs link prediction, on one KG or across multiple KGs, with a single model and a tiny fraction of the parameters required by traditional KGEs. Results also indicate it can achieve reliably higher performance; however, in its current for, it struggles to scale to larger datasets.

This version of TWIG was presented at the Adapt Scientific Conference in 2024.

A quick overview:
- twig/ contains all TWIG-I code and documentation, as well as code for baseline experiments
- install/ contains a script to set up a twig environment that you ca run TWIG-I in.
- twig/kg_learn_1/ contains the full implementation of TWIG-I, its outputs and data, and its documentation
- twig/output/ contains the output of KGE baselines to TWIG-I
- twig/TWIG-I-hyperparameter-search.sh is used to run a hyperparameter search for TWIG-I.
- twig/kgl_pipleline.sh is used to run TWIG-I. Its corresponding .py file does most of the heavy lifting; the bash file exists simply for ease of use on the command line.
- twig/pipeline_test-set.sh is used to run standard KGE models on their test set for evaluation. Its corresponding .py file does most of the heavy lifting; the bash file exists simply for ease of use on the command line.

Contact: sardinaj AT tcd [dot] ie

All rights reserved.
