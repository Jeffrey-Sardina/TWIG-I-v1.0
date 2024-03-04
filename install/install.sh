#!/bin/bash

# Update package list
apt-get update

# Install conda
apt-get install -y wget
if [[ ! -d "/root/miniconda3" ]]
then
    miniconda_name="Miniconda3-latest-Linux-x86_64.sh"
    wget https://repo.anaconda.com/miniconda/$miniconda_name
    chmod u+x $miniconda_name
    ./$miniconda_name -b
    rm $miniconda_name
fi

# Install Python packages
export PATH=$PATH:~/miniconda3/bin
apt-get install -y git # needed for PyKEEN internals
apt install sqlite # needed for optuna, used by PyKEEN
conda create -n "twig" python=3.9 pip
conda run --no-capture-output -n twig pip install torch torchvision torchaudio
conda run --no-capture-output -n twig pip install pykeen numba
conda init bash

# Add to conda to .bashrc
echo "conda activate twig" >> ~/.bashrc
