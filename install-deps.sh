#!/bin/bash

set -xe


git clone https://github.com/xianyi/OpenBLAS.git
cd OpenBLAS
make
sudo make PREFIX=/usr/local/lib/ install
cd ..
rm -fr OpenBLAS

sudo apt-get install gfortran liblapack3 libsuperlu3 libarpack3 libarpack2-dev libsuperlu3-dev gfortran arpack++

echo "Installing armadillo"
git clone https://github.com/conradsnicta/armadillo-code.git
cd armadillo-code/
cmake .
make
sudo make install
cd ..
rm -fr armadillo-code

echo "installing numpy"
sudo pip install numpy nose
python -c 'import numpy; numpy.test()'
