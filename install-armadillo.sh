#!/bin/bash

set -xe

git clone https://github.com/xianyi/OpenBLAS.git
cd OpenBLAS
make
sudo make PREFIX=/usr/local/lib/ install

cd ..
git clone https://github.com/Reference-LAPACK/lapack.git
sudo apt-get install gfortran
sudo apt-get install liblapack3
sudo apt-get install libsuperlu3
sudo apt-get install libarpack3
sudo apt-get install libarpack2-dev libsuperlu3-dev gfortran arpack++
cd armadillo-code/
cmake .
make
sudo make install
