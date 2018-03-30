#!/bin/bash

set -xe

sudo apt-get install gfortran liblapack3 libsuperlu-dev libarpack2-dev gfortran arpack++ python3-pip python-dev build-essential
sudo pip3 install --upgrade pip
echo "installing numpy"
sudo pip3 install numpy nose
python -c 'import numpy; numpy.test()'
exit 0

git clone https://github.com/xianyi/OpenBLAS.git
cd OpenBLAS
make
sudo make PREFIX=/usr/local/lib/ install
cd ..
rm -fr OpenBLAS

echo "Installing armadillo"
git clone https://github.com/conradsnicta/armadillo-code.git
cd armadillo-code/
cmake .
make
sudo make install
cd ..
rm -fr armadillo-code
