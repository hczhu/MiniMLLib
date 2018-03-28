#!/bin/bash

set -ex

make QuasarSpectraLinearRegMain
./QuasarSpectraLinearRegMain

python data_visualizer.py < data/quasar_train_visual.csv
