#!/bin/bash

set -ex

# make QuasarSpectraLinearRegMain
# ./QuasarSpectraLinearRegMain

python data_visualizer.py < data/quasar_train_visual.csv
python data_visualizer.py < data/test_example_1.csv
python data_visualizer.py < data/test_example_6.csv
