#!/bin/bash

set -ex

# make QuasarSpectraLinearRegMain
# ./QuasarSpectraLinearRegMain

python QuasarSpectraVisualizer.py < data/quasar_train_visual.csv
python QuasarSpectraVisualizer.py < data/test_example_1.csv
python QuasarSpectraVisualizer.py < data/test_example_6.csv
