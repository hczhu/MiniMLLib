mkdir mnist-data
kaggle competitions download -c digit-recognizer -p mnist-data
python3 mnist_ffn.py | tee mnist-data/sub.csv
