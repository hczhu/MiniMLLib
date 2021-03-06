if [ ! -r mnist-data ]; then
  mkdir mnist-data
  kaggle competitions download -c digit-recognizer -p mnist-data
fi

log_file='/tmp/mnist.log'
rm -fr $log_file
touch $log_file
python3 mnist_ffn.py 2> $log_file | tee mnist-data/sub.csv
tail -f $log_file
kaggle competitions submit -c digit-recognizer -f mnist-data/sub.csv
