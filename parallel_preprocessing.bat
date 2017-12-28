echo starting preprocessing
date +"%T"
python parallel_preprocessing.py 0 10 &
python parallel_preprocessing.py 11 20 &
python parallel_preprocessing.py 21 30 &
wait
date +"%T"