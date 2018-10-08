echo starting krantiakri
date +"%T"
python data_creation.py 0 10 lcquad &> logs/0.txt &
python data_creation.py 10 20 lcquad &> logs/10.txt &
python data_creation.py 20 30 lcquad &> logs/20.txt &
python data_creation.py 30 -1 lcquad &> logs/30.txt &
date +"%T"