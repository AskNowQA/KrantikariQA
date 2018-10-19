echo starting krantiakri
date +"%T"
python -u data_creator_step1.py 0 500 lcquad &> logs/0.txt &
python -u data_creator_step1.py 500 1000 lcquad &> logs/500.txt &
python -u data_creator_step1.py 1000 1500 lcquad &> logs/1000.txt &
python -u data_creator_step1.py 1500 2000 lcquad &> logs/1500.txt &
python -u data_creator_step1.py 2000 2500 lcquad &> logs/2000.txt &
python -u data_creator_step1.py 2500 3000 lcquad &> logs/2500.txt &
python -u data_creator_step1.py 3000 3500 lcquad &> logs/3000.txt &
python -u data_creator_step1.py 3500 4000 lcquad &> logs/3500.txt &
python -u data_creator_step1.py 4000 4500 lcquad &> logs/4000.txt &
python -u data_creator_step1.py 4500 -1 lcquad &> logs/4500.txt &
date +"%T"