echo starting krantiakri
date +"%T"
python -u data_creator_step1.py 0 500 qg &> logs/0.txt &
python -u data_creator_step1.py 500 1000 qg &> logs/500.txt &
python -u data_creator_step1.py 1000 1500 qg &> logs/1000.txt &
python -u data_creator_step1.py 1500 2000 qg &> logs/1500.txt &
python -u data_creator_step1.py 2000 2500 qg &> logs/2000.txt &
python -u data_creator_step1.py 2500 3000 qg &> logs/2500.txt &
python -u data_creator_step1.py 3000 3500 qg &> logs/3000.txt &
python -u data_creator_step1.py 3500 4000 qg &> logs/3500.txt &
python -u data_creator_step1.py 4000 4500 qg &> logs/4000.txt &
python -u data_creator_step1.py 4500 5000 qg &> logs/4500.txt &
python -u data_creator_step1.py 5000 5500 qg &> logs/5000.txt &
python -u data_creator_step1.py 5500 6500 qg &> logs/5500.txt &
python -u data_creator_step1.py 6500 7500 qg &> logs/6500.txt &
python -u data_creator_step1.py 7500 -1 qg &> logs/7500.txt &
date +"%T"