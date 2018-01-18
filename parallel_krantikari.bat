echo starting krantiakri
date +"%T"
python krantikari.py 500 500 1000 0 &> 500.txt &
python krantikari.py 1000 1000 2000 0 &> 1000.txt &
python krantikari.py 2000 2000 3000 0 &> 2000.txt &
python krantikari.py 3000 3000 4000 0 &> 3000.txt &
python krantikari.py 4000 4000 0 0 &> 4000.txt &
date +"%T"