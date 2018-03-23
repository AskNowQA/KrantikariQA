echo starting krantiakri
date +"%T"
python krantikari_new.py 250 0 250 0 &> 250.txt &
python krantikari_new.py 500 250 500 0 &> 500.txt &
python krantikari_new.py 750 500 750 0 &> 750.txt &
python krantikari_new.py 1000 750 1000 0 &> 1000.txt &
python krantikari_new.py 1250 1000 1250 0 &> 1250.txt &
python krantikari_new.py 1500 1250 1500 0 &> 1500.txt &
python krantikari_new.py 1750 1500 1750 0 &> 1750.txt &
python krantikari_new.py 2000 1750 2000 0 &> 2000.txt &
python krantikari_new.py 2250 2000 2250 0 &> 2250.txt &
python krantikari_new.py 2500 2250 2500 0 &> 2500.txt &
python krantikari_new.py 3000 2500 3000 0 &> 3000.txt &
python krantikari_new.py 3500 3000 3500 0 &> 3500.txt &
python krantikari_new.py 4000 3500 4000 0 &> 4000.txt &
python krantikari_new.py 4500 4000 4500 0 &> 4500.txt &
python krantikari_new.py 5000 4500 5000 0 &> 5000.txt &
date +"%T"