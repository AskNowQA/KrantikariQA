# KrantikariQA
An Information Gain based Question Answering system over knowledge graph systems.


1. chmod +x parallel_data_creation.sh
2. download glove42B and save it in resource folder
3. mkdir logs
4. ./parallel_data_creation.sh
5. python data_creation_step1.py
6. python reduce_data_creation_step2.py
7. CUDA_VISIBLE_DEVICES=3 python corechain.py -model slotptr -device cuda -dataset lcquad -pointwise False



#### Download glove
>wget http://nlp.stanford.edu/data/glove.42B.300d.zip
  save it to resource folder
> unzip it

### Use Anaconda installation (still need to test it)
> conda env create -f environment.yml    

### Setup redis server (this setup is not necessary. Its used for caching)
    For installation https://redis.io/topics/quickstart

### Setup dbpedia and add the url in utils/dbpedia_interface.py

### Setup SPARQL parsing server
    @TODO: add code here 
    Install nodejs (node, nodejs)
    > nodejs app.js

### Setup embedding server
     python ei_server.py (Keep this always on)
     This will need bottle installed (pip install bottle)


> Check for running verison of DBPedia, Redis (if caching),
 SPARQL Parsing server, Embedding interface
 

### Setup Qelos-utils
https://github.com/lukovnikov/qelos-util.git
change into qelos-util dir and python setup.py build/develop/
cp qelos ../

### Install few more things


### A potential bug is that he glove file datatype would be <U32


A rdftype_lookup.json can be created using the keys of relation.pickle (data/data/common) 
```
import numpy as np
mat = np.load('resources/vectors_gl.npy')
mat = mat.astype(np.float64)
np.save('resources/vectors_gl.npy',mat)

#### TODO
change embedding in configs to 300d
```


#### Once the dataset is prepared

To check if all the files are in correct palce run the following command

```
python file_location_check.py
```

Once the data is at appropriate place run the following command. 

```
CUDA_VISIBLE_DEVICES=3 python corechain.py -model slotptr -device cuda -dataset lcquad -pointwise False
```



