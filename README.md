# KrantikariQA
An Information Gain based Question Answering system over knowledge graph systems.


1. chmod +x parallel_data_creation.sh
2. download glove42B and save it in resource folder
3. mkdir logs
4. ./parallel_data_creation.sh
5. python data_creation_step1.py
6. python reduce_data_creation_step2.py
7. CUDA_VISIBLE_DEVICES=3 python corechain.py -model slotptr -device cuda -dataset lcquad -pointwise False