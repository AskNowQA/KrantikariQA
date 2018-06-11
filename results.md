# Core Chain Ranking

### Bidirectional RNN with Dot

*Accuracy*: ~ 53.70%

*Seq Length*: 25 

*Model stored*: ./data/models/core_chain/birnn_dot/lcquad/model_15/model.h5

*EPOCHS*: 180

*DATASET*: lcquad

Epoch 00255: val_metric improved from 0.53300 to 0.54800, saving model to ./data/models/core_chain/birnn_dot/lcquad/model_16/model.h5



*Change log* 
    
    Experiment 1:
        Epoch 00235: val_metric improved from 0.55600 to 0.56100, saving model to ./data/models/core_chain/birnn_dot/lcquad/model_17/model.h5
    @smart save model called
        From LSTM to GRU
        Dropout - (0.3-0.5)
    
    Experiment 2:
        Everything same as experiment 1 but beta2 in adam from .999 -> .9
        Epoch 00235: val_metric improved from 0.57200 to 0.58300, saving model to ./data/models/core_chain/birnn_dot/lcquad/model_17/model.h5@smart save model called

            
    Experiment 3:
        Everything same as experiment 2 but hidden units are 256
        Epoch 00020: val_metric improved from 0.42000 to 0.49000, saving model to ./data/models/core_chain/birnn_dot/lcquad/model_01/model 
    
    EXPERIMENT 4:
        Everything same but with higher batch size = 1760
        Epoch 00270: val_metric improved from 0.57400 to 0.58000, saving model to ./data/models/core_chain/birnn_dot/lcquad/model_01/model.h5@smart save model called

     
     Note - experiment 3 needs to be performed again as exp 4 overwrote it 
     
    

@sda-srv04
@smart save model called
network.py:smart_save_model: Saving model in
 ./data/models/core_chain/birnn_dot/lcquad/model_02/model.h5
Epoch 00110: val_metric improved from 0.57400 to 0.58500
core_chain accuracy = 530/1000
fmeasure = 60.01





### Bidirectional RNN with dense and dense

@qrowdgpu+titan
network.py:smart_save_model: Saving model in 
./data/models/core_chain/birnn_dense/lcquad/model_15/model.h5
*Accuracy*: 33.20
300 epochs 

### Bidirectional RNN with Dot
*Accuracy*: ~ 53%

*Seq Length*: 25

*kwargs*: `{normalizedot: True}`

### Bidirectional RNN with Dense and Dot

@qrowdgpu+titan
Epoch 00250: val_metric improved from 0.46300 to 0.47400,
@smart save model called
network.py:smart_save_model:
Saving model in ./data/models/core_chain/birnn_dense/lcquad/model_14/model.h5


### Bidirectional with Triplet Loss
*Accuracy*: ??

*Seq Length*: 25

### Parikh
*Accuracy*: ~ 33.29%

*Seq Length*: 25

*Model location - ./data/models/core_chain/parikh/lcquad/model_07/model.h5

*Change log- Remove batch normalization; merged ques encoding with the aligned vectors.
            - Decreasing the learning rate does not help.


### Maheshwari
*Accuracy*: ~ 10%

*Seq Length*: 25


### SimplerDense
*Accuracy*: ~ 50.56%

*Seq Length*: 25

*Model location - ./data/models/core_chain/birnn_dense/lcquad/model_07/model.h5


### Cnn
*Accuracy*: ~ 38.056%

*Seq Length*: 25

*Model location -  ./data/models/core_chain/cnn/lcquad/model_00/model.h5


### Parikh_dot
*Accuracy*: ~ 41.986%

*Seq Length*: 25

*Model location - ./data/models/core_chain/parikh_dot/lcquad/model_02/model.h5

    
*Change log- Remove batch normalization; merged ques encoding with the aligned vectors.
            - Decreasing the learning rate does not help.
            - new BidirectionalRNN to include dot.

# RDF Chain Ranking


### Bidirectional RNN with Dot

*Accuracy*: ~ 80.519%

*Seq Length*: 25

*Model stored*: ./data/models/rdf/lcquad/model_04/model.h5

*EPOCHS*: 20

*DATASET*: lcquad



@sda-srv04
network.py:smart_save_model: Saving model in
 ./data/models/rdf/lcquad/model_00/model.h5
 0.83117
 
@qrowdgpu+titan
Epoch 00120: val_metric improved from 0.82857 to 0.83117, 
saving model
@smart save model called
network.py:smart_save_model by calling smart save function: 
Saving model in ./data/models/rdf/lcquad/model_05/model.h5




### Intent Prediction 

*Accuracy*: ~ 99.3%

*Seq Length*: 25

*Model stored*: ./data/models/intent/lcquad/model_03/model.h5

*EPOCHS*: 20

*DATASET*: lcquad


    
# intent prediction
.992 
network.py:smart_save_model: Saving model in ./data/models/intent/lcquad/model_03/model.h5


@sda-srv04
('rnn model results are ', 0.992)
@smart save model called
network.py:smart_save_model: Saving model in ./data/models/intent/lcquad/model_00/model.h5



# type existance 
('rnn model results are ', 0.786)
@smart save model called
network.py:smart_save_model: Saving model in ./data/models/type_existence/lcquad/model_03/model.h5

@sda-srv04
('rnn model results are ', 0.785)
@smart save model called
network.py:smart_save_model: Saving model in ./data/models/type_existence/lcquad/model_00/model.h5

also at@sda-srv04 

('rnn model results are ', 0.782)
@smart save model called
network.py:smart_save_model by calling smart save function: Saving model in ./data/models/type_existence/lcquad/model_03/model.h5



