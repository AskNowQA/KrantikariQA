# Core Chain Ranking

### Bidirectional RNN with Dot
*Accuracy*: ~ 60.16%

*Seq Length*: 25

*kwargs*: `{normalizedot: False}`

### Bidirectional RNN with Dot
*Accuracy*: ~ 55.42%

*Seq Length*: 25

lr = 0.0005`


### Bidirectional RNN with Dot
*Accuracy*: ~ 53%

*Seq Length*: 25

*kwargs*: `{normalizedot: True}`

### Bidirectional RNN with Dot with sigmoid based loss
*Accuracy*: ~ 50%

*Seq Length*: 25

### Bidirectional RNN with Dense
*Accuracy*: ~ 33% --> 49.431

*Seq Length*: 25

*Model Location - ./data/models/core_chain/birnn_dense/lcquad/model_08/model.h5

*Change Log -
            Used relu instead of default activation
            Used SimplerDense (That is just a BIRNN and one layer of dense)
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
