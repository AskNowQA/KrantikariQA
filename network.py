'''
    Calls various components files to create a models/network. Also provides various functionalities to train and predict.
'''

import components as com
import torch

class BiLstmDot():

    def __init__(self,_parameter_dict,_device,_debug):
        self.debug = _debug
        self.parameter_dict = _parameter_dict
        self.device = _device

        if self.debug:
            print("inti encoder model")
        self.encoder = com.Encoder(self.parameter_dict['max_length'], self.parameter_dict['hidden_size']
                          , self.parameter_dict['number_of_layer'], self.parameter_dict['embedding_dim'],
                              self.parameter_dict['vocab_size'],
                          bidirectional=self.parameter_dict['bidirectional'],
                          vectors=self.parameter_dict['vectors']).cuda(self.device)


    def train(self,ques_batch, pos_batch, neg_batch, dummy_y, model, optimizer, loss_fn, device):
        '''
            Given data, passes it through model, inited in constructor, returns loss and updates the weight
            :params ques_batch: batch of question
            :params pos_batch: batch of corresponding positive paths
            :params neg_batch: batch of corresponding negative paths
            :params dummy_y:a batch of ones (same length as that of batch)
        '''

        hidden = model.init_hidden(ques_batch.shape[0], device)
        optimizer.zero_grad()
        #Encoding all the data
        ques_batch, _ = model(ques_batch, hidden)
        pos_batch, _ = model(pos_batch, hidden)
        neg_batch, _ = model(neg_batch, hidden)
        #Calculating dot score
        pos_scores = torch.sum(ques_batch[-1] * pos_batch[-1], -1)
        neg_scores = torch.sum(ques_batch[-1] * neg_batch[-1], -1)
        '''
            If `y == 1` then it assumed the first input should be ranked higher
            (have a larger value) than the second input, and vice-versa for `y == -1`
        '''
        loss = loss_fn(pos_scores, neg_scores, dummy_y)
        loss.backward()
        optimizer.step()
        return loss

    def predict(self,que,paths,model,device):
        hidden = model.init_hidden(que.shape[0], device)
        question, _ = model(que.long(), hidden)
        paths, _ = model(paths.long(), hidden)
        score = torch.sum(question[-1] * paths[-1], -1)
        return score