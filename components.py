'''

    File contains components like encoder (LSTM layer). CNN model etc.
'''

#Torch related functionalities
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):

    """LSTM encoder."""
    def __init__(self, max_length, hidden_dim, number_of_layer, embedding_dim, vocab_size, bidirectional, dropout = 0.0,vectors=None,debug=False):
        '''
            :param max_length: Max length of the sequence.
            :param hidden_dim: dimension of the output of the LSTM.
            :param number_of_layer: Number of LSTM to be stacked.
            :param embedding_dim: The output dimension of the embedding layer/ important only if vectors=none
            :param vocab_size: Size of vocab / number of rows in embedding matrix
            :param bidirectional: boolean - if true creates BIdir LStm
            :param vectors: embedding matrix
            :param debug: Bool/ prints shapes and some other meta data.
        '''
        super(Encoder, self).__init__()

        self.max_length, self.hidden_dim, self.embedding_dim, self.vocab_size = max_length, hidden_dim, embedding_dim, vocab_size
        self.number_of_layer = number_of_layer
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.debug = debug

        if vectors is not None:
            self.embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(vectors))
            self.embedding_layer.weight.requires_grad = True
        else:
            # Embedding layer
            self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)

        # LSTM layer
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.number_of_layer, bidirectional=self.bidirectional,dropout=self.dropout)

    def init_hidden(self, batch_size, device):
        # Return a new hidden layer variable for LSTM
        # The axes semantics are (num_layers * num_directions, minibatch_size, hidden_dim)
        if not self.bidirectional:
            return (torch.zeros((self.number_of_layer, batch_size, self.hidden_dim), device=device),
                    torch.zeros((self.number_of_layer, batch_size, self.hidden_dim), device=device))
        else:
            return (torch.zeros((2 * self.number_of_layer, batch_size, self.hidden_dim), device=device),
                    torch.zeros((2 * self.number_of_layer, batch_size, self.hidden_dim), device=device))

    def forward(self, x, h):
        # x is the input and h is the hidden state.
        if self.debug: print ("input/x shape is :", x.shape)
        if self.debug: print ("hidden state shape is :", h[0].shape)

        x_embedded = self.embedding_layer(x)
        if self.debug: print ("x_embedded transpose shape is :", x_embedded.transpose(1, 0).shape)

        #         output,h = self.lstm(x_embedded.view(-1,self.batch_size,self.embedding_dim),h)
        output, h = self.lstm(x_embedded.transpose(1, 0), h)
        if self.debug: print ("output shape is ", output.shape)
        if self.debug: print ("h[0] shape is ", h[0].shape, "h[1] shape is ", h[1].shape)

        return output, h


class DenseClf(nn.Module):

    def __init__(self, inputdim, hiddendim, outputdim):
        """
            This class has a two layer dense network of changable dims.
            Intended use case is that of

                - *bidir dense*:
                    give it [v_q, v_p] and it gives a score.
                    in this case, have outputdim as 1
                - * bidir dense dot*
                    give it v_q and it gives a condensed vector
                    in this case, have any outputdim, preferably outputdim < inputdim

        :param inputdim: int: #neurons
        :param hiddendim: int: #neurons
        :param outputdim: int: #neurons
        """

        super(DenseClf, self).__init__()

        self.inputdim = inputdim
        self.hiddendim = hiddendim
        self.outputdim = outputdim
        self.hidden = nn.Linear(inputdim, hiddendim)
        self.output = nn.Linear(hiddendim, outputdim)

    def forward(self, x):

        _x = F.relu(self.hidden(x))

        if self.outputdim == 1:
            return F.sigmoid(self.output(_x))

        else:
            return F.softmax(self.output(_x))


class CNN(nn.Module):

    def __init__(self, _vectors, _vocab_size, _embedding_dim, _output_dim,_debug):
        super(CNN, self).__init__()

        self.vectors = _vectors
        self.vocab_size = _vocab_size
        self.output_dim = _output_dim
        self.debug = _debug


        if self.vectors is not None:
            self.embedding_dim = self.vectors.shape[1]
        else:
            self.embedding_dim = _embedding_dim


        self.out_channels = int(self.embedding_dim / 2.0)

        if self.vectors is not None:
            self.embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(self.vectors))
            self.embedding_layer.weight.requires_grad = True
        else:
            # Embedding layer
            self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)


        self.kernel_size_conv1 = 5
        self.kernel_size_max1 = 2


        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.embedding_dim,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size_conv1,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self.kernel_size_max1),
        )


        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.embedding_dim,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size_conv1,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self.kernel_size_max1),
        )


        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.embedding_dim,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size_conv1,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self.kernel_size_max1),
        )


        self.output = nn.Linear(self.out_channels * 3 * self.kernel_size_conv1 * self.kernel_size_max1, self.output_dim)

    def forward(self, x):

        x_embedded = self.embedding_layer(x)
        if self.debug : print("embedded shape is ", x_embedded.shape)

        x_embedded = x_embedded.transpose(2, 1)
        if self.debug : print("transposed shape is ", x_embedded.shape )

        x_conv1 = self.conv1(x_embedded)
        if self.debug: print("x_conv1 shape is ,", x_conv1.shape)

        x_conv2 = self.conv2(x_embedded)
        if self.debug: print("x_conv2 shape is ,", x_conv2.shape)

        x_conv3 = self.conv3(x_embedded)
        if self.debug: print("x_conv1 shape is ,", x_conv3.shape)

        x_cat = torch.cat((x_conv1, x_conv2, x_conv3), 1)
        if self.debug: print("concated x shape is ,", x_cat.shape)

        x_flat = x_cat.view(x_cat.size(0), -1)
        if self.debug: print("flattened x shape is , ", x_flat.shape)

        output = self.output(x_flat)
        if self.debug: print("final output shape is ,", output.shape)

        return output

