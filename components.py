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


class AttendCompareAggregate(nn.Module):
    """
        Corresponds to the equations above. Init needs inputoutput dims.
        Suggestion:
            inputdim = hiddendim of encoder (*2) if bidir

        Forward:
            p, q are output of encoder.
                **shape** = (len, batch, hidden* (2 if bidir)).

        ## Link: https://arxiv.org/pdf/1606.01933.pdf

        ## Attend, Compare, Aggregate

            A class which performs all of the things of decomposible attention.
            The way this works is in following steps:

            ### Attend:
            - Encode all q hidden states with $F$ as follows:
            - $att_{qi} = F(q_i)$
            - Similarly
            - $att_{pj} = F(p_j)$
            - Then we combine them as follows
            - $e_{ij} = att_{qi}^{T} \cdot att_{pj}$
            - Finally we take softamx along two axis as follows:
            - $\beta_i=\sum_j^{l_b} softmax_j(e_{ij})\cdot p_j$
            - $\alpha_j=\sum_i^{l_a} softmax_i(e_{ij}) \cdot q_i$

            ### Compare:
            - Concatenate and feedforward the outputs in this manner:
            - $v_{1,i}=G([q_i, \beta_i])$ for $i \in (1,..l_q)$
            - $v_{2,j}=G([p_j, \alpha_j])$ for $j \in (1,..l_p)$

            ### Aggregate
            - Sum over all the $v_{1/2}$ and pass it through a dense to compute final score
            - $v_1 = \sum^{l_q} v_{1,i}$
            - $v_2 = \sum^{l_p} v_{2,j}$
            - res = $H([v_1, v_2])$
    """

    def __init__(self, inputdim, debug=False):
        super(AttendCompareAggregate, self).__init__()

        self.inputdim = inputdim
        self.debug = debug

        self.F = nn.Linear(self.inputdim, self.inputdim)
        self.G = nn.Linear(self.inputdim * 2, self.inputdim)
        self.H = nn.Linear(self.inputdim * 2, 1)

    def forward(self, q, p):

        # Collect some temp macros
        batch_size = q.shape[1]
        seq_length_q, seq_length_p = q.shape[0], p.shape[0]
        if self.debug:
            print("Input:")
            print("\tq:\t", q.shape)
            print("\tp:\t", p.shape)

        # Create att_p, q matrices. We use view to change the input and the output. VIEW IS TESTED DONT PANIC.
        att_q = self.F(q.view(-1, q.shape[2])).view(seq_length_q, batch_size, -1).transpose(1, 0)
        att_p = self.F(p.view(-1, p.shape[2])).view(seq_length_p, batch_size, -1).transpose(1, 0)
        if self.debug:
            print ("\tatt_p:\t", att_p.shape)
            print ("\tatt_q:\t", att_q.shape)

        # Now we calculate e. To do so, we transpose att_q, and BMM it with att_p
        # Note: correspondence held between q->i & p->j in the matrix e.
        e = torch.bmm(att_q, att_p.transpose(2, 1))
        if self.debug:
            print ("\te:\t", e.shape)

        # We now prepare softmax_j and softmax_i (as defined in eq above)
        softmax_j = F.softmax(e.view(-1, e.shape[2]), dim=1).view(-1, e.shape[1], e.shape[2])
        softmax_i = F.softmax(e.transpose(2, 1).contiguous().view(-1, e.shape[1]), dim=1).transpose(1, 0). \
            view(e.shape[1], -1, e.shape[2]).transpose(1, 0)
        if self.debug:
            print ("       softmaxj:\t\b", softmax_j.shape)
            print ("       softmaxi:\t\b", softmax_i.shape)

        beta = torch.bmm(softmax_j, p.transpose(1, 0))
        alpha = torch.bmm(softmax_i.transpose(2, 1), q.transpose(1, 0))
        if self.debug:
            print ("\tbeta:\t", beta.shape)
            print ("\talpha:\t", alpha.shape)

        """
            Compare
        """
        # Concatenate beta,q && alpha,p and feed it to G to get v1 and v2
        v1 = self.G(torch.cat((q.transpose(1, 0), beta), dim=-1).view(-1, self.inputdim * 2)) \
            .view(batch_size, seq_length_q, -1)
        v2 = self.G(torch.cat((p.transpose(1, 0), alpha), dim=-1).view(-1, self.inputdim * 2)) \
            .view(batch_size, seq_length_p, -1)
        if self.debug:
            print("\tv1:\t", v1.shape)
            print("\tv2:\t", v2.shape)

        """
            Aggregate
        """
        sum_v1 = torch.sum(v1, dim=1)
        sum_v2 = torch.sum(v2, dim=1)
        if self.debug:
            print("\tsum_v1:\t", sum_v1.shape)
            print("\tsum_v2:\t", sum_v2.shape)

        # Finally calculate the sum
        result = self.H(torch.cat((sum_v1, sum_v2), dim=-1))
        if self.debug:
            print("\tresult:\t", result.shape)

        return result