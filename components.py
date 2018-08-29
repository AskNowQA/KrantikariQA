'''

    File contains components like encoder (LSTM layer). CNN model etc.
'''

#Torch related functionalities
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import tensor_utils as tu

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

        self.inputdim = int(inputdim)
        self.hiddendim = int(hiddendim)
        self.outputdim = int(outputdim)
        self.hidden = nn.Linear(self.inputdim, self.hiddendim)
        self.output = nn.Linear(self.hiddendim, self.outputdim)

    def forward(self, x):

        _x = F.relu(self.hidden(x))

        if self.outputdim == 1:
            return F.relu(self.output(_x))

        else:
            return F.relu(self.output(_x))


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


class BetterAttendCompareAggregate(nn.Module):
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
        super(BetterAttendCompareAggregate, self).__init__()

        self.inputdim = inputdim
        self.debug = debug

        self.F = nn.Linear(self.inputdim, self.inputdim, bias=False)
        self.G = nn.Linear(self.inputdim * 2, self.inputdim, bias=False)
        self.H = nn.Linear(self.inputdim * 2, 1, bias=False)

    def forward(self, q, p, qm, pm):

        # Collect some temp macros
        batch_size = q.shape[1]
        seq_length_q, seq_length_p = q.shape[0], p.shape[0]
        if self.debug:
            print("Input:")
            print("\tq:\t", q.shape)
            print("\tp:\t", p.shape)
            print("\tqm:\t", qm.shape)
            print("\tpm:\t", pm.shape)

        # Create att_p, q matrices. We use view to change the input and the output. VIEW IS TESTED DONT PANIC.
        att_q = self.F(q.view(-1, q.shape[2])).view(seq_length_q, batch_size, -1).transpose(1, 0)
        att_p = self.F(p.view(-1, p.shape[2])).view(seq_length_p, batch_size, -1).transpose(1, 0)
        if self.debug:
            print ("\tatt_p:\t", att_p.shape)
            print ("\tatt_q:\t", att_q.shape)

        # Now we calculate e. To do so, we transpose att_q, and BMM it with att_p
        # Note: correspondence held between q->i & p->j in the matrix e.
        e = torch.bmm(att_q, att_p.transpose(2, 1))
        pm = pm.unsqueeze(1).repeat(1, qm.shape[-1], 1)
        qm = qm.unsqueeze(1).repeat(1, pm.shape[-1], 1).transpose(2, 1)
        m = qm * pm
        # Make both masks of the same shape as that of e

        if self.debug:
            print ("\te:\t", e.shape)
            print ("\tqm:\t", qm.shape)
            print ("\tpm:\t", pm.shape)

        # We now prepare softmax_j and softmax_i (as defined in eq above)
        softmax_j = tu.masked_softmax(e.view(-1, e.shape[2]),
                                      m=m.contiguous().view(-1, m.shape[2]),
                                      dim=1).view(-1, e.shape[1],e.shape[2])
        softmax_i = tu.masked_softmax(e.transpose(2, 1).contiguous().view(-1, e.shape[1]),
                                      m=m.transpose(2, 1).contiguous().view(-1, e.shape[1]),
                                      dim=1).view(-1, e.shape[2], e.shape[1])
        #         softmax_j = F.softmax(e.view(-1, e.shape[2]), dim=1)

        #         softmax_i = F.softmax(e.transpose(2, 1).contiguous().view(-1, e.shape[1]), dim=1).transpose(1, 0). \
        #             view(e.shape[1], -1, e.shape[2]).transpose(1, 0)
        if self.debug:
            print ("       softmaxj:\t\b", softmax_j.shape)
            print ("       softmaxi:\t\b", softmax_i.shape)

        beta = torch.bmm(softmax_j, p.transpose(1, 0))
        alpha = torch.bmm(softmax_i, q.transpose(1, 0))
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


class HRBiLSTM(nn.Module):
    """
        ## Improved Relation Detection

        Implementation of the paper here: https://arxiv.org/pdf/1704.06194.pdf.
        In our implementation, we first add then pool instead of the other way round.

        **NOTE: We assume that path encoder's last states are relevant, and we pool from them.**
    """

    def __init__(self, hidden_dim,
                 max_len_ques,
                 max_len_path,
                 embedding_dim,
                 vocab_size,
                 dropout=0.0,
                 vectors=None,
                 debug=False):

        super(HRBiLSTM, self).__init__()

        # Save the parameters locally
        self.max_len_ques = max_len_ques
        self.max_len_path = max_len_path
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.debug = debug

        if vectors is not None:
            self.embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(vectors))
            self.embedding_layer.weight.requires_grad = True
        else:
            # Embedding layer
            self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)

        self.layer1 = nn.LSTM(self.embedding_dim, self.hidden_dim, bidirectional=True, dropout=self.dropout)
        self.layer2 = nn.LSTM(self.hidden_dim * 2, self.hidden_dim, bidirectional=True, dropout=self.dropout)

    def init_hidden(self, batch_size, device):

        return (torch.zeros((2, batch_size, self.hidden_dim), device=device),
                torch.zeros((2, batch_size, self.hidden_dim), device=device))

    def forward(self, ques, path_word, path_rel_1, path_rel_2, _h):
        """
        :params
            :ques: torch.tensor (batch, seq)
            :path_word: torch tensor (batch, seq)
            :path_rel_1: torch.tensor (batch, 1)
            :path_rel_2: torch.tensor (batch, 1)
        """
        batch_size = ques.shape[0]

        # Join two paths into a path rel
        path_rel = torch.cat((path_rel_1, path_rel_2), dim=-1).squeeze()

        if self.debug:
            print("question:\t", ques.shape)
            print("path_word:\t", path_word.shape)
            print("path_rel:\t", path_rel.shape)
            print("hidden_l1:\t", _h[0].shape)

        # Embed all the things!
        q = self.embedding_layer(ques)
        pw = self.embedding_layer(path_word)
        pr = self.embedding_layer(path_rel)

        if self.debug:
            print("\nembedded_q:\t", q.shape)
            print("embedded_pw:\t", pw.shape)
            print("embedded_pr:\t", pr.shape)

        _q, _h2 = self.layer1(q.transpose(1, 0), _h)
        _pw, _ = self.layer1(pw.transpose(1, 0), _h)
        _pr, _ = self.layer1(pr.transpose(1, 0), _h)

        if self.debug:
            print("\nencode_pw:\t", _pw.shape)
            print("encode_pr:\t", _pr.shape)
            print("encode_q:\t", _q.shape)

            # Pass encoded question through another layer
        __q, _ = self.layer2(_q, _h2)
        if self.debug: print("\nencoded__q:\t", __q.shape)

        # Pointwise sum both question representations
        sum_q = _q + __q
        if self.debug: print("\nsum_q:\t\t", sum_q.shape)

        # Pool it along the sequence
        h_q, _ = torch.max(sum_q, dim=0)
        if self.debug: print("\npooled_q:\t", h_q.shape)

        # Now, we pool the last hidden states of _pw and _pr to get h_r
        h_r, _ = torch.max(torch.stack((_pw[-1], _pr[-1]), dim=1), dim=1)
        if self.debug: print("\npooled_p:\t", h_r.shape)

        score = F.cosine_similarity(h_q, h_r)

        return score


class SlotPointer(nn.Module):
    """
        This is an implementation of the model described in our paper (Sec: slot pointer).

        We make certain assumptions namely:
            - only use the last state of paths
            - while calculating energies, we use encoded and not embedded version of the question

    """

    def __init__(self, hidden_dim,
                 max_len_ques,
                 max_len_path,
                 embedding_dim,
                 vocab_size,
                 debug=False):

        super(SlotPointer, self).__init__()

        # Save the parameters locally
        self.embedding_dim = int(embedding_dim)
        self.max_len_ques = int(max_len_ques)
        self.max_len_path = int(max_len_path)
        self.hidden_dim = int(hidden_dim)
        self.vocab_size = int(vocab_size)
        self.debug = debug

        # A dense layer to normalize dimensions
        # self.normalize = nn.Linear(self.embedding_dim, self.hidden_dim * 2, bias=False)

        # Attention parameters
        self.k1 = nn.Parameter(torch.randn((self.hidden_dim * 2,), dtype=torch.float))
        self.k2 = nn.Parameter(torch.randn((self.hidden_dim * 2,), dtype=torch.float))

    @staticmethod
    def compute_emb_mean(emb_path, emb_mask):
        emb_sum = torch.sum(emb_path.transpose(1, 0), dim=1)
        mask_sum = torch.sum(emb_mask, dim=1)
        return emb_sum / mask_sum.unsqueeze(1).repeat(1, emb_sum.shape[1])

    def forward(self, ques_enc, ques_emb, ques_mask, path_1_enc, path_1_emb, path_1_mask, path_2_enc, path_2_emb, path_2_mask):
        """
        :params
            :ques: torch.tensor (batch, seq)
            :path_word: torch tensor (batch, seq)
            :path_rel_1: torch.tensor (batch, 1)
            :path_rel_2: torch.tensor (batch, 1)

            TODO: Put in the mask while calculating mask
        """

        batch_size = ques_enc.shape[1]

        if self.debug:
            print("ques_enc:\t", ques_enc.shape)
            print("ques_emb:\t", ques_emb.shape)
            print("ques_mask:\t", ques_mask.shape)
            print("path_1_enc\t", path_1_enc.shape)
            print("path_2_enc\t", path_1_emb.shape)
            print("path_2_enc\t", path_2_enc.shape)
            print("path_2_emb\t", path_2_emb.shape)

        # Energy. For one path. dot of k and q_T
        e_1 = torch.mv(ques_enc.transpose(1, 0).contiguous().view(-1, ques_enc.shape[-1]), self.k1).view(
            ques_enc.shape[1], ques_enc.shape[0])
        e_2 = torch.mv(ques_enc.transpose(1, 0).contiguous().view(-1, ques_enc.shape[-1]), self.k2).view(
            ques_enc.shape[1], ques_enc.shape[0])

        # Softmax over this axis
        alpha_1 = tu.masked_softmax(e_1, dim=1, m=ques_mask)
        alpha_2 = tu.masked_softmax(e_2, dim=1, m=ques_mask)
        # alpha_1 = F.softmax(e_1, dim=1)
        # alpha_2 = F.softmax(e_2, dim=1)

        # Stack them for ease of use
        #         alpha = torch.stack((alpha_1, alpha_2), dim=1)

        if self.debug:
            print('\nalpha_1:\t', alpha_1.shape)
            print('alpha_2:\t', alpha_2.shape)
            print('sum_input_1:\t', ques_enc.transpose(1, 0).shape)
            print('sum_input_2:\t', self.normalize(ques_emb.transpose(1, 0)).shape)

        # For q, first prepare (q + _q)
        sum_q = ques_enc.transpose(1, 0) + ques_emb.transpose(1, 0)
        q1 = torch.sum(alpha_1.unsqueeze(2).repeat(1, 1, sum_q.shape[2]) * sum_q, dim=1)
        q2 = torch.sum(alpha_2.unsqueeze(2).repeat(1, 1, sum_q.shape[2]) * sum_q, dim=1)
        # q = torch.stack((q1, q2), dim=1)

        if self.debug:
            print('\nsum_q:\t\t', sum_q.shape)
            print('q1:\t\t', q1.shape)
            print('q2:\t\t', q2.shape)
            # print('q:\t\t', q.shape)
            print('p_ input_a:\t', path_1_enc.shape)
            # print('p_ input_b:\t', self.normalize(torch.mean(path_1_emb.transpose(1, 0), dim=1)).shape)

        # for p, we need the last state of encoders, and summed up embeddings.

        p1 = path_1_enc +  self.compute_emb_mean(path_1_emb,path_1_mask)
        p2 = path_2_enc + self.compute_emb_mean(path_2_emb,path_2_mask)
        # p = torch.stack((p1, p2), dim=1)

        if self.debug:
            print('\np1:\t\t', p1.shape)
            print('p2:\t\t', p2.shape)
            # print('p:\t\t', p.shape)
            # print('dot input_a:\t', q.view(-1, q.shape[-1]).shape)
            # print('dot input_b:\t', p.view(-1, p.shape[-1]).shape)
            # print('penultimatesum:\t',
            #       torch.sum(q.view(-1, q.shape[-1]) * p.view(-1, p.shape[-1]), dim=1).view(batch_size, -1).shape)

        # Get the dot of p and q, and add it for both path 1 & 2
        # # Cross check the dot.
        res1 = torch.sum(q1*p1, dim=1)
        res2 = torch.sum(q2*p2, dim=1)
        res = res1 + res2
        # res = torch.sum(torch.sum(q.view(-1, q.shape[-1]) * p.view(-1, p.shape[-1]), dim=1).view(batch_size, -1), dim=1)

        return res


class NotSuchABetterEncoder(nn.Module):
    def __init__(self, max_length, hidden_dim, number_of_layer,
                 embedding_dim, vocab_size, bidirectional,
                 dropout=0.0, mode='LSTM', enable_layer_norm=False,
                 vectors=None, debug=False, residual=False):
        '''
            :param max_length: Max length of the sequence.
            :param hidden_dim: dimension of the output of the LSTM.
            :param number_of_layer: Number of LSTM to be stacked.
            :param embedding_dim: The output dimension of the embedding layer/ important only if vectors=none
            :param vocab_size: Size of vocab / number of rows in embedding matrix
            :param bidirectional: boolean - if true creates BIdir LStm
            :param vectors: embedding matrix
            :param debug: Bool/ prints shapes and some other meta data.
            :param enable_layer_norm: Bool/ layer normalization.
            :param mode: LSTM/GRU.
            :param residual: Bool/ return embedded state of the input.

        TODO: Implement multilayered shit someday.
        '''
        super(NotSuchABetterEncoder, self).__init__()

        self.max_length, self.hidden_dim, self.embedding_dim, self.vocab_size = int(max_length), int(hidden_dim), int(embedding_dim), int(vocab_size)
        self.enable_layer_norm = enable_layer_norm
        self.number_of_layer = number_of_layer
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.debug = debug
        self.mode = mode
        self.residual = residual

        assert self.mode in ['LSTM', 'GRU']

        if vectors is not None:
            self.embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(vectors))
            self.embedding_layer.weight.requires_grad = True
        else:
            # Embedding layer
            self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)

        # Mode
        if self.mode == 'LSTM':
            self.rnn = torch.nn.LSTM(input_size=self.embedding_dim,
                                     hidden_size=self.hidden_dim,
                                     num_layers=1,
                                     bidirectional=self.bidirectional)
        elif self.mode == 'GRU':
            self.rnn = torch.nn.GRU(input_size=self.embedding_dim,
                                    hidden_size=self.hidden_dim,
                                    num_layers=1,
                                    bidirectional=self.bidirectional)
        self.dropout = torch.nn.Dropout(p=self.dropout)
        self.reset_parameters()

    def init_hidden(self, batch_size, device):
        """
            Hidden states to be put in the model as needed.
        :param batch_size: desired batchsize for the hidden
        :param device: torch device
        :return:
        """
        if self.mode == 'LSTM':
            return (torch.ones((2 * 1, batch_size, self.hidden_dim), device=device),
                    torch.ones((2 * 1, batch_size, self.hidden_dim), device=device))
        else:
            return torch.ones((2 * 1, batch_size, self.hidden_dim), device=device)

    def reset_parameters(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

    def forward(self, x, h):
        """

        :param x: input (batch, seq)
        :param h: hiddenstate (depends on mode. see init hidden)
        :param device: torch device
        :return: depends on booleans passed @ init.
        """

        if self.debug:
            print ("\tx:\t", x.shape)
            if self.mode is "LSTM":
                print ("\th[0]:\t", h[0].shape)
            else:
                print ("\th:\t", h.shape)

        mask = tu.compute_mask(x)

        x = self.embedding_layer(x).transpose(0, 1)

        if self.debug: print ("x_emb:\t\t", x.shape)

        if self.enable_layer_norm:
            seq_len, batch, input_size = x.shape
            x = x.view(-1, input_size)
            x = self.layer_norm(x)
            x = x.view(seq_len, batch, input_size)

        if self.debug: print("x_emb bn:\t", x.shape)

        # get sorted v
        lengths = mask.eq(1).long().sum(1)
        lengths_sort, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        x_sort = x.index_select(1, idx_sort)
        h_sort = (h[0].index_select(1, idx_sort), h[1].index_select(1, idx_sort)) \
            if self.mode is "LSTM" else h.index_select(1, idx_sort)

        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x_sort, lengths_sort)
        x_dropout = self.dropout.forward(x_pack.data)
        x_pack_dropout = torch.nn.utils.rnn.PackedSequence(x_dropout, x_pack.batch_sizes)

        if self.debug:
            print("\nidx_sort:", idx_sort.shape)
            print("idx_unsort:", idx_unsort.shape)
            print("x_sort:", x_sort.shape)
            if self.mode is "LSTM":
                print ("h_sort[0]:\t\t", h_sort[0].shape)
            else:
                print ("h_sort:\t\t", h_sort.shape)


        o_pack_dropout, h_sort = self.rnn.forward(x_pack_dropout, h_sort)
        o, _ = torch.nn.utils.rnn.pad_packed_sequence(o_pack_dropout)

        # Unsort o based ont the unsort index we made
        o_unsort = o.index_select(1, idx_unsort)  # Note that here first dim is seq_len
        h_unsort = (h_sort[0].index_select(1, idx_unsort), h_sort[1].index_select(1, idx_unsort)) \
            if self.mode is "LSTM" else h_sort.index_select(1, idx_unsort)


        # @TODO: Do we also unsort h? Does h not change based on the sort?

        if self.debug:
            if self.mode is "LSTM":
                print("h_sort\t\t", h_sort[0].shape)
            else:
                print("h_sort\t\t", h_sort.shape)
            print("o_unsort\t\t", o_unsort.shape)
            if self.mode is "LSTM":
                print("h_unsort\t\t", h_unsort[0].shape)
            else:
                print("h_unsort\t\t", h_unsort.shape)

        # get the last time state
        len_idx = (lengths - 1).view(-1, 1).expand(-1, o_unsort.size(2)).unsqueeze(0)
        o_last = o_unsort.gather(0, len_idx)
        o_last = o_last.squeeze(0)

        if self.debug:
            print("len_idx:\t", len_idx.shape)
            print("o_last:\t", o_last.shape)

        # Need to also return the last embedded state. Wtf. How?

        if self.residual:
            len_idx = (lengths - 1).view(-1, 1).expand(-1, x.size(2)).unsqueeze(0)
            x_last = x.gather(0, len_idx)
            x_last = x_last.squeeze(0)
            return o_unsort, o_last, h_unsort, mask, x, x_last
        else:
            return o_unsort, o_last, h_unsort, mask


class QelosSlotPointer(nn.Module): pass


if __name__ == "__main__":
    max_length = 25
    hidden_dim = 30
    embedding_dim = 300
    vocab_size = 1000,
    bidirectional = True
    batch_size = 10
    device = torch.device('cpu')

    question = torch.randint(1, 1000, (10, 25), device=device, dtype=torch.long)
    question[:5, 20:] = torch.zeros_like(question[:5, 20:])
    question[5:,14:] = torch.zeros_like(question[5:,14:])
    question = question[:,:(question.shape[1] - torch.min(torch.sum(question.eq(0).long(), dim=1))).item()]
    vectors = torch.randn((1000, embedding_dim))

    encoder = NotSuchABetterEncoder(max_length, hidden_dim, 1,
                                    embedding_dim, vocab_size, bidirectional, vectors=vectors).to(device)

    hidden_0 = encoder.init_hidden(question.shape[0], device)
    # hidden_0 = (torch.zeros((2 * 1, question.shape[0], hidden_dim), device=device),
    #                     torch.zeros((2 * 1, question.shape[0], hidden_dim), device=device))
    # hidden_1 = (torch.ones((2 * 1, question.shape[0], hidden_dim), device=device),
    #                     torch.ones((2 * 1, question.shape[0], hidden_dim), device=device))
    # hidden_r = (torch.randint(0,1000,(2 * 1, question.shape[0], hidden_dim), device=device),
    #                     torch.randint(0,1000, (2 * 1, question.shape[0], hidden_dim), device=device))
    out0 = encoder.forward(question, hidden_0, device)
    # out1 = encoder.forward(question, hidden_1, device)[0]
    # outr = encoder.forward(question, hidden_r, device)[0]