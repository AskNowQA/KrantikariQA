'''

    File contains components like encoder (LSTM layer). CNN model etc.
'''

#Torch related functionalities
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import tensor_utils as tu

# import qelos_core as q

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


# class CNN(nn.Module):
#
#     def __init__(self, _vectors, _vocab_size, _embedding_dim, _output_dim,_debug):
#         super(CNN, self).__init__()
#
#         self.vectors = _vectors
#         self.vocab_size = _vocab_size
#         self.output_dim = _output_dim
#         self.debug = _debug
#
#
#         if self.vectors is not None:
#             self.embedding_dim = self.vectors.shape[1]
#         else:
#             self.embedding_dim = _embedding_dim
#
#
#         self.out_channels = int(self.embedding_dim / 2.0)
#
#         if self.vectors is not None:
#             self.embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(self.vectors))
#             self.embedding_layer.weight.requires_grad = True
#         else:
#             # Embedding layer
#             self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
#
#
#         self.kernel_size_conv1 = [3,4,5]
#         self.kernel_size_max1 = 2
#
#
#         self.conv1 = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=self.embedding_dim,
#                 out_channels=self.out_channels,
#                 kernel_size=self.kernel_size_conv1[0],
#             ),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=self.kernel_size_max1),
#         )
#
#
#         self.conv2 = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=self.embedding_dim,
#                 out_channels=self.out_channels,
#                 kernel_size=self.kernel_size_conv1[1],
#             ),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=self.kernel_size_max1),
#         )
#
#
#         self.conv3 = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=self.embedding_dim,
#                 out_channels=self.out_channels,
#                 kernel_size=self.kernel_size_conv1[2],
#             ),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=self.kernel_size_max1),
#         )
#
#         # self.dropout = nn.Dropout(0.1)
#         # self.output = nn.Linear(self.out_channels * 3 * self.kernel_size_conv1 * self.kernel_size_max1, self.output_dim)
#
#     def forward(self, x):
#
#         x_embedded = self.embedding_layer(x)
#         if self.debug : print("embedded shape is ", x_embedded.shape)
#
#         x_embedded = x_embedded.transpose(2, 1)
#         if self.debug : print("transposed shape is ", x_embedded.shape )
#
#         x_conv1 = self.conv1(x_embedded)
#         if self.debug: print("x_conv1 shape is ,", x_conv1.shape)
#
#         x_conv2 = self.conv2(x_embedded)
#         if self.debug: print("x_conv2 shape is ,", x_conv2.shape)
#
#         x_conv3 = self.conv3(x_embedded)
#         if self.debug: print("x_conv1 shape is ,", x_conv3.shape)
#
#         x_cat = torch.cat((x_conv1, x_conv2, x_conv3), 1)
#         if self.debug: print("concated x shape is ,", x_cat.shape)
#
#         x_flat = x_cat.view(x_cat.size(0), -1)
#         if self.debug: print("flattened x shape is , ", x_flat.shape)
#
#         # output = self.output(x_flat)
#         # if self.debug: print("final output shape is ,", output.shape)
#
#         # x_flat = self.dropout(x_flat)
#
#         return x_flat


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
                 max_length,
                 embedding_dim,
                 vocab_size,
                 dropout=0.5,
                 vectors=None,
                 debug=False,
                 bidirectional=True,
                 number_of_layer=1,
                 mode='LSTM',
                 enable_layer_norm=False,
                 residual=False,
                ):

        super(HRBiLSTM, self).__init__()

        # Save the parameters locally
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.debug = debug
        self.bidirectional = bidirectional
        self.mode = mode
        self.enable_layer_norm = enable_layer_norm
        self.residual = residual
        self.number_of_layer = number_of_layer
        if vectors is not None:
            self.embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(vectors))
            self.embedding_layer.weight.requires_grad = True
        else:
            # Embedding layer
            self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)

        self.layer1 = NotSuchABetterEncoder(
            number_of_layer=self.number_of_layer,
            bidirectional=self.bidirectional,
            embedding_dim=self.embedding_dim,
            max_length=self.max_length,
            hidden_dim=self.hidden_dim,
            vocab_size=self.vocab_size,
            dropout=self.dropout,
            vectors=vectors,
            enable_layer_norm=False,
            mode='LSTM',
            debug=False,
            residual=self.residual)

        self.layer2 = NotSuchABetterEncoder_v2(
            number_of_layer=self.number_of_layer,
            bidirectional=self.bidirectional,
            embedding_dim=self.hidden_dim * 2,
            max_length=self.max_length,
            hidden_dim=self.hidden_dim,
            vocab_size=self.vocab_size,
            dropout=self.dropout,
            vectors=vectors,
            enable_layer_norm=False,
            mode='LSTM',
            debug=self.debug,
            residual=self.residual)

    #         self.layer1 = nn.LSTM(self.embedding_dim, self.hidden_dim, bidirectional=True, dropout=self.dropout)
    #         self.layer2 = nn.LSTM(self.hidden_dim * 2, self.hidden_dim, bidirectional=True, dropout=self.dropout)

    def init_hidden(self, batch_size, device):

        return (torch.zeros((2, batch_size, self.hidden_dim), device=device),
                torch.zeros((2, batch_size, self.hidden_dim), device=device))

    def forward(self, ques, path_word, path_rel_1, path_rel_2, _h):
        """
        :params
            :ques: torch.tensor (batch, seq)
            :path_word: torch tenquessor (batch, seq)
            :path_rel_1: torch.tensor (batch, 1)
            :path_rel_2: torch.tensor (batch, 1)_q
        """
        batch_size = ques.shape[0]

        # Join two paths into a path rel
        # print("***********" , torch.cat((path_rel_1, path_rel_2), dim=-1).shape)
        path_rel = torch.cat((path_rel_1, path_rel_2), dim=-1)

        if self.debug:
            print("question:\t", ques.shape)
            print("path_word:\t", path_word.shape)
            print("path_rel:\t", path_rel.shape)
            print("hidden_l1:\t", _h[0].shape)

        _q, _, hidden_ques, ques_mask = self.layer1(tu.trim(ques), _h)
        _pw, _, hidden_word, pw_mask = self.layer1(tu.trim(path_word), _h)
        _pr, _, _, pr_mask = self.layer1(tu.trim(path_rel), hidden_word)

        # Need to transpose the question befor giving it to

        if self.debug:
            print("\nembedded_and_encoded_q:\t", _q.shape)
            print("eembedded_and_encoded_pw:\t", _pw.shape)
            print("embedded_and_encoded_pr:\t", _pr.shape)
            print("hidden h[0] shape is :\t", hidden_ques[0].shape)
            print("hidden 1 shape is:\t", hidden_ques[1].shape)

        #         _q, _h2 = self.layer1(q.transpose(1, 0), _h)
        #         _pw, _ = self.layer1(pw.transpose(1, 0), _h)
        #         _pr, _ = self.layer1(pr.transpose(1, 0), _h)

        #         if self.debug:
        #             print("\nencode_pw:\t", _pw.shape)
        #             print("encode_pr:\t", _pr.shape)
        #             print("encode_q:\t", _q.shape)

        # Pass encoded question through another layer
        # hidden_ques = self.layer2.init_hidden(_q.shape[0])

        #Multiply mask
        __q, _, _, _ = self.layer2(_q, _h, ques_mask)
        if self.debug: print("\nencoded__q:\t", __q.shape)

        # Pointwise sum both question representations
        sum_q = _q + __q
        if self.debug: print("\nsum_q:\t\t", sum_q.shape)

        # Pool it along the se        quence
        h_q = torch.mean(sum_q, dim=0)
        if self.debug: print("\npooled_q:\t", h_q.shape)

        # Now _pw_pw_pw_pw_pwwe pool the pw and pr across time
        _pw, _ = torch.max(_pw, dim=0)
        _pr, _ = torch.max(_pr, dim=0)

        # Now, we pool the last hidden states of _pw and _pr to get h_r
        h_r = torch.mean(torch.stack((_pw, _pr), dim=1), dim=1)
        if self.debug: print("\npooled_p:\t", h_r.shape)

        # score = F.cosine_similarity(h_q, h_r)
        score = torch.sum(h_q *h_r, -1)

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
            return (torch.ones((1+self.bidirectional , batch_size, self.hidden_dim), device=device),
                    torch.ones((1+self.bidirectional, batch_size, self.hidden_dim), device=device))
        else:
            return torch.ones((1+self.bidirectional, batch_size, self.hidden_dim), device=device)

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

        len_idx = (lengths - 1).view(-1, 1).expand(-1, o_unsort.size(2)).unsqueeze(0)

        if self.debug:
            print("len_idx:\t", len_idx.shape)

        # Need to also return the last embedded state. Wtf. How?

        if self.residual:
            len_idx = (lengths - 1).view(-1, 1).expand(-1, x.size(2)).unsqueeze(0)
            x_last = x.gather(0, len_idx)
            x_last = x_last.squeeze(0)
            return o_unsort, h_unsort[0].transpose(1,0).contiguous().view(h_unsort[0].shape[1], -1) , h_unsort, mask, x, x_last
        else:
            return o_unsort, h_unsort[0].transpose(1,0).contiguous().view(h_unsort[0].shape[1], -1) , h_unsort, mask


class QelosFlatEncoder(nn.Module):
    def __init__(self, max_length, hidden_dim, number_of_layer,
                 embedding_dim, vocab_size, bidirectional, device,
                 dropout=0.0, mode='LSTM', enable_layer_norm=False,
                 vectors=None, residual=False, dropout_in=0., dropout_rec=0, debug=False,encoder=False):
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
        super(QelosFlatEncoder, self).__init__()

        self.max_length, self.hidden_dim, self.embedding_dim, self.vocab_size = \
            int(max_length), int(hidden_dim), int(embedding_dim), int(vocab_size)
        self.enable_layer_norm = enable_layer_norm
        self.number_of_layer = number_of_layer
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.dropout_in, self.dropout_rec = dropout_in, dropout_rec
        self.debug = debug
        self.mode = mode
        self.residual = residual
        self.device = device


        # if vectors is not None:
        #     self.embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(vectors))
        #     self.embedding_layer.weight.requires_grad = True
        # else:
        #     self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)

        if encoder:
            self.lstm = encoder
        else:
            self.lstm = NotSuchABetterEncoder(
                number_of_layer=self.number_of_layer,
                bidirectional=self.bidirectional,
                embedding_dim=self.embedding_dim,
                max_length = self.max_length,
                hidden_dim=self.hidden_dim,
                vocab_size=self.vocab_size,
                dropout=self.dropout,
                vectors=vectors,
                enable_layer_norm=False,
                mode = 'LSTM',
                debug = self.debug,
                residual=self.residual)

        self.adapt_lin = None   # Make layer if dims mismatch
        if residual and self.hidden_dim*2 != self.embedding_dim:
            self.adapt_lin = torch.nn.Linear(self.embedding_dim, self.hidden_dim*2, bias=False)

    def forward(self, x):
        # embs = self.embedding_layer(x)
        # mask = tu.compute_mask(x)

        h = self.lstm.init_hidden(x.shape[0],self.device)

        if self.residual:
            _, final_state, _, mask, embs, _ = self.lstm(x, h)
        else:
            _, final_state, _, mask = self.lstm(x, h)
        # final_state = self.lstm.y_n[-1]
        final_state = final_state.contiguous().view(x.size(0), -1)

        # if self.residual:
        #     if self.adapt_lin is not None:
        #         embs = self.adapt_lin(embs)
        #     meanpool = embs.sum(0)
        #     masksum = mask.float().sum(1).unsqueeze(1)
        #     meanpool = meanpool / masksum
        #     final_state = final_state + meanpool
        return final_state


class QelosSlotPtrChainEncoder(nn.Module):
    def __init__(self, max_length, hidden_dim, number_of_layer,
                 embedding_dim, vocab_size, bidirectional, device,
                 dropout=0.0, mode='LSTM', enable_layer_norm=False,
                 vectors=None, residual=False, dropout_in=0., dropout_rec=0,debug=False,encoder=False):
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
        super(QelosSlotPtrChainEncoder, self).__init__()

        self.max_length, self.hidden_dim, self.embedding_dim, self.vocab_size = \
            int(max_length), int(hidden_dim), int(embedding_dim), int(vocab_size)
        self.enable_layer_norm = enable_layer_norm
        self.number_of_layer = number_of_layer
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.dropout_in, self.dropout_rec = dropout_in, dropout_rec
        self.debug = debug
        self.mode = mode
        self.residual = residual
        self.device = device

        self.enc = QelosFlatEncoder(max_length, hidden_dim, number_of_layer,
                 embedding_dim, vocab_size, bidirectional, device, dropout=0.5, mode='LSTM',
                 enable_layer_norm=False, vectors=vectors, residual=self.residual,
                 dropout_in=self.dropout_in, dropout_rec=self.dropout_rec, debug=False,encoder=encoder)#.to(device)

    def forward(self, firstrels, secondrels):
        firstrels_enc = self.enc(firstrels)
        secondrels_enc = self.enc(secondrels)
        # cat???? # TODO
        enc = torch.cat([firstrels_enc, secondrels_enc], 1)
        return enc


class QelosSlotPtrQuestionEncoder(nn.Module):
    # TODO: (1) skip connection, (2) two outputs (summaries weighted by forwards)
    def __init__(self, max_length, hidden_dim, number_of_layer,
                 embedding_dim, vocab_size, bidirectional, device,
                 dropout=0.0, mode='LSTM', enable_layer_norm=False,
                 vectors=None, residual=True, dropout_in=0., dropout_rec=0, debug=False):

        super(QelosSlotPtrQuestionEncoder, self).__init__()
        self.max_length, self.hidden_dim, self.embedding_dim, self.vocab_size = \
            int(max_length), int(hidden_dim), int(embedding_dim), int(vocab_size)
        self.enable_layer_norm = enable_layer_norm
        self.number_of_layer = number_of_layer
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.dropout_in, self.dropout_rec = dropout_in, dropout_rec
        self.debug = debug
        self.mode = mode
        self.residual = residual
        self.device = device

        if vectors is not None:
            self.embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(vectors))
            self.embedding_layer.weight.requires_grad = True
        else:
            self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)

        #self.lstm = q.FastestLSTMEncoder(self.embedding_dim, self.hidden_dim, bidir=self.bidirectional,
        #                                  dropout_in=self.dropout_in, dropout_rec=self.dropout_rec)

        self.lstm = NotSuchABetterEncoder(
            number_of_layer=self.number_of_layer,
            bidirectional=self.bidirectional,
            embedding_dim=self.embedding_dim,
            max_length=self.max_length,
            hidden_dim=self.hidden_dim,
            vocab_size=self.vocab_size,
            dropout=self.dropout,
            vectors=vectors,
            enable_layer_norm=False,
            mode='LSTM',
            debug=self.debug,
            residual=self.residual)


        dims = [self.hidden_dim]
        self.linear = torch.nn.Linear(dims[-1] * (1+self.bidirectional), 2)
        self.sm = torch.nn.Softmax(1)
        # for adapter
        outdim = dims[-1] * (1+self.bidirectional)
        self.adapt_lin = None

        if outdim != self.embedding_dim:
            self.adapt_lin = torch.nn.Linear(self.embedding_dim, outdim, bias=False)

    def return_encoder(self):
        return self.lstm
    def forward(self, x):
        # embs = self.embedding_layer(x)
        # mask = tu.compute_mask(x)

        h = self.lstm.init_hidden(x.shape[0], self.device)

        if self.residual:
            ys, final_state, _, mask, embs, _ = self.lstm(x, h)
        else:
            ys, final_state, _, mask = self.lstm(x, h)

        ys = ys.transpose(1,0)
        # ys = self.lstm(embs, mask=mask)

        # final_state = final_state.contiguous().view(x.size(0), -1)


        # get attention scores
        scores = self.linear(ys)
        # s1 = scores
        scores = scores + torch.log(mask[:, :ys.size(1)].float().unsqueeze(2))
        scores = self.sm(scores)  # (batsize, seqlen, 2)
        # get summaries
        # region skipper
        skipadd = embs[:, :ys.size(1), :]
        if self.adapt_lin is not None:
            skipadd = self.adapt_lin(skipadd)
        if not self.residual:
            ys = ys + skipadd
        # endregion
        ys = ys.unsqueeze(2)  # (batsize, seqlen, 1, dim)
        scores = scores.unsqueeze(3)  # (batsize, seqlen, 2, 1)
        b = ys * scores  # (batsize, seqlen, 2, dim)
        summaries = b.sum(1)  # (batsize, 2, dim)
        ret = torch.cat([summaries[:, 0, :], summaries[:, 1, :]], 1)
        return ret,scores

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


        self.kernel_size_conv1 = [3,4,5]
        self.kernel_size_max1 = 2

        Ci = 1
        Co = 50
        Ks = [3,4,5]
        D = 300
        C = self.output_dim
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(len(Ks) * Co, C)

        # self.dropout = nn.Dropout(0.1)
        # self.output = nn.Linear(self.out_channels * 3 * self.kernel_size_conv1 * self.kernel_size_max1, self.output_dim)

        def conv_and_pool(self, x, conv):
            x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
            x = F.max_pool1d(x, x.size(2)).squeeze(2)
            return x

    def forward(self, x):

        x_embedded = self.embedding_layer(x)
        if self.debug : print("embedded shape is ", x_embedded.shape)

        x_embedded = x_embedded.unsqueeze(1)
        if self.debug : print("transposed shape is ", x_embedded.shape )

        x_embedded = [F.relu(conv(x_embedded)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x_embedded = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x_embedded]  # [(N, Co), ...]*len(Ks)

        x_embedded = torch.cat(x_embedded, 1)

        x_embedded = self.dropout(x_embedded)

        # logit = self.fc1(x_embedded)  # (N, C)
        logit = self.fc1(x_embedded)  # (N, C)

        return logit

class NotSuchABetterEncoder_v2(nn.Module):
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
        super(NotSuchABetterEncoder_v2, self).__init__()

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
            return (torch.ones((1+self.bidirectional , batch_size, self.hidden_dim), device=device),
                    torch.ones((1+self.bidirectional, batch_size, self.hidden_dim), device=device))
        else:
            return torch.ones((1+self.bidirectional, batch_size, self.hidden_dim), device=device)

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

    def forward(self, x, h,mask):
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

#         mask = tu.compute_mask(x)

#         x = self.embedding_layer(x).transpose(0, 1)

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

#         print("YABADABADOOO")
#         print(x_pack_dropout.shape, h_sort[0].shape)
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

        len_idx = (lengths - 1).view(-1, 1).expand(-1, o_unsort.size(2)).unsqueeze(0)

        if self.debug:
            print("len_idx:\t", len_idx.shape)

        # Need to also return the last embedded state. Wtf. How?

        if self.residual:
            len_idx = (lengths - 1).view(-1, 1).expand(-1, x.size(2)).unsqueeze(0)
            x_last = x.gather(0, len_idx)
            x_last = x_last.squeeze(0)
            return o_unsort, h_unsort[0].transpose(1,0).contiguous().view(h_unsort[0].shape[1], -1) , h_unsort, mask, x, x_last
        else:
            return o_unsort, h_unsort[0].transpose(1,0).contiguous().view(h_unsort[0].shape[1], -1) , h_unsort, mask



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


# def file_reader(file_number):
#     return pickle.load(open(str(file_number) + '/model_info.pickle','rb'),encoding='bytes')