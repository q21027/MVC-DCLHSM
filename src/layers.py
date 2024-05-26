import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

# GCN
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

        self.linear = nn.Linear(in_features, out_features)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input_ =input
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)#torch.mm是矩阵乘法
        output = torch.spmm(adj, support) #spmm是稀疏矩阵乘法，只支持两种形式：1.稀疏矩阵在前，普通矩阵在后2.两个普通矩阵
        output = self.act(output)

        input_res = self.linear(input_)

        output = output + input_res

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

# multi-head-GAT
class Multi_head_GATLayer(Module):
    def __init__(self, in_features, out_features, heads=8, dropout=0.2, act=F.relu,add_skip_connection = False):
        super(Multi_head_GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_rate = dropout
        self.act = act
        self.num_of_heads = heads
        self.add_skip_connection = add_skip_connection


        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.linear_proj = torch.nn.Linear(self.in_features, self.num_of_heads * self.out_features, bias = False)
        # self.lamRes = torch.nn.Parameter(torch.zeros(1))
        # self.lamX = torch.nn.Parameter(torch.ones(1))

        #shape: 1 * NH * Fout
        self.score_of_source = torch.nn.Parameter(torch.Tensor(1,self.num_of_heads, self.out_features))
        self.score_of_target = torch.nn.Parameter(torch.Tensor(1,self.num_of_heads, self.out_features))

        self.skip_proj1 = torch.nn.Linear(self.in_features, int(self.out_features/2), bias=False)
        self.skip_proj2 = torch.nn.Linear(int(self.out_features/2),  self.out_features, bias=False)


        self.leaklyReLU = torch.nn.LeakyReLU(0.2)
        self.softmax = torch.nn.Softmax(dim = -1)
        self.dropout = torch.nn.Dropout(p = self.dropout_rate)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.score_of_source)
        torch.nn.init.xavier_uniform_(self.score_of_target)
        torch.nn.init.xavier_uniform_(self.linear_proj.weight)
        torch.nn.init.xavier_uniform_(self.skip_proj1.weight)
        torch.nn.init.xavier_uniform_(self.skip_proj2.weight)


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

    def forward(self, input_A, adj_mask):
        ################################################################################################################
        # step1. Linear projection + regularization(linear layer)
        ################################################################################################################
        # dropout
        input_A = self.dropout(input_A)

        #regularization
        #input_A_proj.shape & input_B_proj.shape [N, NH, FOUT]
        input_A_proj = self.linear_proj(input_A).view(-1,self.num_of_heads,self.out_features)

        #dropout
        input_A_proj = self.dropout(input_A_proj)

        ################################################################################################################
        # step2. calculate edge attention
        ################################################################################################################
        #cal scores
        #scores_source_A & scores_target_A & scores_source_P & scores_targrt_P    shape:[N, NH, 1]
        scores_source_A = torch.sum((input_A_proj * self.score_of_source),dim=-1, keepdim=True)
        scores_target_A = torch.sum((input_A_proj * self.score_of_target),dim=-1, keepdim=True)


        #source shape:[NH, N, 1]  target shape:[NH. 1 ,N]
        scores_source_A = scores_source_A.transpose(0,1)
        scores_target_A = scores_target_A.permute(1,2,0)

        #shape:[NH, N, N]
        all_scores_A = self.leaklyReLU(scores_source_A + scores_target_A)

        #shaoe: [NE, N, N] For each head, for each node, its neighbours' attention coefficients are regularized.
        all_attention_coeff_A = self.softmax(all_scores_A + adj_mask)

        ################################################################################################################
        #step3. feature aggregation
        ################################################################################################################
        # [NH, N, N] * [NH, N, Fout] -> [NH, N, Fout]
        output_A = torch.bmm(all_attention_coeff_A, input_A_proj.transpose(0,1))

        #shape:[N, NH. Fout]
        output_A = output_A.permute(1,0,2)

        ################################################################################################################
        #step4. residual connection and average
        ################################################################################################################
        output_A = output_A.mean(dim = 1)

        ################################################################################################################
        #step5. activation
        ################################################################################################################

        output_A = self.act(output_A)
        if self.add_skip_connection:
            if self.in_features == self.out_features:
                # output_A = self.lamX * output_A + self.lamRes * input_A
                # output_A = output_A + self.lamRes * input_A
                output_A = output_A + input_A

            else:
                # output_A = self.lamX * output_A + self.lamRes * self.skip_proj2(self.skip_proj1(input_A)).view(-1, self.out_features)
                # output_A = output_A + self.lamRes * self.skip_proj2(self.skip_proj1(input_A)).view(-1, self.out_features)
                output_A = output_A +  self.skip_proj2(self.skip_proj1(input_A)).view(-1, self.out_features)

            # print(self.lamRes)
            res = 0
            # print(res)

        return output_A

#GAT
class GraphAttentionLayer(Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout=0.3, alpha = 0.2, concat=True, res = True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.res = res

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.linear = nn.Linear(in_features,out_features, bias = False)
        torch.nn.init.xavier_uniform_(self.linear.weight)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj_mask):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        # zero_vec = -9e15*torch.ones_like(e)
        # attention = torch.where(adj > 0, e, zero_vec)
        attention = e + adj_mask
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)



        if self.concat:
            out = F.elu(h_prime)
        else:
            out = h_prime

        # if (self.res):
        #     out = h_prime + self.linear(h)
        #     # h_prime = h_prime + self.leakyrelu(self.linear(h))

        return out

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

