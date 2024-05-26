import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn import preprocessing

from src.layers import *
from src.Utils_PreProcess import normalizeAdj
from src.randomWalk import diffusion_fun_improved_ppmi_dynamic_sparsity, sparse_mx_to_torch_sparse_tensor

class GCN_model(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(GCN_model, self).__init__()

        self.dim_in = dim_in
        self.out = dim_out

        self.hidden_dim = (dim_in + dim_out) // 2

        self.gcn_layer1 = GraphConvolution(dim_in, self.hidden_dim)
        # self.gcn_layer1 = GraphConvolution(dim_in, dim_out)
        self.gcn_layer2 = GraphConvolution(self.hidden_dim, dim_out)

        # self.gcn_layer3 = GraphConvolution(dim_out, self.hidden_dim)
        self.gcn_layer3 = GraphConvolution(dim_out, dim_in)
        # self.gcn_layer4 = GraphConvolution(self.hidden_dim, dim_in)

        self.linear1 = nn.Linear(dim_in,self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim,dim_out)

    def encode(self, x, adj):

        h = self.gcn_layer1(x, adj)
        embed = self.gcn_layer2(h, adj)
        # embed = self.gcn_layer1(x, adj)
        #
        # h = self.linear1(x)
        # embed = self.linear2(h)

        return embed

    def decode(self, embed, adj):

        # h = self.gcn_layer3(embed, adj)
        # recon = self.gcn_layer4(h, adj)
        recon = self.gcn_layer3(embed,adj)
        return recon


    def forward(self, x, adj):

        embed = self.encode(x, adj)
        # recon = self.decode(embed, adj)

        return embed

class MemoryFusion(nn.Module):
    def __init__(self):
        super(MemoryFusion, self).__init__()
        self.activation = nn.Sigmoid()

    def forward(self, embeds, embed_Oldfusion):
        x_readout = self.activation(torch.Tensor.mean(embed_Oldfusion, dim=0))
        xes = []
        m, n = embeds[0].shape
        # xe_all = torch.zeros(m)
        xe_all = torch.zeros(1)

        re = torch.zeros(m, n)

        for i in range(len(embeds)):
            # xe_i = torch.Tensor.min(torch.matmul(embeds[i], embeds_fusion_readout.T))
            xe_i = torch.matmul(embeds[i], x_readout.T).view(1, -1)
            # xe_i = torch.clamp(xe_i,min = 1, max = 30)
            xes.append(xe_i)

        xe = torch.cat(xes, dim=0)
        xe = F.softmax(xe, dim=0)
        xes = []

        for i in range(len(embeds)):
            xe_i = xe[i, :].squeeze()
            xe_i = torch.Tensor.mean(xe_i)
            xes.append(xe_i)
            xe_all += xe_i
            # xes.append(torch.exp(xe_i))
            # xe_all += torch.exp(xe_i)

        for i in range(len(embeds)):
            alpha = xes[i] / xe_all
            # alpha = torch.diag(alpha)
            re += alpha * embeds[i]
            # re += torch.matmul(alpha, embeds[i])
        return re


class LG_cross(nn.Module):
    def __init__(self, dim_in, dim_out, addRes = True):
        super(LG_cross, self).__init__()

        self.dim_in = dim_in
        self.out = dim_out
        self.addRes = addRes

        self.hidden_dim = (dim_in + dim_out) // 2
        self.hidden_dim2 = self.hidden_dim // 2

        self.gcn_layer1 = GraphConvolution(dim_in, self.hidden_dim)
        self.gcn_layer2 = GraphConvolution(dim_in, self.hidden_dim)
        # self.gcn_layer1 = GraphConvolution(dim_in, dim_out)
        self.gcn_layer3 = GraphConvolution(self.hidden_dim, dim_out)

        # self.gcn_layer3 = GraphConvolution(dim_out, self.hidden_dim)
        self.gcn_layer4 = GraphConvolution(self.hidden_dim, dim_out)
        # self.gcn_layer4 = GraphConvolution(self.hidden_dim, dim_in)

        self.linear_local = nn.Linear(self.hidden_dim,dim_out)
        self.linear_global = nn.Linear(self.hidden_dim,dim_out)

        self.res1 = nn.Linear(dim_in,self.hidden_dim)
        self.res2 = nn.Linear(self.hidden_dim,dim_out)

    def calGlobalAttention(self, x):

        x = x / torch.norm(x, dim=-1, keepdim=True)
        att = torch.mm(x, x.T).numpy()
        att = torch.FloatTensor(normalizeAdj(att))

        return att



    def forward(self, x, adj):
        # global_att = self.calGlobalAttention(x)

        # adj = 0.8 * adj + 0.2 * global_att

        # gnn
        # diffusion_hidden = self.gcn_layer1(x, global_att)
        local_hidden = self.gcn_layer2(x, adj)

        # if (self.addRes):
        #     # diffusion_cross = diffusion_cross + self.linear_global(diffusion_hidden)
        #     local_hidden = local_hidden + self.res1(x)

        # cross
        # diffusion_cross = self.gcn_layer3(diffusion_hidden, 0.5 * adj + 0.5 * global_att)
        local_cross = self.gcn_layer4(local_hidden, adj)

        if(self.addRes):
            # diffusion_cross = diffusion_cross + self.linear_global(diffusion_hidden)
            local_cross = local_cross + self.res2(local_hidden)

        # embed
        # embed = 0.5 * diffusion_cross + 0.5 * local_cross
        embed = local_cross

        return embed

class GAT_model(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(GAT_model, self).__init__()

        self.dim_in = dim_in
        self.out = dim_out

        self.hidden_dim = (dim_in + dim_out) // 2
        h = (dim_in + dim_out) // 4

        # self.gcn_layer1 = GraphAttentionLayer(dim_in, self.hidden_dim + h)
        # self.gcn_layer2 = GraphAttentionLayer(self.hidden_dim + h, self.hidden_dim)
        # self.gcn_layer3 = GraphAttentionLayer(self.hidden_dim, self.hidden_dim - h)
        # self.gcn_layer4 = GraphAttentionLayer(self.hidden_dim - h, self.out)

        self.gcn_layer1 = GraphAttentionLayer(dim_in, self.hidden_dim)
        # self.gcn_layer1 = GraphConvolution(dim_in, dim_out)
        self.gcn_layer2 = GraphAttentionLayer(self.hidden_dim, dim_out)

        # self.gcn_layer3 = GraphConvolution(dim_out, self.hidden_dim)
        # self.gcn_layer3 = GraphConvolution(dim_out, dim_in)
        # self.gcn_layer4 = GraphConvolution(self.hidden_dim, dim_in)

        self.linear1 = nn.Linear(dim_in,self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim,dim_out)

    def encode(self, x, adj):

        h = self.gcn_layer1(x, adj)
        # h1 = self.gcn_layer2(h, adj)
        # h2 = self.gcn_layer3(h1, adj)
        embed = self.gcn_layer2(h, adj)
        # embed = self.gcn_layer1(x, adj)
        #
        # h = self.linear1(x)
        # embed = self.linear2(h)

        return embed

    def decode(self, embed, adj):

        # h = self.gcn_layer3(embed, adj)
        # recon = self.gcn_layer4(h, adj)
        recon = self.gcn_layer3(embed,adj)
        return recon


    def forward(self, x, adj):

        embed = self.encode(x, adj)
        # recon = self.decode(embed, adj)

        return embed

class inter_attentionModule(nn.Module):
    def __init__(self):
        super(inter_attentionModule, self).__init__()
        self.activation = nn.Sigmoid()

    def forward(self, embeds):

        embeds_fusion = embeds[0].clone()
        for i in range(len(embeds)):
            if(i > 0):
                embeds_fusion += embeds[i]

        embeds_fusion = embeds_fusion / len(embeds)
        embeds_fusion_readout = self.activation(torch.Tensor.mean(embeds_fusion, dim = 0))

        xes = []
        m,n = embeds[0].shape
        # xe_all = torch.zeros(m)
        xe_all = torch.zeros(1)

        re = torch.zeros(m,n)

        for i in range(len(embeds)):
            # xe_i = torch.Tensor.min(torch.matmul(embeds[i], embeds_fusion_readout.T))
            xe_i = torch.matmul(embeds[i], embeds_fusion_readout.T)
            xe_i = torch.clamp(xe_i,min = 1, max = 30)
            xe_i = torch.mean(xe_i)
            xes.append(xe_i)
            xe_all += xe_i
            # xes.append(torch.exp(xe_i))
            # xe_all += torch.exp(xe_i)



        for i in range(len(embeds)):
            alpha = xes[i] / xe_all
            # alpha = torch.diag(alpha)
            re += alpha * embeds[i]
            # re += torch.matmul(alpha, embeds[i])
        return re




class intra_attentionModule(nn.Module):
    def __init__(self):
        super(intra_attentionModule, self).__init__()

        self.activation = nn.Sigmoid()

    def forward(self, x1, x2):
        x_fusion = ( x1 + x2 ) / 2
        x_readout = self.activation(torch.Tensor.mean(x_fusion, dim = 0))

        xe_1 = torch.matmul(x1, x_readout.T)
        xe_2 = torch.matmul(x2, x_readout.T)

        xe_1 = torch.clamp(xe_1,min = 1, max = 30)
        xe_2 = torch.clamp(xe_2,min = 1, max = 30)

        xe_1 = torch.Tensor.mean(xe_1)
        xe_2 = torch.Tensor.mean(xe_2)


        # alpha1 = torch.exp(xe_1) / (torch.exp(xe_1) + torch.exp(xe_2))
        # alpha2 = torch.exp(xe_2) / (torch.exp(xe_1) + torch.exp(xe_2))

        alpha1 = xe_1 / (xe_1 + xe_2)
        alpha2 = xe_2 / (xe_1 + xe_2)

        # alpha1 = torch.exp(xe_1) / (torch.exp(xe_1) + torch.exp(xe_2))
        # alpha2 = torch.exp(xe_2) / (torch.exp(xe_1) + torch.exp(xe_2))

        # alpha1 = torch.diag(alpha1)
        # alpha2 = torch.diag(alpha2)

        # return torch.matmul(alpha1, x1) + torch.matmul(alpha2, x2)
        return alpha1 * x1 + alpha2 * x2

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 0)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 1)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 1)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 0)

        return logits

class FCMNet(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(FCMNet, self).__init__()
        step = ( input_dim - output_dim ) // 3
        self.fc1 = nn.Linear(input_dim, input_dim - step)
        self.fc2 = nn.Linear(input_dim - step, input_dim - 2 * step)
        self.fc3 = nn.Linear(input_dim -  2 * step, output_dim)
        self.last = nn.Softmax(dim=1)
        self.reset_parameters()
        # self.k = centers.shape[0]

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)




    def forward(self, embed):
        x = self.fc1(embed)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        p = self.last(x)

        # q = self.T_student(embed, self.u_mean)
        #
        # lw,lb = self.get_loss(embed, p)

        return p

class AdaptiveReceptiveField(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(AdaptiveReceptiveField, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out

        self.hidden_dim = (dim_in + dim_out) // 2

        self.gcn_layer1 = GraphConvolution(dim_in, self.hidden_dim)
        self.gcn_layer2 = GraphConvolution(self.hidden_dim, self.dim_out)
        # self.gcn_layer1 = GraphConvolution(dim_in, self.dim_out)

        self.linear = nn.Linear(dim_in, dim_out)


    def forward(self, x, adj, ppmi):
        x_base1 = self.gcn_layer1(x, adj)
        x_base2 = self.gcn_layer2(x_base1, adj)

        x_rf1 = self.gcn_layer1(x, ppmi)
        x_rf2 = self.gcn_layer2(x_rf1, ppmi)

        attM = intra_attentionModule()
        x_enhanced = attM(x_base2, x_rf2)
        # x_enhanced = (x_base2 + x_rf2) / 2
        # x_enhanced = (x_base1 + x_rf1) / 2

        return x_enhanced

        


