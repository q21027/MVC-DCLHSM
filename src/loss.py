import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import diffdist
import torch.distributed as dist

class NormalContrastiveLoss(nn.Module):
    LARGE_NUMBER = 1e4

    def __init__(self, multiplier=2, tau=0.5, distributed=False):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed

    def forward(self, z, get_map=False):
        n = z.shape[0]

        # print("n:",n)
        # print("multiplier:",self.multiplier)
        assert n % self.multiplier == 0

        z = z / np.sqrt(self.tau)

        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # TODO: try to rewrite it with pytorch official tools
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            z_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.multiplier + m])
            z = torch.cat(z_sorted, dim=0)
            n = z.shape[0]

        logits = z @ z.t()

        #把对角线元素设置为负的最大值
        logits[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER

        #归一化取对数
        logprob = F.log_softmax(logits, dim=1)

        # choose all positive objects for an example, for i it would be (i + k * n/m), where k=0...(m-1)
        m = self.multiplier

        #np.repeat(np.arange(n), m)是为了获得每组正对
        #n//m 得到的是每个视图中节点的数目
        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n // m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)

        loss = -logprob[np.repeat(np.arange(n), m - 1), labels].sum() / n / (m - 1)

        return loss


class interViewContrastiveLoss(nn.Module):

    def __init_(self):
        super(intraViewContrastiveLoss, self).__init__()

    def generate_cluster_matrix(self, labels):
        n_samples = len(labels)
        cluster_matrix = np.zeros((n_samples, n_samples), dtype=int)

        # 创建一个与 labels 相同的矩阵，每行都是 labels
        label_matrix = np.tile(labels, (n_samples, 1))

        # 使用向量化操作判断是否相等，相等则为1，不相等则为-1
        cluster_matrix = np.where(label_matrix == label_matrix.T, 1, -1)

        return cluster_matrix
    def forward(self, embeds, adjs, labels=None, centers=None):
        self.largeNum = 1e4
        n_views = len(embeds)

        loss = 0

        for i in range(n_views):
            logits = torch.matmul(embeds[i], embeds[i].t())
            adj = adjs[i].astype('int64')
            adj += np.eye(adj.shape[0]).astype('int64')

            # sims = F.cosine_similarity(embeds[i].unsqueeze(1), embeds[i].unsqueeze(0), dim = -1)
            if(labels is None):

                logprob = F.log_softmax(logits, dim=1)
                row_index = np.nonzero(adj)[0]
                col_index = np.nonzero(adj)[1]
                loss += -torch.sum(logprob[row_index, col_index])
                loss /= adj.shape[0]
            else:
                labels_matrix = self.generate_cluster_matrix(labels)

                indictor_matrix = adj - labels_matrix

                idx_p2n_row = np.where(indictor_matrix==2)[0]
                idx_p2n_col = np.where(indictor_matrix==2)[1]

                idx_rmn_row = np.where(indictor_matrix==-1)[0]
                idx_rmn_col = np.where(indictor_matrix==-1)[1]

                adj[idx_p2n_row, idx_p2n_col] = 0
                logits[idx_p2n_row, idx_p2n_col] = -self.largeNum
                logits[idx_rmn_row, idx_rmn_col] = -self.largeNum

                logprob = F.log_softmax(logits, dim=1)

                row_index = np.nonzero(adj)[0]
                col_index = np.nonzero(adj)[1]
                loss += -torch.sum(logprob[row_index, col_index])
                loss /= adj.shape[0]



        loss /= n_views

        return loss



class intraViewContrastiveLoss(nn.Module):

    def __init__(self):
        super(intraViewContrastiveLoss, self).__init__()

    def forward(self, embeds):

        loss = 0
        n = embeds[0].shape[0]
        n_views = len(embeds)
        for i in range(len(embeds)):
            for j in range(len(embeds)):
                if(~(j==i)):
                    logits = torch.matmul(embeds[i], embeds[j].t())
                    logprob = F.log_softmax(logits, dim=1)
                    sum = -torch.sum(torch.diag(logprob))
                    loss += sum / n


        return loss / (n_views * (n_views - 1))