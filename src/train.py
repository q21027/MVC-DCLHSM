import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import datetime
import scipy.sparse as sp
from torch import optim
import warnings
import pickle
import os

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.spatial.distance import pdist, squareform
from torch.autograd import Variable
from src.clusteringVisilization import *

from src.Utils_PreProcess import *
from src.dataManager import dataManager
from src.model import *
from src.loss import NormalContrastiveLoss, intraViewContrastiveLoss, interViewContrastiveLoss
from src.clustering import Clustering
from src.Utils_train import *
from src.Utils_PreProcess import *
warnings.filterwarnings("ignore")


class train:
    def __init__(self,args):
        super(train, self).__init__()
        self.args = args


    def update_centers(self):
        for para in self.models[0].parameters():
            para.requires_grad = False
        for para in self.FCMNet.parameters():
            para.requires_grad = False

        p = self.FCMNet(self.embed_fusion)
        p = torch.pow(p, self.m)

        new_centers = torch.zeros([self.dataset.K, self.embed_fusion.shape[1]])
        for i in range(self.dataset.K):
            sum1 = torch.sum(p[:, i])
            sum2 = torch.matmul(p[:, i].view(1,self.embed_fusion.shape[0]), self.embed_fusion)
            new_centers[i, :] = torch.div(sum2, sum1)
        #update
        self.centers = Variable(new_centers)

        return self.centers

    def DEC_train(self):
        self.models[0].eval()


        for para in self.models[0].parameters():
            para.requires_grad = False
        for para in self.FCMNet.parameters():
            para.requires_grad = True

        optimizer = optim.Adam(self.FCMNet.parameters(), lr=self.args.lr)


        for i in range(5):

            optimizer.zero_grad()
            p = self.FCMNet(self.embed_fusion)
            p = torch.pow(p, self.m)

        # loss = self.args.lambda1 * lw(embed1, self.centers, p) + self.args.lambda2 * lb(p, embed.shape[0])
            loss = lw(self.embed_fusion, self.centers, p)
        #     loss =lw(embed, self.centers, p)

            loss.backward()
            optimizer.step()

        return self.FCMNet

    def get_membership(self):
        p = self.FCMNet(self.embed_fusion)
        labels = torch.argmax(p, dim=1).detach().numpy()
        return labels



    def run(self):

        # load data
        self.dataset = dataManager(self.args.dataset)
        self.dataset.printDatasetInfo()
        features, adjs_raw, labels = self.dataset.getData()  #type np,list,np

        self.n_views = len(adjs_raw)
        self.labels = None
        # labels = np.tile(labels, n_views)
        # self.FCMNet = FCMNet(self.args.embed_size, self.dataset.K)

        print("\n-------------------------------Self Loops---------------------------------")
        self_loop = checkSelfLoops(adjs_raw)
        if(self.args.selfLoop == False and self_loop == True):
            adjs = eliminate_self_loops(adjs_raw)
            checkSelfLoops(adjs)
        elif(self.args.selfLoop == True and self_loop == False):
            adjs = add_self_loops(adjs_raw)
            checkSelfLoops(adjs)
        print("--------------------------------------------------------------------------")


        # preprocess
        # features, adjs = graphFilter(self.dataset.dataset_name, features, adjs)
        adjs_mask = []
        ppmis = []
        ppmis1_path = 'ppmis_{}_1.pkl'.format(self.dataset.dataset_name)
        acc1_path = 'acc_{}_1.txt'.format(self.dataset.dataset_name)

        features = torch.FloatTensor(features)

        # 计算节点相似性
        # S = similaruty(features)
        for i in range(self.n_views):
            # A->D^(-0.5)AD^(-0.5)
            adjs[i] = torch.FloatTensor(normalizeAdj(adjs_raw[i]))
            # if(os.path.exists(ppmis1_path)):
            #     with open(ppmis1_path, 'rb') as file:
            #         ppmis = pickle.load(file)
            # else:
            if(self.args.model == "ARE"):
                ppmi = diffusion_fun_improved_ppmi_dynamic_sparsity(adjs[i])
                ppmi= sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(ppmi))
                # ppmis.append(torch.FloatTensor(normalizeAdj(ppmi)))
                ppmis.append(ppmi)
            # adjs[i] = torch.FloatTensor(adjs[i])
            # adjs_mask.append(matrix_to_mask(adjs[i]))



        # train
        self.models = []
        self.paras = []
        embed_fusions = []
        recons = []

        accs = []
        nmis = []
        aris = []
        f1s = []
        losses = []

        for n in range(self.n_views):
            if(self.args.model == "LG_cross"):
                self.models.append(LG_cross(features.shape[1], self.args.embed_size))
            else:
                self.models.append(AdaptiveReceptiveField(features.shape[1], self.args.embed_size))
            self.paras.append({'params':self.models[n].parameters()})

        optimizer = optim.Adam(self.paras, lr=self.args.lr)
        # loss_model = NormalContrastiveLoss(self.n_views)
        loss_model1 = intraViewContrastiveLoss()
        loss_model2 = interViewContrastiveLoss()

        acc_epoch = 0
        nmi_epoch = 0
        ari_epoch = 0
        f1_epoch = 0

        start_time = time.time()
        print("\n-------------------------------Start Train--------------------------------")
        for epoch in range(self.args.epoches):
            # for para in self.models[0].parameters():
            #     para.requires_grad = True
            # for para in self.FCMNet.parameters():
            #     para.requires_grad = False
            # model
            optimizer.zero_grad()
            embed_fusion = 0
            recon_fusion = 0
            embeds = []

            for n in range(self.n_views):
                # print("n:",n)
                # print("features:",features)
                # print("adj:",adjs_mask[n])
                if (self.args.model == "LG_cross"):
                    embed = self.models[n](features, adjs[n])
                else:
                    embed = self.models[n](features, adjs[n], ppmis[n])
                # embed = models[n](features, adjs_mask[n])
                # print("embed:",embed)
                # print("embed:",embed)
                embeds.append(embed)
                embed_fusion += embed
                # recon_fusion += recon

            # embed_fusion = torch.cat(embeds,dim=1)
            # embed_fusion = embed_fusion / self.n_views
            fusionModel = MemoryFusion()
            if(epoch == 0):
                embed_fusion_memory = embed_fusion / self.n_views
            else:
                embed_fusion_memory = embed_fusions[-1]
            embed_fusion = fusionModel(embeds, embed_fusion_memory)
            embed_fusions.append(embed_fusion)


            # loss
            if(self.args.model == "LG_cross" or self.args.model == "ARE"):
                # clustering_loss = kmeans.get_loss(embed_fusion)
            # recon_loss = calReconLosses(features, recons)
            # recon_loss = calReconLoss(features, recon_fusion)
            # print(torch.cat(embeds,dim=0).shape)

                # # Deep Embedding Clustering
                # if(self.args.model == "ARE"):
                #     if(~hasattr(self, 'centers')):
                #         self.centers = Variable(kmeans.u)
                #
                #     self.FCMNet.train()
                #     self.update_centers()
                clustering_loss = get_ClusteringLoss(embed_fusion)

                interContrastive_loss = loss_model1(embeds)
                # intraContrastive_loss = loss_model2(embeds, adjs_raw)
                intraContrastive_loss = loss_model2(embeds, adjs_raw,self.labels)
            # recon_loss = recon_losses(features, recons)
            #     loss = 0.1 * clustering_loss + contrastive_loss
                loss = interContrastive_loss + 0.1 * intraContrastive_loss + 0.1 * clustering_loss
                # loss = interContrastive_loss
            # print("cluster_loss:",clustering_loss)
            # print("recon_loss:",recon_loss)
            # backward
            loss.backward(retain_graph=True)
            optimizer.step()

            losses.append(loss.detach().numpy())

            # kmeans
            kmeans = Clustering(self.dataset.K)
            # kmeans.cluster(torch.cat(embeds,dim=0))
            kmeans.cluster(embed_fusion)
            self.labels = kmeans.M






            # metrics
            acc, nmi, ari, f1 = evaluation(labels, kmeans.M)
            # DEC
            # acc, nmi, ari, f1 = evaluation(labels, self.get_membership())
            if(acc_epoch < acc):
                acc_epoch = acc
                nmi_epoch = nmi
                ari_epoch = ari
                f1_epoch = f1


            accs.append(acc)
            nmis.append(nmi)
            aris.append(ari)
            f1s.append(f1)



            print('\r训练进度: [%-100s]%.2f%%  ' % ('#' * ((epoch + 1) * 100//self.args.epoches), (epoch + 1) * 100//self.args.epoches), end='')
        print("\n-------------------------------Train End---------------------------------")
        # plot
        # print(embeds[0]==embeds[1])
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

        cv = cluster_result_visiualization(embed_fusions[-1].detach().numpy(), self.labels, self.dataset.K)
        cv.main()

        # save data
        if(self.args.model == "ARE"):
            with open(acc1_path, 'r') as file:
                acc = file.read()

            if(acc_epoch > float(acc)):
                with open(acc1_path, 'w') as file:
                    file.write(str(acc_epoch))
                with open(ppmis1_path, 'wb') as file:
                    pickle.dump(ppmis, file)



        print("-------------------------------Result Info--------------------------------")
        print("\033[1;36m ACC: \033[0;30m", acc_epoch)
        print("\033[1;36m NMI: \033[0;30m", nmi_epoch)
        print("\033[1;36m ARI: \033[0;30m", ari_epoch)
        print("\033[1;36m F1: \033[0;30m", f1_epoch)
        print("--------------------------------------------------------------------------")
        # plotLossAndMetrics(losses, accs, nmis, aris, f1s)







