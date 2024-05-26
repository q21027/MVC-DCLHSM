import torch

from sklearn.cluster import KMeans


class Clustering(object):

    def __init__(self, K, n_init=20):
        self.K = K
        self.n_init = n_init


        self.u = None  # centers
        self.M = None  # labels

    def cluster(self, embed):
        embed_np = embed.detach().cpu().numpy()
        clustering = KMeans(n_clusters=self.K, n_init=self.n_init, max_iter=500)
        clustering.fit(embed_np)

        self.M = clustering.labels_
        self.u = torch.from_numpy(clustering.cluster_centers_)


    def get_loss(self, embed):  # loss = sigma(|xhat-x|^2)
        loss = torch.Tensor([0.])
        # TODO: This may be slightly inefficient, we can fix it later to use matrix multiplications
        for i, clusteridx in enumerate(self.M):
            x = embed[i]
            c = self.u[clusteridx]
            difference = x - c
            err = torch.sum(torch.mul(difference, difference))  # torch.mul is element-mul
            loss += err

        return loss / len(self.M)

    def get_membership(self):
        return self.M

    def _compute_centers(self, labels, embed):
        """
        sklearn kmeans may not give accurate cluster centers in some cases (see doc), so we compute ourselves
        """
        clusters = {}
        for i, lbl in enumerate(labels):
            if clusters.get(lbl) is None:
                clusters[lbl] = []
            clusters[lbl].append(torch.FloatTensor(embed[i]))

        centers = {}
        for k in clusters:
            all_embed = torch.stack(clusters[k])
            center = torch.mean(all_embed, 0)
            centers[k] = center

        return centers

    # def show(self):


