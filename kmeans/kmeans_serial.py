import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, init=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init = init

    def fit(self, X):
        self.X = X
        self.labels_ = np.zeros([len(X)], dtype=int)

        # set random initial cluster centers (centroids)
        if self.init is None:
            initial_cluster_centers_idx = np.random.choice(len(self.X), self.n_clusters, replace=False)
            initial_cluster_centers = [self.X[idx] for idx in initial_cluster_centers_idx]
            self.cluster_centers_ = initial_cluster_centers
        else:
            self.cluster_centers_ = self.init

        clusters_counters = [0 for _ in range(self.n_clusters)]
        clusters_totals = [0 for _ in range(self.n_clusters)]

        for iter in range(self.max_iter):
            print(f"Started iteration[{iter}]")
            cluster_centers_old = self.cluster_centers_[:]

            for x_idx, x_value in enumerate(self.X):
                cluster_idx = np.argmin(((cluster_centers_old - x_value) ** 2).sum(axis=1))
                self.labels_[x_idx] = cluster_idx
                clusters_counters[cluster_idx] += 1
                clusters_totals[cluster_idx] += x_value

            cluster_centers_new = []
            for cluster_idx in range(self.n_clusters):
                if clusters_counters[cluster_idx] > 0:
                    cluster_centers_new.append(clusters_totals[cluster_idx] / clusters_counters[cluster_idx])
                else:
                    cluster_centers_new.append(cluster_centers_old[cluster_idx])
                self.cluster_centers_[cluster_idx] = cluster_centers_new[cluster_idx]
                clusters_counters[cluster_idx] = 0
                clusters_totals[cluster_idx] = 0

            if np.allclose(cluster_centers_new, cluster_centers_old):
                break

        return self