import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.cluster import KMeans


class EntityEncoder(LatentDirichletAllocation):
    def __init__(self, num_topics=10, num_clusters=100, reduce_dim=False, num_lower_dims=None, random_state=0):
        super(EntityEncoder, self).__init__(n_components=num_topics, random_state=random_state)
        self.num_topics = num_topics
        self.num_clusters = num_clusters
        self.reduce_dim = reduce_dim
        self.num_lower_dims = num_lower_dims
        self.random_state = random_state

    def create_corpus(self, X, num_entities, missing_idx, num_clusters, cluster_centers):
        dists = np.linalg.norm(X[:, None] - cluster_centers, axis=2)
        labels = dists.argmin(1)
        labels_onehot = np.zeros((len(labels), num_clusters))
        labels_onehot[np.arange(len(labels)), labels] = 1

        X_words = np.zeros((missing_idx.size, num_clusters))
        X_words[~missing_idx] = labels_onehot
        X_words = X_words.reshape(num_entities, -1, num_clusters)
        X_corpus = X_words.sum(1)
        return X_corpus
    
    def fit(self, X, fit_kwargs):
        num_entities, _, num_features = X.shape
        X_missing = X.reshape(-1, num_features)
        missing_idx = np.isnan(X_missing.sum(1))
        X_flt = X_missing[~missing_idx]

        self.reducer_matrix = np.eye(num_features)
        if self.reduce_dim:
            dim_reducer = TruncatedSVD(n_components=num_features, random_state=self.random_state).fit(X_flt)
            if self.num_lower_dims is None: self.num_lower_dims = dim_reducer.explained_variance_ratio_.cumsum().searchsorted(0.9) + 1
            self.reducer_matrix = dim_reducer.components_[:self.num_lower_dims, :]
            X_flt = np.dot(X_flt, self.reducer_matrix.T)
        
        clusterer = KMeans(n_clusters=self.num_clusters, random_state=self.random_state, init='k-means++', n_init='auto').fit(X_flt)
        self.cluster_centers = clusterer.cluster_centers_
        X_corpus = self.create_corpus(X_flt, num_entities, missing_idx, self.num_clusters, self.cluster_centers)
        self.doc_lengths = X_corpus.sum(1)

        for key, value in fit_kwargs["lda"].items(): setattr(self, key, value)
        super().fit(X_corpus)
        self.lambda_matrix = self.components_
        return self

    def transform(self, X):
        num_entities, _, num_features = X.shape
        X_missing = X.reshape(-1, num_features)
        missing_idx = np.isnan(X_missing.sum(1))
        X_flt = X_missing[~missing_idx]
        if self.reduce_dim: X_flt = np.dot(X_flt, self.reducer_matrix.T)
        X_corpus = self.create_corpus(X_flt, num_entities, missing_idx, self.num_clusters, self.cluster_centers)
        gamma_matrix = self._unnormalized_transform(X_corpus)
        return gamma_matrix