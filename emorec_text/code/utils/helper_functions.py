from hdbscan.hdbscan_ import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster._kmeans import KMeans


def cluster_hdbscan(arr,
                    min_cluster_size=10,
                    min_samples=1,
                    cluster_selection_epsilon=0.1):
    labels = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=cluster_selection_epsilon).fit_predict(arr)
    return labels


def cluster_kmeans(arr,
                   seed=0,
                   n_clusters=2):
    labels = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(arr)
    return labels


def embed_tsne(x,
               seed=0,
               perplexity=30,
               n_components=2):
    tsne = TSNE(n_components=n_components, verbose=0, random_state=seed, perplexity=perplexity, n_iter=2000)
    z = tsne.fit_transform(x)

    return z


def embed_pca(x,
              seed=0,
              n_components=2):
    pca = PCA(n_components=n_components, random_state=seed)
    z = pca.fit_transform(x)

    return z