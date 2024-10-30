import argparse
import numpy as np
from utility_service import get_dataset
from utility_service import get_centroids
from kmeans_serial import KMeans as KMeansSerial
from sklearn.cluster import KMeans as KMeansSerialSklearn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--centroids_init', type=int, required=False)
    parser.add_argument('--ignore_last_column', type=int, required=False)
    args = parser.parse_args()

    centroids_initialized = (args.centroids_init == 1)
    ignore_last_column = (args.ignore_last_column == 1)
    centroids = get_centroids(centroids_initialized)

    with open(args.dataset) as f:
        X = get_dataset(f.read(), ignore_last_column)

    kmeans_serial = KMeansSerial(n_clusters=args.k, init=centroids).fit(X)
    if centroids_initialized == True:
        kmeans_serial_sklearn = KMeansSerialSklearn(n_clusters=args.k, init=centroids).fit(X)
    else:
        kmeans_serial_sklearn = KMeansSerialSklearn(n_clusters=args.k).fit(X)

    labels_equal = np.allclose(kmeans_serial.labels_, kmeans_serial_sklearn.labels_)
    cluster_centers_equal = np.allclose(kmeans_serial.cluster_centers_, kmeans_serial_sklearn.cluster_centers_)

    print("Dataset X: " + args.dataset)
    print("Number clusters K: " + str(args.k))
    print("Centroids initialized: " + str(centroids_initialized))
    print("Ignore last column: " + str(ignore_last_column))
    print("The labels are equal" if labels_equal else "The labels are not equal")
    print("The cluster centers are equal" if cluster_centers_equal else "The cluster centers are not equal")
    print("\nLabels serial: " + str(kmeans_serial.labels_))
    print("Cluster centers serial: " + str(kmeans_serial.cluster_centers_))
    print("Labels serial sklearn: " + str(kmeans_serial_sklearn.labels_))
    print("Cluster centers serial sklearn: " + str(kmeans_serial_sklearn.cluster_centers_))