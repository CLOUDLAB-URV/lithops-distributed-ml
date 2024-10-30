import argparse
import time
from kmeans_serial import KMeans
from utility_service import get_dataset
from utility_service import get_centroids

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

    start_time = time.time()

    with open(args.dataset) as f:
        X = get_dataset(f.read(), ignore_last_column)

    kmeans = KMeans(n_clusters=args.k, init=centroids).fit(X)

    end_time = time.time()
    execution_time = end_time - start_time

    print("Dataset X: " + args.dataset)
    print("Number clusters K: " + str(args.k))
    print("Centroids initialized: " + str(centroids_initialized))
    print("Ignore last column: " + str(ignore_last_column))
    print("Total duration serial: " + str(execution_time))
    print("Centroids: " + str(kmeans.cluster_centers_))