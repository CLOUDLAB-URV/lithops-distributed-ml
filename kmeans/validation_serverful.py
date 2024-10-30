import argparse
from utility_service import get_dataset
from utility_service import get_centroids
from sklearn.cluster import KMeans as KMeansSerial
from kmeans_serverful import KMeans as KMeansParallel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--n_workers', type=int, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--centroids_init', type=int, required=False)
    parser.add_argument('--ignore_last_column', type=int, required=False)
    args = parser.parse_args()

    centroids_initialized = (args.centroids_init == 1)
    ignore_last_column = (args.ignore_last_column == 1)
    centroids = get_centroids(centroids_initialized)

    with open(args.dataset) as f:
        X = get_dataset(f.read(), ignore_last_column)

    if centroids_initialized == True:
        kmeans_serial = KMeansSerial(n_clusters=args.k, init=centroids)
    else:
        kmeans_serial = KMeansSerial(n_clusters=args.k)

    kmeans_serial.fit(X)

    kmeans_parallel = KMeansParallel(n_clusters=args.k, n_workers=args.n_workers, init=centroids, ignore_last_column=ignore_last_column)
    kmeans_parallel.fit(args.dataset)

    print("\nNumber clusters K: " + str(args.k))
    print("Number workers: " + str(args.n_workers))
    print("Centroids initialized: " + str(centroids_initialized))
    print("Ignore last column: " + str(ignore_last_column))
    print("Dataset: " + args.dataset)
    print("\nLabels serial: " + str(kmeans_serial.labels_))
    print("Cluster centers serial: " + str(kmeans_serial.cluster_centers_))
    print("\nLabels parallel: " + str(kmeans_parallel.labels_))
    print("Cluster centers parallel: " + str(kmeans_parallel.cluster_centers_))