import argparse
from lithops import Storage
from utility_service import get_dataset
from utility_service import get_centroids
from sklearn.cluster import KMeans as KMeansSerial

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--n_workers', type=int, required=True)
    parser.add_argument('--dataset_bucket', type=str, required=True)
    parser.add_argument('--dataset_key', type=str, required=True)
    parser.add_argument('--centroids_init', type=int, required=False)
    parser.add_argument('--ignore_last_column', type=int, required=False)
    parser.add_argument('--n_processes', type=int, required=False)
    args = parser.parse_args()

    centroids_initialized = (args.centroids_init == 1)
    ignore_last_column = (args.ignore_last_column == 1)
    centroids = get_centroids(centroids_initialized)
    dataset_serial = Storage().get_object(args.dataset_bucket, args.dataset_key).decode("utf-8")

    X_serial = get_dataset(dataset_serial, ignore_last_column)
    X_parallel = [f'{args.dataset_bucket}/{args.dataset_key}']

    if centroids_initialized == True:
        kmeans_serial = KMeansSerial(n_clusters=args.k, init=centroids)
    else:
        kmeans_serial = KMeansSerial(n_clusters=args.k)

    kmeans_serial.fit(X_serial)

    if args.algorithm == "kmeans_v1":
        from kmeans_v1 import KMeans as KMeansParallel
        kmeans_parallel = KMeansParallel(n_clusters=args.k, n_workers=args.n_workers, init=centroids, ignore_last_column=ignore_last_column)
    elif args.algorithm == "kmeans_v2":
        from kmeans_v2 import KMeans as KMeansParallel
        kmeans_parallel = KMeansParallel(n_clusters=args.k, n_workers=args.n_workers, init=centroids, ignore_last_column=ignore_last_column)
    elif args.algorithm == "kmeans_v1_parallel":
        from kmeans_v1_parallel import KMeans as KMeansParallel
        kmeans_parallel = KMeansParallel(n_clusters=args.k, n_workers=args.n_workers, n_processes=args.n_processes, init=centroids, ignore_last_column=ignore_last_column)
    elif args.algorithm == "kmeans_v2_parallel":
        from kmeans_v2_parallel import KMeans as KMeansParallel
        kmeans_parallel = KMeansParallel(n_clusters=args.k, n_workers=args.n_workers, n_processes=args.n_processes, init=centroids, ignore_last_column=ignore_last_column)

    kmeans_parallel.fit(X_parallel)

    print("\nAlgorithm: " + args.algorithm)
    print("Number clusters K: " + str(args.k))
    print("Number workers: " + str(args.n_workers))
    print("Number processes: " + str(args.n_processes))
    print("Centroids initialized: " + str(centroids_initialized))
    print("Ignore last column: " + str(ignore_last_column))
    print("Dataset bucket: " + args.dataset_bucket)
    print("Dataset key: " + args.dataset_key)
    print("\nLabels serial: " + str(kmeans_serial.labels_))
    print("Cluster centers serial: " + str(kmeans_serial.cluster_centers_))
    print("\nLabels parallel: " + str(kmeans_parallel.labels_))
    print("Cluster centers parallel: " + str(kmeans_parallel.cluster_centers_))