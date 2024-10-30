import sys
import numpy as np
from sklearn.datasets import make_blobs

if __name__ == "__main__":
    K = int(sys.argv[1])
    n_samples = int(sys.argv[2])
    n_features = int(sys.argv[3])
    X, y = make_blobs(centers=K, n_samples=n_samples, n_features=n_features, shuffle=True, random_state=40)
    NEWLINE_SIZE_IN_BYTES = -1
    with open(f'kmeans_k={K}_samples={n_samples}_features={n_features}.csv', 'wb') as fout:
        np.savetxt(fout, X, delimiter=",")
        fout.seek(NEWLINE_SIZE_IN_BYTES, 2)
        fout.truncate()