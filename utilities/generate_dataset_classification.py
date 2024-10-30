import sys
import numpy as np
from sklearn.datasets import make_classification

if __name__ == "__main__":
    n_samples = int(sys.argv[1])

    X, y = make_classification(n_samples=n_samples, n_features=5, n_informative=2, n_classes=2, random_state=40)
    dataset = np.column_stack((X, y))

    NEWLINE_SIZE_IN_BYTES = -1
    with open(f'lgr_samples={n_samples}.csv', 'wb') as fout:
        np.savetxt(fout, dataset, delimiter=",")
        fout.seek(NEWLINE_SIZE_IN_BYTES, 2)
        fout.truncate()