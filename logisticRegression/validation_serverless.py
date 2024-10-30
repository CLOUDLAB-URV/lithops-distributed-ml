import argparse
import numpy as np
from utility_service import get_datasetAndLabels
from sklearn.linear_model import SGDClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--max_iter', type=int, required=True)
    parser.add_argument('--n_features', type=int, required=True)
    parser.add_argument('--n_workers', type=int, required=True)
    parser.add_argument('--dataset_serial', type=str, required=True)
    parser.add_argument('--dataset_parallel', type=str, required=True)
    parser.add_argument('--n_processes', type=int, required=False)
    args = parser.parse_args()

    with open(args.dataset_serial) as f:
       X_serial, y_serial = get_datasetAndLabels(f.read())

    if args.algorithm == "logisticRegression_v1":
        from logisticRegression_v1 import LogisticRegression
        logisticRegression = LogisticRegression(learning_rate=args.learning_rate, max_iter=args.max_iter, n_features=args.n_features, n_workers=args.n_workers)
    elif args.algorithm == "logisticRegression_v2":
        from logisticRegression_v2 import LogisticRegression
        logisticRegression = LogisticRegression(learning_rate=args.learning_rate, max_iter=args.max_iter, n_features=args.n_features, n_workers=args.n_workers)
    elif args.algorithm == "logisticRegression_v1_parallel":
        from logisticRegression_v1_parallel import LogisticRegression
        logisticRegression = LogisticRegression(learning_rate=args.learning_rate, max_iter=args.max_iter, n_features=args.n_features, n_workers=args.n_workers, n_processes=args.n_processes)
    elif args.algorithm == "logisticRegression_v2_parallel":
        from logisticRegression_v2_parallel import LogisticRegression
        logisticRegression = LogisticRegression(learning_rate=args.learning_rate, max_iter=args.max_iter, n_features=args.n_features, n_workers=args.n_workers, n_processes=args.n_processes)

    logisticRegression.fit([args.dataset_parallel])
    predictions1 = logisticRegression.predict(X_serial)

    sgdClassifier = SGDClassifier(loss="log_loss", max_iter=args.max_iter, learning_rate="constant", eta0=args.learning_rate, alpha=0, tol=None)
    sgdClassifier.fit(X_serial, y_serial)
    predictions2 = sgdClassifier.predict(X_serial)

    predictions_equal = np.allclose(predictions1, predictions2)

    print("\nAlgorithm: " + args.algorithm)
    print("Dataset serial: " + args.dataset_serial)
    print("Dataset parallel: " + str(args.dataset_parallel))
    print("Number workers: " + str(args.n_workers))
    print("Number processes: " + str(args.n_processes))
    print("Number features: " + str(args.n_features))
    print("Learning rate: " + str(args.learning_rate))
    print("Max iterations: " + str(args.max_iter))
    print("The predictions are equal" if predictions_equal else "The predictions are not equal")