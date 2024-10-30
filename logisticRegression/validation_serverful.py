import argparse
import numpy as np
from utility_service import get_datasetAndLabels
from sklearn.linear_model import SGDClassifier
from logisticRegression_serverful import LogisticRegression

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--max_iter', type=int, required=True)
    parser.add_argument('--n_features', type=int, required=True)
    parser.add_argument('--n_workers', type=int, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    with open(args.dataset) as f:
       X, y = get_datasetAndLabels(f.read())

    logisticRegression = LogisticRegression(learning_rate=args.learning_rate, max_iter=args.max_iter, n_features=args.n_features, n_workers=args.n_workers)
    logisticRegression.fit(args.dataset)
    predictions1 = logisticRegression.predict(X)

    sgdClassifier = SGDClassifier(loss="log_loss", max_iter=args.max_iter, learning_rate="constant", eta0=args.learning_rate, alpha=0, tol=None)
    sgdClassifier.fit(X, y)
    predictions2 = sgdClassifier.predict(X)

    predictions_equal = np.allclose(predictions1, predictions2)

    print("\nDataset: " + str(args.dataset))
    print("Number workers: " + str(args.n_workers))
    print("Number features: " + str(args.n_features))
    print("Learning rate: " + str(args.learning_rate))
    print("Max iterations: " + str(args.max_iter))
    print("The predictions are equal" if predictions_equal else "The predictions are not equal")