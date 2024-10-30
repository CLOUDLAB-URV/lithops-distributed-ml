import argparse
import time
from utility_service import get_datasetAndLabels
from logisticRegression_serial import LogisticRegression

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--max_iter', type=int, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    breakdown = {}

    start_time = time.time()

    loadDataset_start = time.time()
    with open(args.dataset) as f:
       X,y = get_datasetAndLabels(f.read())
    loadDataset_end = time.time()
    breakdown["loadDataset"] = (loadDataset_end - loadDataset_start)

    logisticRegression = LogisticRegression(learning_rate=args.learning_rate, max_iter=args.max_iter)

    compute_start = time.time()
    logisticRegression.fit(X, y)
    compute_end = time.time()
    breakdown["compute"] = (compute_end - compute_start)

    end_time = time.time()
    execution_time = end_time - start_time

    print("Dataset: " + args.dataset)
    print("Learning rate: " + str(args.learning_rate))
    print("Max iterations: " + str(args.max_iter))
    print("Total duration serial: " + str(execution_time))
    print("Breakdown: " + str(breakdown))