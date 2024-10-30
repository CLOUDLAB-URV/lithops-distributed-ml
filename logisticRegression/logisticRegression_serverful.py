import time
import numpy as np
import multiprocessing as mp
from utility_service import sigmoid, partition_dataset, get_datasetAndLabels

class LogisticRegression:

    def __init__(self, learning_rate=0.001, max_iter=100000, n_features=2, n_workers=1):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.n_features = n_features
        self.n_workers = n_workers
        self.workers = []

    def launch_worker(self, worker_id):
        X = self.partitioned_X[worker_id]
        y = self.partitioned_y[worker_id]
        self.workers[worker_id].set_dataset(X, y)
        return self.workers[worker_id].run()

    def initialize_workers(self):
        for worker_idx in range(self.n_workers):
            worker = Worker(worker_idx, self.learning_rate, self.max_iter, self.weights, self.gradients, self.barrier, self.lock)
            self.workers.append(worker)

    def fit(self, dataset):
        start_time = time.time()

        with open(dataset) as f:
            X, y = get_datasetAndLabels(f.read())
            self.partitioned_X = partition_dataset(X, self.n_workers)
            self.partitioned_y = partition_dataset(y, self.n_workers)

        # shared state
        self.weights = mp.Array('d', [0 for _ in range(self.n_features + 1)])
        self.gradients = mp.Array('d', [0 for _ in range(self.n_features + 1)])
        self.barrier = mp.Barrier(self.n_workers)
        self.lock = mp.Lock()

        self.initialize_workers()

        processes = []
        for process_idx in range(self.n_workers):
            process = mp.Process(target=self.launch_worker, args=(process_idx,))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()
            process.close()

        self.weights = self.weights[:]

        end_time = time.time()
        self.total_duration = end_time - start_time


    def predict(self, X):
        X = np.c_[np.ones((np.shape(X)[0], 1)), X]
        z = np.dot(X, self.weights)
        y_pred = sigmoid(z)
        predictions = [1 if i > 0.5 else 0 for i in y_pred]
        return predictions


class Worker:
    def __init__(self, worker_id, learning_rate, max_iter, weights, gradients, barrier, lock):
        self.worker_id = worker_id
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = weights
        self.gradients = gradients
        self.empty_gradients = [0 for _ in range(len(gradients))]
        self.barrier = barrier
        self.lock = lock

    def set_dataset(self, X, y):
        self.X = np.c_[np.ones((np.shape(X)[0], 1)), X]
        self.y = y

    def run(self):
        print(f"Worker[{self.worker_id}] - execution started")

        for iter in range(self.max_iter):
            print(f"Worker[{self.worker_id}] - started iteration[{iter}]")
            weights_local = self.weights[:]
            z = np.dot(self.X, weights_local)
            y_pred = sigmoid(z)
            errors = y_pred - self.y
            gradients_local = np.dot(self.X.T, errors)

            with self.lock:
                self.gradients[:] += gradients_local

            self.barrier.wait()

            if self.worker_id == 0:
                gradients_total = np.array(self.gradients[:])
                weights_local = weights_local - self.learning_rate * gradients_total
                self.weights[:] = weights_local
                self.gradients[:] = self.empty_gradients

            self.barrier.wait()