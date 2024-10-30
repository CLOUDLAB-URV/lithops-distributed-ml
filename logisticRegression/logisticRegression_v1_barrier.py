import time
import numpy as np
from lithops import FunctionExecutor
from lithops.multiprocessing import Array, Barrier, Lock
from utility_service import sigmoid
from utility_service import get_datasetAndLabels


class LogisticRegression:

    def __init__(self, learning_rate=0.001, max_iter=100000, n_features=2, n_workers=1):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.n_features = n_features
        self.n_workers = n_workers
        self.workers = []

    def launch_worker(self, obj):
        worker_id = obj.part - 1
        X, y = get_datasetAndLabels(obj.data_stream.read().decode("utf-8"))
        self.workers[worker_id].set_dataset(X, y)
        return self.workers[worker_id].run()

    def initialize_workers(self):
        for worker_idx in range(self.n_workers):
            worker = Worker(worker_idx, self.learning_rate, self.max_iter, self.weights, self.gradients, self.barrier, self.lock)
            self.workers.append(worker)

    def fit(self, dataset):
        start_time = time.time()

        # shared state
        self.weights = Array('d', [0 for _ in range(self.n_features + 1)])
        self.gradients = Array('d', [0 for _ in range(self.n_features + 1)])
        self.barrier = Barrier(self.n_workers)
        self.lock = Lock()

        self.initialize_workers()

        fexec = FunctionExecutor()
        fexec.map(self.launch_worker, dataset, obj_chunk_number=self.n_workers)
        workers_results = fexec.get_result()
        self.weights = self.weights[:]

        end_time = time.time()
        self.total_duration = end_time - start_time
        self.barrier_duration = sum(workers_results) / len(workers_results)


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

        total_times_barrier = []

        for iter in range(self.max_iter):
            print(f"Worker[{self.worker_id}] - started iteration[{iter}]")
            weights_local = self.weights[:]
            z = np.dot(self.X, weights_local)
            y_pred = sigmoid(z)
            errors = y_pred - self.y
            gradients_local = np.dot(self.X.T, errors)

            with self.lock:
                self.gradients[:] += gradients_local

            start_time_barrier = time.time()
            self.barrier.wait()
            end_time_barrier = time.time()
            total_time_barrier = end_time_barrier - start_time_barrier
            total_times_barrier.append(total_time_barrier)

            if self.worker_id == 0:
                gradients_total = np.array(self.gradients[:])
                weights_local = weights_local - self.learning_rate * gradients_total
                self.weights[:] = weights_local
                self.gradients[:] = self.empty_gradients

            self.barrier.wait()

        self.barrier_duration = sum(total_times_barrier) / len(total_times_barrier)

        return self.barrier_duration