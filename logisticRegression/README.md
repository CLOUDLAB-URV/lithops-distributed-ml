This directory contains the following scripts:

- **logisticRegression_serial.py** - serial implementation of the Logistic Regression algorithm based on Gradient Descent

- **logisticRegression_serverful.py** - serverful (distributed) implementation of the serial Logistic Regression algorithm, based on Python Processes via the Multiprocessing API

- **logisticRegression_v1.py** - serverless (distributed) implementation of the serial Logistic Regression algorithm, based on Lithops; this version uses Locks for updating the shared state

- **logisticRegression_v2.py** - serverless (distributed) implementation of the serial Logistic Regression algorithm, based on Lithops; this version is Lock-free

- **logisticRegression_v1_parallel.py** - implementation with inner workers of the algorithm from *logisticRegression_v1.py*

- **logisticRegression_v2_parallel.py** - implementation with inner workers of the algorithm from *logisticRegression_v2.py*

- **logisticRegression_v1_breakdown.py** - implementation which includes the breakdown of execution times for the algorithm from *logisticRegression_v1.py*

- **logisticRegression_v1_parallel_breakdown.py** - implementation which includes the breakdown of execution times for the algorithm from *logisticRegression_v1_parallel.py*

- **logisticRegression_v1_barrier.py** - implementation which includes the breakdown of time spent waiting on a barrier for the algorithm from *logisticRegression_v1.py*


- **experiment_serial.py** - executes *logisticRegression_serial.py* and it outputs the total execution time of the algorithm

- **experiment_serverful.py** - executes *logisticRegression_serverful.py* and it outputs the total execution time of the algorithm

- **experiment_serverless.py** - executes the serverless algorithms of Logistic Regression (version 1 and 2) and it outputs total execution times of the algorithm and of workers

- **experiment_breakdown.py** - executes *logisticRegression_v1_breakdown.py* and it outputs the breakdown and the total execution times of the algorithm and of workers

- **experiment_breakdown_parallel.py** - executes *logisticRegression_v1_parallel_breakdown.py* and it outputs the breakdown and the total execution times of the algorithm and of workers

- **experiment_serial_breakdown.py** - executes *logisticRegression_serial.py* and it outputs the breakdown and the total execution time of the algorithm

- **experiment_barrier.py** - executes *logisticRegression_v1_barrier.py* and it outputs the breakdown of time spent waiting on a barrier


- **validation_serial.py** - validates that the algorithm from *logisticRegression_serial.py* yields correct results by comparing them with the results from scikit-learn

- **validation_serverful.py** - validates that the algorithm from *logisticRegression_serverful.py* yields correct results by comparing them with the results from scikit-learn

- **validation_serverless.py** - validates that the serverless algorithms of Logistic Regression yield correct results by comparing them with the results from scikit-learn


- **utility_service.py** - includes utility functions used by the Logistic Regression algorithms