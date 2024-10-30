This directory contains the following scripts:

- **kmeans_serial.py** - serial implementation of the KMeans algorithm (standard Lloyd's version)

- **kmeans_serverful.py** - serverful (distributed) implementation of the serial KMeans algorithm, based on Python Processes via the Multiprocessing API

- **kmeans_v1.py** - serverless (distributed) implementation of the serial KMeans algorithm, based on Lithops; this version uses Locks for updating the shared state

- **kmeans_v2.py** - serverless (distributed) implementation of the serial KMeans algorithm, based on Lithops; this version is Lock-free

- **kmeans_v1_parallel.py** - implementation with inner workers of the algorithm from *kmeans_v1.py*

- **kmeans_v2_parallel.py** - implementation with inner workers of the algorithm from *kmeans_v2.py*

- **kmeans_serverful_breakdown.py** - implementation which includes the breakdown of execution times for the algorithm from *kmeans_serverful.py*

- **kmeans_v1_breakdown.py** - implementation which includes the breakdown of execution times for the algorithm from *kmeans_v1.py*

- **kmeans_v2_breakdown.py** - implementation which includes the breakdown of execution times for the algorithm from *kmeans_v2.py*

- **kmeans_v1_parallel_breakdown.py** - implementation which includes the breakdown of execution times for the algorithm from *kmeans_v1_parallel.py*

- **kmeans_v1_barrier.py** - implementation which includes the breakdown of time spent waiting on a barrier for the algorithm from *kmeans_v1.py*


- **experiment_serial.py** - executes *kmeans_serial.py* and it outputs the total execution time of the algorithm

- **experiment_serverful.py** - executes *kmeans_serverful.py* and it outputs the total execution time of the algorithm

- **experiment_serverless.py** - executes the serverless algorithms of KMeans (version 1 and 2) and it outputs total execution times of the algorithm and of workers

- **experiment_breakdown.py** - executes *kmeans_v1_breakdown.py* and it outputs the breakdown and the total execution times of the algorithm and of workers

- **experiment_breakdown_parallel.py** - executes *kmeans_v1_parallel_breakdown.py* and it outputs the breakdown and the total execution times of the algorithm and of workers

- **experiment_barrier.py** - executes *kmeans_v1_barrier.py* and it outputs the breakdown of time spent waiting on a barrier


- **validation_serial.py** - validates that the algorithm from *kmeans_serial.py* yields correct results by comparing them with the results from scikit-learn

- **validation_serverful.py** - validates that the algorithm from *kmeans_serverful.py* yields correct results by comparing them with the results from scikit-learn

- **validation_serverless.py** - validates that the serverless algorithms of KMeans yield correct results by comparing them with the results from scikit-learn


- **utility_service.py** - includes utility functions used by the KMeans algorithms