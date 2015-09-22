import re, gzip, pickle, time
from multiprocessing import Queue, Lock
import threading
import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})

from .glove_inner import train_glove

class Glove(object):
    def __init__(self, cooccurence, alpha=0.75, x_max=100.0, d=50, seed=1234):
        """
        Glove model for obtaining dense embeddings from a
        co-occurence (sparse) matrix.
        """
        self.alpha           = alpha
        self.x_max           = x_max
        self.d               = d
        self.cooccurence     = cooccurence
        self.seed            = seed
        np.random.seed(seed)
        self.W               = np.random.uniform(-0.5/d, 0.5/d, (len(cooccurence), d)).astype(np.float64)
        self.ContextW        = np.random.uniform(-0.5/d, 0.5/d, (len(cooccurence), d)).astype(np.float64)
        self.b               = np.random.uniform(-0.5/d, 0.5/d, (len(cooccurence), 1)).astype(np.float64)
        self.ContextB        = np.random.uniform(-0.5/d, 0.5/d, (len(cooccurence), 1)).astype(np.float64)
        self.gradsqW         = np.ones_like(self.W, dtype=np.float64)
        self.gradsqContextW  = np.ones_like(self.ContextW, dtype=np.float64)
        self.gradsqb         = np.ones_like(self.b, dtype=np.float64)
        self.gradsqContextB  = np.ones_like(self.ContextB, dtype=np.float64)

    def train(self, step_size=0.05, workers = 9, batch_size=50, verbose=False):
        jobs = Queue(maxsize=2 * workers)
        lock = threading.Lock()  # for shared state (=number of words trained so far, log reports...)
        total_error = [0.0]
        total_done  = [0]

        total_els = 0
        for key in self.cooccurence:
            for subkey in self.cooccurence[key]:
                total_els += 1

        # Worker function:
        def worker_train():
            error = np.zeros(1, dtype = np.float64)
            while True:
                job = jobs.get()
                if job is None:  # data finished, exit
                    break
                train_glove(self, job, step_size, error)
                with lock:
                    total_error[0] += error[0]
                    total_done[0] += len(job[0])
                    if verbose:
                        if total_done[0] % 1000 == 0:
                            print("Completed %.3f%%\r" % (100.0 * total_done[0] / total_els))
                error[0] = 0.0

        # Create workers
        workers_threads = [threading.Thread(target=worker_train) for _ in range(workers)]
        for thread in workers_threads:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        # Batch co-occurence pieces
        batch_length = 0
        batch = []
        num_examples = 0
        for key in self.cooccurence:
            for subkey in self.cooccurence[key]:
                batch.append((key, subkey, self.cooccurence[key][subkey]))
                batch_length += 1
                if batch_length >= batch_size:
                    jobs.put(
                        (
                            np.array([k for k,s,c in batch], dtype=np.int32),
                            np.array([s for k,s,c in batch], dtype=np.int32),
                            np.array([c for k,s,c in batch], dtype=np.float64)
                        )
                    )
                    num_examples += len(batch)
                    batch = []
                    batch_length = 0
        if len(batch) > 0:
            jobs.put(
                (
                    np.array([k for k,s,c in batch], dtype=np.int32),
                    np.array([s for k,s,c in batch], dtype=np.int32),
                    np.array([c for k,s,c in batch], dtype=np.float64)
                )
            )
            num_examples += len(batch)
            batch = []
            batch_length = 0

        for _ in range(workers):
            jobs.put(None)  # give the workers heads up that they can finish -- no more work!

        for thread in workers_threads:
            thread.join()

        return total_error[0] / num_examples
