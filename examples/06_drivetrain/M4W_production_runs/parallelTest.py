#%%
# === imports  ===
# import os
# disbale nested threading: force thread limits before importing numpy
# -> so, 1 process = 1 core
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# print(f"PID {os.getpid()} started")

from random import random
from time import sleep
import time
import multiprocessing as mp
from multiprocessing.pool import Pool
n_procs = 6 # 6-8 workers best
len_cases = 10

# !!! Absolutely NOTHING heavy should execute at top level.
# --- instead, everything inside `driver` (cf. WISDEM\examples\13_design_of_experiments\doe_custom.py)
# --- (cf. https://chatgpt.com/s/t_699717e4aec081918ca7f5778ca14cfe)

#%%
# task executed in a worker process
def task(identifier, value):
    # report a message
    print(f'Task {identifier} executing with {value}', flush=True)
    # block for a moment
    sleep(value)
    return (identifier, value)

# test function
# result = task( items[0][0], items[0][1] )
#%% # `driver` guard: prevents recurring spawning
# - as each worker's __name__ != __main__, so driver() not called again (recurring)

def driver( n_procs, len_cases ):
    # prepare arguments (as an iterable (or list) of tuples (also an iterable) )
    items = [(i, random()) for i in range( len_cases )]
    # parallel
    mp.set_start_method("spawn", force=True) # explicit set #TODO

    ts = time.time()
    # create and configure the process pool
    with Pool( n_procs ) as pool: #NOTE: 
        # execute tasks and process results in order
        for results in pool.starmap(task, items):
            print(f'Got result: {results}', flush=True)
    # process pool is closed automatically
    te = time.time()
    print(f'Execution time: {te - ts:.2f} seconds')

if __name__ == '__main__': # protect the entry point
    driver( n_procs, len_cases )

# %%
