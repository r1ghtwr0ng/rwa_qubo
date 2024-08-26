import random
import traceback
import multiprocessing
import numpy as np
from numba import njit, prange
from .simcim import simcim, check_coloring
from .array_manipulation import qubo_to_ising

# Simple Timer class
class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start

# Get a baseline value to improve upon
@njit(parallel=True)
def benchmark_quality(Q, A, W, params, runs):
    solution_quality = 0
    nodes = A.shape[0]

    # Run multiple iterations and then record the median solution quality
    for i in prange(runs):
        #seed = random.randint(1, 2**32 - 1)
        # Run the SimCIM algorithm
        h, J = qubo_to_ising(Q)
        solution = simcim(J, A, W,
            params['steps'],
            params['nu0'],
            params['zeta'],
            params['noise'],
            params['h'],
            params['quant_level'],
            False)

        # Calculate and add solution quality to array
        test_state = np.sign(solution[W:]).astype('int64').reshape(nodes, W)
        solution_quality += 1 if check_coloring(test_state, A, W) else 0 

    # Return the average solution quality score
    return solution_quality / runs

@njit(['float64(uint8[:, :], int64[:, :])'])
def evaluate_solution_quality(adjacency, solution_matrix):
    n, w = solution_matrix.shape
    node_quality = np.ones(n, dtype='float64')
    fraction = 1 / (w-1)

    for i in prange(n):
        # Check if the node has at least one color set
        colors = np.sum(solution_matrix[i] == 1)
        node_quality[i] = (1+fraction) - (colors * fraction) if (colors > 0 and colors < w) else -1

        # Check for color conflicts with neighbors
        if colors > 0:
            neighbors = np.where(adjacency[i] == 1)[0]
            for neighbor in neighbors:
                # Find the colors set for both nodes
                node_colors = np.where(solution_matrix[i] == 1)[0]
                neighbor_colors = np.where(solution_matrix[neighbor] == 1)[0]

                # Check for conflicts manually
                for color in node_colors:
                    if color in neighbor_colors:
                        node_quality[i] -= fraction
                        break

        #if node_quality[i] < 0:
            #print(f"[DEBUG] Node quality {node_quality[i]} at idx {i}")
            #node_quality[i] = 0

    quality_score = np.mean(node_quality)
    return quality_score

# TODO handle exception thrown when problem is too large
def run_with_timeout(func, timeout, *args, **kwargs):
    def wrapper(return_dict, *args, **kwargs):
        try:
            return_dict['result'] = func(*args, **kwargs)
        except Exception as e:
            return_dict['result'] = "error"
            return_dict['traceback'] = traceback.format_exc()

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    
    p = multiprocessing.Process(target=wrapper, args=(return_dict, *args), kwargs=kwargs)
    p.start()
    p.join(timeout)
    
    if p.is_alive():
        print("[!] Function took too long. Terminating...")
        p.terminate()
        p.join()
        return "timeout"
    elif 'traceback' in return_dict:
        print("[!] An exception occurred:")
        print(return_dict['traceback'])
        return "error"
    else:
        return return_dict['result']

# Run benchmark and return the number of colors used for solving GCP
def benchmark_coloring(name, funct_call, timeout, *args, **kwargs):
    try:
        result = run_with_timeout(funct_call, timeout, *args, **kwargs)
        if result == "timeout":
            print(f"{name} timed out after {timeout} seconds")
            return None
        elif result["valid"]:
            #print(f"\n{name}:")
            #print("Time taken:", result["time_taken"])
            #print("Number of colors used:", result["num_colors"])
            #print("Valid coloring:", result["valid"])
            return [result["num_colors"], result["time_taken"]]
        else:
            print(f"{name} provided invalid results")
            return None
    except Exception as e:
        print(f"[!] An exception occured when running {name}: {e}")
        return None