import math
import random
import numpy as np
from numba import cuda, float32
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float64
from .array_manipulation import format_solution

# ---------------- KERNELS --------------------
# CUDA implementation of SimCIM
@cuda.jit
def simcim_cuda(Q, s, n, steps, nu0, zeta, noise, h, rng_states, iter_flag):
    tx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if tx < n and iter_flag[0] == -1:  # Check if solution is already found
        for t in range(steps):
            # Compute nu and Gaussian noise for each thread
            nu = nu0 * (1 - math.tanh(t / steps * 6 - 3))
            f = xoroshiro128p_normal_float64(rng_states, tx) * noise

            # Compute feed-forward term
            feed_forward = 0
            for j in range(n):
                feed_forward += zeta * Q[tx, j] * s[j]

            feed_forward += h

            # Update state vector
            s[tx] += nu * s[tx] + feed_forward + f
            s[tx] = max(-1, min(1, s[tx]))  # Clip values to [-1, 1]

# CUDA kernel to reset a vector
@cuda.jit
def reset_vector(d_vec, n, val):
    tx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if tx < n:
        d_vec[tx] = val

# CUDA kernel to cap numbers
@cuda.jit
def cap_solution(d_s, n):
    tx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if tx < n:
        if d_s[tx] < 0:
            d_s[tx] = -1
        else:
            d_s[tx] = 1

# Verify coloring correctness of first color
@cuda.jit
def check_coloring_kernel(s, A, flags, num_nodes, W, solution_storage, iter_flag, iteration):
    node = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    # Step 1: Individual Node Check
    if node < num_nodes and iter_flag[0] == -1:
        if node == 0: # TODO DEBUG BREAKPOINT
            x = 1
        #    from pdb import set_trace; set_trace()
        colors = 0
        for color in range(W):
            node_idx = (1 + node) * W + color
            if s[node_idx] == 1:
                colors += 1
                # Check neighbors for the same color
                for neighbor in range(num_nodes):
                    neighbor_idx = (1 + neighbor) * W + color
                    c1 = A[node, neighbor] == 1
                    c2 = s[neighbor_idx] == 1
                    if c1 and c2 and neighbor != node:
                        flags[node] = False
                        break
                #break # Only check first color
        
        # Invalid if missing coloring
        if colors == 0:
            flags[node] = False

        # Ensure all threads have reached this point
        cuda.syncthreads()

        # Step 2: Aggregation of Results
        # This is done by only one thread to avoid race conditions
        if node == 0:
            all_flags_true = True
            for i in range(num_nodes):
                if not flags[i]:
                    all_flags_true = False
                    break
            
            if all_flags_true:
                # Update iteration flag and copy solution
                cuda.atomic.compare_and_swap(iter_flag, -1, iteration)
                for i in range((1 + num_nodes) * W):
                    solution_storage[i] = s[i]

# ---------------- WRAPPERS -------------------
# Return the RNG state array and handle seeding
def create_rng_states(n, iteration, base_seed):
    if base_seed is None:
        base_seed = random.randint(1, 2**32 - 1)
    seed = base_seed + iteration
    return create_xoroshiro128p_states(n, seed=seed)

# Function to call the kernels
def graph_color_cuda(A, Q, n, W, steps, nu0, zeta, noise, h, runs, seed, debug=False):
    node_count = A.shape[0]

    # Grid and block dimensions
    threads_per_block = 128
    blocks_per_grid = max(1, (n + (threads_per_block - 1)) // threads_per_block)
    
    # Allocate device memory for Q and A matrices
    d_Q = cuda.to_device(Q)
    d_A = cuda.to_device(A)

    # Allocate memory for flags and initialize them
    flags = np.ones(node_count, dtype=np.bool_)
    d_flags = cuda.to_device(flags)

    # Initialize iteration flag and solution storage
    iter_flag = np.array([-1], dtype=np.int32) # -1 indicates no solution found yet
    d_iter_flag = cuda.to_device(iter_flag)
    d_solution_storage = cuda.device_array(n, dtype=np.float64)

    d_s = cuda.device_array(n, dtype=np.float64)  # State vector

    # Loop for multiple runs
    for run in range(runs):    
        reset_vector[blocks_per_grid, threads_per_block](d_s, n, 0)  # Reset state vector
        reset_vector[blocks_per_grid, threads_per_block](d_flags, node_count, False)  # Reset the flags
        rng_states = create_rng_states(n, run, seed) # Create CUDA rng array

        # Launch SimCIM kernel
        simcim_cuda[blocks_per_grid, threads_per_block](d_Q, d_s, n, steps, nu0, zeta, noise, h, rng_states, d_iter_flag)
        cap_solution[blocks_per_grid, threads_per_block](d_s, n)
    
        # Launch result checking kernel
        check_coloring_kernel[blocks_per_grid, threads_per_block](d_s, d_A, d_flags, node_count, W, d_solution_storage, d_iter_flag, run)
        
        if run % int(runs/100) == 0:
            print(f"[i] Progress: {int(((1+run)/runs)*100)}%    ", end="\r")
            # Periodically check if a solution has been found
            iteration = d_iter_flag.copy_to_host()[0]
            #print("\n----------------------------------------")
            #print(A)
            if debug:
                print("----------------------------------------")
                print(d_s.copy_to_host()[W:].reshape(node_count, W))
                print("----------------------------------------\n")
            
            if iteration != -1:
                # Solution found, copy it back to host
                print(f" solution found on iteration: {iteration}")
                solution = d_solution_storage.copy_to_host()
                break
    
    # Cleanup device memory
    del d_Q
    del d_A
    del d_flags
    del d_iter_flag
    del d_solution_storage
    del d_s
    
    # No solution found
    if iteration == -1:
        print(" no valid solution found.")
        return None, None

    #print(f"[+] Solution found on iteration: {iteration}")
    _, solution = format_solution(solution, node_count, W)
    return iteration, solution