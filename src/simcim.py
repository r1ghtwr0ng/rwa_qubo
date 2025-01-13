import numpy as np
import random
from numba import njit, boolean, prange
from .array_manipulation import qubo_to_ising

@njit()
def quantize(values, levels):
    """
    Quantize the input values into a specified number of levels.

    Parameters:
    values (np.ndarray): The array of values to be quantized.
    levels (int): The number of discrete levels to quantize the values into.

    Returns:
    np.ndarray: The quantized values.
    """
    # Determine the min and max to scale the values to [0, 1]
    min_val, max_val = np.min(values), np.max(values)
    if min_val == max_val:
        return values  # Avoid division by zero if all values are the same
    
    # Scale values to the [0, 1] range
    scaled = (values - min_val) / (max_val - min_val)
    
    # Quantize the scaled values to the nearest level
    quantized = np.round(scaled * (levels - 1)) / (levels - 1)
    
    # Rescale back to the original range
    rescaled = quantized * (max_val - min_val) + min_val
    
    return rescaled

def make_qubo(c_vals, A, W):
    """
    Create the spin-spin interaction QUBO matrix from the Hamiltonian for a given problem.

    Parameters:
    c_vals (list or np.ndarray): Coefficients for the QUBO construction, typically including penalty terms.
    A (np.ndarray): The adjacency matrix representing the graph structure of the problem.
    W (int): The number of possible states or colors in the graph coloring problem.

    Returns:
    np.ndarray: The constructed QUBO matrix representing the problem.

    This function constructs the QUBO matrix using the provided coefficients and graph structure. It is designed for problems like graph coloring, where the matrix captures both the objective function and constraints.
    """
    degrees = np.sum(A, axis=1)  # Node degrees
    # Identity matrices
    E_w = np.identity(W)  # WxW identity matrix
    E_nv = np.identity(A.shape[0])
    I_w = np.ones((W, W))  # WxW matrix of ones
    # Submatrices for QUBO construction
    Q_11 = E_w * c_vals[0]  # Coefficient for self-interactions
    Q_12 = np.kron(-0.5 * c_vals[2] * degrees, E_w)  # Interaction between nodes and colors
    Q_21 = Q_12.T  # Transpose of Q_12
    Q_22 = np.kron((c_vals[1] * E_nv), (I_w - 2 * E_w)) + np.kron(c_vals[1] * A, E_w)  # Penalty terms
    
    # Combine submatrices to form the full Q matrix
    Q = np.block([[Q_11, Q_12], [Q_21, Q_22]])
    
    return Q

@njit()
def calculate_total_energy(s, Q):
    """
    Calculate the total energy of a given solution s with respect to the QUBO matrix Q.

    Parameters:
    s (np.ndarray): The solution vector.
    Q (np.ndarray): The QUBO matrix.

    Returns:
    float: The total energy of the solution.
    """
    return np.dot(s.T, np.dot(Q, s))

@njit(['float64[:](float64, float64, int64)'])
def custom_linspace(start, stop, num):
    step = (stop - start) / (num - 1)
    arr = np.empty(num, dtype=np.float64)
    for i in range(num):
        arr[i] = start + step * i
    return arr

@njit(['float64[:](int64, float64)'])
def calculate_v(steps, nu0=-15):
    end_v = 0
    t = custom_linspace(-3, 3, steps)
    v = (np.tanh(t) + 1) / 2 * (end_v - nu0) + nu0
    return v

@njit()
def simcim(J, A, W, steps, nu0, zeta, noise, h, quant_level, bmark=False, seed=None):
    """
    Execute the SimCIM algorithm to solve an optimization problem.

    Parameters:
    J (np.ndarray): The Ising matrix representing the optimization problem.
    A (np.ndarray): The adjacency matrix used for feasibility checks.
    W (int): The partition parameter, like the number of colors in graph coloring.
    n (int): The size of the solution vector.
    steps (int): The number of iterations for the SimCIM algorithm.
    nu0 (float): Initial value of the dynamic parameter nu.
    zeta (float): The feed-forward factor.
    noise (float): Standard deviation of the Gaussian noise for simulation.
    h (np.ndarray): Vector for the external field applied to each spin.
    seed (int, optional): Seed for the random number generator.
    quan_level (int, optional): The level for gradient quantization. If 0, quantization is not applied.

    Returns:
    np.ndarray: The optimal spin configuration found by the algorithm, or an array of -1s if no feasible solution is found.

    This function implements the SimCIM algorithm with enhancements for stability near optimum points, including noise normalization and optional gradient quantization.
    """

    # Matrix size
    n = J.shape[0]
    nodes = A.shape[0]

    # Convert QUBO to Ising
    np.random.seed(random.randint(1, 2**32 - 1))  # Convert seed to int and seed the RNG for reproducibility

    if W == 0:
        # Return an array of -1s indicating no feasible solution exists
        return -1 * np.ones(n, dtype='int64')

    s = np.zeros(n, dtype=np.float64)  # Initialize state vector
    J = J.astype(np.float64)  # Ensure Ising matrix is in float64 for calculations

    #h = 0 * np.ones(n, dtype=np.float64)  # External field vector

    feasible_solutions = []  # List to store feasible solutions
    min_energy = np.inf  # Track minimum energy for optimality
    optimal_solution = None  # Store the optimal solution

    # Pump loss factor calculation
    v = calculate_v(steps, nu0)

    for t in range(steps):
        #print(f"[i] Progress: {t}/{steps}")
        # Mean field calculation
        f = np.random.normal(0, noise, size=n)  # Sample Gaussian noise
        feed_forward = zeta * (J @ s) + h  # Calculate feed-forward term
        gradient_norm = np.linalg.norm(feed_forward)  # Norm of feed-forward term

        if gradient_norm != 0:
            f /= gradient_norm  # Normalize noise by gradient norm
        if quant_level > 0:  # Apply quantization if specified
            f = quantize(f, quant_level)

        s += v[t] * s + feed_forward + f  # Update state vector
        np.clip(s, -1, 1, out=s)  # Ensure values are within [-1, 1]

        # Feasibility check and energy calculation
        test_solution = np.sign(s[W:]).astype('int64').reshape(nodes, W)
        if check_coloring(test_solution, A, W):
            energy = calculate_total_energy(s, J)
            #print(f"[!] Found feasible solution on iteration: {t}")
            feasible_solutions.append((s.copy(), energy))
            if energy < min_energy:
                min_energy = energy
                optimal_solution = s.copy()
    # DEBUG
    #print(f"[i] Feasible solutions: {len(feasible_solutions)}")

    # Return the optimal solution if found, else an array of -1s
    if optimal_solution is not None:
        #print("[+] Found solution!")
        return np.sign(optimal_solution).astype('int64')
    elif bmark:
        # Return sign configuration for benchmarking
        return np.sign(s).astype('int64')
    else:
        # Return an array of -1s indicating no feasible solution was found
        return -1 * np.ones(n, dtype='int64')

def run_simcim(Q, A, W, steps, nu0, zeta, noise, h_val, quant_level, debug=False, seed=None):
    """
    Execute the SimCIM algorithm to solve an optimization problem and format the solution.

    Parameters:
    J (np.ndarray): The Ising matrix representing the optimization problem.
    A (np.ndarray): The adjacency matrix of the graph, used for checking solution feasibility.
    steps (int): The number of iterations to run the SimCIM algorithm.
    nu0 (float): Initial value of the dynamic parameter nu.
    zeta (float): The feed-forward factor.
    noise (float): Standard deviation of the Gaussian noise.
    h (np.ndarray): External field applied to each spin.
    W (int): The number of colors (in graph coloring) or other partition parameter.
    seed (int, optional): Seed for the random number generator.

    Returns:
    tuple: A tuple containing two numpy arrays:
        - w_arr: The solution vector for the weight or partition decision.
        - x_arr: The reshaped solution array representing node assignments or colors.

    This function solves the optimization problem using the SimCIM algorithm, then extracts and formats the solution vector into meaningful components.
    """

    h, J = qubo_to_ising(Q)
    #h = h_val * np.ones(len(Q))

    solution = simcim(J, A, W, steps, nu0, zeta, noise, h, quant_level, debug)
    nodes = A.shape[0]
    w_arr = solution[:W]
    x_arr = np.array(solution[W:]).reshape(nodes, W)
    return w_arr, x_arr

@njit()
def check_coloring(s, A, W):
    """
    Verify if a graph coloring solution is valid by ensuring no adjacent nodes share the same color.

    Parameters:
    s (np.ndarray): The solution vector representing node colors.
    A (np.ndarray): The adjacency matrix of the graph.
    W (int): The number of colors used in the coloring problem.

    Returns:
    bool: True if the solution is valid (no adjacent nodes share the same color), False otherwise.

    This function iterates over each node in the graph, checking if it has been assigned a color and if so, verifies that none of its adjacent nodes have the same color, ensuring the solution's validity.
    """
    num_nodes = A.shape[0]
    for node in range(num_nodes):
        node_has_color = False
        node_color = -1

        for color in range(W):
            if s[node][color] == 1:  # Adjusted indexing to access flat array
                node_color = color
                node_has_color = True
                break

        if node_has_color:
            for neighbor in range(num_nodes):
                if A[node][neighbor] == 1 and s[neighbor][node_color] == 1:  # Adjusted indexing
                    return False
        else:
            return False

    return True