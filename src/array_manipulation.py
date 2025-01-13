import numpy as np
from numba import njit, prange
from itertools import permutations

# Sort a solution matrix by the number of 1s in the columns in descending order
def sort_columns_by_ones(matrix):
    return matrix[:, np.argsort(-np.sum(matrix == 1, axis=0))]

# Check if a recursed solution fits into the initial solution constraints
def fits(recursive_col, initial_matrix):
    for col in range(initial_matrix.shape[1]):
        if np.all(np.logical_or(recursive_col != 1, initial_matrix[:, col] != -1)):
            return True
    return False

# Brute-force mappings from recursive solution to initial constraints, O(N!) complexity but N is usually < 8 so its chill
def find_column_mapping(recursive_solution, initial_solution):
    # Sorting the recursive solution columns
    sorted_recursive = sort_columns_by_ones(recursive_solution)

    # Check if each column can fit
    for col in range(sorted_recursive.shape[1]):
        if not fits(sorted_recursive[:, col], initial_solution):
            print(f"No permutation can resolve this issue {initial_solution} <- {recursive_soltuion}")
            return None

    # Finding a column remapping configuration
    for perm in permutations(range(sorted_recursive.shape[1])):
        permuted_recursive = sorted_recursive[:, perm]
        if all(fits(permuted_recursive[:, col], initial_solution) for col in range(permuted_recursive.shape[1])):
            print(f"Found an applicable permutation: {initial_solution} <- {recursive_soltuion} perm: {perm}")
            return perm
    print("Quitting with no permutation selection")
    return None

# Attempt to resolve the recursed solution with the current solution by shifting assigned colors.
def resolve_recursed(solution, recursed, remaining):
    perm = find_column_mapping(solution[remaining], recursed)
    if perm is not None:
        solution[remaining] = recursed[perm]

    return solution

# Identify and remove columns where all entries are -1 (no available color to assign)
def filter_columns(adj_matrix):
    non_viable_cols = np.all(adj_matrix == -1, axis=0)
    return adj_matrix[:, ~non_viable_cols]

# Convert and extract solution values into a more readable format
def format_solution(solution, nv, W):
    # Replace -1 with 0
    solution[solution == -1] = 0
    w = solution[:W]
    x = np.array(solution[W:]).reshape(nv, W)
    return w, x

# Convert a matrix QUBO solution vector color configuration
def solution_to_coloring(solution):
    if solution is None:
        return None

    nodes = len(solution)
    coloring = np.full(nodes, -1, dtype=int)  # Ensure the dtype is int for indices
    for node, row in enumerate(solution):
        colors = np.where(row == 1)[0]
        if colors.size > 0:  # Check if the array is not empty
            coloring[node] = colors[0]  # Assign the first index of 1

    return coloring

# Collapse coloring configurations to a single color per node
@njit
def clean_solution(solution):
    rows, _ = solution.shape
    # Iterate over each row
    for i in prange(rows):
        first_one_index = np.argmax(solution[i])  # Find the index of the first '1'

        # Set all entries to '0' and then set the first '1' found
        solution[i, :] = 0
        solution[i, first_one_index] = 1

    return solution

# Retun the number of columns with at least one entry equal to 1 (not an all zero column)
def count_colors(solution):
    return np.sum(np.any(solution, axis=0))

# QUBO to Ising matrix transformation function
@njit
def qubo_to_ising(Q):
    """
    Convert a QUBO matrix to Ising model parameters h and J.

    Parameters:
    Q (np.ndarray): The QUBO matrix.

    Returns:
    h (np.ndarray): The linear coefficients for the Ising model.
    J (np.ndarray): The quadratic coefficients (interaction terms) for the Ising model.
    """
    n = Q.shape[0]
    
    # Initialize Ising model parameters
    h = np.zeros(n, dtype='float64')
    J = np.zeros((n, n), dtype='float64')
    
    for i in range(n):
        for j in range(n):
            if i == j:
                h[i] += Q[i, j] / 2
            else:
                J[i, j] += Q[i, j] / 4
                h[i] += Q[i, j] / 4
                h[j] += Q[i, j] / 4

    # Ensure J is symmetric
    J = (J + J.T) / 2
    
    # Normalize J to ensure values are within the desired range
    J_max = np.max(np.abs(J))
    if J_max > 1:
        J /= J_max
        h /= J_max
    
    return h, J