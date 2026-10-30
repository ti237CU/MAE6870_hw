import numpy as np
from scipy.linalg import null_space


# Compute all eigen values of the graph laplacians for each graph

laplacian_a = np.array([[4, -1 ,-1, 0, 0, -1, -1],
                        [-1, 4, -1, -1, 0, 0, -1],
                        [-1, -1, 4, -1, -1, 0, 0],
                        [0, -1, -1, 4, -1, -1, 0],
                        [0, 0, -1, -1, 4, -1, -1],
                        [-1, 0, 0, -1, -1, 4, -1],
                        [-1, -1, 0, 0, -1, -1, 4]])

eigenvalues_a = np.round(np.linalg.eigvals(laplacian_a), 3)

laplacian_b = np.array([[6, -1, -1, -1, -1, -1, -1],
                        [-1, 1, 0, 0, 0, 0, 0],
                        [-1, 0, 1, 0, 0, 0, 0],
                        [-1, 0, 0, 1, 0, 0, 0],
                        [-1, 0, 0, 0, 1, 0, 0],
                        [-1, 0, 0, 0 , 0, 1, 0],
                        [-1, 0, 0, 0, 0, 0, 1]])
eigenvalues_b = np.round(np.linalg.eigvals(laplacian_b),3)

laplacian_c = np.array([[1, 0, 0, 0, 0, 0, -1],
                        [-1, 1, 0, 0, 0, 0, 0],
                        [-1, -1, 2, 0, 0, 0, 0,],
                        [0, 0, -1, 1, 0, 0,0],
                        [-1, 0, 0, -1, 2, 0, 0],
                        [0, 0, 0, 0, -1, 1, 0],
                        [0, 0, 0, 0, 0, -1, 1]])

eigenvalues_c = np.round(np.linalg.eigvals(laplacian_c),3)

laplacian_d = np.array([[1, -1, 0, 0, 0, 0, 0,],
                        [0, 1, -1, 0, 0, 0, 0],
                        [-1, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, -1, 0, 0],
                        [0, 0, 0, -1, 1, 0, 0],
                        [-1, 0, 0, -1, 0, 3, -1],
                        [-1, 0, 0, -1, 0, 0, 2]])

eigenvalues_d = np.round(np.linalg.eigvals(laplacian_d), 3)

print(
    f"Graph A: {sorted(eigenvalues_a)}",
    f"Graph B: {sorted(eigenvalues_b)}",
    f"Graph C: {sorted(eigenvalues_c)}",
    f"Graph D: {sorted(eigenvalues_d)}",
    sep="\n"
)

# Find w a left null eigenvector of L satisfying the conditions

def left_null(L):
    N = null_space(L.T)                  # columns span the null space of L^T
    if N.shape[1] > 1:
        print(f"null space has dimension {N.shape[1]}, so w is NOT unique")
        print("basis vectors (columns):\n", np.round(N, 4))
        return None
    w = N[:, 0] / N[:, 0].sum()          # normalize so the entries sum to 1
    print("w =", np.round(w, 4))
    print("check w^T L =", np.round(w @ L, 10))
    return w

laplacians = {"a": laplacian_a, "b": laplacian_b, "c": laplacian_c, "d": laplacian_d}

w_values = {}
for name, L in laplacians.items():
    print(f"\nGraph {name}:")
    w_values[name] = left_null(L)
    