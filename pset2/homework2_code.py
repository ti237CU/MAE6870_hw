import numpy as np
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


script_dir = Path(__file__).resolve().parent
csv_path = script_dir / "room_2x2.csv"

# Load csv map
binary_map = np.loadtxt(csv_path, delimiter=",")

## PROBLEM 2
# True temperature field
theta = 20 + 20 * binary_map

def generate_measurement(theta, sigma):
    noise = np.random.normal(0, sigma, size=np.shape(theta))
    return theta + noise

# Define given values
R_s = 3.5
R_c = 4.5
P = np.array([
    [4.5, 4.5], 
    [1.5, 1.5], 
    [1.5, 7.5], 
    [4.5, 8.5], 
    [8.5, 7.5], 
    [8.5, 3.5], 
    [5.5, 1.5]
])

# print(P.shape)
# print(P)

# Determine sensing range for each robot
rows, cols = np.meshgrid(np.arange(10)+0.5, np.arange(10)+0.5, indexing="ij")
robot_rows = P[:, 0, None, None]
robot_cols = P[:, 1, None, None]

row_diff = rows - robot_rows
col_diff = cols - robot_cols

d_r = np.sqrt(row_diff**2 + col_diff**2)
robot_range = d_r <= R_s

# True temperature field
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(theta, origin="lower", cmap="inferno",
               vmin=20, vmax=40)
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xlabel("Column c")
ax.set_ylabel("Row r")
ax.set_title("True Temperature Field")
fig.colorbar(im, ax=ax, label="Temperature (°C)")
plt.show()


# Individual robot sensing neighborhoods
N = P.shape[0]
fig, axes = plt.subplots(2, 4, figsize=(14, 7))

for i, ax in enumerate(axes.flat):
    if i >= N:
        ax.axis("off")
        continue

    ax.imshow(robot_range[i], origin="lower",
              cmap="Blues", vmin=0, vmax=1)
    ax.scatter(P[i, 1] - 0.5, P[i, 0] - 0.5,
               color="red", marker="x", s=70)

    ax.set_title(f"Robot {i+1}: {robot_range[i].sum()} cells")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))

fig.suptitle("Individual Robot Sensing Neighborhoods")
plt.tight_layout()
plt.show()

# Determine what robots can communicate with eachother
d_c = np.linalg.norm(P[:, None, :] - P[None, :, :], axis = 2)
robot_com = d_c <= R_c
np.fill_diagonal(robot_com, False)  # i = j -> False
# print(robot_com)

# Determine degree of each robot
robot_degree = np.sum(robot_com, axis=1)
print(robot_degree)

# Create communication graph
comm_graph = nx.from_numpy_array(robot_com)

pos = {i: (P[i, 1], P[i, 0]) for i in range(P.shape[0])}
labels = {i: f"Robot {i+1}" for i in range(P.shape[0])}

connected = nx.is_connected(comm_graph)
art_points = list(nx.articulation_points(comm_graph))
art_robots = [i + 1 for i in art_points]
print("Graph connected:", connected)
print("Articulation points:", art_robots)

nx.draw(
    comm_graph,
    pos=pos,
    with_labels=True,
    labels=labels,
    node_color="lightblue",
    node_size=1000,
    font_size=9
)

plt.xlabel("Column c (m)")
plt.ylabel("Row r (m)")
plt.title("Robot Communication Graph")
plt.axis("on")
plt.show()

# Metropolis weights
W = np.zeros((P.shape[0], P.shape[0]))

max_degree = np.maximum(robot_degree[:, None], robot_degree[None, :])
W = robot_com / (1 + max_degree)
diagonal_weights = 1 - np.sum(W, axis=1)
np.fill_diagonal(W, diagonal_weights)

print(np.sum(W, axis=1))  # Row sums
print(np.sum(W, axis=0))  # Column sums
print(np.all(W >= 0))

# Determine how many robots sense each tile and if the whole floor is covered
coverage_map = np.sum(robot_range, axis=0)

coverage_counts = np.sum(robot_range, axis=(1, 2))

print("\nIndividual robot coverage")
for i, count in enumerate(coverage_counts):
    print(f"Robot {i+1}: {count} cells ({count}%)")

print("\nCooperative coverage:",
      np.count_nonzero(coverage_map), "cells")

if np.all(coverage_map > 0):
    print("Entire floor covered")
else:
    print("Entire floor is NOT covered")

plt.imshow(coverage_map, origin="lower", cmap="viridis")
plt.colorbar(label="Number of robots sensing each cell")
plt.xlabel("Column c (m)")
plt.ylabel("Row r (m)")
plt.title("Robot Sensing Coverage")
plt.show()

## PROBLEM 3
# Problem 3b
K = 100
sigma = 5.0
N = P.shape[0]
n_rows, n_cols = theta.shape

theta_all = np.broadcast_to(theta, (N, n_rows, n_cols))
measurements = generate_measurement(theta_all, sigma)
measurements = np.where(robot_range, measurements, 0.0)

a = measurements.copy()
b = robot_range.astype(float)

x_history = np.full((K + 1, N), np.nan)

# Helper function record the history of estimates
def record_estimates(a, b):
    estimates = np.full(N, np.nan)

    np.divide(
        a[:, 5, 5],
        b[:, 5, 5],
        out=estimates,
        where=b[:, 5, 5] > 0
    )

    return estimates

x_history[0] = record_estimates(a,b)

for k in range(K):
    a_next = np.einsum("ij, jrc->irc", W, a)
    b_next = np.einsum("ij, jrc->irc", W, b)
    
    a = a_next
    b = b_next
    
    x_history[k + 1] = record_estimates(a, b)

x_distributed = np.full(a.shape, np.nan)
np.divide(a, b, out=x_distributed, where=b>0)

coverage_map = np.sum(robot_range, axis=0)
measurements_sum = np.sum(measurements, axis=0)

x_centralized = np.full(theta.shape, np.nan)
np.divide(measurements_sum, coverage_map, out=x_centralized, where=coverage_map>0)

abs_error = np.abs(x_distributed - x_centralized[None, :, :])
max_abs_diff = np.nanmax(abs_error)

print("\nProblem 3(b) Results")
print("Consensus iterations:", K)
print("Sensor variance:", sigma**2)
print("Observed cells:", np.sum(coverage_map > 0))
print("Maximum absolute difference:", max_abs_diff, "°C")

# Problem 3c
for i in range(N):
    plt.plot(range(K + 1), x_history[:, i], label=f"Robot {i+1}")

plt.axhline(
    y=x_centralized[5, 5],
    color="black",
    linestyle="--",
    label="Centralized estimate"
)

plt.xlabel("Consensus iteration k")
plt.ylabel("Estimated temperature (°C)")
plt.title("Consensus Convergence at Cell (5, 5)")
plt.legend()
plt.grid(True)
plt.show()


# Problem 3d
V1 = robot_range[0]

# Number of robots observing each cell
n_rc = coverage_map[V1]

# Theoretical per-cell variances
local_variance = np.full(n_rc.shape, sigma**2)
fused_variance = sigma**2 / n_rc

# Expected MSE averaged over V(1)
local_expected_mse = np.mean(local_variance)
fused_expected_mse = np.mean(fused_variance)

print("Local expected MSE:", local_expected_mse)
print("Fused expected MSE:", fused_expected_mse)



# PROBLEMS 4 & 5 SETUP
N = P.shape[0]
K_comparison = 100
M_comparison = 100

mask = robot_range.astype(float)
coverage_map = robot_range.sum(axis=0)

observed = coverage_map > 0
O1 = robot_range[0] & (coverage_map >= 2)


def run_consensus_fusion(measurements, alpha, W, K):
    alpha_grid = alpha[:, None, None]

    a = measurements * alpha_grid
    b = mask * alpha_grid

    for k in range(K):
        a = np.einsum("ij,jrc->irc", W, a)
        b = np.einsum("ij,jrc->irc", W, b)

    estimates = np.full(a.shape, np.nan)

    np.divide(a, b, out=estimates, where=b > 0)

    return estimates


def compare_fusion_methods(sensor_variances, problem, seed):
    rng = np.random.default_rng(seed)

    sensor_std = np.sqrt(sensor_variances)
    alpha_weighted = 1.0 / sensor_variances
    alpha_unweighted = np.ones(N)

    mse_u_all = np.zeros(M_comparison)
    mse_w_all = np.zeros(M_comparison)

    mse_u_O1 = np.zeros(M_comparison)
    mse_w_O1 = np.zeros(M_comparison)

    saved_maps = {}

    for trial in range(M_comparison):

        noise = rng.normal(
            loc=0.0,
            scale=sensor_std[:, None, None],
            size=(N, *theta.shape)
        )

        measurements = theta[None, :, :] + noise
        measurements = np.where(robot_range, measurements, 0.0)

        x_u = run_consensus_fusion(measurements, alpha_unweighted, W, K_comparison)
        x_w = run_consensus_fusion(measurements, alpha_weighted,W, K_comparison)

        estimate_u = x_u[0]
        estimate_w = x_w[0]

        error_u = estimate_u - theta
        error_w = estimate_w - theta

        mse_u_all[trial] = np.mean(error_u[observed] ** 2)
        mse_w_all[trial] = np.mean(error_w[observed] ** 2)
        mse_u_O1[trial] = np.mean(error_u[O1] ** 2)
        mse_w_O1[trial] = np.mean(error_w[O1] ** 2)

        if trial == 0:
            saved_maps["Unweighted"] = estimate_u.copy()
            saved_maps["Weighted"] = estimate_w.copy()

    results = {
        "Unweighted MSE (all)": np.mean(mse_u_all),
        "Weighted MSE (all)": np.mean(mse_w_all),
        "Unweighted MSE (O1)": np.mean(mse_u_O1),
        "Weighted MSE (O1)": np.mean(mse_w_O1),
        "Unweighted RMSE (O1)": np.sqrt(np.mean(mse_u_O1)),
        "Weighted RMSE (O1)": np.sqrt(np.mean(mse_w_O1))
    }

    print(f"\nPROBLEM {problem}: RESULTS")
    print("-" * 45)
    print("Sensor variances:", sensor_variances)
    print("Number of trials:", M_comparison)
    print("Cells in O1:", np.sum(O1))

    for name, value in results.items():
        units = "°C" if "RMSE" in name else "°C²"
        print(f"{name}: {value:.6f} {units}")


    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    for ax, method in zip(axes, ["Unweighted", "Weighted"]):
        data = saved_maps[method]

        im = ax.imshow(
            data,
            origin="lower",
            cmap="inferno",
            vmin=10,
            vmax=50,
            interpolation="nearest"
        )

        ax.contour(
            np.arange(-1, 11),
            np.arange(-1, 11),
            np.pad(O1.astype(float), 1),
            levels=[0.5],
            colors="cyan",
            linewidths=1.5
        )

        handle = Line2D(
            [0], [0],
            color="cyan",
            linewidth=1.5,
            label=r"$O_1$ (Overlapping Region)"
        )

        ax.legend(handles=[handle], loc="upper right", fontsize=8)
        ax.set_title(f"{method} Fusion")
        ax.set_xlabel("Column c")
        ax.set_ylabel("Row r")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xlim(-0.5, 9.5)
        ax.set_ylim(-0.5, 9.5)

    fig.colorbar(
        im, ax=axes,
        label="Estimated Temperature (°C)",
        shrink=0.8
    )

    fig.suptitle(f"Problem {problem}: Fused Temperature Maps")
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5),sharex=True, sharey=True)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("lightgray")

    for ax, method in zip(axes, ["Unweighted", "Weighted"]):
        error = saved_maps[method] - theta
        masked_error = np.ma.masked_where(~O1, error)

        im = ax.imshow(
            masked_error,
            origin="lower",
            cmap=cmap,
            vmin=-15,
            vmax=15,
            interpolation="nearest"
        )

        ax.set_title(f"{method} Signed Error")
        ax.set_xlabel("Column c")
        ax.set_ylabel("Row r")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xlim(-0.5, 9.5)
        ax.set_ylim(-0.5, 9.5)

    fig.colorbar(im, ax=axes, label="Signed Error (°C)", shrink=0.8)
    fig.suptitle(f"Problem {problem}: Signed Errors in O1")

    plt.show()
    return results



# PROBLEM 4
variances_4 = np.full(N, 225.0)
variances_4[0] = 1.0

results_4 = compare_fusion_methods(
    sensor_variances=variances_4,
    problem=4,
    seed=42
)


# PROBLEM 5
variances_5 = np.ones(N)
variances_5[0] = 225.0

results_5 = compare_fusion_methods(
    sensor_variances=variances_5,
    problem=5,
    seed=43
)


## PROBLEM 6
T = 20
K = 100
M = 100
N = P.shape[0]

rng = np.random.default_rng(42)
sensor_variances = np.full(N, 225.0)
sensor_variances[0] = 1.0

sensor_std = np.sqrt(sensor_variances)
alpha = 1.0 / sensor_variances

mask = robot_range.astype(float)    # m_irc

coverage_map = robot_range.sum(axis=0)
observed = coverage_map > 0

O1 = robot_range[0] & (coverage_map >= 2)

mse_unweighted_all = np.zeros((M, T))
mse_weighted_all = np.zeros((M, T))

mse_unweighted_O1 = np.zeros((M, T))
mse_weighted_O1 = np.zeros((M, T))

saved_maps = {}

def consensus_ratio(numerator, denominator, W, K):
    a = numerator.copy()
    b = denominator.copy()

    for k in range(K):
        a_next = np.einsum("ij,jrc->irc", W, a)
        b_next = np.einsum("ij,jrc->irc", W, b)

        a = a_next
        b = b_next

    estimates = np.full(a.shape, np.nan)

    np.divide(a, b, out=estimates, where=b > 0)
    return estimates

for experiment in range(M):

    H = np.zeros((N, *theta.shape))
    J = np.zeros((N, *theta.shape))

    S = np.zeros((N, *theta.shape))
    C = np.zeros((N, *theta.shape))

    for t in range(1, T + 1):
        noise = rng.normal(
            loc=0.0,
            scale=sensor_std[:, None, None],
            size=(N, *theta.shape)
        )

        measurements = theta[None, :, :] + noise
        measurements = np.where(robot_range, measurements, 0.0)

        H += measurements * alpha[:, None, None]
        J += mask * alpha[:, None, None]

        S += measurements
        C += mask

        x_weighted = consensus_ratio(H, J, W, K)
        x_unweighted = consensus_ratio(S, C, W, K)

        estimate_w = x_weighted[0]
        estimate_u = x_unweighted[0]

        error_w = estimate_w - theta
        error_u = estimate_u - theta

        mse_weighted_all[experiment, t - 1] = np.mean(error_w[observed]**2)
        mse_unweighted_all[experiment, t - 1] = np.mean(error_u[observed]**2)
        mse_weighted_O1[experiment, t - 1] = np.mean(error_w[O1]**2)
        mse_unweighted_O1[experiment, t - 1] = np.mean(error_u[O1]**2)

        if experiment == 0 and t in (1, 20):
            saved_maps[("Unweighted", t)] = estimate_u.copy()
            saved_maps[("Weighted", t)] = estimate_w.copy()

# MSE
mean_mse_u_all = mse_unweighted_all.mean(axis=0)
mean_mse_w_all = mse_weighted_all.mean(axis=0)

mean_mse_u_O1 = mse_unweighted_O1.mean(axis=0)
mean_mse_w_O1 = mse_weighted_O1.mean(axis=0)

time = np.arange(1, T + 1)

print("\nProblem 6 Results")
print("Independent experiments:", M)
print("Sensing rounds:", T)
print("Consensus rounds per sensing round:", K)

print("\nFinal MSE at t = 20")
print("Unweighted (all):", mean_mse_u_all[-1])
print("Weighted (all):", mean_mse_w_all[-1])

print("Unweighted (O1):", mean_mse_u_O1[-1])
print("Weighted (O1):", mean_mse_w_O1[-1])

# MSE vs time for all observed cells
plt.figure(figsize=(9, 5))

plt.plot(
    time, mean_mse_u_all,
    marker="o",
    markersize=3,
    label="Unweighted Fusion"
)

plt.plot(
    time, mean_mse_w_all,
    marker="s",
    markersize=3,
    label="Uncertainty-Weighted Fusion"
)

plt.xlabel("Sensing Round t")
plt.ylabel("Mean Squared Error (°C²)")
plt.title("Sequential Fusion: MSE Over All Observed Cells")
plt.xticks(np.arange(1, T + 1))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# MSE vs time for region O1
plt.figure(figsize=(9, 5))

plt.plot(
    time, mean_mse_u_O1,
    marker="o",
    markersize=3,
    label="Unweighted Fusion"
)

plt.plot(
    time, mean_mse_w_O1,
    marker="s",
    markersize=3,
    label="Uncertainty-Weighted Fusion"
)

plt.xlabel("Sensing Round t")
plt.ylabel("Mean Squared Error (°C²)")
plt.title("Sequential Fusion: MSE Over O1")
plt.xticks(np.arange(1, T + 1))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Temperature Maps
fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 10),
    sharex=True,
    sharey=True
)

methods = ["Unweighted", "Weighted"]
times = [1, 20]

for row, method in enumerate(methods):
    for col, t in enumerate(times):

        ax = axes[row, col]
        data = saved_maps[(method, t)]

        im = ax.imshow(
            data,
            origin="lower",
            cmap="inferno",
            vmin=10,
            vmax=50,
            interpolation="nearest"
        )

        # Outline the O1 region
        ax.contour(
            np.arange(-1, 11),
            np.arange(-1, 11),
            np.pad(O1.astype(float), 1),
            levels=[0.5],
            colors="cyan",
            linewidths=1.5
        )

        o1_handle = Line2D(
            [0], [0],
            color="cyan",
            linewidth=1.5,
            label=r"$O_1$ (Overlapping Sensing Region)"
        )

        ax.legend(handles=[o1_handle], loc="upper right", fontsize=8)

        ax.set_title(f"{method} Fusion, t = {t}")

        ax.set_xticks(np.arange(10))
        ax.set_yticks(np.arange(10))
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.set_xlabel("Column c (m)")
        ax.set_ylabel("Row r (m)")
        ax.set_xlim(-0.5, 9.5)
        ax.set_ylim(-0.5, 9.5)

fig.colorbar(im, ax=axes, label="Estimated Temperature (°C)", shrink=0.8)
fig.suptitle("Sequential Temperature Mapping: t = 1 vs t = 20", fontsize=14)

plt.show()
