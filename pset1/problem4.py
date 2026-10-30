import numpy as np
import matplotlib.pyplot as plt


# Graph Laplacians and Initial Conditions
x0 = np.array([1, 2, 3, 4, 5, 6, 7], dtype=float)

L_a = np.array([
    [ 4, -1, -1,  0,  0, -1, -1],
    [-1,  4, -1, -1,  0,  0, -1],
    [-1, -1,  4, -1, -1,  0,  0],
    [ 0, -1, -1,  4, -1, -1,  0],
    [ 0,  0, -1, -1,  4, -1, -1],
    [-1,  0,  0, -1, -1,  4, -1],
    [-1, -1,  0,  0, -1, -1,  4]
], dtype=float)

L_b = np.array([
    [ 6, -1, -1, -1, -1, -1, -1],
    [-1,  1,  0,  0,  0,  0,  0],
    [-1,  0,  1,  0,  0,  0,  0],
    [-1,  0,  0,  1,  0,  0,  0],
    [-1,  0,  0,  0,  1,  0,  0],
    [-1,  0,  0,  0,  0,  1,  0],
    [-1,  0,  0,  0,  0,  0,  1]
], dtype=float)


# Sim Parameters
dt = 0.1

# Switch graphs every m communication rounds
m = 5

max_rounds = 500
tolerance = 1e-3

I = np.eye(7)

W_a = I - dt * L_a
W_b = I - dt * L_b



x = x0.copy()

history = [x.copy()]
graph_history = []

consensus_round = None

def simulate_switching_consensus(L_a, L_b, dt, m, max_rounds=500, tolerance=1e-3):
    W_a = np.eye(7) - dt * L_a
    W_b = np.eye(7) - dt * L_b

    x = x0.copy()

    history = [x.copy()]

    consensus_round = None

    for k in range(max_rounds):

        block = k // m

        if block % 2 == 0:
            W = W_a
        else:
            W = W_b

        x = W @ x

        history.append(x.copy())

        spread = np.max(x) - np.min(x)

        if spread < tolerance:
            consensus_round = k + 1
            break

    history = np.array(history)

    return history, consensus_round

history_switch, rounds_switch = simulate_switching_consensus(
    L_a,
    L_b,
    dt=0.1,
    m=5
)

print("Switching consensus round:", rounds_switch)
print("Final state:", history_switch[-1])

for m in [1, 2, 5, 10]:
    history, rounds = simulate_switching_consensus(L_a, L_b, dt=0.1, m=m)
    print(f"m = {m}: {rounds} rounds")