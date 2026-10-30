import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Part A
x0 = np.array([1, 2, 3, 4, 5, 6, 7], dtype=float)


def linear_consesus(t, x, L):
    return -L @ x


def simulate_consensus(L, graph_name):

    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 1000)

    solution = solve_ivp(
        linear_consesus,
        t_span,
        x0,
        args=(L,),
        method='RK45',
        t_eval=t_eval
    )

    print(solution.y[:, -1])

    plt.figure()

    for i in range(7):
        plt.plot(
            solution.t,
            solution.y[i],
            label=f'Node {i+1}'
        )

    plt.xlabel('Time')
    plt.ylabel('$x_i(t)$')
    plt.title(f'Continuous-Time Consensus: Graph {graph_name}')
    plt.legend()
    plt.grid()

    plt.show()

    return solution

laplacian_a = np.array([[4, -1 ,-1, 0, 0, -1, -1],
                        [-1, 4, -1, -1, 0, 0, -1],
                        [-1, -1, 4, -1, -1, 0, 0],
                        [0, -1, -1, 4, -1, -1, 0],
                        [0, 0, -1, -1, 4, -1, -1],
                        [-1, 0, 0, -1, -1, 4, -1],
                        [-1, -1, 0, 0, -1, -1, 4]])

laplacian_b = np.array([[6, -1, -1, -1, -1, -1, -1],
                        [-1, 1, 0, 0, 0, 0, 0],
                        [-1, 0, 1, 0, 0, 0, 0],
                        [-1, 0, 0, 1, 0, 0, 0],
                        [-1, 0, 0, 0, 1, 0, 0],
                        [-1, 0, 0, 0 , 0, 1, 0],
                        [-1, 0, 0, 0, 0, 0, 1]])

laplacian_c = np.array([[1, 0, 0, 0, 0, 0, -1],
                        [-1, 1, 0, 0, 0, 0, 0],
                        [-1, -1, 2, 0, 0, 0, 0,],
                        [0, 0, -1, 1, 0, 0,0],
                        [-1, 0, 0, -1, 2, 0, 0],
                        [0, 0, 0, 0, -1, 1, 0],
                        [0, 0, 0, 0, 0, -1, 1]])

laplacian_d = np.array([[1, -1, 0, 0, 0, 0, 0,],
                        [0, 1, -1, 0, 0, 0, 0],
                        [-1, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, -1, 0, 0],
                        [0, 0, 0, -1, 1, 0, 0],
                        [-1, 0, 0, -1, 0, 3, -1],
                        [-1, 0, 0, -1, 0, 0, 2]])


sol_a = simulate_consensus(laplacian_a, "A")
sol_b = simulate_consensus(laplacian_b, "B")
sol_c = simulate_consensus(laplacian_c, "C")
sol_d = simulate_consensus(laplacian_d, "D")

# Part B: Graph C is the only one that can be reweighted as A and B are symmetric and D can not be reweighted
L_c_weighted = np.array([
    [ 3,  0,  0,  0,  0,  0, -3],
    [-1,  1,  0,  0,  0,  0,  0],
    [-1, -1,  2,  0,  0,  0,  0],
    [ 0,  0, -2,  2,  0,  0,  0],
    [-1,  0,  0, -2,  3,  0,  0],
    [ 0,  0,  0,  0, -3,  3,  0],
    [ 0,  0,  0,  0,  0, -3,  3]
], dtype=float)

sol_c_weighted = simulate_consensus(L_c_weighted, "C Weighted")

# Part C: Simulate Discrete Consensus

def discrete_consensus(L, dt, graph_name, max_rounds=500,tolerance=1e-3):

    W = np.eye(7) - dt * L

    # Current state
    x = x0.copy()

    # Store initial condition
    history = [x.copy()]

    consensus_round = None

    for k in range(max_rounds):
        x = W @ x

        history.append(x.copy())

        # Check consensus
        spread = np.max(x) - np.min(x)

        if spread < tolerance:
            consensus_round = k + 1
            break

    history = np.array(history)

    # Plot each node
    plt.figure()

    for i in range(7):
        plt.plot(
            range(len(history)),
            history[:, i],
            label=f'Node {i+1}'
        )

    plt.xlabel('Communication Round k')
    plt.ylabel('$x_i(k)$')
    plt.title(
        f'Discrete-Time Consensus: Graph {graph_name}, '
        f'$\\Delta t$ = {dt}'
    )
    plt.legend()
    plt.grid()
    plt.show()

    # Print results
    print(f'Graph {graph_name}')
    print(f'dt = {dt}')
    print('Final state:')
    print(history[-1])

    if consensus_round is not None:
        print(f'Consensus reached after {consensus_round} rounds.')
    else:
        print(f'Global consensus was not reached within {max_rounds} rounds.')

    print()

    return history, consensus_round

dt = 0.1

history_a, rounds_a = discrete_consensus(laplacian_a, dt, '(a)')
history_b, rounds_b = discrete_consensus(laplacian_b, dt, '(b)')
history_c, rounds_c = discrete_consensus(laplacian_c, dt, '(c)')
history_d, rounds_d = discrete_consensus(laplacian_d, dt, '(d)')

# Compare smaller delta t
history_b_small, rounds_b_small = discrete_consensus(laplacian_b, 0.05,'(b), smaller dt')
print('Graph (b) communication-round comparison')
print('-----------------------------------------')
print(f'dt = 0.10: {rounds_b} rounds')
print(f'dt = 0.05: {rounds_b_small} rounds')