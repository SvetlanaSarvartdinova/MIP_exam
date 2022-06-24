import numpy as np
import matplotlib.pyplot as plt


def get_coefs(data, coef):
    for j in range(len(data) - 1):
        Tj = data[j]['time']
        T_next = data[j + 1]['time']
        delta_T = T_next - Tj

        b_j = data[j]['b']
        b_dot_j = data[j]['b_dot']
        b_next = data[j + 1]['b']
        b_dot_next = data[j + 1]['b_dot']

        a0 = b_j
        a1 = b_dot_j
        a2 = (3 * b_next - 3 * b_j - 2 * b_dot_j * delta_T - b_dot_next * delta_T) / delta_T ** 2
        a3 = (2 * b_j + (b_dot_j + b_dot_next) * delta_T - 2 * b_next) / delta_T ** 3
        coef[Tj] = [a0, a1, a2, a3]


def get_polynomial_params(coef, t):
    for dt, coefficients in reversed(coef.items()):
        if t >= dt:
            a0, a1, a2, a3 = coefficients
            trajectory = a0 + a1 * (t - dt) + a2 * (t - dt) ** 2 + a3 * (t - dt) ** 3
            velocity = a1 + 2 * a2 * (t - dt) + 3 * a3 * (t - dt) ** 2
            return trajectory, velocity


def graphics_polynomial(data, coef):
    fig = plt.figure()
    pl_coords = fig.add_subplot(2, 2, 1)
    pl_vel = fig.add_subplot(2, 2, 2)
    pl_traj = fig.add_subplot(2, 2, 3)

    step = 0.05
    times = [t['time'] for t in data]
    t = np.arange(min(times), max(times) + step, step).tolist()

    coords = list()
    vel = list()
    for i in range(len(t)):
        trajectory, velocity = get_polynomial_params(coef, t[i])
        coords.append(trajectory)
        vel.append(velocity)

    coords = np.array(coords)

    pl_coords.set_title('coordinates')
    pl_coords.plot(t, coords)
    pl_coords.legend(['x', 'y'])
    pl_coords.grid()

    pl_vel.set_title('velocity')
    pl_vel.plot(t, vel)
    pl_vel.legend(['dx', 'dy'])
    pl_vel.grid()

    pl_traj.set_title('trajectory')
    pl_traj.plot(coords[:, 0], coords[:, 1])
    pl_traj.grid()

    for i in range(len(data)):
        x = data[i]['b'][0]
        y = data[i]['b'][1]
        pl_traj.scatter(x, y, color="blue")
        pl_traj.annotate(f"({x},{y})", (x, y))
    plt.tight_layout()
    plt.show()


data = [
    {
        'time': 0,
        'b': np.array([0, 0]),
        'b_dot': np.array([0, 0])
    },
    {
        'time': 1,
        'b': np.array([0, 1]),
        'b_dot': np.array([1, 1])
    },
    {
        'time': 2,
        'b': np.array([1, 1]),
        'b_dot': np.array([1, -1])
    },
    {
        'time': 3,
        'b': np.array([1, 0]),
        'b_dot': np.array([0, 0])
    }
]
coef = dict()
get_coefs(data, coef)
graphics_polynomial(data, coef)
