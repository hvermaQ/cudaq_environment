import cudaq
from cudaq import spin
import cudaq_solvers as solvers

# Define quantum kernel (ansatz)
@cudaq.kernel
def ansatz(theta: float):
    q = cudaq.qvector(2)
    x(q[0])
    ry(theta, q[1])
    x.ctrl(q[1], q[0])

# Define Hamiltonian
H = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - \
    2.1433 * spin.y(0) * spin.y(1) + \
    0.21829 * spin.z(0) - 6.125 * spin.z(1)


# Using L-BFGS-B optimizer with parameter-shift gradients
energy, parameters, data = solvers.vqe(
    lambda thetas: ansatz(thetas[0]),
    H,
    initial_parameters=[0.0],
    optimizer='lbfgs',
    gradient='parameter_shift',
    verbose=True
)

# Using SciPy optimizer directly
from scipy.optimize import minimize

def callback(xk):
    exp_val = cudaq.observe(ansatz, H, xk[0]).expectation()
    print(f"Energy at iteration: {exp_val}")

energy, parameters, data = solvers.vqe(
    lambda thetas: ansatz(thetas[0]),
    H,
    initial_parameters=[0.0],
    optimizer=minimize,
    callback=callback,
    method='COBYQA',
    jac='3-point',
    tol=1e-4,
    options={'disp': True}
)