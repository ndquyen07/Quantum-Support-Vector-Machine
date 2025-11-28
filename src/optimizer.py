from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, SLSQP, SPSA


cobyla = COBYLA(
    maxiter=5000,
    tol=1e-10,
    rhobeg=0.5
)

lbfgsb = L_BFGS_B(
    maxiter=1000
)

slsqp = SLSQP(
    maxiter=1000
)

spsa = SPSA(
    maxiter=1000
)

