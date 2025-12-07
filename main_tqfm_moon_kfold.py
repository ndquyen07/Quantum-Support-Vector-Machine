import os
import argparse
from src.feature_map import TrainableQuantumFeatureMap
import numpy as np
import json
from src.kernel_estimate import KernelMatrix
from qiskit_algorithms.optimizers import COBYLA, SPSA, ADAM
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_moons
from src.utils import calculate_accuracy1



# Parse command line arguments
parser = argparse.ArgumentParser(description='Run TQFM training on Cancer dataset with configurable parameters')
parser.add_argument("--run_id", type=int, default=0, help="Run ID for experiment tracking")
parser.add_argument("--depth", type=int, default=1, help="Depth of the TQFM circuit")
parser.add_argument("--type_ansatz", type=str, default="TwoLocal", 
                    choices=["TwoLocal", "RealAmplitudes", "EfficientSU2"],
                    help="Type of ansatz to use: TwoLocal, RealAmplitudes, EfficientSU2")
parser.add_argument("--optimizer", type=str, default="COBYLA",
                    choices=["COBYLA", "SPSA", "ADAM"],
                    help="Optimizer to use: COBYLA, SPSA, ADAM")
parser.add_argument("--maxiter", type=int, default=10000, help="Max iteration of optimizer")
args = parser.parse_args()

print("="*70)
print("TQFM Training - Circle Data")
print("="*70)
print(f"Configuration:")
print(f"  Run ID: {args.run_id}")
print(f"  Depth: {args.depth}")
print(f"  Type Ansatz: {args.type_ansatz}")
print(f"  Optimizer: {args.optimizer}")
print(f"  Max Iterations: {args.maxiter}")
print("="*70)

# Load data
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)


def main():

    # Create results_tqfm directory if it doesn't exist
    results_dir = "results_tqfm_moon_kfold"
    os.makedirs(results_dir, exist_ok=True)
    
    
    # Create optimizer instance based on argument
    optimizer_map = {
        "COBYLA": COBYLA(maxiter=args.maxiter),
        "SPSA": SPSA(maxiter=args.maxiter),
        "ADAM": ADAM(maxiter=args.maxiter, tol= 0.001, lr= 0.1)
    }
    optimizer = optimizer_map[args.optimizer]
    init_theta = None
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    losses = []

    for train_idx, val_idx in skf.split(X, y):

        X_train = X[train_idx] 
        y_train = y[train_idx]
        X_val   = X[val_idx]
        y_val   = y[val_idx]


        tqfm = TrainableQuantumFeatureMap(depth=args.depth, type_ansatz=args.type_ansatz)
    
        # Train TQFM
        tqfm.fit(X_train, y_train, init_theta=init_theta, optimizer=optimizer)
        losses.append(tqfm.optimal_value)
        init_theta = tqfm.init_theta

        # Compute kernel matrices
        kernel_train = KernelMatrix.compute_kernel_matrix_with_inner_products(X_train, X_train, tqfm.optimal_params, tqfm.circuit)
        kernel_val = KernelMatrix.compute_kernel_matrix_with_inner_products(X_val, X_train, tqfm.optimal_params, tqfm.circuit)

        #
        acc = calculate_accuracy1(kernel_train, kernel_val, y_train, y_val)
        accuracies.append(acc)

    mean_acc = np.mean(accuracies)
    mean_loss = np.mean(losses)

    data = {
        "run_id" : args.run_id,
        "depth" : args.depth,
        "mean_loss" : mean_loss,
        "mean_acc" : mean_acc
    }

    with open(f"{results_dir}/result_depth{args.depth}_run{args.run_id}.json", "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
