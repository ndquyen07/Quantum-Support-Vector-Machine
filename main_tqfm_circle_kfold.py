import os
import argparse
from src.feature_map import TrainableQuantumFeatureMap
import numpy as np
import json
from src.kernel_estimate import KernelMatrix
from qiskit_algorithms.optimizers import COBYLA, SPSA, ADAM
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.datasets import make_circles
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
X, y = make_circles(n_samples=400, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def main():

    # Create results_tqfm directory if it doesn't exist
    results_dir = "results_tqfm_circle_kfold"
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

    accuracies_val_before = []
    accuracies_test_before = []
    accuracies_val_after = []
    accuracies_test_after = []
    losses = []

    for train_idx, val_idx in skf.split(X_train, y_train):

        X_train_fold = X_train[train_idx] 
        y_train_fold = y_train[train_idx]
        X_val   = X_train[val_idx]
        y_val   = y_train[val_idx]


        tqfm = TrainableQuantumFeatureMap(depth=args.depth, type_ansatz=args.type_ansatz)
    
        # Train TQFM
        tqfm.fit(X_train_fold, y_train_fold, init_theta=init_theta, optimizer=optimizer)
        init_theta = tqfm.init_theta

        #
        kernel_train_before = KernelMatrix.compute_kernel_matrix_with_inner_products(X_train_fold, X_train_fold, tqfm.init_theta, tqfm.circuit)
        kernel_val_before = KernelMatrix.compute_kernel_matrix_with_inner_products(X_val, X_train_fold, tqfm.init_theta, tqfm.circuit)
        kernel_test_before = KernelMatrix.compute_kernel_matrix_with_inner_products(X_test, X_train_fold, tqfm.init_theta, tqfm.circuit)


        # Compute kernel matrices
        kernel_train_after = KernelMatrix.compute_kernel_matrix_with_inner_products(X_train_fold, X_train_fold, tqfm.optimal_params, tqfm.circuit)
        kernel_val_after = KernelMatrix.compute_kernel_matrix_with_inner_products(X_val, X_train_fold, tqfm.optimal_params, tqfm.circuit)
        kernel_test_after = KernelMatrix.compute_kernel_matrix_with_inner_products(X_test, X_train_fold, tqfm.optimal_params, tqfm.circuit)

        #
        acc_val_before = calculate_accuracy1(kernel_train_before, kernel_val_before, y_train_fold, y_val)
        acc_test_before = calculate_accuracy1(kernel_train_before, kernel_test_before, y_train_fold, y_test)
        #
        acc_val_after = calculate_accuracy1(kernel_train_after, kernel_val_after, y_train_fold, y_val)
        acc_test_after = calculate_accuracy1(kernel_train_after, kernel_test_after, y_train_fold, y_test)
        
        #
        accuracies_val_before.append(acc_val_before)
        accuracies_test_before.append(acc_test_before)
        #
        accuracies_val_after.append(acc_val_after)
        accuracies_test_after.append(acc_test_after)
        
        losses.append(tqfm.optimal_value)

    mean_val_before = np.mean(accuracies_val_before)
    mean_test_before = np.mean(accuracies_test_before)
    mean_val_after = np.mean(accuracies_val_after)
    mean_test_after = np.mean(accuracies_test_after)
    mean_loss = np.mean(losses)

    data = {
        "run_id" : args.run_id,
        "depth" : args.depth,
        "mean_loss" : mean_loss,
        "mean_val_before" : mean_val_before,
        "mean_test_before" : mean_test_before,
        "mean_val_after" : mean_val_after,
        "mean_test_after" : mean_test_after
    }

    with open(f"{results_dir}/result_depth{args.depth}_run{args.run_id}.json", "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
