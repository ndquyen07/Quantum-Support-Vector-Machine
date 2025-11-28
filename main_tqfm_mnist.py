import time
import os
import argparse
from datetime import datetime
from src.feature_map_0 import TrainableQuantumFeatureMap
import numpy as np
import pickle
from src.kernel_estimate import KernelMatrix
from qiskit_algorithms.optimizers import COBYLA, SPSA, ADAM, L_BFGS_B, SLSQP


# Parse command line arguments
parser = argparse.ArgumentParser(description='Run TQFM training on MNIST dataset with configurable parameters')
parser.add_argument("--run_id", type=int, default=0, help="Run ID for experiment tracking")
parser.add_argument("--depth", type=int, default=1, help="Depth of the TQFM circuit")
parser.add_argument("--type_ansatz", type=str, default="TwoLocal", 
                    choices=["TwoLocal", "RealAmplitudes", "EfficientSU2"],
                    help="Type of ansatz to use: TwoLocal, RealAmplitudes, EfficientSU2")
parser.add_argument("--optimizer", type=str, default="COBYLA",
                    choices=["COBYLA", "SPSA", "ADAM", "L_BFGS_B", "SLSQP"],
                    help="Optimizer to use: COBYLA, SPSA, ADAM, L_BFGS_B, SLSQP")
parser.add_argument("--maxiter", type=int, default=10000, help="Max iteration of optimizer")
args = parser.parse_args()

print("="*70)
print("TQFM Training - MNIST Dataset (28x28 Handwritten Digits)")
print("="*70)
print(f"Configuration:")
print(f"  Run ID: {args.run_id}")
print(f"  Dataset: MNIST (subset)")
print(f"  Depth: {args.depth}")
print(f"  Type Ansatz: {args.type_ansatz}")
print(f"  Optimizer: {args.optimizer}")
print(f"  Max Iterations: {args.maxiter}")
print("="*70)

# Load data from mnist_8features_data.npz
print("\n Loading training and test data...")
loaded_data = np.load('data/mnist_8features_data.npz')

# Extract the data (300 train, 100 val, 100 test)
X_train = loaded_data['X_train']
y_train = loaded_data['y_train']
X_val = loaded_data['X_val']
y_val = loaded_data['y_val']
X_test = loaded_data['X_test']
y_test = loaded_data['y_test']

print(f" Data loaded successfully:")
print(f"  Training set: {X_train.shape[0]} samples x {X_train.shape[1]} features")
print(f"  Validation set: {X_val.shape[0]} samples x {X_val.shape[1]} features")
print(f"  Test set: {X_test.shape[0]} samples x {X_test.shape[1]} features")

# Check label distribution
print(f"\n Label distribution (10 classes: 0-9):")
train_labels, train_counts = np.unique(y_train, return_counts=True)
for label, count in zip(train_labels, train_counts):
    print(f"  Digit {label}: {count} samples ({count/len(y_train)*100:.1f}%)")


def main():
    """
    Main training function for TQFM on MNIST dataset
    
    Steps:
    1. Initialize TQFM with specified parameters
    2. Train on training data
    3. Compute kernel matrices (before/after training, test)
    4. Save trained model and kernel matrices
    """
    
    # Create results_tqfm directory if it doesn't exist
    results_dir = "/data/ndquyen/svqsvm/results_tqfm"
    os.makedirs(results_dir, exist_ok=True)
    print(f"\nUsing output directory: {results_dir}")
    
    # Initialize TQFM circuit and parameters
    print(f"\n Initializing TQFM...")
    print(f"  Number of qubits: {X_train.shape[1]}")
    print(f"  Circuit depth: {args.depth}")
    print(f"  Ansatz type: {args.type_ansatz}")
    
    # Create optimizer instance based on argument
    optimizer_map = {
        "COBYLA": COBYLA(maxiter=args.maxiter),
        "SPSA": SPSA(maxiter=args.maxiter),
        "ADAM": ADAM(maxiter=args.maxiter),
        "L_BFGS_B": L_BFGS_B(maxiter=args.maxiter),
        "SLSQP": SLSQP(maxiter=args.maxiter)
    }
    optimizer = optimizer_map[args.optimizer]
    
    tqfm = TrainableQuantumFeatureMap(
        depth=args.depth, 
        type_ansatz=args.type_ansatz
    )
    
    # Train TQFM
    print(f"\n Starting TQFM training...")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Max iterations: {args.maxiter}")
    start_time = time.time()
    
    tqfm.fit(X_train, y_train, optimizer=optimizer)
    
    training_time = time.time() - start_time
    print(f" Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Compute kernel matrices
    print(f"\n Computing kernel matrices...")
    
    print("  Computing kernel matrix (before training)...")
    kernel_train_before = KernelMatrix.compute_kernel_matrix_with_inner_products(
        X_train, X_train, tqfm.init_theta, tqfm.circuit
    )
    
    print("  Computing kernel matrix (after training)...")
    kernel_train_after = KernelMatrix.compute_kernel_matrix_with_inner_products(
        X_train, X_train, tqfm.optimal_params, tqfm.circuit
    )

    print("  Computing validation kernel matrix...")
    kernel_val = KernelMatrix.compute_kernel_matrix_with_inner_products(
        X_val, X_train, tqfm.optimal_params, tqfm.circuit
    )

    print("  Computing test kernel matrix...")
    kernel_test = KernelMatrix.compute_kernel_matrix_with_inner_products(
        X_test, X_train, tqfm.optimal_params, tqfm.circuit
    )
    
    print(f" Kernel matrices computed:")
    print(f"  Kernel (before): {kernel_train_before.shape}")
    print(f"  Kernel (after): {kernel_train_after.shape}")
    print(f"  Kernel (val): {kernel_val.shape}")
    print(f"  Kernel (test): {kernel_test.shape}")
    
    
    # Save the trained model
    model_filename = f"{results_dir}/tqfm_mnist_depth{args.depth}_ansatz{args.type_ansatz}_iter{args.maxiter}_run{args.run_id}.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(tqfm, f)
    
    print(f"\n Trained model saved to: {model_filename}")
    
    # Save kernel matrices
    kernel_filename = f"{results_dir}/kernels_mnist_depth{args.depth}_ansatz{args.type_ansatz}_iter{args.maxiter}_run{args.run_id}.npz"
    np.savez(
        kernel_filename,
        kernel_train_before=kernel_train_before,
        kernel_train_after=kernel_train_after,
        kernel_val=kernel_val,
        kernel_test=kernel_test
    )
    
    print(f"💾 Kernel matrices saved to: {kernel_filename}")
    
    
    # Print final summary
    print("\n" + "="*70)
    print(" TQFM Training Complete!")
    print("="*70)
    print(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Model saved: {model_filename}")
    print(f"Kernels saved: {kernel_filename}")
    print("="*70 + "\n")

    return tqfm, kernel_train_before, kernel_train_after, kernel_val, kernel_test


if __name__ == "__main__":
    main()
