import time
import os
import argparse
from src.feature_map import TrainableQuantumFeatureMap
import numpy as np
import pickle
from src.kernel_estimate import KernelMatrix
from qiskit_algorithms.optimizers import COBYLA, SPSA, ADAM


# Parse command line arguments
parser = argparse.ArgumentParser(description='Run TQFM training on Iris dataset with configurable parameters')
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
print("TQFM Training - Iris Dataset")
print("="*70)
print(f"Configuration:")
print(f"  Run ID: {args.run_id}")
print(f"  Dataset: Iris")
print(f"  Depth: {args.depth}")
print(f"  Type Ansatz: {args.type_ansatz}")
print(f"  Optimizer: {args.optimizer}")
print(f"  Max Iterations: {args.maxiter}")
print("="*70)

# Load data from iris_4features_data.npz
loaded_data = np.load('data/iris_4features_data.npz')

# Extract the data (341 train, 114 val, 114 test)
X_train = loaded_data['X_train']
y_train = loaded_data['y_train']
X_val = loaded_data['X_val']
y_val = loaded_data['y_val']
X_test = loaded_data['X_test']
y_test = loaded_data['y_test']

print(f"  Training set: {X_train.shape[0]} samples x {X_train.shape[1]} features")
print(f"  Validation set: {X_val.shape[0]} samples x {X_val.shape[1]} features")
print(f"  Test set: {X_test.shape[0]} samples x {X_test.shape[1]} features")


def main():
    """
    Main training function for TQFM on Iris dataset
    
    Steps:
    1. Initialize TQFM with specified parameters
    2. Train on training data
    3. Compute kernel matrices (before/after training, test)
    4. Save trained model and kernel matrices
    """
    
    # Create results_tqfm directory if it doesn't exist
    results_dir = "/data/ndquyen/svqsvm/results_tqfm_iris"
    os.makedirs(results_dir, exist_ok=True)
    print(f"\nUsing output directory: {results_dir}")
    
    # Initialize TQFM circuit and parameters
    # Create optimizer instance based on argument
    optimizer_map = {
        "COBYLA": COBYLA(maxiter=args.maxiter),
        "SPSA": SPSA(maxiter=args.maxiter),
        "ADAM": ADAM(maxiter=args.maxiter, tol= 0.001, lr= 0.1)
    }

    optimizer = optimizer_map[args.optimizer]
    
    tqfm = TrainableQuantumFeatureMap(
        depth=args.depth, 
        type_ansatz=args.type_ansatz
    )
    

    # Train TQFM
    start_time = time.time()
    
    tqfm.fit(X_train, y_train, optimizer=optimizer)
    
    training_time = time.time() - start_time
    print(f" Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Compute kernel matrices
    
    start_time = time.time()
    kernel_train_before = KernelMatrix.compute_kernel_matrix_with_inner_products(
        X_train, X_train, tqfm.init_theta, tqfm.circuit, use_aer=False, verbose=False
    )
    kernel_time = time.time() - start_time
    print(f" Calculate kernel_train_before completed in {kernel_time:.2f} seconds ({kernel_time/60:.2f} minutes)")

    
    start_time = time.time()
    kernel_train_after = KernelMatrix.compute_kernel_matrix_with_inner_products(
        X_train, X_train, tqfm.optimal_params, tqfm.circuit, use_aer=False, verbose=False
    )
    kernel_time = time.time() - start_time
    print(f" Calculate kernel_train_after completed in {kernel_time:.2f} seconds ({kernel_time/60:.2f} minutes)")


    start_time = time.time()
    kernel_val = KernelMatrix.compute_kernel_matrix_with_inner_products(
        X_val, X_train, tqfm.optimal_params, tqfm.circuit, use_aer=False, verbose=False
    )
    kernel_time = time.time() - start_time
    print(f" Calculate kernel_val completed in {kernel_time:.2f} seconds ({kernel_time/60:.2f} minutes)")


    start_time = time.time()
    kernel_test = KernelMatrix.compute_kernel_matrix_with_inner_products(
        X_test, X_train, tqfm.optimal_params, tqfm.circuit, use_aer=False, verbose=False
    )
    kernel_time = time.time() - start_time
    print(f" Calculate kernel_test completed in {kernel_time:.2f} seconds ({kernel_time/60:.2f} minutes)")
    
    
    # Save the trained model
    model_filename = f"{results_dir}/tqfm_iris_depth{args.depth}_ansatz{args.type_ansatz}_optimizer{args.optimizer}_iter{args.maxiter}_run{args.run_id}.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(tqfm, f)
    
    print(f"\n Trained model saved to: {model_filename}")
    
    # Save kernel matrices
    kernel_filename = f"{results_dir}/kernels_iris_depth{args.depth}_ansatz{args.type_ansatz}_optimizer{args.optimizer}_iter{args.maxiter}_run{args.run_id}.npz"
    np.savez(
        kernel_filename,
        kernel_train_before=kernel_train_before,
        kernel_train_after=kernel_train_after,
        kernel_val=kernel_val,
        kernel_test=kernel_test,
    )
    
    print(f" Kernel matrices saved to: {kernel_filename}")
    

    return tqfm, kernel_train_before, kernel_train_after, kernel_val, kernel_test


if __name__ == "__main__":
    main()
