import time
import os
import argparse
from datetime import datetime
from src.qsvm import QSVC
import numpy as np
import pickle

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run QSVC experiment with configurable ansatz')
parser.add_argument("--run_id", type=int, default=0, help="Run ID for experiment tracking")
parser.add_argument("--type_ansatz", type=str, default="custom1", 
                    choices=["custom1", "custom2", "TwoLocal", "RealAmplitudes", "EfficientSU2"],
                    help="Type of ansatz to use: custom1, custom2, TwoLocal, RealAmplitudes, EfficientSU2")
parser.add_argument("--depth", type=int, default=1, help="Depth of the ansatz circuit")
parser.add_argument("--C", type=float, default=1.0, help="Regularization parameter C")
parser.add_argument("--gamma", type=float, default=1.0, help="Gamma parameter")
parser.add_argument("--max_iter", type=int, default=100, help="Maximum iterations for optimization")
args = parser.parse_args()

print(f"Starting QSVC experiment - Run ID: {args.run_id}")
print(f"Configuration:")
print(f"  Type Ansatz: {args.type_ansatz}")
print(f"  Depth: {args.depth}")
print(f"  C: {args.C}")
print(f"  Gamma: {args.gamma}")
print(f"  Max Iterations: {args.max_iter}")

# Load data from cancer_data.npz
print("Loading training and test data...")
loaded_data = np.load('data/iris_data.npz')

# Extract the data
X_train = loaded_data['X_train']
y_train = loaded_data['y_train']
X_test = loaded_data['X_test']
y_test = loaded_data['y_test']

print(f"Data loaded - Training: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

# Load circuit and optimal parameters
print("Loading TQFM circuit and parameters...")
with open(f"data/tqfm_depth1_iris.pkl", "rb") as f:
	tqfm = pickle.load(f)


tqfm_circuit = tqfm.circuit
tqfm_optimal_params = tqfm.optimal_params

print(f"TQFM loaded - Qubits: {tqfm.num_qubits}, Depth: {tqfm.depth}")

# Load kernel matrix
print("Loading kernel matrix...")
kernel = np.load("data/kernel_matrix_after_iris.npy")
print(f"Kernel matrix shape: {kernel.shape}")

# Results storage
results = []

def main():
    """Run QSVM experiment with configurable ansatz
    Type ansatz options: custom1, custom2, TwoLocal, RealAmplitudes, EfficientSU2
    """
    print("="*50)
    print("Starting QSVC Training")
    print("="*50)
    
    start_time = time.time()
    
    # Train and evaluate QSVC with command line parameters
    print("Initializing QSVC...")
    print(f"  Using ansatz: {args.type_ansatz} with depth: {args.depth}")
    qsvc = QSVC(C=args.C, gamma=args.gamma, depth=args.depth, 
                type_ansatz=args.type_ansatz, optimizer='COBYLA', max_iter=args.max_iter)
    
    print("Fitting QSVC model...")
    qsvc.fit(X=X_train,
        y=y_train,
        kernel_matrix=kernel,
        theta_optimal=tqfm_optimal_params,
        parametrized_circuit=tqfm_circuit)

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Evaluate model
    print("Evaluating model on test set...")
    accuracy = qsvc.score(X_test, y_test)
    
    print("="*50)
    print("RESULTS")
    print("="*50)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Optimal Loss Value: {qsvc.optimal_value:.6f}")
    print(f"Training Time: {training_time:.2f} seconds")
    
    # Create result dictionary
    result = {
        'run_id': args.run_id,
        'type_ansatz': args.type_ansatz,
        'depth': args.depth,
        'C': args.C,
        'gamma': args.gamma,
        'max_iter': args.max_iter,
        'accuracy': accuracy,
        'optimal_value': qsvc.optimal_value,
        'optimal_params': qsvc.optimal_params,
        'training_time': training_time,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }
    
    # Store in results list
    results.append(result)
    
    # Create results directory if it doesn't exist
    results_dir = "results_iris"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Save individual run results with ansatz info in filename
    results_file = os.path.join(results_dir, f"qsvc_{args.type_ansatz}_d{args.depth}_run_{args.run_id}_{result['timestamp']}.npz")
    np.savez(results_file,
             run_id=args.run_id,
             type_ansatz=args.type_ansatz,
             depth=args.depth,
             C=args.C,
             gamma=args.gamma,
             max_iter=args.max_iter,
             accuracy=accuracy,
             optimal_value=qsvc.optimal_value,
             optimal_params=qsvc.optimal_params,
             training_time=training_time)
    
    print(f"Results saved to: {results_file}")
    print("QSVC experiment completed successfully!")

    return accuracy, qsvc.optimal_value, qsvc.optimal_params, training_time

    


if __name__ == "__main__":
    try:
        accuracy, optimal_value, optimal_params, training_time = main()
        print("Program finished successfully!")
        
        # Print final summary
        print("\n" + "="*50)
        print("FINAL SUMMARY")
        print("="*50)
        print(f"Run ID: {args.run_id}")
        print(f"Ansatz Type: {args.type_ansatz}")
        print(f"Depth: {args.depth}")
        print(f"C: {args.C}, Gamma: {args.gamma}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Optimal Value: {optimal_value:.6f}")
        print(f"Training Time: {training_time:.2f} seconds")
        print(f"Parameters shape: {np.array(optimal_params).shape}")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        print("Program terminated with errors!")
        raise

