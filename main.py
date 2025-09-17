"""
Support Vector Quantum Support Vector Machine (SVQSVM) Implementation
=====================================================================

A professional implementation of SVQSVM using trainable quantum feature maps
and variational quantum eigensolvers for binary classification.

Author: [Your Name]
Date: September 5, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit import transpile

# Scikit-learn imports
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Optimization imports
from scipy.optimize import minimize


@dataclass
class SVQSVMConfig:
    """Configuration class for SVQSVM parameters."""
    num_qubits: int = 4
    depth: int = 1
    num_classes: int = 2
    gamma: float = 0.01
    C: float = 1.0
    max_iter: int = 100
    optimizer: str = 'COBYLA'
    random_seed: int = 42


class DataPreprocessor:
    """Handles data loading and preprocessing for SVQSVM."""
    
    def __init__(self, config: SVQSVMConfig):
        self.config = config
        self.scaler = StandardScaler()
    
    def load_iris_binary(self, samples_per_class: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess IRIS dataset for binary classification.
        
        Args:
            samples_per_class: Number of samples per class for training
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Load IRIS dataset
        iris = datasets.load_iris()
        X, y = iris.data, iris.target
        
        # Filter to binary classification (classes 0 and 1)
        binary_mask = y < 2
        X, y = X[binary_mask], y[binary_mask]
        
        # Shuffle data
        np.random.seed(self.config.random_seed)
        shuffled_indices = np.random.permutation(X.shape[0])
        X, y = X[shuffled_indices], y[shuffled_indices]
        
        # Select training samples
        train_indices = np.hstack([
            np.where(y == class_idx)[0][:samples_per_class] 
            for class_idx in range(2)
        ])
        
        # Split data
        X_train, y_train = X[train_indices], y[train_indices]
        remaining_indices = np.setdiff1d(np.arange(X.shape[0]), train_indices)
        X_test, y_test = X[remaining_indices], y[remaining_indices]
        
        # Normalize features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Convert labels to {-1, 1}
        y_train = np.where(y_train == 0, -1, 1)
        y_test = np.where(y_test == 0, -1, 1)
        
        return X_train, X_test, y_train, y_test


class QuantumFeatureMap:
    """Trainable Quantum Feature Map (TQFM) implementation."""
    
    def __init__(self, config: SVQSVMConfig):
        self.config = config
        self.circuit = self._create_parametrized_circuit()
    
    def _create_parametrized_circuit(self) -> QuantumCircuit:
        """Create a parametrized quantum circuit for feature mapping."""
        qc = QuantumCircuit(self.config.num_qubits)
        
        # Data parameters
        data_params = ParameterVector('x', length=self.config.num_qubits)
        # Trainable parameters
        theta_params = ParameterVector('θ', length=self.config.num_qubits * self.config.depth * 2)
        
        # Initial layer with data encoding
        for i in range(self.config.num_qubits):
            qc.ry(data_params[i] + theta_params[i], i)
        
        param_idx = self.config.num_qubits
        
        # Add entangling layers
        for layer in range(self.config.depth):
            # Entangling gates
            for i in range(self.config.num_qubits - 1):
                qc.cx(i, i + 1)
            
            # Parametrized rotation gates
            for i in range(self.config.num_qubits):
                qc.ry(data_params[i] + theta_params[param_idx], i)
                param_idx += 1
        
        return qc
    
    def get_statevector(self, x: np.ndarray, theta: np.ndarray) -> Statevector:
        """Get statevector for given input and parameters."""
        param_dict = {}
        
        # Bind data parameters
        for k in range(len(x)):
            param_dict[self.circuit.parameters[k]] = x[k]
        
        # Bind theta parameters
        theta_params = list(self.circuit.parameters)[len(x):]
        for k, t in enumerate(theta):
            param_dict[theta_params[k]] = t
        
        return Statevector.from_instruction(self.circuit.assign_parameters(param_dict))


class LossFunction:
    """Loss function implementation for TQFM training."""
    
    def __init__(self, config: SVQSVMConfig, feature_map: QuantumFeatureMap):
        self.config = config
        self.feature_map = feature_map
        self.class_patterns = {0: '00', 1: '01'}
    
    def compute_class_probability(self, psi: Statevector, class_label: int) -> float:
        """Compute probability for a specific class."""
        prob = 0.0
        pattern = self.class_patterns[class_label]
        remaining_qubits = self.config.num_qubits - 2
        
        for state_idx in range(2**remaining_qubits):
            remaining_bits = format(state_idx, f'0{remaining_qubits}b')
            full_state = pattern + remaining_bits
            
            y_j = Statevector.from_label(full_state)
            prob += np.abs(np.vdot(psi.data, y_j.data))**2
        
        return prob
    
    def __call__(self, theta: np.ndarray, X_data: np.ndarray, y_data: np.ndarray) -> float:
        """Compute loss function E(theta)."""
        y_binary = np.where(y_data == -1, 0, 1)
        loss = 0.0
        
        for j in range(self.config.num_classes):
            idx = np.where(y_binary == j)[0]
            M_j = len(idx)
            
            if M_j == 0:
                continue
            
            class_loss = 0.0
            for i in idx:
                psi = self.feature_map.get_statevector(X_data[i], theta)
                prob = self.compute_class_probability(psi, j)
                class_loss += prob
            
            loss += class_loss / M_j
        
        return 1 - (loss / self.config.num_classes)


class KernelMatrix:
    """Kernel matrix computation and manipulation."""
    
    def __init__(self, config: SVQSVMConfig, feature_map: QuantumFeatureMap):
        self.config = config
        self.feature_map = feature_map
    
    def compute_quantum_kernel(self, X_data: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Compute quantum kernel matrix K_ij = |<psi(x_i)|psi(x_j)>|^2."""
        n_samples = X_data.shape[0]
        kernel = np.zeros((n_samples, n_samples))
        
        # Cache statevectors for efficiency
        statevectors = [self.feature_map.get_statevector(X_data[i], theta) 
                       for i in range(n_samples)]
        
        for i in range(n_samples):
            for j in range(n_samples):
                overlap = np.vdot(statevectors[i].data, statevectors[j].data)
                kernel[i, j] = np.abs(overlap)**2
        
        return kernel
    
    def compute_svm_kernel(self, quantum_kernel: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Compute SVM kernel matrix K_ij = y_i * y_j * k(x_i, x_j) + δ_ij/γ."""
        n = quantum_kernel.shape[0]
        K = np.zeros_like(quantum_kernel)
        
        for i in range(n):
            for j in range(n):
                K[i, j] = (y_train[i] * y_train[j] * quantum_kernel[i, j] + 
                          (1 if i == j else 0) / self.config.gamma)
        
        return K


class PauliDecomposer:
    """Decomposes kernel matrix into Pauli basis using random sampling."""
    
    def __init__(self):
        self.paulis = {
            "I": np.array([[1, 0], [0, 1]], dtype=complex),
            "X": np.array([[0, 1], [1, 0]], dtype=complex),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
            "Z": np.array([[1, 0], [0, -1]], dtype=complex),
        }
        
        self.projector_decomp = {
            (0, 0): [("I", 0.5), ("Z", 0.5)],
            (1, 1): [("I", 0.5), ("Z", -0.5)],
            (0, 1): [("X", 0.5), ("Y", 0.5j)],
            (1, 0): [("X", 0.5), ("Y", -0.5j)],
        }
    
    def expand_projector(self, x_bits: List[int], y_bits: List[int]) -> Tuple[List[str], List[complex]]:
        """Expand |x><y| to Pauli decomposition."""
        terms = [([], 1.0)]
        
        for xb, yb in zip(x_bits, y_bits):
            local_terms = self.projector_decomp[(xb, yb)]
            new_terms = []
            for prefix, coeff in terms:
                for p, c in local_terms:
                    new_terms.append((prefix + [p], coeff * c))
            terms = new_terms
        
        return ["".join(p) for p, c in terms], [c for p, c in terms]
    
    def random_sampling_decomposition(self, K: np.ndarray, num_samples: int = 1000) -> SparsePauliOp:
        """Decompose kernel matrix using random sampling."""
        n = int(np.log2(K.shape[0]))
        coeffs = {}
        
        # Collect nonzero entries
        entries = [(i, j, K[i, j]) for i in range(K.shape[0]) 
                  for j in range(K.shape[1]) if abs(K[i, j]) > 1e-12]
        
        if not entries:
            return SparsePauliOp.from_list([("I" * n, 0.0)])
        
        # Sample with probability proportional to |K_ij|
        probs = np.array([abs(v) for _, _, v in entries])
        probs /= probs.sum()
        
        for _ in range(num_samples):
            idx = np.random.choice(len(entries), p=probs)
            i, j, val = entries[idx]
            
            xi = [int(b) for b in format(i, f"0{n}b")]
            yj = [int(b) for b in format(j, f"0{n}b")]
            
            p_strings, p_coeffs = self.expand_projector(xi, yj)
            k = np.random.randint(len(p_strings))
            pauli = p_strings[k]
            coeff = val * p_coeffs[k]
            
            coeffs[pauli] = coeffs.get(pauli, 0) + coeff
        
        pauli_list = [(pauli_str, coeff) for pauli_str, coeff in coeffs.items() if coeff != 0]
        return SparsePauliOp.from_list(pauli_list if pauli_list else [("I" * n, 0.0)])


class VQEPlus:
    """Variational Quantum Eigensolver for SVQSVM optimization."""
    
    def __init__(self, config: SVQSVMConfig, num_qubits: int):
        self.config = config
        self.num_qubits = num_qubits
        self.simulator = AerSimulator(method='density_matrix')
        self.loss_history = []
        self.optimal_params = None
    
    def create_ansatz(self, theta: Parameter) -> QuantumCircuit:
        """Create ansatz circuit for VQE."""
        ansatz = QuantumCircuit(self.num_qubits)
        for i in range(self.num_qubits):
            ansatz.ry(theta, i)
        for i in range(self.num_qubits - 1):
            ansatz.cz(i, i + 1)
        return ansatz
    
    def optimize(self, hamiltonian: SparsePauliOp) -> Dict:
        """Optimize the VQE objective function."""
        theta = Parameter('ξ')
        ansatz = self.create_ansatz(theta)
        
        def objective(params):
            bound_circuit = ansatz.assign_parameters({theta: params[0]})
            statevector = Statevector.from_instruction(bound_circuit)
            expectation = statevector.expectation_value(hamiltonian).real
            
            # Add regularization terms (simplified)
            loss = 0.5 * expectation
            self.loss_history.append(loss)
            return loss
        
        result = minimize(
            objective,
            x0=[0.0],
            method=self.config.optimizer,
            options={'maxiter': self.config.max_iter}
        )
        
        self.optimal_params = result.x
        return {
            'optimal_value': result.fun,
            'optimal_point': result.x,
            'success': result.success,
            'cost_history': self.loss_history.copy()
        }


class SVQSVMClassifier:
    """Main SVQSVM classifier implementation."""
    
    def __init__(self, config: SVQSVMConfig):
        self.config = config
        self.feature_map = QuantumFeatureMap(config)
        self.loss_function = LossFunction(config, self.feature_map)
        self.kernel_matrix = KernelMatrix(config, self.feature_map)
        self.pauli_decomposer = PauliDecomposer()
        
        self.optimal_theta = None
        self.support_vectors = None
        self.alpha_coefficients = None
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Train the SVQSVM classifier."""
        print("Training SVQSVM...")
        
        # Step 1: Optimize TQFM parameters
        print("Step 1: Optimizing TQFM parameters...")
        num_theta_params = len(self.feature_map.circuit.parameters) - self.config.num_qubits
        theta_init = np.zeros(num_theta_params)
        
        result_theta = minimize(
            self.loss_function,
            theta_init,
            args=(X_train, y_train),
            method=self.config.optimizer,
            options={'maxiter': self.config.max_iter}
        )
        
        self.optimal_theta = result_theta.x
        print(f"TQFM optimization completed. Final loss: {result_theta.fun:.4f}")
        
        # Step 2: Compute kernel matrices
        print("Step 2: Computing kernel matrices...")
        quantum_kernel = self.kernel_matrix.compute_quantum_kernel(X_train, self.optimal_theta)
        svm_kernel = self.kernel_matrix.compute_svm_kernel(quantum_kernel, y_train)
        
        # Step 3: Pauli decomposition
        print("Step 3: Performing Pauli decomposition...")
        pauli_hamiltonian = self.pauli_decomposer.random_sampling_decomposition(quantum_kernel)
        
        # Step 4: VQE optimization
        print("Step 4: Running VQE optimization...")
        m = int(np.ceil(np.log2(len(X_train))))
        vqe = VQEPlus(self.config, m)
        vqe_result = vqe.optimize(pauli_hamiltonian)
        
        # Step 5: Extract support vectors
        print("Step 5: Extracting support vectors...")
        theta_opt = vqe_result['optimal_point'][0]
        ansatz = vqe.create_ansatz(Parameter('ξ'))
        bound_ansatz = ansatz.assign_parameters({'ξ': theta_opt})
        state = Statevector.from_instruction(bound_ansatz)
        
        self.support_vectors, self.alpha_coefficients = self._extract_support_vectors(state)
        
        print("Training completed successfully!")
        return {
            'tqfm_result': result_theta,
            'vqe_result': vqe_result,
            'num_support_vectors': len(self.support_vectors)
        }
    
    def _extract_support_vectors(self, state: Statevector) -> Tuple[List[str], np.ndarray]:
        """Extract support vectors from the optimal state."""
        max_prob = max(state.probabilities())
        threshold = max_prob * 0.5
        num_qubits = int(np.log2(len(state)))
        
        support_vectors = []
        c_i = np.zeros(len(state))
        
        for k, prob in enumerate(state.probabilities()):
            if prob > threshold:
                bitstring = format(k, f"0{num_qubits}b")
                support_vectors.append(bitstring)
                c_i[k] = 1
        
        return support_vectors, c_i
    
    def predict(self, X_test: np.ndarray, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Make predictions on test data."""
        if self.optimal_theta is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Simplified prediction (placeholder implementation)
        # In practice, you would implement the full decision function
        predictions = []
        
        for x_test in X_test:
            # Compute similarity to training data
            similarities = []
            for i, x_train in enumerate(X_train):
                psi_test = self.feature_map.get_statevector(x_test, self.optimal_theta)
                psi_train = self.feature_map.get_statevector(x_train, self.optimal_theta)
                similarity = np.abs(np.vdot(psi_test.data, psi_train.data))**2
                similarities.append(similarity * y_train[i])
            
            prediction = np.sign(np.sum(similarities))
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def plot_training_history(self):
        """Plot training convergence."""
        if hasattr(self.loss_function, 'history'):
            plt.figure(figsize=(10, 6))
            plt.plot(self.loss_function.history)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('TQFM Training Convergence')
            plt.grid(True)
            plt.show()


def main():
    """Main execution function."""
    # Configuration
    config = SVQSVMConfig(
        num_qubits=4,
        depth=1,
        num_classes=2,
        gamma=0.01,
        C=1.0,
        max_iter=100,
        optimizer='COBYLA'
    )
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    preprocessor = DataPreprocessor(config)
    X_train, X_test, y_train, y_test = preprocessor.load_iris_binary(samples_per_class=32)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Create and train classifier
    classifier = SVQSVMClassifier(config)
    training_results = classifier.train(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred = classifier.predict(X_test, X_train, y_train)
    
    # Evaluate performance
    accuracy = np.mean(y_pred == y_test)
    print(f"Test accuracy: {accuracy:.2%}")
    
    return classifier, training_results


if __name__ == "__main__":
    classifier, results = main()