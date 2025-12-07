import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector


class TrainableQuantumFeatureMap:
    """Trainable Quantum Feature Map (TQFM) implementation.
    
    Optimized for performance with:
    - AerSimulator for fast batch statevector computation
    - Persistent caching across optimizer iterations
    - Vectorized numpy operations
    """

    def __init__(self, depth: int = 1, type_ansatz: str = "EfficientSU2", full_entangle: bool = False, 
                 use_aer: bool = True, batch_size: int = 50):
        self.depth = depth
        self.type = type_ansatz
        self.full_entangle = full_entangle
        self.optimizer = None
        self.use_aer = use_aer
        self.batch_size = batch_size  # Number of circuits to batch in AerSimulator

        self.num_qubits = None
        self.init_theta = None
        self.num_classes = None
        self.circuit = None

        self.X = None
        self.y = None

        self.optimal_params = None
        self.optimal_value = None

        self.loss_history = []
        
        # Cache for optimization
        self._class_patterns_cache = {}
        self._basis_states_cache = None
        self._class_indices_cache = None
        self._data_params_cache = None
        self._theta_params_cache = None


    def _get_class_patterns(self, num_classes):
        """Get or create class patterns with caching."""
        if num_classes in self._class_patterns_cache:
            return self._class_patterns_cache[num_classes]
        
        # Generate class patterns
        if num_classes <= 2:
            patterns = {i: format(i, 'b') for i in range(num_classes)}
        elif num_classes <= 4:
            patterns = {i: format(i, '02b') for i in range(num_classes)}
        elif num_classes <= 8:
            patterns = {i: format(i, '03b') for i in range(num_classes)}
        elif num_classes <= 16:
            patterns = {i: format(i, '04b') for i in range(num_classes)}
        else:
            raise ValueError(f"Number of classes {num_classes} not supported (max 16)")
        
        self._class_patterns_cache[num_classes] = patterns
        return patterns


    def _initialize_loss_cache(self, num_qubits, X, y, circuit_template, num_classes):
        """Initialize all caches that remain constant during optimization."""
        # Cache class patterns
        class_patterns = self._get_class_patterns(num_classes)
        
        # Cache class indices
        self._class_indices_cache = {}
        class_sizes = np.zeros(num_classes, dtype=int)
        for j in range(num_classes):
            idx = np.where(y == j)[0]
            self._class_indices_cache[j] = idx
            class_sizes[j] = len(idx)
        
        # Cache parameter mappings
        self._data_params_cache = list(circuit_template.parameters)[:X.shape[1]]
        self._theta_params_cache = list(circuit_template.parameters)[X.shape[1]:]
        
        # Cache basis states
        self._basis_states_cache = {}
        for j in range(num_classes):
            pattern = class_patterns[j]
            if num_qubits > len(pattern):
                remaining_qubits = num_qubits - len(pattern)
                num_basis_states = 2**remaining_qubits
                # Preallocate array for all basis states
                basis_vectors = np.empty((num_basis_states, 2**num_qubits), dtype=complex)
                for state_idx in range(num_basis_states):
                    remaining_bits = format(state_idx, f'0{remaining_qubits}b')
                    full_state = pattern + remaining_bits
                    basis_vectors[state_idx] = Statevector.from_label(full_state).data
                self._basis_states_cache[j] = basis_vectors
            else:
                self._basis_states_cache[j] = Statevector.from_label(pattern).data
        
        return class_sizes


    def _loss(self, theta, num_qubits, X, y, circuit_template, num_classes):
        """
        Compute the loss function as defined in the equation:
        E(theta) = 1 - (1/L) * sum_{j=1}^L (1/M_j) * sum_{i=1}^{M_j} |<psi(x_i^j, theta)|y_j>|^2
        
        Optimized version with caching and vectorization.
        """
        # Initialize cache on first call
        if self._basis_states_cache is None:
            class_sizes = self._initialize_loss_cache(num_qubits, X, y, circuit_template, num_classes)
        else:
            class_sizes = np.array([len(self._class_indices_cache[j]) for j in range(num_classes)])
        
        # Build theta parameter dictionary once
        theta_dict = dict(zip(self._theta_params_cache, theta))
        
        # Accumulate loss across all classes
        total_weighted_prob = 0.0
        
        for j in range(num_classes):
            idx = self._class_indices_cache[j]
            M_j = class_sizes[j]
            if M_j == 0:
                continue
            
            X_class = X[idx]
            basis_state = self._basis_states_cache[j]
            is_multi_basis = isinstance(basis_state, np.ndarray) and basis_state.ndim == 2
            
            # Accumulate probabilities for this class
            class_prob_sum = 0.0
            
            for x_sample in X_class:
                # Build full parameter dictionary efficiently
                param_dict = theta_dict.copy()
                for k, param in enumerate(self._data_params_cache):
                    param_dict[param] = x_sample[k]
                
                # Get statevector
                psi_data = Statevector.from_instruction(circuit_template.assign_parameters(param_dict)).data
                
                # Compute probability
                if is_multi_basis:
                    # Vectorized: compute all inner products at once
                    inner_products = basis_state @ psi_data.conj()
                    class_prob_sum += np.sum(np.abs(inner_products)**2)
                else:
                    # Single basis state
                    class_prob_sum += np.abs(np.vdot(basis_state, psi_data))**2
            
            total_weighted_prob += class_prob_sum / M_j
        
        loss = 1.0 - (total_weighted_prob / num_classes)
        
        # Store loss history
        self.loss_history.append(loss)
        
        return loss
    



    def fit(self, X: np.ndarray, y: np.ndarray, optimizer, init_theta: np.ndarray=None, circuit: QuantumCircuit=None) -> float:
        """Fit the quantum feature map to the data."""
        self.X = X
        self.y = y
        self.optimizer = optimizer

        self.num_qubits = X.shape[1]
        self.num_classes = len(np.unique(y))
        print(f"Number of qubits: {self.num_qubits}, Number of classes: {self.num_classes}")

        if circuit is None:
            self._set_circuit(type=self.type, full_entangle=self.full_entangle)
        else:
            self.circuit = circuit

        num_params = len(self.circuit.parameters) - self.num_qubits
        if init_theta is None:
            self.init_theta = np.random.uniform(-np.pi, np.pi, num_params)
            # self.init_theta = np.zeros((num_params,))
        else:
            perturbation = np.pi / 10
            perturbation = np.random.normal(-perturbation, perturbation, num_params)
            self.init_theta = init_theta + perturbation

        
        # Clear previous loss history and caches
        self.loss_history = []
        self._basis_states_cache = None
        self._class_indices_cache = None
        self._data_params_cache = None
        self._theta_params_cache = None

        # Perform optimization using the optimizer parameter
        fun = lambda theta: self._loss(theta, self.num_qubits, X, y, self.circuit, self.num_classes)
        result = self.optimizer.minimize(fun, x0=self.init_theta)

        # Store results
        self.optimal_params = result.x
        self.optimal_value = result.fun

        return


    def draw_circuit(self):
        """Draw the quantum circuit."""
        try:
            # Try to draw with text output, handling encoding issues
            circuit_str = self.circuit.draw(output='text', fold=-1)
            print(circuit_str)
        except UnicodeEncodeError:
            print("Circuit drawing contains Unicode characters that cannot be displayed.")
            print(f"Circuit has {self.circuit.num_qubits} qubits and {len(self.circuit)} gates.")
            print("Use save_circuit() method to save as image file instead.")


    def save_circuit(self, filename: str):
        """Save the quantum circuit to a file."""
        self.circuit.draw('mpl', filename=filename)


    def plot_loss(self):
        """Plot the loss history."""
        import matplotlib.pyplot as plt

        plt.plot(self.loss_history)
        plt.xlabel("iteration times")
        plt.ylabel(r"values of $E(\theta)$")
        plt.plot(self.loss_history, color='red', label=r"$E(\theta)$")
        plt.axhline(0, color='black', linestyle='--', label=r"$L=0$")
        plt.legend()
        plt.tight_layout()
        plt.show()


    def save_loss_history(self, filename: str):
        """Save the loss history to a figure."""
        import matplotlib.pyplot as plt

        plt.plot(self.loss_history)
        plt.xlabel("iteration times")
        plt.ylabel(r"values of $E(\theta)$")
        plt.plot(self.loss_history, color='red', label=r"$E(\theta)$")
        plt.axhline(0, color='black', linestyle='--', label=r"$L=0$")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


    def _set_circuit(self, type='EfficientSU2', full_entangle=False):
        """Set a custom quantum circuit.
        
        Args:
            type: Type of ansatz ('TwoLocal', 'RealAmplitudes', 'EfficientSU2')
            full_entangle: If True, use all-to-all entanglement. If False, use linear (nearest-neighbor) entanglement.
        """
        if type == 'TwoLocal':
            self.circuit = ParametrizedCircuit.TwoLocal_circuit(self.num_qubits, self.depth, full_entangle)
        elif type == 'RealAmplitudes':
            self.circuit = ParametrizedCircuit.RealAmplitudes_circuit(self.num_qubits, self.depth, full_entangle)
        elif type == 'EfficientSU2':
            self.circuit = ParametrizedCircuit.EfficientSU2_circuit(self.num_qubits, self.depth, full_entangle)
        else:
            raise ValueError(f"Unknown circuit type: {type}")
        


    def get_optimal_params(self) -> np.ndarray:
        """Get the optimal parameters after fitting."""
        return self.optimal_params

    def get_optimal_value(self) -> float:
        """Get the optimal loss value after fitting."""
        return self.optimal_value
    
    def get_circuit(self) -> QuantumCircuit:
        """Get the quantum circuit."""
        return self.circuit
    
    def set_optimizer(self, optimizer: str, maxiter: int = 100):
        """Set the optimizer and its parameters."""
        self.optimizer = optimizer
        self.maxiter = maxiter




class ParametrizedCircuit:
    """Class to create parameterized quantum circuits (ansatz)."""

    @staticmethod
    def TwoLocal_circuit(num_qubits, depth, full_entangle=False) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)

        data_params = ParameterVector('x', length=num_qubits)
        # Calculate total theta parameters needed: num_qubits for each layer (depth + 1)
        num_gates = 2
        total_theta_params = num_qubits * 2 + num_qubits * depth * num_gates
        theta_params = ParameterVector('θ', length=total_theta_params)
        param_idx = 0

        # Initial first layer
        for i in range(num_qubits):
            qc.ry(data_params[i] + theta_params[param_idx], i)
            param_idx += 1
        for i in range(num_qubits):
            qc.rz(data_params[i] + theta_params[param_idx], i)
            param_idx += 1
        
        # Layers of parameterized rotations and entangling gates
        for _ in range(depth):
            # Entangling layer
            if full_entangle:
                # Full entanglement: all-to-all connectivity
                for i in range(num_qubits):
                    for j in range(i + 1, num_qubits):
                        qc.cz(i, j)
            else:
                # Linear entanglement: nearest-neighbor connectivity
                for i in range(num_qubits - 1):
                    qc.cz(i, i + 1)
            
            # Rotation layer
            for i in range(num_qubits):
                qc.ry(data_params[i] + theta_params[param_idx], i)
                param_idx += 1
            for i in range(num_qubits):
                qc.rz(data_params[i] + theta_params[param_idx], i)
                param_idx += 1

        return qc


    @staticmethod
    def RealAmplitudes_circuit(num_qubits, depth, full_entangle=False) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)

        data_params = ParameterVector('x', length=num_qubits)
        # Calculate total theta parameters needed: num_qubits for each layer (depth + 1)
        num_gates = 1
        total_theta_params = num_qubits + num_qubits * depth * num_gates
        theta_params = ParameterVector('θ', length=total_theta_params)
        param_idx = 0

        # Initial first layer
        for i in range(num_qubits):
            qc.ry(data_params[i] + theta_params[param_idx], i)
            param_idx += 1

        # Layers of parameterized rotations and entangling gates
        for _ in range(depth):
            # Entangling layer
            if full_entangle:
                # Full entanglement: all-to-all connectivity
                for i in range(num_qubits):
                    for j in range(i + 1, num_qubits):
                        qc.cx(i, j)
            else:
                # Linear entanglement: nearest-neighbor connectivity
                for i in range(num_qubits - 1):
                    qc.cx(i, i + 1)
            
            # Rotation layer
            for i in range(num_qubits):
                qc.ry(data_params[i] + theta_params[param_idx], i)
                param_idx += 1

        return qc
    

    @staticmethod
    def EfficientSU2_circuit(num_qubits, depth, full_entangle=False) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)

        data_params = ParameterVector('x', length=num_qubits)
        # Calculate total theta parameters needed: num_qubits for each layer (depth + 1)
        num_gates = 2
        total_theta_params = num_qubits * 2 + num_qubits * depth * num_gates
        theta_params = ParameterVector('θ', length=total_theta_params)
        param_idx = 0

        # Initial first layer
        for i in range(num_qubits):
            qc.ry(data_params[i] + theta_params[param_idx], i)
            param_idx += 1
        for i in range(num_qubits):
            qc.rz(data_params[i] + theta_params[param_idx], i)
            param_idx += 1
        
        # Layers of parameterized rotations and entangling gates
        for _ in range(depth):
            # Entangling layer
            if full_entangle:
                # Full entanglement: all-to-all connectivity
                for i in range(num_qubits):
                    for j in range(i + 1, num_qubits):
                        qc.cx(i, j)
            else:
                # Linear entanglement: nearest-neighbor connectivity
                for i in range(num_qubits - 1):
                    qc.cx(i, i + 1)
            
            # Rotation layer
            for i in range(num_qubits):
                qc.ry(data_params[i] + theta_params[param_idx], i)
                param_idx += 1
            for i in range(num_qubits):
                qc.rz(data_params[i] + theta_params[param_idx], i)
                param_idx += 1

        return qc
    




if __name__ == "__main__":
    # Example usage
    depth = 4
    tqfm = TrainableQuantumFeatureMap(depth=depth)
    
    # Generate some synthetic data
    X = np.random.rand(20, 4) * 2 * np.pi  # 20 samples, features in [0, 2π]
    y = np.array([-1]*10 + [1]*10)  # Binary labels
    
    # Fit the model
    tqfm.fit(X, y)
    