import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector


class TrainableQuantumFeatureMap:
    """Trainable Quantum Feature Map (TQFM) implementation."""

    def __init__(self, depth: int = 1, type_ansatz: str = "EfficientSU2"):
        self.depth = depth
        self.type = type_ansatz
        self.optimizer = None

        self.num_qubits = None
        self.init_theta = None
        self.num_classes = None
        self.circuit = None

        self.X = None
        self.y = None

        self.optimal_params = None
        self.optimal_value = None

        self.loss_history = []


    def _loss(self, theta, num_qubits, X, y, circuit_template, num_classes):
        """
        Compute the loss function as defined in the equation:
        E(theta) = 1 - (1/L) * sum_{j=1}^L (1/M_j) * sum_{i=1}^{M_j} |<psi(x_i^j, theta)|y_j>|^2

        """
    
        if num_classes == 2:
            class_patterns = {0: '0', 1: '1'}
        elif num_classes == 3:
            class_patterns = {0: '00', 1: '01', 2: '10'}
        elif num_classes == 4:
            class_patterns = {0: '00', 1: '01', 2: '10', 3: '11'}
        elif num_classes == 5:
            class_patterns = {0: '000', 1: '001', 2: '010', 3: '011', 4: '100'}
        elif num_classes == 6:
            class_patterns = {0: '000', 1: '001', 2: '010', 3: '011', 4: '100', 5: '101'}
        elif num_classes == 7:
            class_patterns = {0: '000', 1: '001', 2: '010', 3: '011', 4: '100', 5: '101', 6: '110'}
        elif num_classes == 8:
            class_patterns = {0: '000', 1: '001', 2: '010', 3: '011', 4: '100', 5: '101', 6: '110', 7: '111'}
        elif num_classes == 9:
            class_patterns = {0: '0000', 1: '0001', 2: '0010', 3: '0011', 4: '0100', 5: '0101', 6: '0110', 7: '0111', 8: '1000'}
        elif num_classes == 10:
            class_patterns = {0: '0000', 1: '0001', 2: '0010', 3: '0011', 4: '0100', 5: '0101', 6: '0110', 7: '0111', 8: '1000', 9: '1001'}

        loss = 0.0

        for j in range(num_classes):
            # Select samples of class j
            idx = np.where(y == j)[0]
            M_j = len(idx)
            if M_j == 0:
                continue
            class_loss = 0.0
            for i in idx:
                # Bind Data parameters
                param_dict = {}
                for k in range(X.shape[1]):
                    param_dict[circuit_template.parameters[k]] = X[i, k]
                # Bind Theta parameters
                theta_params = list(circuit_template.parameters)[X.shape[1]:]
                for k, t in enumerate(theta):
                    param_dict[theta_params[k]] = t
                # Get statevector
                psi = Statevector.from_instruction(circuit_template.assign_parameters(param_dict))

                # Create basis state for class j
                pattern = class_patterns[j]
                if num_qubits > len(pattern):
                    # Pad with zeros for remaining qubits
                    remaining_qubits = num_qubits - len(pattern)
                    pattern = pattern + '0' * remaining_qubits
                
                y_j = Statevector.from_label(pattern)
                prob = np.abs(np.vdot(psi.data, y_j.data))**2

                class_loss += prob
            loss += class_loss / M_j
        loss = 1 - (loss / num_classes)

        # Store loss history    
        self.loss_history.append(loss)
        # print(f"Iteration {len(self.loss_history)}: Loss = {loss}")
        
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
            self._set_circuit(type=self.type)
        else:
            self.circuit = circuit

        if init_theta is None:
            num_params = len(self.circuit.parameters) - self.num_qubits
            self.init_theta = np.random.uniform(-np.pi, np.pi, num_params)
            # self.init_theta = np.zeros((num_params,))
        else:
            self.init_theta = init_theta

        
        # Clear previous loss history
        self.loss_history = []

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


    def _set_circuit(self, type= 'EfficientSU2'):

        """Set a custom quantum circuit."""
        if type == 'TwoLocal':
            self.circuit = ParametrizedCircuit.TwoLocal_circuit(self.num_qubits, self.depth)
        elif type == 'RealAmplitudes':
            self.circuit = ParametrizedCircuit.RealAmplitudes_circuit(self.num_qubits, self.depth)
        elif type == 'EfficientSU2':
            self.circuit = ParametrizedCircuit.EfficientSU2_circuit(self.num_qubits, self.depth)
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
    def TwoLocal_circuit(num_qubits, depth) -> QuantumCircuit:
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
            for i in range(num_qubits - 1):
                qc.cz(i, i + 1)
            for i in range(num_qubits):
                qc.ry(data_params[i] + theta_params[param_idx], i)
                param_idx += 1
            for i in range(num_qubits):
                qc.rz(data_params[i] + theta_params[param_idx], i)
                param_idx += 1

        return qc


    @staticmethod
    def RealAmplitudes_circuit(num_qubits, depth) -> QuantumCircuit:
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
            for i in range(num_qubits - 1):
                qc.cx(i, i + 1)
            for i in range(num_qubits):
                qc.ry(data_params[i] + theta_params[param_idx], i)
                param_idx += 1

        return qc
    

    @staticmethod
    def EfficientSU2_circuit(num_qubits, depth) -> QuantumCircuit:
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
            for i in range(num_qubits - 1):
                qc.cx(i, i + 1)
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
    