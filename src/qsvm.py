from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit import transpile

from src.decompose import Decomposer
from src.ansatz import Ansatz

import numpy as np
from scipy.optimize import minimize



class QSVC:
    """
    Custom Variational Quantum Support Vector Classifier (QSVC) implementation.
    """

    def __init__(self, C=1.0, gamma=1.0, depth=1, type_ansatz='EfficientSU2', random=False, num_samples=None, optimizer="COBYLA", max_iter=100):
        """
        Initialize the QSVC instance.

        """
        self.penalty = C
        self.gamma = gamma
        self.random = random
        self.num_samples = num_samples

        self.m = None
        self.ansatz = None
        self.type_ansatz = type_ansatz
        self.depth = depth
        self.initial_xi = None

        self.support_vectors = None
        self.c = None
        self.state_alpha = None
        self.alpha = None

        self.X = None
        self.y = None
        self.hamiltonian = None
        self.kernel_matrix = None
        self.theta_optimal = None
        self.parametrized_circuit = None

        self.ket_mu = None
        self.ket_upsilon = None

        self.transpile = transpile
        self.simulator = AerSimulator()

        self.optimizer = optimizer
        self.max_iter = max_iter

        self.optimal_params = None
        self.optimal_value = None

        self.basis_states = None

        self.loss_history = []


    @staticmethod
    def hadamard_test_circuit(num_qubit, bound_circuit, with_cz=False):
        """
        Create a Hadamard test circuit.
        """

        # total_qubits = ancilla (1) + data (m)
        qc = QuantumCircuit(1 + num_qubit, 1)

        anc = 0
        data = list(range(1, 1 + num_qubit))

        qc.h(anc)
        controlled_ansatz = bound_circuit.to_gate().control(1)
        qc.append(controlled_ansatz, [anc] + data)

        for i in data:
            qc.h(i)

        # Optional controlled-Z for l(α)
        if with_cz:
            qc.cz(anc, data[0]) 

        qc.h(anc)
        qc.measure(anc, 0)

        return qc
    

    @staticmethod
    def _compute_expectation(bound_circuit, hamiltonian):
        """
        Compute the expectation value <ψ(θ)|H|ψ(θ)>
        """
            
        try:
            # Debug: Check if bound_circuit and hamiltonian are valid
            if bound_circuit is None:
                print("Error: bound_circuit is None")
                return 0.0
                
            if hamiltonian is None:
                print("Error: hamiltonian is None")
                return 0.0
            
            # Check if bound_circuit has any unbound parameters
            if bound_circuit.parameters:
                print(f"Warning: bound_circuit still has unbound parameters: {bound_circuit.parameters}")
                return 0.0
                
            statevector = Statevector.from_instruction(bound_circuit)
            expectation = statevector.expectation_value(hamiltonian).real
        except Exception as e:
            print(f"Error computing expectation value: {repr(e)}")
            # Additional debugging info
            print(f"  bound_circuit type: {type(bound_circuit)}")
            print(f"  bound_circuit num_qubits: {getattr(bound_circuit, 'num_qubits', 'N/A')}")
            print(f"  hamiltonian type: {type(hamiltonian)}")
            print(f"  hamiltonian num_qubits: {getattr(hamiltonian, 'num_qubits', 'N/A')}")
            return 0.0

        return expectation
            
    

    def _compute_l1_norm(self, qc_template):
        """
        Compute the L1 norm || |α> ||_1 = (P0 - P1) * √2^m
        """

        transpiled_qc = self.transpile(qc_template, self.simulator)
        
        try:
            job = self.simulator.run(transpiled_qc, shots=4096)
            result = job.result()
            counts = result.get_counts()
            prob_0 = counts.get('0', 0) / 4096
            prob_1 = counts.get('1', 0) / 4096
        except Exception as e:
            print(f"Simulation error: {e}")
            return 0.0
        
        # Estimate ||α||_1 (approximate)
        l1_norm = (prob_0 - prob_1) * np.sqrt(2 ** self.m)

        return l1_norm

    
    def _compute_l_alpha(self, qc_template):
        """
        Compute l(α) = (P0 - P1) * √2^m
        """


        transpiled_qc = self.transpile(qc_template, self.simulator)

        try:
            job = self.simulator.run(transpiled_qc, shots=4096)
            result = job.result()
            counts = result.get_counts()
            prob_0 = counts.get('0', 0) / 4096
            prob_1 = counts.get('1', 0) / 4096
        except Exception as e: 
            print(f"Simulation error: {e}")
            return 0.0
        
        # Estimate l(α) (approximate)
        l_alpha = (prob_0 - prob_1) * np.sqrt(2 ** self.m)

        return l_alpha

        
    def _loss_function(self, xi, hamiltonian):
        r"""
        Compute the loss function
        L(\alpha) = \min_{\alpha} \frac{1}{2}\langle\alpha|K|\alpha\rangle - || | \alpha \rangle||_1 + Cl^2(\alpha)
        """
        try:
            param_dict = {param: xi[i] for i, param in enumerate(self.ansatz.parameters)}
            bound_circuit = self.ansatz.assign_parameters(param_dict)

            # Compute expectation value
            expectation = self._compute_expectation(bound_circuit, hamiltonian)

            # Compute || |α> ||_1
            qc_2 = self.hadamard_test_circuit(self.m ,bound_circuit, with_cz=False)
            l1_norm = self._compute_l1_norm(qc_2)

            # Compute l(α)
            qc_3 = self.hadamard_test_circuit(self.m, bound_circuit, with_cz=True)
            l_alpha = self._compute_l_alpha(qc_3)

            # Compute L(α)
            loss = 0.5 * expectation - l1_norm + self.penalty * (l_alpha ** 2)

            # Store loss history 
            self.loss_history.append(loss)
            
            # Print iteration progress
            # print(f"Iteration {len(self.loss_history)}: Loss = {loss}")

            return loss

        except Exception as e:
            print(f"Error in cost function: {e}")
            return float('inf')


    def _kernel_to_hamiltonian(self, kernel_matrix):
        """
        Convert kernel matrix to Hamiltonian using Pauli decomposition.
        """
        K = np.zeros_like(kernel_matrix)
        for i in range(kernel_matrix.shape[0]):
            for j in range(kernel_matrix.shape[1]):
                K[i, j] = self.y[i] * self.y[j] * kernel_matrix[i, j] + (1 if i == j else 0) / self.gamma

        if self.random:
            sparse_op = Decomposer.decompose_random(K, num_samples=self.num_samples)
        else:
            sparse_op = Decomposer.decompose_exact(K)
            
        return sparse_op

    

    def fit(self, X, y, kernel_matrix, theta_optimal, parametrized_circuit, initial_xi=None, ansatz=None):
        """
        Optimize the parameters to minimize the loss function.
        """
        self.X = X
        self.y = y
        self.kernel_matrix = kernel_matrix
        self.hamiltonian = self._kernel_to_hamiltonian(kernel_matrix)
        self.theta_optimal = theta_optimal
        self.parametrized_circuit = parametrized_circuit

        self.m = int(np.ceil(np.log2(len(self.X))))
        if ansatz is None:
            self._set_ansatz(self.type_ansatz, self.m, self.depth)
        else:
            self.ansatz = ansatz

        self.basis_states = [Statevector.from_label(bv) for bv in [format(i, f"0{self.m}b") for i in range(2**self.m)]]

        # Set initial point with proper validation
        expected_param_count = len(self.ansatz.parameters)
        
        if initial_xi is None:
            self.initial_xi = np.random.uniform(-np.pi, np.pi, expected_param_count)
            print(f"Using random initialization with {expected_param_count} parameters")
        else:
            # Validate and fix parameter dimensions if needed
            if len(initial_xi) != expected_param_count:
                print(f"Warning: initial_xi has {len(initial_xi)} parameters, expected {expected_param_count}")
                if len(initial_xi) < expected_param_count:
                    # Pad with random values
                    padding = np.random.uniform(0, 2*np.pi, expected_param_count - len(initial_xi))
                    self.initial_xi = np.concatenate([initial_xi, padding])
                    print(f"Padded initial_xi to {expected_param_count} parameters")
                else:
                    # Truncate
                    self.initial_xi = initial_xi[:expected_param_count]
                    print(f"Truncated initial_xi to {expected_param_count} parameters")
            else:
                self.initial_xi = initial_xi.copy()
                print(f"Using provided initialization with {len(initial_xi)} parameters")
            
            # Add small perturbation to avoid exact reuse (prevents local minima sticking)
            perturbation = 0.05  # Small perturbation (5% of 2π range)
            perturbation = np.random.normal(-perturbation, perturbation, expected_param_count)
            self.initial_xi = self.initial_xi + perturbation


        # Clear previous 
        self.loss_history = []
        self.ket_mu = None

        # Define optimization options
        options = {'maxiter': self.max_iter}
        if self.optimizer == 'COBYLA':
            options['disp'] = True

        # Perform optimization
        result = minimize(
            fun=self._loss_function,
            x0=self.initial_xi,
            args=(self.hamiltonian,),
            method=self.optimizer,
            options=options
        )

        # Store results
        self.optimal_params = result.x
        self.optimal_value = result.fun
        self.state_alpha = Statevector.from_instruction(
            self.ansatz.assign_parameters({self.ansatz.parameters[i]: self.optimal_params[i] for i in range(len(self.initial_xi))})
        )
        self.alpha = self.state_alpha.probabilities()
        self._set_support_vectors()

        return

    def set_alpha(self, parameters):
        """
        Set the alpha statevector using the provided parameters.
        """
        self.state_alpha = Statevector.from_instruction(
            self.ansatz.assign_parameters({self.ansatz.parameters[i]: parameters[i] for i in range(len(parameters))})
        )
        self.alpha = self.state_alpha.probabilities()
        self._set_support_vectors()


    def get_state_alpha(self):
        """
        Get the alpha statevector.
        """
        return self.state_alpha


    def _set_support_vectors(self):

        max_prob = max(self.state_alpha.probabilities())
        threshold = max_prob * 0.1

        self.support_vectors = []
        self.c = np.zeros(len(self.state_alpha))

        for k, v in enumerate(self.state_alpha.probabilities()):
            bitstring = format(k, f"0{self.m}b")
            if v > threshold:
                self.support_vectors.append(bitstring)
                self.c[k] = 1


    def _create_psi(self, x):
        """
        Create the quantum state |ψ(x, θ)⟩ using the parametrized circuit.
        """
        param_dict = {}
        x = x.reshape(1, -1)
        
        for k in range(x.shape[1]):
            param_dict[self.parametrized_circuit.parameters[k]] = x[0, k]
        theta_params_list = list(self.parametrized_circuit.parameters)[x.shape[1]:]
        for k, t in enumerate(self.theta_optimal):
            param_dict[theta_params_list[k]] = t
        psi = Statevector.from_instruction(self.parametrized_circuit.assign_parameters(param_dict))

        return psi


    def _prepare_ket_mu(self):
        """
        Prepare the state |μ> from data training set
        """
        ket_mu = 0
        for i in range(self.X.shape[0]):
            ket_mu += self.alpha[i] * self.y[i] * self.c[i] * np.kron(self.basis_states[i], self._create_psi(self.X[i]))

        return ket_mu


    def _prepare_ket_upsilon(self, x_new):
        """
        Prepare the state |υ(x_new)>
        """
        ket_upsilon = 0
        for i in range(self.X.shape[0]):
            ket_upsilon += self.c[i] * np.kron(self.basis_states[i], self._create_psi(x_new))
        ket_upsilon = ket_upsilon / np.sqrt(len(self.support_vectors))

        return ket_upsilon


    def predict(self, x_new):
        if self.ket_mu is None:
            self.ket_mu = self._prepare_ket_mu()
        else:
            self.ket_mu = self.ket_mu
        self.ket_upsilon = self._prepare_ket_upsilon(x_new)

        return np.sign(np.vdot(self.ket_mu, self.ket_upsilon).real)


    def score(self, X_test, y_test):
        """
        Compute the accuracy of the QSVC on the test set.

        Returns:
            float: Accuracy score
        """
        correct = 0
        for i in range(len(X_test)):
            pred = self.predict(X_test[i])
            if pred == y_test[i]:
                correct += 1
        accuracy = correct / len(X_test)
        return accuracy
    

    def get_alpha(self):
        """
        Get the alpha  of the QSVM.
        """
        return self.alpha


    def get_ansatz(self):
        """
        Get the ansatz circuit.

        Returns:
            QuantumCircuit: The ansatz circuit
        """
        return self.ansatz



    def get_optimal_circuit(self):
        """
        Get the quantum circuit with optimal parameters bound.

        Returns:
            QuantumCircuit: Circuit with optimal parameters
        """
        if self.optimal_params is None:
            raise ValueError("Must run compute_minimum_loss first!")

        param_dict = {param: self.optimal_params[i] 
                     for i, param in enumerate(self.ansatz.parameters)}
        return self.ansatz.assign_parameters(param_dict)



    def plot_convergence(self):
        """
        Plot the convergence of the loss function.
        """
        import matplotlib.pyplot as plt

        if not self.loss_history:
            print("No optimization history available. Run compute_minimum_loss first.")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, 'b-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Loss Function Value')
        plt.title('VQEplus Convergence')
        plt.grid(True)
        plt.show()


    def save_ansatz(self, filename):
        """
        Save the ansatz circuit to a file.
        """
        self.ansatz.draw('mpl', filename=filename)


    def save_convergence(self, filename):
        """
        Save the convergence plot of the loss function to a file.
        """
        import matplotlib.pyplot as plt

        if not self.loss_history:
            print("No optimization history available. Run compute_minimum_loss first.")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, 'b-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Loss Function Value')
        plt.title('VQEplus Convergence')
        plt.grid(True)
        plt.savefig(filename)
        plt.close()


    def plot_state_alpha(self, shots=1024):
        """
        Measure the state |α⟩ in the computational basis.
        """
        import matplotlib.pyplot as plt
        from qiskit.visualization import plot_histogram


        if self.state_alpha is None:
            raise ValueError("State |α⟩ is not prepared. Run fit() first.")

        qc = self.ansatz.copy()
        param_dict = {param: self.optimal_params[i]
                      for i, param in enumerate(self.ansatz.parameters)}
        qc = qc.assign_parameters(param_dict)
        qc.measure_all()

        transpiled_qc = self.transpile(qc, self.simulator)
        job = self.simulator.run(transpiled_qc, shots=shots)
        result = job.result()
        counts = result.get_counts()
        return plot_histogram(counts)


    def get_optimal_params(self):
        """
        Get the optimal parameters after fitting.

        Returns:
            np.array: Optimal parameter values
        """
        return self.optimal_params


    def get_optimal_value(self):
        """
        Get the optimal loss value after fitting.

        Returns:
            float: Optimal loss value
        """
        return self.optimal_value
    

    def set_parameter(self, C: float, gamma: float, initial_xi: np.array = None, ansatz: QuantumCircuit = None):
        """Set the initial parameter values."""
        self.penalty = C
        self.gamma = gamma
        self.initial_xi = initial_xi if initial_xi is not None else self.initial_xi

        if ansatz is not None:
            self.ansatz = ansatz


    def set_optimizer(self, optimizer: str, maxiter: int = 100):
        """Set the optimization method."""
        self.optimizer = optimizer
        self.max_iter = maxiter


    def _set_ansatz(self, type: str = 'EfficientSU2', num_qubits: int = 4, depth: int = 1, **kwargs):
        """
        Set ansatz circuit.
        """
        if type == 'TwoLocal':
            self.ansatz = Ansatz.TwoLocal(num_qubits, depth=depth, **kwargs)
        elif type == 'RealAmplitudes':
            self.ansatz = Ansatz.RealAmplitudes(num_qubits, depth=depth, **kwargs)
        elif type == 'EfficientSU2':
            self.ansatz = Ansatz.EfficientSU2(num_qubits, depth=depth, **kwargs)


if __name__ == "__main__":
    # Example usage
    pass