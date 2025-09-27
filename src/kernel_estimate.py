import numpy as np
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit import transpile


class KernelMatrix:
    """Class to compute the kernel matrix"""

    @staticmethod
    def compute_kernel_matrix_with_inner_products(X1, X2, theta_params, circuit):
        """
        Compute the kernel matrix K_ij = |<psi(x_i, theta)|psi(x_j, theta)>|^2 using inner products of statevectors.
        """
        kernel_matrix = np.zeros((X1.shape[0], X2.shape[0]))
        for i in range(X1.shape[0]):
            for j in range(X2.shape[0]):
                # Prepare statevector for x_i
                param_dict_i = {}
                for k in range(X1.shape[1]):
                    param_dict_i[circuit.parameters[k]] = X1[i, k]
                theta_params_list = list(circuit.parameters)[X1.shape[1]:]
                for k, t in enumerate(theta_params):
                    param_dict_i[theta_params_list[k]] = t
                sv_i = Statevector.from_instruction(circuit.assign_parameters(param_dict_i))

                # Prepare statevector for x_j
                param_dict_j = {}
                for k in range(X2.shape[1]):
                    param_dict_j[circuit.parameters[k]] = X2[j, k]
                for k, t in enumerate(theta_params):
                    param_dict_j[theta_params_list[k]] = t
                sv_j = Statevector.from_instruction(circuit.assign_parameters(param_dict_j))

                # Compute squared overlap
                kernel_matrix[i, j] = np.abs(np.vdot(sv_i.data, sv_j.data))**2

        return kernel_matrix


    @staticmethod
    def compute_kernel_matrix_by_adjoint_method(X1, X2, theta_params, circuit, shots=1024):
        """
        Compute kernel matrix using the adjoint method: K_ij = |<psi(x_i)|psi(x_j)>|^2
        
        Theory: Apply U(x_i) then U†(x_j) to |0⟩. The probability of measuring |0⟩^n 
        gives Re[⟨psi(x_i)|psi(x_j)⟩]. For |⟨psi(x_i)|psi(x_j)⟩|², we need the 
        swap test or use this method only when states are real.
        
        Note: This method assumes quantum states are real-valued. For complex states,
        use the swap test or inner product method instead.
        """
        kernel_matrix = np.zeros((X1.shape[0], X2.shape[0]))
        
        for i in range(X1.shape[0]):
            for j in range(X2.shape[0]):
                # Create quantum circuit with only quantum registers (no classical bits for now)
                qc = QuantumCircuit(circuit.num_qubits)

                # Apply U(x_i) to |0⟩ to prepare |ψ(x_i)⟩
                param_dict_i = {}
                for k in range(X1.shape[1]):
                    param_dict_i[circuit.parameters[k]] = X1[i, k]
                theta_params_list = list(circuit.parameters)[X1.shape[1]:]
                for k, t in enumerate(theta_params):
                    param_dict_i[theta_params_list[k]] = t

                circuit_i = circuit.assign_parameters(param_dict_i)
                qc.compose(circuit_i, inplace=True)

                # Apply U†(x_j) to get ⟨ψ(x_j)|ψ(x_i)⟩ amplitude in |0⟩ state
                param_dict_j = {}
                for k in range(X2.shape[1]): 
                    param_dict_j[circuit.parameters[k]] = X2[j, k]
                for k, t in enumerate(theta_params):
                    param_dict_j[theta_params_list[k]] = t

                circuit_j = circuit.assign_parameters(param_dict_j)
                qc.compose(circuit_j.inverse(), inplace=True)

                # Add measurements
                qc.measure_all()
                
                # Execute circuit
                backend = AerSimulator()
                qc_t = transpile(qc, backend)
                result = backend.run(qc_t, shots=shots).result()
                counts = result.get_counts()
                
                # Get probability of measuring |0⟩^n state
                zero_state = "0" * circuit.num_qubits
                prob_zero = counts.get(zero_state, 0) / shots
                
                # For real quantum states: |⟨ψ_i|ψ_j⟩|² ≈ P(|0⟩^n)
                # For complex states, this gives only Re[⟨ψ_i|ψ_j⟩]
                kernel_matrix[i, j] = prob_zero
        return kernel_matrix


    @staticmethod
    def compute_kernel_matrix_swap_test(X1, X2, theta_params, circuit, shots=1024):
        """
        Compute kernel matrix using the swap test: K_ij = |⟨ψ(x_i)|ψ(x_j)⟩|²
        
        Theory: The swap test uses an ancilla qubit and controlled-SWAP gates to 
        measure the squared overlap between two quantum states exactly.
        
        Circuit: |0⟩_anc ⊗ U(x_i)|0⟩ ⊗ U(x_j)|0⟩ 
                → H_anc → cSWAP → H_anc → measure ancilla
        
        Result: P(ancilla = 0) = (1 + |⟨ψ(x_i)|ψ(x_j)⟩|²)/2
        Therefore: |⟨ψ(x_i)|ψ(x_j)⟩|² = 2×P(ancilla = 0) - 1
        """
        kernel_matrix = np.zeros((X1.shape[0], X2.shape[0]))

        n_qubits = circuit.num_qubits

        for i in range(X1.shape[0]):
            for j in range(X2.shape[0]):
                if i == j:
                    kernel_matrix[i, j] = 1.0  # Perfect overlap for same states
                    continue
                    
                # Create circuit: ancilla + 2 copies of the original circuit
                qc = QuantumCircuit(1 + 2 * n_qubits, 1)  # ancilla + 2 register copies
                
                # Prepare |ψ(x_i)⟩ in first register
                param_dict_i = {}
                for k in range(X1.shape[1]):
                    param_dict_i[circuit.parameters[k]] = X1[i, k]
                theta_params_list = list(circuit.parameters)[X1.shape[1]:]
                for k, t in enumerate(theta_params):
                    param_dict_i[theta_params_list[k]] = t

                circuit_i = circuit.assign_parameters(param_dict_i)
                qc.compose(circuit_i, qubits=range(1, 1 + n_qubits), inplace=True)
                
                # Prepare |ψ(x_j)⟩ in second register  
                param_dict_j = {}
                for k in range(X2.shape[1]):
                    param_dict_j[circuit.parameters[k]] = X2[j, k]
                for k, t in enumerate(theta_params):
                    param_dict_j[theta_params_list[k]] = t

                circuit_j = circuit.assign_parameters(param_dict_j)
                qc.compose(circuit_j, qubits=range(1 + n_qubits, 1 + 2 * n_qubits), inplace=True)
                
                # Swap test protocol
                qc.h(0)  # Hadamard on ancilla
                
                # Controlled-SWAP gates between registers
                for k in range(n_qubits):
                    qc.cswap(0, 1 + k, 1 + n_qubits + k)
                
                qc.h(0)  # Final Hadamard on ancilla
                qc.measure(0, 0)  # Measure ancilla
                
                # Execute circuit
                backend = AerSimulator()
                qc_t = transpile(qc, backend)
                result = backend.run(qc_t, shots=shots).result()
                counts = result.get_counts()
                
                # Extract kernel value from swap test
                prob_0 = counts.get('0', 0) / shots
                kernel_value = 2 * prob_0 - 1
                kernel_matrix[i, j] = max(0, kernel_value)  # Clamp negative values due to shot noise
        
        return kernel_matrix


    @staticmethod
    def plot_kernel_matrix(kernel_matrix, title="Quantum Kernel Matrix", filename=None, cmap='Greys'):
        """Plot the kernel matrix as a heatmap."""
        import matplotlib.pyplot as plt
        plt.imshow(kernel_matrix, cmap=cmap, interpolation='nearest')
        plt.colorbar(label='Kernel Value')
        plt.title(title)
        if filename:
            plt.savefig(filename)
        plt.show()


    @staticmethod
    def plot_multi_kernel_matrices(matrices, titles, filename=None, cmap='Greys'):
        """Plot multiple kernel matrices side by side."""
        import matplotlib.pyplot as plt
        n = len(matrices)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        for i in range(n):
            ax = axes[i] if n > 1 else axes
            im = ax.imshow(matrices[i], cmap=cmap, interpolation='nearest')
            ax.set_title(titles[i])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if filename:
            plt.savefig(filename)
        plt.show()


    
if __name__ == "__main__":
    # Example usage
    pass