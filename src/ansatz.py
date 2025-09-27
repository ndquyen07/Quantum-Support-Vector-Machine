
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

class Ansatz:
    @staticmethod
    def TwoLocal(num_qubits, depth=1):
        """Creates a TwoLocal ansatz."""
        qc = QuantumCircuit(num_qubits, name='Ansatz')
        num_gates = 2
        xi = ParameterVector('ξ', length= num_gates * depth + 2)

        param_idx = 0
        # Initial first layer
        for i in range(num_qubits):
            qc.ry(xi[param_idx], i)
        param_idx += 1
        for i in range(num_qubits):
            qc.rz(xi[param_idx], i)
        param_idx += 1

        # Layers of parameterized rotations and entangling gates
        for _ in range(depth):
            for i in range(num_qubits - 1):
                qc.cz(i, i + 1)
            for i in range(num_qubits):
                qc.ry(xi[param_idx], i)
            param_idx += 1
            for i in range(num_qubits):
                qc.rz(xi[param_idx], i)
            param_idx += 1

        return qc
    
    
    @staticmethod
    def RealAmplitudes(num_qubits, depth=1):
        """Creates a RealAmplitudes ansatz."""
        qc = QuantumCircuit(num_qubits, name='Ansatz')
        num_gates = 1
        xi = ParameterVector('ξ', length= num_gates * depth + 1)

        param_idx = 0
        # Initial first layer
        for i in range(num_qubits):
            qc.ry(xi[param_idx], i)
        param_idx += 1

        # Layers of parameterized rotations and entangling gates
        for _ in range(depth):
            for i in range(num_qubits - 1):
                qc.cx(i, i + 1)
            for i in range(num_qubits):
                qc.ry(xi[param_idx], i)
            param_idx += 1

        return qc
    
    

    @staticmethod
    def EfficientSU2(num_qubits, depth=1, parameter_prefix='ξ', entanglement='linear'):
        """Creates an EfficientSU2 ansatz."""
        qc = QuantumCircuit(num_qubits, name='Ansatz')
        num_gates = 2
        xi = ParameterVector('ξ', length= num_gates * depth + 2)

        param_idx = 0
        # Initial first layer
        for i in range(num_qubits):
            qc.ry(xi[param_idx], i)
        param_idx += 1
        for i in range(num_qubits):
            qc.rz(xi[param_idx], i)
        param_idx += 1

        # Layers of parameterized rotations and entangling gates
        for _ in range(depth):
            for i in range(num_qubits - 1):
                qc.cx(i, i + 1)
            for i in range(num_qubits):
                qc.ry(xi[param_idx], i)
            param_idx += 1
            for i in range(num_qubits):
                qc.rz(xi[param_idx], i)
            param_idx += 1

        return qc
    

    @staticmethod
    def custom_ansatz1(num_qubits, depth=1):
        """Creates a custom parameterized quantum circuit (ansatz) with the given depth and number of qubits."""
        qc = QuantumCircuit(num_qubits, name='Ansatz')
        num_gates = 3
        xi = ParameterVector('ξ', length= num_gates * depth)
        param_idx = 0

        # Initial layer of Hadamard gates
        for i in range(num_qubits):
            qc.h(i)

        # Layers of parameterized rotations and entangling gates
        for _ in range(depth):
            for i in range(num_qubits):
                qc.rz(xi[param_idx], i)
            param_idx += 1
            for i in range(num_qubits):
                qc.ry(xi[param_idx], i)
            param_idx += 1
            for i in range(num_qubits):
                qc.rz(xi[param_idx], i)
            for i in range(num_qubits - 1):
                qc.cz(i, i + 1)
        return qc


    @staticmethod
    def custom_ansatz2(num_qubits, depth=1):
        qc = QuantumCircuit(num_qubits, name='Ansatz')
        num_gates = 3
        xi = ParameterVector('ξ', length= num_gates * depth)
        param_idx = 0

        # Initial layer of Hadamard gates
        for i in range(num_qubits):
            qc.h(i)

        # Layers of parameterized rotations and entangling gates
        for _ in range(depth):
            for i in range(num_qubits - 1):
                qc.cz(i, i + 1)
            for i in range(num_qubits):
                qc.rz(xi[param_idx], i)
            param_idx += 1
            for i in range(num_qubits):
                qc.ry(xi[param_idx], i)
            param_idx += 1
            for i in range(num_qubits):
                qc.rz(xi[param_idx], i)
        return qc

if __name__ == "__main__":
    # Example usage
    ansatz = Ansatz.EfficientSU2(4, 1)