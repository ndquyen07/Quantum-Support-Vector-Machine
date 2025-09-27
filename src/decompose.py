import numpy as np
from qiskit.quantum_info import SparsePauliOp

class Decomposer:
    """
    A class to decompose a Kernel Matrix into Pauli operators by Random Sampling.
    """

    # --- Pauli matrices ---
    paulis = {
        "I": np.array([[1, 0],[0, 1]], dtype=complex),
        "X": np.array([[0, 1],[1, 0]], dtype=complex),
        "Y": np.array([[0, -1j],[1j, 0]], dtype=complex),
        "Z": np.array([[1, 0],[0, -1]], dtype=complex),
    }

    # Projectors in Pauli basis
    projector_decomp = {
        (0,0): [("I",0.5), ("Z",0.5)],
        (1,1): [("I",0.5), ("Z",-0.5)],
        (0,1): [("X",0.5), ("Y",0.5j)],
        (1,0): [("X",0.5), ("Y",-0.5j)],
    }


    @classmethod
    def _expand_projector(cls, x_bits, y_bits):
        """Expand the projector |x><y| into Pauli terms."""
        terms = [([], 1.0)]
        for xb, yb in zip(x_bits, y_bits):
            local_terms = cls.projector_decomp[(xb, yb)]
            new_terms = []
            for prefix, coeff in terms:
                for (p, c) in local_terms:
                    new_terms.append((prefix+[p], coeff*c))
            terms = new_terms
        return ["".join(p) for p,c in terms], [c for p,c in terms]


    @classmethod
    def _random_sampling(cls, kernel_matrix, num_samples=200):
        """Standard random sampling method."""
        n = int(np.log2(kernel_matrix.shape[0]))
        coeffs = {}

        # Collect nonzero entries
        entries = []
        for i in range(kernel_matrix.shape[0]):
            for j in range(kernel_matrix.shape[1]):
                if abs(kernel_matrix[i,j]) > 0.0:
                    entries.append((i,j,kernel_matrix[i,j]))

        if not entries:
            return coeffs

        # Probability ∝ |K_ij|
        probs = np.array([abs(v) for _,_,v in entries])
        probs /= probs.sum()

        # Sample matrix entries and include ALL their Pauli contributions
        for _ in range(num_samples):
            idx = np.random.choice(len(entries), p=probs)
            i, j, val = entries[idx]

            # Binary expansion of i,j
            xi = [int(b) for b in format(i, f"0{n}b")]
            yj = [int(b) for b in format(j, f"0{n}b")]

            # Decompose |xi><yj| into ALL Pauli terms
            p_strings, p_coeffs = cls._expand_projector(xi, yj)
            entry_weight = len(entries) / num_samples
            
            for pauli, pauli_coeff in zip(p_strings, p_coeffs):
                contribution = val * pauli_coeff * entry_weight
                coeffs[pauli] = coeffs.get(pauli, 0) + contribution

        return coeffs


    @classmethod
    def _full_decompose(cls, kernel_matrix):
        """Compute full exact Pauli decomposition."""
        n = int(np.log2(kernel_matrix.shape[0]))
        coeffs = {}

        nonzero_indices = np.where(np.abs(kernel_matrix) > 0.0)
        
        for idx in range(len(nonzero_indices[0])):
            i, j = nonzero_indices[0][idx], nonzero_indices[1][idx]
            val = kernel_matrix[i, j]

            xi = [int(b) for b in format(i, f"0{n}b")]
            yj = [int(b) for b in format(j, f"0{n}b")]

            p_strings, p_coeffs = cls._expand_projector(xi,yj)

            for p, c in zip(p_strings, p_coeffs):
                coeffs[p] = coeffs.get(p, 0) + val * c

        return coeffs
    

    
    @classmethod
    def decompose_exact(cls, kernel_matrix):
        """Exact decomposition wrapper."""
        n = kernel_matrix.shape[0]
        if not (n > 0 and (n & (n - 1)) == 0):
            raise ValueError(f"Kernel matrix size {n} must be a power of 2")
        
    
        coeffs = cls._full_decompose(kernel_matrix)

        filtered_coeffs = {p:c for p,c in coeffs.items() if abs(c) > 0.0}
        pauli_list = []
        coeff_list = []
        for p,c in filtered_coeffs.items():
            pauli_list.append(p)
            coeff_list.append(c)

        return SparsePauliOp(pauli_list, coeff_list)
    


    @classmethod
    def decompose_random(cls, kernel_matrix, num_samples=500):
        """Random sampling decomposition wrapper."""
        n = kernel_matrix.shape[0]
        if not (n > 0 and (n & (n - 1)) == 0):
            raise ValueError(f"Kernel matrix size {n} must be a power of 2")
        

        coeffs = cls._random_sampling(kernel_matrix, num_samples=num_samples)

        filtered_coeffs = {p:c for p,c in coeffs.items() if abs(c) > cls.threshold}
        pauli_list = []
        coeff_list = []
        for p,c in filtered_coeffs.items():
            pauli_list.append(p)
            coeff_list.append(c)

        return SparsePauliOp(pauli_list, coeff_list)
    

    @staticmethod
    def convert_to_matrix(sparse_op):
        """Convert SparsePauliOp to matrix."""
        return sparse_op.to_matrix()



if __name__ == "__main__":
    # Example usage
    K = np.array([[1, 0.5], [0.5, 1]])

