# 🚀 Quantum Support Vector Machine (QSVM)

**The first fully quantum implementation of Quantum Support Vector Machine – from feature mapping to classification, no classical solver involved.**

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://python.org)
[![Qiskit](https://img.shields.io/badge/Qiskit-Latest-purple.svg)](https://qiskit.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Development-yellow.svg)]()
[![GitHub issues](https://img.shields.io/github/issues/ndquyen07/Quantum-Support-Vector-Machine)](https://github.com/ndquyen07/Quantum-Support-Vector-Machine/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/ndquyen07/Quantum-Support-Vector-Machine)](https://github.com/ndquyen07/Quantum-Support-Vector-Machine/pulls)


## 📖 Overview

This repository implements a **fully Quantum Support Vector Machine** using variational quantum algorithms. Unlike hybrid classical-quantum approaches, our QSVM performs both feature mapping and optimization entirely on quantum circuits, potentially offering exponential advantages for high-dimensional classification problems.

### 🔬 Key Features

- **🔵 Trainable Quantum Feature Maps**: Parametric quantum circuits that learn optimal data representations
- **🧠 Core QSVM Algorithm**: End-to-end QuantumSVM implementation, from feature mapping to classification 


## 🏗️ Architecture

```
📁 src/
├── 🧮 feature_map.py      # Trainable Quantum Feature Maps
├── 🔗 kernel.py           # Quantum Kernel Computation  
├── 🎯 qsvm.py             # Main QSVM Implementation
├── 🔧 decompose.py       # Circuit decomposition tools


```

## 🚀 Quick Start

### Prerequisites

```bash
pip install qiskit numpy scikit-learn matplotlib scipy
```

### Basic Usage

```python
from src.qsvm import QSVC
from src.feature_map import TrainableQuantumFeatureMap
from src.kernel import QuantumKernel

# 1. Train quantum feature map
tqfm = TrainableQuantumFeatureMap(depth=4, optimizer='COBYLA')
tqfm.fit(X_train, y_train)

# 2. Compute quantum kernel matrix
qkernel = QuantumKernel()
kernel = qkernel.compute_kernel_matrix_with_inner_products(
    X_train, X_train, tqfm.optimal_params, tqfm.circuit
)

# 3. Train QSVM classifier
qsvc = QSVC(C=1.0, gamma=1.0, optimizer='COBYLA', max_iter=100)
qsvc.fit(X_train, y_train, kernel_matrix=kernel, 
         theta_optimal=tqfm.optimal_params, 
         parametrized_circuit=tqfm.circuit)

# 4. Evaluate performance
accuracy = qsvc.score(X_test, y_test)
print(f"Quantum SVM Accuracy: {accuracy:.3f}")
```

## 🔬 Algorithm Details

### Quantum Feature Mapping

Our trainable quantum feature maps use parametric quantum circuits to embed classical data into a high-dimensional Hilbert space:

```
|Ψ(x,θ)⟩ = U(θ)|x⟩
```

Where `U(θ)` is a parameterized unitary transformation optimized during training.

### Quantum Kernel Estimation

The quantum kernel is computed using quantum state overlap:

```
K(x_i, x_j) = |⟨Ψ(x_i,θ)|Ψ(x_j,θ)⟩|²
```

This enables direct quantum advantage for kernel-based classification.


## 📚 References

1. Li Xu, Xiao-yu Zhang, Ming Li, Shu-qian Shen. "Quantum Classifiers with Trainable Kernel.",(2025).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏷️ Citation

**arXiv:** [arXiv:2505.04234v1 [quant-ph]](https://arxiv.org/abs/2505.04234v1)

---

**🌟 Star this repository if you find it useful!**

For questions or collaboration opportunities, feel free to open an issue or contact the maintainers.
