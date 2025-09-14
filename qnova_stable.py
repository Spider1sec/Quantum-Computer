import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Union, Any
import warnings
from scipy.linalg import sqrtm
from scipy.optimize import minimize
import json
import time
from dataclasses import dataclass
from enum import Enum

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import QFT, GroverOperator, TwoLocal, EfficientSU2
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, entropy, mutual_information
from qiskit.quantum_info import state_fidelity, process_fidelity, average_gate_fidelity
from qiskit.quantum_info import random_statevector, random_unitary, Pauli, SparsePauliOp
from qiskit.quantum_info.operators import Operator

# --- UPDATED IMPORTS FOR QISKIT AER ---
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import pauli_error, depolarizing_error, amplitude_damping_error, phase_damping_error
from qiskit_aer.noise.errors import thermal_relaxation_error, coherent_unitary_error
# --- Import for visualizations ---
from qiskit.visualization import plot_histogram, plot_bloch_multivector, plot_state_qsphere
from qiskit.visualization import plot_state_city, plot_state_paulivec, plot_state_hinton
from qiskit.visualization import circuit_drawer, plot_circuit_layout
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
# --- END UPDATED IMPORTS ---

# Circuit analysis
from qiskit.transpiler import PassManager, passes
from qiskit.transpiler.passes import Depth, CountOps, Size

# Primitives for modern Qiskit
from qiskit.primitives import Sampler, Estimator

# Try to import display for Jupyter environments
try:
    from IPython.display import display, HTML
except ImportError:
    def display(fig):
        if hasattr(fig, 'show'):
            fig.show()
        else:
            print(fig)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class GateType(Enum):
    """Enumeration of available quantum gates."""
    # Single-qubit gates
    H = "hadamard"
    X = "pauli_x"
    Y = "pauli_y"
    Z = "pauli_z"
    S = "phase_s"
    T = "t_gate"
    SDG = "s_dagger"
    TDG = "t_dagger"
    RX = "rotation_x"
    RY = "rotation_y"
    RZ = "rotation_z"
    U = "universal"
    I = "identity"
    SX = "sqrt_x"
    
    # Two-qubit gates
    CX = "cnot"
    CY = "controlled_y"
    CZ = "controlled_z"
    CH = "controlled_h"
    SWAP = "swap"
    ISWAP = "iswap"
    CRX = "controlled_rx"
    CRY = "controlled_ry"
    CRZ = "controlled_rz"
    CU = "controlled_u"
    RXX = "rxx"
    RYY = "ryy"
    RZZ = "rzz"
    
    # Three-qubit gates
    CCX = "toffoli"
    CSWAP = "fredkin"
    CCZ = "controlled_cz"
    
    # Multi-qubit gates
    MCX = "multi_cnot"
    MCZ = "multi_cz"

    # New additions: Echo Cross-Resonance (ECR) and others for advanced hardware simulation
    ECR = "ecr"

@dataclass
class CircuitMetrics:
    """Data class for storing circuit metrics and statistics."""
    depth: int
    gate_count: Dict[str, int]
    qubit_count: int
    classical_bit_count: int
    two_qubit_gates: int
    multi_qubit_gates: int
    parameterized: bool
    connected_components: int
    critical_depth: int

class QuantumAlgorithmLibrary:
    """Library of pre-built quantum algorithms and subroutines."""
    
    @staticmethod
    def quantum_fourier_transform(n_qubits: int) -> QuantumCircuit:
        """Create a Quantum Fourier Transform circuit."""
        qc = QuantumCircuit(n_qubits)
        qft = QFT(n_qubits, do_swaps=True)
        qc.append(qft, range(n_qubits))
        return qc
    
    @staticmethod
    def grover_search(n_qubits: int, marked_states: List[str]) -> QuantumCircuit:
        """
        Create Grover's search algorithm circuit.
        
        Args:
            n_qubits: Number of qubits
            marked_states: List of binary strings representing marked states
        """
        qc = QuantumCircuit(n_qubits, n_qubits)
        
        # Initialize in superposition
        qc.h(range(n_qubits))
        
        # Calculate optimal number of iterations
        N = 2**n_qubits
        M = len(marked_states)
        iterations = int(np.pi/4 * np.sqrt(N/M))
        
        for _ in range(iterations):
            # Oracle
            for state in marked_states:
                qc.barrier()
                # Multi-controlled Z gate for marking
                controls = []
                for i, bit in enumerate(state):
                    if bit == '0':
                        qc.x(i)
                    controls.append(i)
                if n_qubits > 1:
                    qc.mcp(np.pi, controls[:-1], controls[-1])
                else:
                    qc.z(0)
                for i, bit in enumerate(state):
                    if bit == '0':
                        qc.x(i)
            
            # Diffusion operator
            qc.barrier()
            qc.h(range(n_qubits))
            qc.x(range(n_qubits))
            qc.h(n_qubits-1)
            if n_qubits > 1:
                qc.mcx(list(range(n_qubits-1)), n_qubits-1)
            else:
                qc.z(0)
            qc.h(n_qubits-1)
            qc.x(range(n_qubits))
            qc.h(range(n_qubits))
        
        qc.measure(range(n_qubits), range(n_qubits))
        return qc
    
    @staticmethod
    def quantum_phase_estimation(n_counting: int, unitary: QuantumCircuit) -> QuantumCircuit:
        """
        Create a Quantum Phase Estimation circuit.
        
        Args:
            n_counting: Number of counting qubits
            unitary: The unitary operator to estimate phase for
        """
        n_state = unitary.num_qubits
        qpe = QuantumCircuit(n_counting + n_state, n_counting)
        
        # Initialize counting qubits in superposition
        for q in range(n_counting):
            qpe.h(q)
        
        # Controlled unitary operations
        repetitions = 1
        for counting_qubit in range(n_counting):
            for _ in range(repetitions):
                qpe.append(unitary.control(), [counting_qubit] + list(range(n_counting, n_counting + n_state)))
            repetitions *= 2
        
        # Inverse QFT on counting qubits
        qpe.append(QFT(n_counting, inverse=True), range(n_counting))
        
        # Measure counting qubits
        qpe.measure(range(n_counting), range(n_counting))
        return qpe
    
    @staticmethod
    def variational_circuit(n_qubits: int, depth: int = 2) -> QuantumCircuit:
        """Create a parameterized variational circuit for VQE/QAOA."""
        params = ParameterVector('θ', n_qubits * depth * 3)
        qc = QuantumCircuit(n_qubits)
        
        param_idx = 0
        for d in range(depth):
            # Rotation layer
            for q in range(n_qubits):
                qc.rx(params[param_idx], q)
                qc.ry(params[param_idx + 1], q)
                qc.rz(params[param_idx + 2], q)
                param_idx += 3
            
            # Entanglement layer
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
            if n_qubits > 2:
                qc.cx(n_qubits - 1, 0)  # Circular entanglement
        
        return qc
    
    @staticmethod
    def quantum_teleportation() -> QuantumCircuit:
        """Create a quantum teleportation circuit."""
        qc = QuantumCircuit(3, 3)
        
        # Create entangled pair (Bell state) between qubits 1 and 2
        qc.h(1)
        qc.cx(1, 2)
        
        qc.barrier()
        
        # Bell measurement on qubits 0 and 1
        qc.cx(0, 1)
        qc.h(0)
        qc.measure([0, 1], [0, 1])
        
        qc.barrier()
        
        # Apply corrections to qubit 2 based on measurement results
        qc.cx(1, 2)
        qc.cz(0, 2)
        
        return qc
    
    @staticmethod
    def bb84_key_distribution(n_bits: int = 8) -> QuantumCircuit:
        """Create a BB84 quantum key distribution protocol circuit."""
        qc = QuantumCircuit(n_bits, n_bits * 2)
        
        # Alice prepares random bits in random bases
        np.random.seed(42)  # For reproducibility
        alice_bits = np.random.randint(0, 2, n_bits)
        alice_bases = np.random.randint(0, 2, n_bits)
        
        for i in range(n_bits):
            if alice_bits[i] == 1:
                qc.x(i)
            if alice_bases[i] == 1:
                qc.h(i)
        
        qc.barrier()
        
        # Bob measures in random bases
        bob_bases = np.random.randint(0, 2, n_bits)
        for i in range(n_bits):
            if bob_bases[i] == 1:
                qc.h(i)
            qc.measure(i, i)
        
        return qc

    @staticmethod
    def qaoa(n_qubits: int, depth: int = 1) -> QuantumCircuit:
        """Create a QAOA circuit for optimization problems (example: MaxCut-like)."""
        qc = QuantumCircuit(n_qubits)
        gamma = ParameterVector('γ', depth)  # Cost parameters
        beta = ParameterVector('β', depth)   # Mixer parameters
        
        # Initial superposition
        qc.h(range(n_qubits))
        
        for p in range(depth):
            # Cost layer (example: ring topology)
            for i in range(n_qubits):
                qc.rzz(gamma[p], i, (i + 1) % n_qubits)
            
            # Mixer layer
            for i in range(n_qubits):
                qc.rx(2 * beta[p], i)  # Factor of 2 for standard QAOA
        
        qc.measure_all()
        return qc

    @staticmethod
    def shors_algorithm(N: int, a: Optional[int] = None) -> QuantumCircuit:
        """
        Placeholder for Shor's algorithm. Full implementation requires modular exponentiation.
        This is a simplified conceptual version; for production, use specialized libraries.
        """
        if a is None:
            a = np.random.randint(2, N)
        
        # Determine size of registers
        n = int(np.ceil(np.log2(N)))
        m = 2 * n  # Control register size
        
        qc = QuantumCircuit(m + n, m)
        
        # Superposition on control register
        qc.h(range(m))
        
        # Modular exponentiation (placeholder: actual implementation is complex)
        qc.x(m)  # Set base to 1 (simplified)
        for i in range(m):
            qc.append(Operator(random_unitary(2**(n+1))), [i] + list(range(m, m+n)))  # Placeholder
        
        # Inverse QFT
        qc.append(QFT(m, inverse=True), range(m))
        
        # Measure
        qc.measure(range(m), range(m))
        
        print(f"Shor's algorithm placeholder for factoring {N} with base {a}.")
        return qc

class AdvancedQuantumSimulator:
    """
    Advanced Quantum Computer Simulator with comprehensive features.
    
    Upgrades in this version (September 2025):
    - Added support for modern Qiskit primitives (Sampler, Estimator) for hybrid algorithms.
    - Expanded algorithm library with QAOA and Shor's (placeholder).
    - Added run_vqe method for Variational Quantum Eigensolver with classical optimization.
    - Enhanced simulation with additional methods like 'matrix_product_state' for larger systems.
    - Added state visualization options (Bloch, Q-sphere, etc.).
    - Added fidelity calculation.
    - Added coherent noise model.
    - Support for dynamic circuits (mid-circuit measurements).
    - Improved optimization with higher levels and more passes.
    - Added batch simulation for multiple circuits.
    
    This class provides:
    - Extended gate set including ECR
    - Advanced noise modeling including coherent errors
    - Quantum algorithms library with new additions
    - State tomography and analysis
    - Circuit optimization
    - Error correction (basic)
    - Visualization tools including state visualizations
    - Hybrid quantum-classical capabilities
    """
    
    def __init__(self, num_qubits: int, name: str = "QuantumSimulator"):
        """
        Initialize the Advanced Quantum Simulator.
        
        Args:
            num_qubits: Number of qubits for quantum circuits
            name: Name identifier for the simulator
        """
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise ValueError("Number of qubits must be a positive integer.")
        
        self.num_qubits = num_qubits
        self.name = name
        self.circuit = None
        self.classical_bits = 0
        self.noise_model = None
        self.backend = None
        self.last_result = None
        self.circuit_history = []
        self.optimization_level = 1
        self.algorithm_library = QuantumAlgorithmLibrary()
        
        print(f"🎛️ Advanced Quantum Simulator '{name}' initialized with {num_qubits} qubits.")
        print(f"   Features: Extended gates, Noise modeling, Algorithms, Optimization, Analysis, Hybrid Quantum-Classical, State Visualizations")

    def create_circuit(self, classical_bits: Optional[int] = None, 
                      name: Optional[str] = None) -> QuantumCircuit:
        """
        Create a new quantum circuit with optional classical register.
        
        Args:
            classical_bits: Number of classical bits (defaults to num_qubits)
            name: Optional name for the circuit
        
        Returns:
            Created QuantumCircuit object
        """
        self.classical_bits = classical_bits if classical_bits is not None else self.num_qubits
        
        # Create quantum and classical registers
        qreg = QuantumRegister(self.num_qubits, 'q')
        creg = ClassicalRegister(self.classical_bits, 'c') if self.classical_bits > 0 else None
        
        if creg:
            self.circuit = QuantumCircuit(qreg, creg, name=name or f"Circuit_{len(self.circuit_history)}")
        else:
            self.circuit = QuantumCircuit(qreg, name=name or f"Circuit_{len(self.circuit_history)}")
        
        print(f"✅ Created circuit '{self.circuit.name}' with {self.num_qubits} qubits and {self.classical_bits} classical bits")
        return self.circuit
    
    def add_gate(self, gate_type: Union[str, GateType], qubits: Union[int, List[int]], 
                 params: Optional[Union[float, List[float]]] = None, 
                 controls: Optional[List[int]] = None):
        """
        Add a quantum gate to the circuit with enhanced functionality.
        
        Args:
            gate_type: Type of gate to add
            qubits: Target qubit(s)
            params: Parameters for parameterized gates
            controls: Additional control qubits for multi-controlled gates
        """
        if self.circuit is None:
            raise ValueError("No circuit created. Call create_circuit() first.")
        
        # Convert gate_type to string if it's an enum
        if isinstance(gate_type, GateType):
            gate_type = gate_type.name
        
        gate_type = gate_type.upper()
        qubit_indices = [qubits] if isinstance(qubits, int) else qubits
        
        # Validate qubit indices
        all_qubits = qubit_indices + (controls or [])
        for q in all_qubits:
            if not (0 <= q < self.num_qubits):
                raise IndexError(f"Qubit index {q} out of bounds for {self.num_qubits} qubits.")
        
        try:
            # Single-qubit gates
            if gate_type == 'H':
                for q in qubit_indices:
                    self.circuit.h(q)
            elif gate_type == 'X':
                for q in qubit_indices:
                    self.circuit.x(q)
            elif gate_type == 'Y':
                for q in qubit_indices:
                    self.circuit.y(q)
            elif gate_type == 'Z':
                for q in qubit_indices:
                    self.circuit.z(q)
            elif gate_type == 'S':
                for q in qubit_indices:
                    self.circuit.s(q)
            elif gate_type == 'T':
                for q in qubit_indices:
                    self.circuit.t(q)
            elif gate_type == 'SDG':
                for q in qubit_indices:
                    self.circuit.sdg(q)
            elif gate_type == 'TDG':
                for q in qubit_indices:
                    self.circuit.tdg(q)
            elif gate_type == 'SX':
                for q in qubit_indices:
                    self.circuit.sx(q)
            elif gate_type == 'I':
                for q in qubit_indices:
                    self.circuit.id(q)
            
            # Rotation gates
            elif gate_type == 'RX':
                if params is None:
                    raise ValueError("RX gate requires angle parameter")
                for q in qubit_indices:
                    self.circuit.rx(params[0] if isinstance(params, list) else params, q)
            elif gate_type == 'RY':
                if params is None:
                    raise ValueError("RY gate requires angle parameter")
                for q in qubit_indices:
                    self.circuit.ry(params[0] if isinstance(params, list) else params, q)
            elif gate_type == 'RZ':
                if params is None:
                    raise ValueError("RZ gate requires angle parameter")
                for q in qubit_indices:
                    self.circuit.rz(params[0] if isinstance(params, list) else params, q)
            elif gate_type == 'U':
                if params is None or len(params) < 3:
                    raise ValueError("U gate requires 3 angle parameters [theta, phi, lambda]")
                for q in qubit_indices:
                    self.circuit.u(params[0], params[1], params[2], q)
            
            # Two-qubit gates
            elif gate_type == 'CX' or gate_type == 'CNOT':
                if len(qubit_indices) != 2:
                    raise ValueError("CX gate requires exactly 2 qubits")
                self.circuit.cx(qubit_indices[0], qubit_indices[1])
            elif gate_type == 'CY':
                if len(qubit_indices) != 2:
                    raise ValueError("CY gate requires exactly 2 qubits")
                self.circuit.cy(qubit_indices[0], qubit_indices[1])
            elif gate_type == 'CZ':
                if len(qubit_indices) != 2:
                    raise ValueError("CZ gate requires exactly 2 qubits")
                self.circuit.cz(qubit_indices[0], qubit_indices[1])
            elif gate_type == 'CH':
                if len(qubit_indices) != 2:
                    raise ValueError("CH gate requires exactly 2 qubits")
                self.circuit.ch(qubit_indices[0], qubit_indices[1])
            elif gate_type == 'SWAP':
                if len(qubit_indices) != 2:
                    raise ValueError("SWAP gate requires exactly 2 qubits")
                self.circuit.swap(qubit_indices[0], qubit_indices[1])
            elif gate_type == 'ISWAP':
                if len(qubit_indices) != 2:
                    raise ValueError("iSWAP gate requires exactly 2 qubits")
                self.circuit.iswap(qubit_indices[0], qubit_indices[1])
            elif gate_type == 'ECR':
                if len(qubit_indices) != 2:
                    raise ValueError("ECR gate requires exactly 2 qubits")
                self.circuit.ecr(qubit_indices[0], qubit_indices[1])
            
            # Controlled rotation gates
            elif gate_type == 'CRX':
                if len(qubit_indices) != 2 or params is None:
                    raise ValueError("CRX gate requires 2 qubits and angle parameter")
                self.circuit.crx(params[0] if isinstance(params, list) else params, 
                                qubit_indices[0], qubit_indices[1])
            elif gate_type == 'CRY':
                if len(qubit_indices) != 2 or params is None:
                    raise ValueError("CRY gate requires 2 qubits and angle parameter")
                self.circuit.cry(params[0] if isinstance(params, list) else params, 
                                qubit_indices[0], qubit_indices[1])
            elif gate_type == 'CRZ':
                if len(qubit_indices) != 2 or params is None:
                    raise ValueError("CRZ gate requires 2 qubits and angle parameter")
                self.circuit.crz(params[0] if isinstance(params, list) else params, 
                                qubit_indices[0], qubit_indices[1])
            
            # Two-qubit rotation gates
            elif gate_type == 'RXX':
                if len(qubit_indices) != 2 or params is None:
                    raise ValueError("RXX gate requires 2 qubits and angle parameter")
                self.circuit.rxx(params[0] if isinstance(params, list) else params, 
                                qubit_indices[0], qubit_indices[1])
            elif gate_type == 'RYY':
                if len(qubit_indices) != 2 or params is None:
                    raise ValueError("RYY gate requires 2 qubits and angle parameter")
                self.circuit.ryy(params[0] if isinstance(params, list) else params, 
                                qubit_indices[0], qubit_indices[1])
            elif gate_type == 'RZZ':
                if len(qubit_indices) != 2 or params is None:
                    raise ValueError("RZZ gate requires 2 qubits and angle parameter")
                self.circuit.rzz(params[0] if isinstance(params, list) else params, 
                                qubit_indices[0], qubit_indices[1])
            
            # Three-qubit gates
            elif gate_type == 'CCX' or gate_type == 'TOFFOLI':
                if len(qubit_indices) != 3:
                    raise ValueError("CCX gate requires exactly 3 qubits")
                self.circuit.ccx(qubit_indices[0], qubit_indices[1], qubit_indices[2])
            elif gate_type == 'CSWAP' or gate_type == 'FREDKIN':
                if len(qubit_indices) != 3:
                    raise ValueError("CSWAP gate requires exactly 3 qubits")
                self.circuit.cswap(qubit_indices[0], qubit_indices[1], qubit_indices[2])
            elif gate_type == 'CCZ':
                if len(qubit_indices) != 3:
                    raise ValueError("CCZ gate requires exactly 3 qubits")
                self.circuit.ccz(qubit_indices[0], qubit_indices[1], qubit_indices[2])
            
            # Multi-controlled gates
            elif gate_type == 'MCX':
                if controls is None or len(controls) < 1:
                    raise ValueError("MCX gate requires control qubits")
                self.circuit.mcx(controls, qubit_indices[0])
            elif gate_type == 'MCZ':
                if controls is None or len(controls) < 1:
                    raise ValueError("MCZ gate requires control qubits")
                # Build MCZ using MCX and H gates
                self.circuit.h(qubit_indices[0])
                self.circuit.mcx(controls, qubit_indices[0])
                self.circuit.h(qubit_indices[0])
            
            # Special operations
            elif gate_type == 'BARRIER':
                if qubit_indices:
                    self.circuit.barrier(qubit_indices)
                else:
                    self.circuit.barrier()
            elif gate_type == 'MEASURE':
                if self.classical_bits == 0:
                    raise ValueError("Cannot measure without classical bits")
                for i, q in enumerate(qubit_indices[:self.classical_bits]):
                    self.circuit.measure(q, i)
            elif gate_type == 'RESET':
                for q in qubit_indices:
                    self.circuit.reset(q)
            
            else:
                raise ValueError(f"Unsupported gate type: {gate_type}")
            
            print(f"   Added {gate_type} gate to qubit(s): {qubit_indices}")
            
        except Exception as e:
            print(f"❌ Error adding gate {gate_type}: {e}")
            raise
    
    def add_custom_unitary(self, unitary_matrix: np.ndarray, qubits: List[int], label: str = "U"):
        """
        Add a custom unitary gate to the circuit.
        
        Args:
            unitary_matrix: Unitary matrix to apply
            qubits: Qubits to apply the unitary to
            label: Label for the gate
        """
        if self.circuit is None:
            raise ValueError("No circuit created.")
        
        # Validate unitary
        n_qubits = len(qubits)
        expected_dim = 2**n_qubits
        if unitary_matrix.shape != (expected_dim, expected_dim):
            raise ValueError(f"Unitary matrix dimensions {unitary_matrix.shape} don't match {n_qubits} qubits")
        
        # Check if matrix is unitary
        if not np.allclose(unitary_matrix @ unitary_matrix.conj().T, np.eye(expected_dim)):
            raise ValueError("Matrix is not unitary")
        
        gate = Operator(unitary_matrix)
        self.circuit.append(gate, qubits)
        print(f"   Added custom unitary '{label}' to qubits {qubits}")
    
    def add_algorithm(self, algorithm: str, **kwargs):
        """
        Add a pre-built quantum algorithm to the circuit.
        
        Args:
            algorithm: Name of the algorithm ('qft', 'grover', 'qpe', 'teleportation', 'bb84', 'vqe', 'qaoa', 'shor')
            **kwargs: Algorithm-specific parameters
        """
        if self.circuit is None:
            self.create_circuit()
        
        algorithm = algorithm.lower()
        
        if algorithm == 'qft':
            n_qubits = kwargs.get('n_qubits', self.num_qubits)
            qft_circuit = self.algorithm_library.quantum_fourier_transform(n_qubits)
            self.circuit.compose(qft_circuit, inplace=True)
            print(f"   Added Quantum Fourier Transform on {n_qubits} qubits")
            
        elif algorithm == 'grover':
            marked_states = kwargs.get('marked_states', ['11'])
            grover_circuit = self.algorithm_library.grover_search(self.num_qubits, marked_states)
            self.circuit = grover_circuit
            print(f"   Added Grover's algorithm searching for states: {marked_states}")
            
        elif algorithm == 'qpe':
            n_counting = kwargs.get('n_counting', 3)
            unitary = kwargs.get('unitary', QuantumCircuit(1))
            qpe_circuit = self.algorithm_library.quantum_phase_estimation(n_counting, unitary)
            self.circuit = qpe_circuit
            print(f"   Added Quantum Phase Estimation with {n_counting} counting qubits")
            
        elif algorithm == 'teleportation':
            if self.num_qubits < 3:
                raise ValueError("Teleportation requires at least 3 qubits")
            teleport_circuit = self.algorithm_library.quantum_teleportation()
            self.circuit = teleport_circuit
            print("   Added Quantum Teleportation circuit")
            
        elif algorithm == 'bb84':
            n_bits = kwargs.get('n_bits', min(8, self.num_qubits))
            bb84_circuit = self.algorithm_library.bb84_key_distribution(n_bits)
            self.circuit = bb84_circuit
            print(f"   Added BB84 Key Distribution protocol for {n_bits} bits")
            
        elif algorithm == 'vqe' or algorithm == 'variational':
            depth = kwargs.get('depth', 2)
            var_circuit = self.algorithm_library.variational_circuit(self.num_qubits, depth)
            self.circuit = var_circuit
            print(f"   Added Variational circuit with depth {depth}")
            
        elif algorithm == 'qaoa':
            depth = kwargs.get('depth', 1)
            qaoa_circuit = self.algorithm_library.qaoa(self.num_qubits, depth)
            self.circuit = qaoa_circuit
            print(f"   Added QAOA circuit with depth {depth}")
            
        elif algorithm == 'shor':
            N = kwargs.get('N', 15)
            a = kwargs.get('a', None)
            shor_circuit = self.algorithm_library.shors_algorithm(N, a)
            self.circuit = shor_circuit
            print(f"   Added Shor's algorithm (placeholder) for N={N}")
            
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def create_bell_state(self, qubit_pair: Tuple[int, int] = (0, 1), bell_state: int = 0):
        """
        Create one of the four Bell states.
        
        Args:
            qubit_pair: Pair of qubits to entangle
            bell_state: Which Bell state (0-3)
                0: |Φ+⟩ = (|00⟩ + |11⟩)/√2
                1: |Φ-⟩ = (|00⟩ - |11⟩)/√2
                2: |Ψ+⟩ = (|01⟩ + |10⟩)/√2
                3: |Ψ-⟩ = (|01⟩ - |10⟩)/√2
        """
        if self.circuit is None:
            self.create_circuit()
        
        q1, q2 = qubit_pair
        
        # Create Bell state
        if bell_state in [2, 3]:  # Ψ states
            self.circuit.x(q2)
        
        self.circuit.h(q1)
        self.circuit.cx(q1, q2)
        
        if bell_state in [1, 3]:  # Negative phase states
            self.circuit.z(q1)
        
        bell_names = ['|Φ+⟩', '|Φ-⟩', '|Ψ+⟩', '|Ψ-⟩']
        print(f"   Created Bell state {bell_names[bell_state]} on qubits {qubit_pair}")
    
    def create_ghz_state(self, qubits: Optional[List[int]] = None):
        """Create a GHZ state on specified qubits."""
        if self.circuit is None:
            self.create_circuit()
        
        if qubits is None:
            qubits = list(range(self.num_qubits))
        
        if len(qubits) < 2:
            raise ValueError("GHZ state requires at least 2 qubits")
        
        self.circuit.h(qubits[0])
        for i in range(1, len(qubits)):
            self.circuit.cx(qubits[0], qubits[i])
        
        print(f"   Created GHZ state on qubits {qubits}")
    
    def create_w_state(self, qubits: Optional[List[int]] = None):
        """Create a W state on specified qubits."""
        if self.circuit is None:
            self.create_circuit()
        
        if qubits is None:
            qubits = list(range(self.num_qubits))
        
        n = len(qubits)
        if n < 2:
            raise ValueError("W state requires at least 2 qubits")
        
        # Initialize first qubit
        angle = 2 * np.arccos(1/np.sqrt(n))
        self.circuit.ry(angle, qubits[0])
        
        # Create W state recursively
        for i in range(1, n-1):
            angle = 2 * np.arccos(1/np.sqrt(n-i))
            self.circuit.cry(angle, qubits[i-1], qubits[i])
        
        # Final CNOT
        self.circuit.cx(qubits[-2], qubits[-1])
        
        print(f"   Created W state on qubits {qubits}")
    
    def add_oracle(self, oracle_type: str, target_state: Optional[str] = None):
        """
        Add various types of oracles to the circuit.
        
        Args:
            oracle_type: Type of oracle ('marking', 'phase', 'boolean')
            target_state: Target state for the oracle (binary string)
        """
        if self.circuit is None:
            raise ValueError("No circuit created")
        
        if oracle_type == 'marking':
            if target_state is None:
                target_state = '1' * self.num_qubits
            
            # Add X gates for 0s in target state
            for i, bit in enumerate(target_state):
                if bit == '0':
                    self.circuit.x(i)
            
            # Multi-controlled Z gate
            if self.num_qubits > 1:
                controls = list(range(self.num_qubits - 1))
                target = self.num_qubits - 1
                self.circuit.h(target)
                self.circuit.mcx(controls, target)
                self.circuit.h(target)
            else:
                self.circuit.z(0)
            
            # Undo X gates
            for i, bit in enumerate(target_state):
                if bit == '0':
                    self.circuit.x(i)
            
            print(f"   Added marking oracle for state |{target_state}⟩")
            
        elif oracle_type == 'phase':
            # Simple phase oracle
            if target_state:
                for i, bit in enumerate(target_state):
                    if bit == '1':
                        self.circuit.z(i)
            print("   Added phase oracle")
            
        else:
            raise ValueError(f"Unknown oracle type: {oracle_type}")
    
    def add_dynamic_measurement(self, qubit: int, classical_bit: int, condition: Optional[Any] = None):
        """
        Add a mid-circuit measurement for dynamic circuits.
        
        Args:
            qubit: Qubit to measure
            classical_bit: Classical bit to store result
            condition: Optional condition for if-else (using qiskit.circuit.if_test)
        """
        if self.circuit is None:
            raise ValueError("No circuit created")
        
        if condition is not None:
            with self.circuit.if_test(condition) as else_:
                self.circuit.measure(qubit, classical_bit)
            with else_:
                # Placeholder for else branch
                pass
        else:
            self.circuit.measure(qubit, classical_bit)
        
        print(f"   Added dynamic measurement on qubit {qubit} to classical bit {classical_bit}")
    
    def add_advanced_noise_model(self, noise_type: str = 'comprehensive', **kwargs):
        """
        Add advanced noise models to the simulator.
        
        Args:
            noise_type: Type of noise model ('comprehensive', 'thermal', 'decoherence', 'coherent', 'custom')
            **kwargs: Noise model parameters
        """
        self.noise_model = NoiseModel()
        
        if noise_type == 'comprehensive':
            # Get parameters with defaults
            t1 = kwargs.get('t1', 50e-6)  # Relaxation time
            t2 = kwargs.get('t2', 70e-6)  # Dephasing time
            gate_time_1q = kwargs.get('gate_time_1q', 50e-9)
            gate_time_2q = kwargs.get('gate_time_2q', 300e-9)
            prob_1 = kwargs.get('single_qubit_error', 0.001)
            prob_2 = kwargs.get('two_qubit_error', 0.01)
            
            # Single-qubit gate errors
            error_1q = depolarizing_error(prob_1, 1)
            thermal_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
            
            # Two-qubit gate errors  
            error_2q = depolarizing_error(prob_2, 2)
            thermal_2q = thermal_relaxation_error(t1, t2, gate_time_2q).tensor(
                         thermal_relaxation_error(t1, t2, gate_time_2q))
            
            # Add errors to noise model
            self.noise_model.add_all_qubit_quantum_error(error_1q.compose(thermal_1q), 
                                                         ['h', 'x', 'y', 'z', 's', 't', 'sx'])
            self.noise_model.add_all_qubit_quantum_error(error_2q.compose(thermal_2q), 
                                                         ['cx', 'cz'])
            
            # Measurement error
            prob_meas = kwargs.get('measurement_error', 0.01)
            error_meas = pauli_error([('X', prob_meas), ('I', 1 - prob_meas)])
            self.noise_model.add_all_qubit_quantum_error(error_meas, "measure")
            
            print(f"   Added comprehensive noise model:")
            print(f"     T1={t1*1e6:.1f}μs, T2={t2*1e6:.1f}μs")
            print(f"     1Q error={prob_1}, 2Q error={prob_2}, Meas error={prob_meas}")
            
        elif noise_type == 'thermal':
            t1 = kwargs.get('t1', 50e-6)
            t2 = kwargs.get('t2', 70e-6)
            gate_time = kwargs.get('gate_time', 50e-9)
            
            thermal_error = thermal_relaxation_error(t1, t2, gate_time)
            self.noise_model.add_all_qubit_quantum_error(thermal_error, 
                                                         ['h', 'x', 'y', 'z', 's', 't'])
            print(f"   Added thermal relaxation noise: T1={t1*1e6:.1f}μs, T2={t2*1e6:.1f}μs")
            
        elif noise_type == 'decoherence':
            # Amplitude and phase damping
            gamma_amp = kwargs.get('amplitude_damping', 0.01)
            gamma_phase = kwargs.get('phase_damping', 0.01)
            
            amp_error = amplitude_damping_error(gamma_amp)
            phase_error = phase_damping_error(gamma_phase)
            combined_error = amp_error.compose(phase_error)
            
            self.noise_model.add_all_qubit_quantum_error(combined_error, 
                                                         ['h', 'x', 'y', 'z'])
            print(f"   Added decoherence noise: γ_amp={gamma_amp}, γ_phase={gamma_phase}")
            
        elif noise_type == 'coherent':
            angle_error = kwargs.get('angle_error', 0.05)
            error_unitary = np.diag([1, np.exp(1j * angle_error)])
            coherent_err = coherent_unitary_error(error_unitary)
            self.noise_model.add_all_qubit_quantum_error(coherent_err, ['rx', 'ry', 'rz'])
            print(f"   Added coherent noise with angle error {angle_error}")
            
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
    
    def optimize_circuit(self, level: int = 2) -> QuantumCircuit:
        """
        Optimize the quantum circuit using transpiler passes.
        
        Args:
            level: Optimization level (0-3, higher = more optimization)
        
        Returns:
            Optimized circuit
        """
        if self.circuit is None:
            raise ValueError("No circuit to optimize")
        
        print(f"   Optimizing circuit with level {level}...")
        
        # Create pass manager with optimization passes
        pm = PassManager()
        
        if level >= 1:
            # Basic optimizations
            pm.append(passes.Unroller(['cx', 'u']))
            pm.append(passes.Optimize1qGates())
            pm.append(passes.CXCancellation())
        
        if level >= 2:
            # More aggressive optimizations
            pm.append(passes.CommutativeCancellation())
            pm.append(passes.OptimizeSwapBeforeMeasure())
            pm.append(passes.RemoveDiagonalGatesBeforeMeasure())
        
        if level >= 3:
            # Maximum optimization
            pm.append(passes.Depth())
            pm.append(passes.FixedPoint('depth'))
            pm.append(passes.Collect2qBlocks())
            pm.append(passes.ConsolidateBlocks())
            pm.append(passes.UnitarySynthesis())
        
        # Run optimization
        optimized = pm.run(self.circuit)
        
        # Report improvements
        original_depth = self.circuit.depth()
        optimized_depth = optimized.depth()
        original_gates = self.circuit.count_ops()
        optimized_gates = optimized.count_ops()
        
        print(f"   Optimization complete:")
        print(f"     Depth: {original_depth} → {optimized_depth} ({100*(1-optimized_depth/original_depth):.1f}% reduction)" if original_depth > 0 else "Depth unchanged")
        print(f"     Gate count: {sum(original_gates.values())} → {sum(optimized_gates.values())}")
        
        return optimized
    
    def analyze_circuit(self) -> CircuitMetrics:
        """
        Analyze the circuit and return comprehensive metrics.
        
        Returns:
            CircuitMetrics object with circuit statistics
        """
        if self.circuit is None:
            raise ValueError("No circuit to analyze")
        
        # Basic metrics
        depth = self.circuit.depth()
        gate_counts = dict(self.circuit.count_ops())
        
        # Count two-qubit and multi-qubit gates
        two_qubit_gates = sum(gate_counts.get(g, 0) for g in ['cx', 'cy', 'cz', 'swap', 'iswap', 'ecr'])
        multi_qubit_gates = sum(gate_counts.get(g, 0) for g in ['ccx', 'cswap', 'mcx'])
        
        # Check if parameterized
        parameterized = len(self.circuit.parameters) > 0
        
        # Convert to DAG for more analysis
        dag = circuit_to_dag(self.circuit)
        
        # Count connected components (approximate)
        connected_components = 1  # Simplified
        
        # Critical depth (longest path)
        critical_depth = dag.depth()
        
        metrics = CircuitMetrics(
            depth=depth,
            gate_count=gate_counts,
            qubit_count=self.num_qubits,
            classical_bit_count=self.classical_bits,
            two_qubit_gates=two_qubit_gates,
            multi_qubit_gates=multi_qubit_gates,
            parameterized=parameterized,
            connected_components=connected_components,
            critical_depth=critical_depth
        )
        
        print("\n📊 Circuit Analysis:")
        print(f"   Depth: {metrics.depth}")
        print(f"   Total gates: {sum(metrics.gate_count.values())}")
        print(f"   Two-qubit gates: {metrics.two_qubit_gates}")
        print(f"   Multi-qubit gates: {metrics.multi_qubit_gates}")
        print(f"   Parameterized: {metrics.parameterized}")
        print(f"   Gate breakdown: {metrics.gate_count}")
        
        return metrics
    
    def simulate(self, shots: int = 1024, method: str = 'automatic', 
                 include_noise: bool = False, seed: Optional[int] = None,
                 memory: bool = False) -> Dict:
        """
        Simulate the quantum circuit with various simulation methods.
        
        Args:
            shots: Number of measurement shots (for sampler methods)
            method: Simulation method ('automatic', 'statevector', 'density_matrix', 'unitary', 'stabilizer', 'extended_stabilizer', 'matrix_product_state')
            include_noise: Whether to include noise model (not for all methods)
            seed: Random seed for reproducibility
            memory: Whether to return individual measurement results (for sampler)
        
        Returns:
            Simulation results dictionary
        """
        if self.circuit is None:
            raise ValueError("No circuit to simulate")
        
        print(f"\n🔬 Simulating with {method} method...")
        
        # Configure backend
        backend_options = {
            'max_parallel_threads': 0,
            'max_parallel_experiments': 0
        }
        
        if seed is not None:
            backend_options['seed_simulator'] = seed
        
        self.backend = AerSimulator(method=method)
        
        if include_noise and self.noise_model and method in ['automatic', 'density_matrix', 'matrix_product_state']:
            print("   Including noise model in simulation")
            backend_options['noise_model'] = self.noise_model
        
        if method in ['statevector', 'density_matrix', 'unitary', 'matrix_product_state', 'extended_stabilizer', 'stabilizer']:
            circuit_no_measure = self.circuit.remove_final_measurements(inplace=False)
            job = self.backend.run(circuit_no_measure, shots=1, **backend_options)
            result = job.result()
            
            if method == 'statevector':
                self.last_result = {
                    'statevector': result.get_statevector(),
                    'backend': method,
                    'success': result.success
                }
                print(f"   Statevector computed: {2**self.num_qubits} amplitudes")
                
            elif method == 'density_matrix':
                self.last_result = {
                    'density_matrix': result.get_density_matrix(),
                    'backend': method,
                    'success': result.success
                }
                print(f"   Density matrix computed: {2**self.num_qubits}×{2**self.num_qubits}")
                
            elif method == 'unitary':
                self.last_result = {
                    'unitary': result.get_unitary(),
                    'backend': method,
                    'success': result.success
                }
                print(f"   Unitary matrix computed: {2**self.num_qubits}×{2**self.num_qubits}")
                
            else:
                # For other methods, fall back to counts if possible
                self.last_result = {
                    'counts': result.get_counts(),
                    'backend': method,
                    'success': result.success
                }
                print(f"   Simulation complete with {method}")
            
        else:
            # Default to qasm-like simulation
            job = self.backend.run(self.circuit, shots=shots, memory=memory, **backend_options)
            result = job.result()
            
            counts = result.get_counts()
            self.last_result = {
                'counts': counts,
                'shots': shots,
                'backend': method,
                'success': result.success
            }
            
            if memory:
                self.last_result['memory'] = result.get_memory()
            
            print(f"   Simulation complete: {shots} shots")
            print(f"   Unique outcomes: {len(counts)}")
        
        return self.last_result
    
    def batch_simulate(self, circuits: List[QuantumCircuit], shots: int = 1024, method: str = 'automatic', 
                       include_noise: bool = False, seed: Optional[int] = None) -> List[Dict]:
        """
        Simulate a batch of quantum circuits in parallel.
        
        Args:
            circuits: List of QuantumCircuit objects
            shots: Number of measurement shots
            method: Simulation method
            include_noise: Whether to include noise model
            seed: Random seed
        
        Returns:
            List of simulation results dictionaries
        """
        results = []
        for idx, circ in enumerate(circuits):
            print(f"Simulating circuit {idx + 1}/{len(circuits)}")
            self.circuit = circ
            res = self.simulate(shots=shots, method=method, include_noise=include_noise, seed=seed)
            results.append(res)
        return results
    
    def run_vqe(self, hamiltonian: SparsePauliOp, ansatz: Optional[QuantumCircuit] = None,
                initial_params: Optional[np.ndarray] = None, optimizer: str = 'COBYLA',
                shots: int = 1024, use_primitives: bool = True) -> Dict:
        """
        Run Variational Quantum Eigensolver using hybrid quantum-classical optimization.
        
        Args:
            hamiltonian: The Hamiltonian as SparsePauliOp
            ansatz: Parameterized circuit (uses default variational if None)
            initial_params: Initial parameter values
            optimizer: Scipy optimizer method
            shots: Shots for estimation (if using Sampler/Estimator)
            use_primitives: Use modern Qiskit primitives for estimation
        
        Returns:
            Optimization result dictionary
        """
        if ansatz is None:
            ansatz = self.algorithm_library.variational_circuit(self.num_qubits)
        
        if initial_params is None:
            initial_params = np.random.random(ansatz.num_parameters) * 0.01  # Small initialization
        
        if use_primitives:
            estimator = Estimator()
            def objective(params):
                bound_circuit = ansatz.assign_parameters(params)
                job = estimator.run([bound_circuit], [hamiltonian], shots=shots)
                return job.result().values[0]
        else:
            def objective(params):
                bound_circuit = ansatz.assign_parameters(params)
                job = AerSimulator(method='statevector').run(bound_circuit)
                state = job.result().get_statevector()
                return state.expectation_value(hamiltonian).real
        
        print(f"Running VQE with optimizer {optimizer}...")
        result = minimize(objective, initial_params, method=optimizer)
        
        print(f"VQE complete. Minimum energy: {result.fun:.6f}")
        return {'energy': result.fun, 'params': result.x, 'success': result.success}
    
    def calculate_entanglement(self, statevector: Optional[np.ndarray] = None,
                              subsystems: Tuple[List[int], List[int]] = None) -> float:
        """
        Calculate entanglement entropy between subsystems.
        
        Args:
            statevector: State vector to analyze (uses last result if None)
            subsystems: Tuple of two lists specifying the partition
        
        Returns:
            Von Neumann entropy of entanglement
        """
        if statevector is None:
            if self.last_result and 'statevector' in self.last_result:
                statevector = self.last_result['statevector']
            else:
                raise ValueError("No statevector available. Run statevector simulation first.")
        
        if subsystems is None:
            # Default: bipartition at middle
            n = self.num_qubits
            subsystems = (list(range(n//2)), list(range(n//2, n)))
        
        # Create density matrix and trace out subsystem
        psi = Statevector(statevector)
        rho = DensityMatrix(psi)
        
        # Calculate reduced density matrix
        qubits_to_trace = subsystems[1]
        rho_reduced = partial_trace(rho, qubits_to_trace)
        
        # Calculate von Neumann entropy
        entanglement = entropy(rho_reduced, base=2)
        
        print(f"   Entanglement entropy: {entanglement:.4f} ebits")
        print(f"   Subsystems: {subsystems[0]} | {subsystems[1]}")
        
        return entanglement
    
    def state_tomography(self, shots: int = 8192) -> DensityMatrix:
        """
        Perform quantum state tomography on the circuit.
        
        Args:
            shots: Number of shots for each measurement basis
        
        Returns:
            Reconstructed density matrix
        """
        if self.circuit is None:
            raise ValueError("No circuit for tomography")
        
        print("   Performing state tomography...")
        print(f"   Warning: Full tomography requires {3**self.num_qubits} measurement settings")
        
        # Simplified tomography using density matrix simulation
        circuit_no_measure = self.circuit.remove_final_measurements(inplace=False)
        
        backend = AerSimulator(method='density_matrix')
        job = backend.run(circuit_no_measure)
        result = job.result()
        
        rho = DensityMatrix(result.get_density_matrix())
        
        # Calculate purity
        purity = np.real(np.trace(rho.data @ rho.data))
        print(f"   Reconstructed state purity: {purity:.4f}")
        
        return rho
    
    def calculate_fidelity(self, state1: Optional[Union[Statevector, DensityMatrix]] = None,
                           state2: Optional[Union[Statevector, DensityMatrix]] = None) -> float:
        """
        Calculate fidelity between two states.
        
        Args:
            state1: First state (uses last result if None)
            state2: Second state (defaults to |0...0>)
        
        Returns:
            State fidelity
        """
        if state1 is None:
            if self.last_result and 'statevector' in self.last_result:
                state1 = self.last_result['statevector']
            elif 'density_matrix' in self.last_result:
                state1 = self.last_result['density_matrix']
            else:
                raise ValueError("No state available in last result")
        
        if state2 is None:
            state2 = Statevector.from_label('0' * self.num_qubits)
        
        fidelity = state_fidelity(state1, state2)
        print(f"   Fidelity: {fidelity:.4f}")
        return fidelity
    
    def visualize_circuit(self, style: str = 'default', filename: Optional[str] = None,
                         output: str = 'mpl', scale: float = 1.0):
        """
        Visualize the quantum circuit with various styles.
        
        Args:
            style: Drawing style ('default', 'iqx', 'iqx-dark', 'textbook')
            filename: Save to file if specified
            output: Output format ('mpl', 'text', 'latex')
            scale: Scale factor for the image
        """
        if self.circuit is None:
            raise ValueError("No circuit to visualize")
        
        fig = circuit_drawer(self.circuit, style=style, filename=filename, output=output, scale=scale)
        display(fig)
        print("   Circuit visualization displayed/saved.")
    
    def visualize_state(self, vis_type: str = 'bloch', state: Optional[Union[Statevector, DensityMatrix]] = None):
        """
        Visualize the quantum state with various representations.
        
        Args:
            vis_type: Visualization type ('bloch', 'qsphere', 'city', 'paulivec', 'hinton')
            state: State to visualize (uses last result if None)
        """
        if state is None:
            if self.last_result and 'statevector' in self.last_result:
                state = self.last_result['statevector']
            elif 'density_matrix' in self.last_result:
                state = self.last_result['density_matrix']
            else:
                raise ValueError("No state available for visualization")
        
        if vis_type == 'bloch':
            fig = plot_bloch_multivector(state)
        elif vis_type == 'qsphere':
            fig = plot_state_qsphere(state)
        elif vis_type == 'city':
            fig = plot_state_city(state)
        elif vis_type == 'paulivec':
            fig = plot_state_paulivec(state)
        elif vis_type == 'hinton':
            fig = plot_state_hinton(state)
        else:
            raise ValueError(f"Unknown visualization type: {vis_type}")
        
        display(fig)
        print(f"   State visualization ({vis_type}) displayed.")
