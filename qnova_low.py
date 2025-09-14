import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit 

# --- UPDATED IMPORTS FOR QISKIT AER ---
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import pauli_error, depolarizing_error
# --- Import for plotting histogram ---
from qiskit.visualization import plot_histogram 
# --- END UPDATED IMPORTS ---

# This might be needed if running in a non-Jupyter environment and 'display' is not defined
try:
    from IPython.display import display
except ImportError:
    def display(fig):
        fig.show() # Fallback for non-Jupyter environments

class QuantumSimulator:
    """
    A comprehensive Quantum Computer Simulator using Qiskit.

    This class provides tools to build, simulate, and analyze quantum circuits,
    including features for visualization, statevector analysis, and basic noise modeling.
    """

    def __init__(self, num_qubits: int):
        """
        Initializes the QuantumSimulator with a specified number of qubits.

        Args:
            num_qubits (int): The number of qubits for the quantum circuits.
        """
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise ValueError("Number of qubits must be a positive integer.")
        self.num_qubits = num_qubits
        self.circuit = None
        self.classical_bits = 0
        self.noise_model = None
        print(f"QuantumSimulator initialized with {num_qubits} qubits.")

    def create_circuit(self, classical_bits: int = None) -> QuantumCircuit:
        """
        Creates a new quantum circuit.

        Args:
            classical_bits (int, optional): Number of classical bits.
                                           Defaults to num_qubits if None.

        Returns:
            QuantumCircuit: The newly created Qiskit QuantumCircuit object.
        """
        if classical_bits is None:
            self.classical_bits = self.num_qubits
        else:
            self.classical_bits = classical_bits
        
        self.circuit = QuantumCircuit(self.num_qubits, self.classical_bits)
        print(f"New QuantumCircuit created with {self.num_qubits} qubits and {self.classical_bits} classical bits.")
        return self.circuit

    def add_gate(self, gate_type: str, qubits, *args):
        """
        Adds a quantum gate to the current circuit.

        Args:
            gate_type (str): Type of gate (e.g., 'h', 'x', 'cx', 'measure', 'rz').
            qubits (int or list[int]): The qubit(s) to apply the gate to.
            *args: Additional arguments for certain gates (e.g., angle for 'rz').
        """
        if self.circuit is None:
            raise ValueError("No circuit created. Call create_circuit() first.")
        
        # Ensure qubits is always a list for consistent iteration
        qubit_indices = [qubits] if isinstance(qubits, int) else qubits

        for q in qubit_indices:
            if not (0 <= q < self.num_qubits):
                raise IndexError(f"Qubit index {q} out of bounds for {self.num_qubits} qubits.")

        try:
            if gate_type.lower() == 'h':
                for q in qubit_indices: self.circuit.h(q)
            elif gate_type.lower() == 'x':
                for q in qubit_indices: self.circuit.x(q)
            elif gate_type.lower() == 'y':
                for q in qubit_indices: self.circuit.y(q)
            elif gate_type.lower() == 'z':
                for q in qubit_indices: self.circuit.z(q)
            elif gate_type.lower() == 'cx': # CNOT
                if len(qubit_indices) != 2: raise ValueError("CX gate requires exactly two qubits.")
                self.circuit.cx(qubit_indices[0], qubit_indices[1])
            elif gate_type.lower() == 'ccx': # Toffoli
                if len(qubit_indices) != 3: raise ValueError("CCX gate requires exactly three qubits.")
                self.circuit.ccx(qubit_indices[0], qubit_indices[1], qubit_indices[2])
            elif gate_type.lower() == 'rz':
                if not args: raise ValueError("RZ gate requires an angle argument.")
                for q in qubit_indices: self.circuit.rz(args[0], q)
            elif gate_type.lower() == 'barrier':
                # Apply barrier to specified qubits or all if empty list
                if qubit_indices:
                    self.circuit.barrier(*qubit_indices)
                else:
                    self.circuit.barrier()
            elif gate_type.lower() == 'measure':
                if self.classical_bits == 0:
                    raise ValueError("Cannot measure without classical bits. Create circuit with classical_bits.")
                
                # Ensure the number of qubits to measure does not exceed available classical bits
                if len(qubit_indices) > self.classical_bits:
                    print(f"Warning: Attempting to measure {len(qubit_indices)} qubits, but only {self.classical_bits} classical bits are available. Measuring the first {self.classical_bits} qubits.")
                    self.circuit.measure(qubit_indices[:self.classical_bits], list(range(self.classical_bits)))
                else:
                    self.circuit.measure(qubit_indices, list(range(len(qubit_indices)))) # Measure into corresponding classical bits

            else:
                raise ValueError(f"Unsupported gate type: {gate_type}")
            print(f"Added {gate_type.upper()} gate to qubit(s): {qubits}")
        except Exception as e:
            print(f"Error adding gate {gate_type}: {e}")

    def visualize_circuit(self, filename: str = None, output_format: str = 'mpl'):
        """
        Draws the current quantum circuit.

        Args:
            filename (str, optional): If provided, saves the circuit diagram to this file.
            output_format (str): 'mpl' for Matplotlib (default), 'text' for ASCII, 'latex' for LaTeX.
        """
        if self.circuit is None:
            print("No circuit to visualize.")
            return
        
        print("\n--- Quantum Circuit Diagram ---")
        
        if output_format == 'text':
            print(self.circuit.draw(output='text', initial_state=True))
            if filename:
                with open(filename, 'w') as f:
                    f.write(self.circuit.draw(output='text', initial_state=True))
                print(f"Circuit diagram saved to {filename}")
        else: # Default to 'mpl' or other graphical outputs
            fig = self.circuit.draw(output=output_format, initial_state=True)
            if filename:
                if output_format == 'mpl':
                    fig.savefig(filename)
                elif output_format == 'latex': # For latex output, you'd usually compile .tex file
                    with open(filename, 'w') as f:
                        f.write(fig)
                print(f"Circuit diagram saved to {filename}")
            else:
                display(fig) # Use display for inline in environments like Jupyter
                # plt.show() # For regular Python scripts to pop up window
            if output_format == 'mpl':
                plt.close(fig) # Close the figure to prevent multiple displays if not in interactive mode

    def add_noise_model(self, single_qubit_error_rate=0.001, two_qubit_error_rate=0.01):
        """
        Creates and applies a simple depolarizing noise model to the simulator.

        Args:
            single_qubit_error_rate (float): Error rate for single-qubit gates.
            two_qubit_error_rate (float): Error rate for two-qubit gates (e.g., CX).
        """
        # Depolarizing error for single-qubit gates
        p1q = single_qubit_error_rate
        error_1 = depolarizing_error(p1q, 1)

        # Depolarizing error for two-qubit gates
        p2q = two_qubit_error_rate
        error_2 = depolarizing_error(p2q, 2)

        self.noise_model = NoiseModel()
        # Note: Qiskit's internal gate names like 'u1', 'u2', 'u3' are now 'rz', 'sx', 'x' or 'u'
        # For simplicity, using common gates like h, x, y, z, rz for single-qubit errors
        self.noise_model.add_all_qubit_quantum_error(error_1, ['h', 'x', 'y', 'z', 'rz', 'id', 'sx', 'u']) 
        self.noise_model.add_all_qubit_quantum_error(error_2, ['cx', 'cz', 'swap'])
        
        print(f"Noise model added: Single-qubit error rate={p1q}, Two-qubit error rate={p2q}")
        print("Note: Noise models significantly increase simulation time for larger circuits.")

    def simulate(self, shots: int = 1024, backend_type: str = 'qasm_simulator', include_noise: bool = False):
        """
        Simulates the current quantum circuit.

        Args:
            shots (int): Number of times to run the circuit (for 'qasm_simulator').
            backend_type (str): 'qasm_simulator' for probabilistic results,
                                'statevector_simulator' for the final quantum state.
            include_noise (bool): Whether to include the defined noise model in the simulation.

        Returns:
            dict or np.array: Measurement counts for 'qasm_simulator', or statevector for 'statevector_simulator'.
        """
        if self.circuit is None:
            raise ValueError("No circuit to simulate. Call create_circuit() and add_gates() first.")

        if backend_type == 'qasm_simulator':
            backend = AerSimulator() # Use AerSimulator directly
            options = {'shots': shots}
            if include_noise and self.noise_model:
                options['noise_model'] = self.noise_model
                print(f"Simulating with QASM simulator and {shots} shots, including noise.")
            else:
                print(f"Simulating with QASM simulator and {shots} shots (no noise).")
            
            job = backend.run(self.circuit, **options) 
            result = job.result()
            counts = result.get_counts(self.circuit)
            print("Simulation complete. Measurement results obtained.")
            return counts
        
        elif backend_type == 'statevector_simulator':
            # Remove measurements for statevector simulation, as they collapse the state
            circuit_no_measure = self.circuit.remove_final_measurements(inplace=False)
            
            # Ensure the backend is initialized correctly for statevector simulation
            backend = AerSimulator(method='statevector') 
            
            if include_noise and self.noise_model:
                print("Warning: Noise models are typically not applied to statevector simulations "
                      "as they model probabilistic outcomes. Simulating without noise for statevector backend.")
            
            print("Simulating with Statevector simulator.")
            job = backend.run(circuit_no_measure)
            result = job.result()
            
            # The key change is here: get_statevector() needs to be called after the simulation
            # The argument to get_statevector is optional if there's only one experiment.
            statevector = result.get_statevector() # Removed the circuit argument
            print("Simulation complete. Final statevector obtained.")
            return statevector
        
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")

    def plot_counts(self, counts: dict, title: str = "Measurement Results"):
        """
        Plots a histogram of the measurement counts.

        Args:
            counts (dict): A dictionary of measurement outcomes and their frequencies.
            title (str): Title for the histogram plot.
        """
        if not counts:
            print("No counts to plot.")
            return
        print("\n--- Plotting Measurement Results ---")
        fig = plot_histogram(counts, title=title)
        display(fig) # For Jupyter environments
        plt.close(fig) # Close the figure to prevent multiple displays

    def print_statevector(self, statevector: np.array):
        """
        Prints the final statevector in a readable format.

        Args:
            statevector (np.array): The statevector obtained from simulation.
        """
        print("\n--- Final Statevector ---")
        num_qubits = int(np.log2(len(statevector)))
        for i, amp in enumerate(statevector):
            if np.isclose(amp, 0, atol=1e-9): continue # Skip printing negligible amplitudes
            binary_state = bin(i)[2:].zfill(num_qubits)
            print(f"|{binary_state}>: {amp:.4f}")

# --- Helper Functions for Demonstrations ---

def create_bell_state_circuit(simulator: QuantumSimulator):
    """Creates a Bell state circuit (|00> + |11>) / sqrt(2)."""
    if simulator.num_qubits < 2:
        raise ValueError("Bell state requires at least 2 qubits.")
    simulator.create_circuit()
    simulator.add_gate('h', 0)
    simulator.add_gate('cx', [0, 1])
    simulator.add_gate('measure', [0, 1])
    print("\nBell state circuit created.")

def create_superposition_circuit(simulator: QuantumSimulator):
    """Creates a circuit putting the first qubit in superposition."""
    simulator.create_circuit(classical_bits=1)
    simulator.add_gate('h', 0)
    simulator.add_gate('measure', 0)
    print("\nSuperposition circuit created.")

def create_deutsch_jozsa_circuit(simulator: QuantumSimulator, oracle_type: str = 'constant'):
    """
    Creates a Deutsch-Jozsa circuit for 2-qubit input function.

    Args:
        simulator (QuantumSimulator): The simulator instance.
        oracle_type (str): 'constant' (f(x)=0 or f(x)=1) or 'balanced' (f(x)=x or f(x)=not x).
    """
    if simulator.num_qubits < 2:
        raise ValueError("Deutsch-Jozsa requires at least 2 qubits (input + ancilla).")
    
    # We'll use N input qubits + 1 ancilla qubit. For simplicity, let's use 1 input qubit + 1 ancilla here
    # to fit within typical small simulation limits. So total 2 qubits.
    input_qubits = simulator.num_qubits - 1
    ancilla_qubit = simulator.num_qubits - 1 # The last qubit is the ancilla

    print(f"\nCreating Deutsch-Jozsa circuit for {input_qubits} input qubit(s) and 1 ancilla.")
    simulator.create_circuit(classical_bits=input_qubits)

    # Initialize ancilla qubit to |-> state
    simulator.add_gate('x', ancilla_qubit)
    simulator.add_gate('h', ancilla_qubit)

    # Apply Hadamard to all input qubits
    for i in range(input_qubits):
        simulator.add_gate('h', i)
    
    simulator.add_gate('barrier', list(range(simulator.num_qubits)))

    # Apply the Oracle
    print(f"Applying {oracle_type} oracle...")
    if oracle_type == 'constant':
        # For DJ, if f(x)=0, oracle acts as identity (no gates).
        # If f(x)=1, oracle acts as Z on ancilla, provided ancilla is in |->.
        # This causes a phase kickback, but for 1-input qubit, it effectively means no output change (still '0')
        # So we'll implement a f(x)=0 (identity) oracle here for 'constant'
        pass 
        # To simulate f(x)=1 constant (for demonstration clarity, though result will still be '0'):
        # simulator.add_gate('z', ancilla_qubit) # Apply Z to ancilla if f(x)=1 constant

    elif oracle_type == 'balanced':
        # Example of a balanced oracle: f(x) = x (CNOT) for 1 input qubit
        # For 1 input qubit (qubit 0), f(0)=0, f(1)=1. This is a CNOT gate where input is control (0), ancilla is target (1).
        if input_qubits == 1:
            simulator.add_gate('cx', [0, ancilla_qubit])
        else:
            print("Warning: Balanced oracle for multiple input qubits is more complex (e.g., using CCX gates). "
                  "Using a simple CNOT for the first input qubit as an example.")
            simulator.add_gate('cx', [0, ancilla_qubit]) # Simple example for 1 input qubit
    else:
        raise ValueError("Unsupported oracle_type. Choose 'constant' or 'balanced'.")
    
    simulator.add_gate('barrier', list(range(simulator.num_qubits)))

    # Apply Hadamard to input qubits again
    for i in range(input_qubits):
        simulator.add_gate('h', i)
    
    simulator.add_gate('barrier', list(range(simulator.num_qubits)))

    # Measure input qubits
    simulator.add_gate('measure', list(range(input_qubits)))
    print("Deutsch-Jozsa circuit constructed.")


# --- Main Execution Block ---
if __name__ == "__main__":
    print("🚀 Starting the 'Best Quantum Simulator in the World' Script! 🚀")

    # --- Part 1: Simulate Superposition ---
    print("\n--- Part 1: Demonstrating Superposition ---")
    superposition_sim = QuantumSimulator(num_qubits=1)
    create_superposition_circuit(superposition_sim)
    superposition_sim.visualize_circuit(filename="superposition_circuit.png")
    
    # Simulate and plot with QASM simulator
    superposition_counts = superposition_sim.simulate(shots=1000)
    superposition_sim.plot_counts(superposition_counts, "Superposition Result (QASM)")

    # Simulate with statevector simulator (no measurement needed for this)
    superposition_sim.create_circuit(classical_bits=0) # Recreate without
    
