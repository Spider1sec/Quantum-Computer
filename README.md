# Quantum-Computer


QNOVA is a set of Python scripts designed to build simulate and test quantum circuits using Qiskit . The main idea it provides a layered framework for quantum simulation — starting from a lightweight version for quick testing moving to a stable version with more features and finally a high-end version that supports advanced algorithms noise modeling and near-hardware realistic scenarios


this project made for simulator in normal computer
General Architecture :

The project is divided into three main files :

qnova_low.py → lightweight version

qnova_stable.py → stable balanced version

qnova_high.py → advanced version


Each file contains classes and methods to : 

Build circuits

Add gates ( basic and advanced )

Apply noise models

Run simulations

Analyze results ( fidelity tomography entanglement )

What each version does ( Low  —  Stable  — High )

Low ( qnova_low.py ) :

For beginners and quick tests Includes a simple QuantumSimulator class that allows adding basic gates ( H X CX measurement ) visualizing the circuit and simulating with or without simple depolarizing noise

Stable ( qnova_stable.py ) :

Builds on the lightweight version by adding a Quantum Algorithm Library ( QFT Grover QPE Teleportation BB84 QAOA ) more complex noise models ( thermal relaxation amplitude damping phase damping coherent errors ) and better measurement/analysis tools.

High ( qnova_high.py ) :

Integrates modern Qiskit primitives ( Estimator, Sampler ) supports hybrid methods like VQE provides advanced analysis ( state tomography entanglement entropy ) and includes a broader set of gates and simulation methods ( density matrix, matrix product state )

The Algorithm Library — What’s inside ?

The Quantum Algorithm Library makes it easier to run well-known algorithms without starting from scratch :

QFT ( Quantum Fourier Transform )

Grover’s Search ( with customizable Oracle )

QPE ( Quantum Phase Estimation )

Teleportation and BB84 ( fundamental communication protocols )

VQE / QAOA ( variational parameterized algorithms for optimization and energy minimization )

This gives users ready-to-use building blocks for both research and teaching

Noise and Device Modeling

One of the strongest parts of QNOVA is realistic noise simulation :

Lightweight version → simple depolarizing error

Stable & High versions → full noise models including :

Thermal relaxation ( T1 T2 )

Amplitude/phase damping

Coherent rotation errors

Readout errors

**** These noise models bring the simulation closer to real hardware behavior, especially superconducting qubits ****

Analysis Tools : VQE — Tomography — Entanglement — Fidelity

VQE (Variational Quantum Eigensolver) : Uses Estimator and optimizers from SciPy to minimize energy of a given Hamiltonian. Useful for quantum chemistry problems

State Tomography : Reconstructs the density matrix of a state and computes purity

Entanglement Entropy : Calculates Von Neumann entropy for subsystems — a direct measure of quantum entanglement

Fidelity : Measures how close a noisy/experimental state is to a reference state


These tools move the project beyond just “ running circuits ” — they help evaluate quality and reliability

Analysis Tools : VQE — Tomography — Entanglement — Fidelity

VQE ( Variational Quantum Eigensolver ) : Uses Estimator and optimizers from SciPy to minimize energy of a given Hamiltonian. Useful for quantum chemistry problems

State Tomography : Reconstructs the density matrix of a state and computes purity

Entanglement Entropy : Calculates Von Neumann entropy for subsystems — a direct measure of quantum entanglement

Fidelity : Measures how close a noisy/experimental state is to a reference state


These tools move the project beyond just “ running circuits ” — they help evaluate quality and reliability

{ Practical Examples }

Bell State Test :
In the low version, you can build a Bell circuit (H + CX + measure). The simulator shows ~50/50 distribution on 00 and 11.

Noise Impact :
Run the same Bell circuit under the stable version’s noise model — fidelity drops, and the histogram shows more errors.

VQE Demo :
Use the high version’s run_vqe method on a Hamiltonian. It automatically builds parameterized circuits evaluates with Estimator and runs a classical optimizer to find the minimum energy

Technical Notes :

Works with modern Qiskit and qiskit-aer

Some algorithms (like Shor’s) are placeholders for educational purposes ( full implementations are heavy and complex )

Advanced simulations ( density matrix tomography ) scale exponentially with qubit count — so they’re practical only for small systems


Conclusion — Why is QNOVA special ?

QNOVA isn’t just one script — it’s a framework that grows with your needs :

Start with lightweight learning tools

Move into stable simulations with real-world noise

Advance into near-hardware experiments with entanglement tomography and variational algorithms


It bridges the gap between education research and practical experimentation in quantum computing
