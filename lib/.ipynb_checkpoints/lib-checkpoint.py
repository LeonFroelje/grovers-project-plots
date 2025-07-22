import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFTGate, grover_operator, MCMTGate, ZGate, HGate
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix
from math import log2, ceil, pi, sqrt
import numpy as np
from enum import Enum
import pandas as pd

def encode_number(n: int, d: int) -> str:
    return (format(n, f"0{d}b").replace("-", "1"))[::-1]

def encode_target(T, d) -> str:
    if T < 0:
        return format(2 ** d + T, f"0{d}b")[::-1]
    return format(T, f"0{d}b")[::-1]


def RotationAdd(m, d, old=False):
    n = encode_number(abs(m), d)
    qc = QuantumCircuit(d + 1)
    k = len(n)
    # iterate over bits of n
    angles = np.zeros(d)
    for j in range(k):
        if n[j] == '1':
            for l in range(d - j):
                if not old:
                    angles[l] += np.sign(m) *  2 * pi * 2 ** (l + j) / 2 ** d
                else:
                    qc.cp(np.sign(m) *  2 * pi * 2 ** (l + j) / 2 ** d, 0,1 + l)
    for k in range(k):
        if angles[k] != 0:
            qc.cp(angles[k], 0, 1 + k)
    return qc.to_gate(label=f"R({m})")


def oracle(S, T, old=False, uncompute=True):
    s_0 = sum([abs(i) for i in S])
    d = ceil(log2(1 + max(s_0, T))) + (1-(old or T >= 0))

    qc = QuantumCircuit()
    data_register = QuantumRegister(len(S), "index")
    qc.add_register(data_register)
    target_register = QuantumRegister(d, "target")
    data_measure = ClassicalRegister(len(S), "target_M")
    qc.add_register(target_register, data_measure)

    encoded_target = encode_target(T, d)
    for i in range(d):
        if encoded_target[i] == '1':
            qc.x(target_register[i])
    qc.append(QFTGate(d), target_register)

    for i in range(len(S)):
        n = S[i]
        rotation_adder = RotationAdd(-n, d, old=old)
        qc.append(rotation_adder, [data_register[i]] + list(target_register))
    qc.append(QFTGate(d).inverse(), target_register)
    qc.x(target_register)
    
    qc.append(MCMTGate(gate=ZGate(), num_ctrl_qubits=d, num_target_qubits=1),[0] + list(range(qc.num_qubits - d, qc.num_qubits)) )
    qc.mcx(target_register, 0)
    qc.append(MCMTGate(gate=ZGate(), num_ctrl_qubits=d, num_target_qubits=1),[0] + list(range(qc.num_qubits - d, qc.num_qubits)))
    qc.mcx(target_register, 0)
    if uncompute:
        qc.x(target_register)
    
        qc.append(QFTGate(d), target_register)
    
        for i in range(len(S)):
            n = S[i]
            rotation_adder = RotationAdd(n, d, old=old)
            qc.append(rotation_adder, [data_register[i]] + list(target_register))
                        
        qc.append(QFTGate(d).inverse(), target_register)
        for i in range(d):
            if encoded_target[i] == '1':
                qc.x(target_register[i])
    return qc

def grovers(S,T,n = 1):
    orcl = oracle(S,T)
    qc = QuantumCircuit(orcl.num_qubits, orcl.num_clbits)
    qc.h(list(range(len(S))))
    
    qc.compose(grover_operator(orcl, reflection_qubits=list(range(len(S)))).power(n).decompose(),
              list(range(orcl.num_qubits)), list(range(orcl.num_clbits)), inplace=True, wrap=False)
    return qc



def test_phase_flip(S,T):
    orcl = oracle(S,T)
    qc = QuantumCircuit(orcl.num_qubits, orcl.num_clbits)
    qc.h(list(range(len(S))))
    qc.append(orcl, list(range(orcl.num_qubits)), list(range(orcl.num_clbits)))
    
    qc.save_statevector()
    
    simulator = AerSimulator(method='matrix_product_state')
    simulator.set_max_qubits(qc.num_qubits)
    circ = qiskit.transpile(qc, simulator)
    result = simulator.run(circ, shots=4096).result().get_statevector()
    return result


def test_grovers(S,T,n = 1):
    orcl = oracle(S,T)
    statevectors = []
    qc = QuantumCircuit(orcl.num_qubits, orcl.num_clbits)
    qc.h(list(range(len(S))))
    qc.compose(orcl, list(range(orcl.num_qubits)), list(range(orcl.num_clbits)),inplace=True, wrap=False)
    
    qc.save_statevector()
    
    simulator = AerSimulator(method='matrix_product_state')
    simulator.set_max_qubits(qc.num_qubits)
    circ = qiskit.transpile(qc, simulator)
    result = simulator.run(circ, shots=4096).result().get_statevector()
    return result

def bitflip_oracle(S, T):
    s_0 = sum([abs(i) for i in S])
    d = ceil(log2(1 + max(s_0, T))) + 1

    qc = QuantumCircuit()
    data_register = QuantumRegister(len(S), "index")
    qc.add_register(data_register)
    target_register = QuantumRegister(d, "target")
    target_measure = ClassicalRegister(d, "target_M")
    qc.add_register(target_register, target_measure)
    bitflip_register = QuantumRegister(1, "out")
    bitflip_measure = ClassicalRegister(1, "out_M")
    qc.add_register(bitflip_register, bitflip_measure)
    
    encoded_target = encode_target(T, d)
    
    for i in range(d):
        if encoded_target[i] == '1':
            qc.x(target_register[i])
    qc.append(QFTGate(d), target_register)
    for i in range(len(S)):
        qc.x(data_register[i])
    for i in range(len(S)):
        n = S[i]
        rotation_adder = RotationAdd(-n, d)
        qc.append(rotation_adder, [data_register[i]] + list(target_register))

    qc.append(QFTGate(d).inverse(), target_register)
    qc.x(target_register)

    qc.mcx(target_register, bitflip_register)

    qc.x(target_register)

    qc.append(QFTGate(d), target_register)

    for i in range(len(S)):
        n = S[i]
        rotation_adder = RotationAdd(n, d)
        qc.append(rotation_adder, [data_register[i]] + list(target_register))
                    
    qc.append(QFTGate(d).inverse(), target_register)
    for i in range(d):
        if encoded_target[i] == '1':
            qc.x(target_register[i])
    qc.measure(bitflip_register, bitflip_measure)
    qc.measure(target_register, target_measure)
    
    return qc

def simulate_bitflip(S,T):
    orcl = bitflip_oracle(S,T)
    simulator = AerSimulator(method='matrix_product_state')
    simulator.set_max_qubits(orcl.num_qubits)
    circ = qiskit.transpile(orcl, simulator)
    result = simulator.run(circ, shots=4096).result().get_counts()
    return pd.Series(result).sort_values(ascending=False)

def simulate(S, T, num_sols, old=False):
    if num_sols > 0:
        num_iter = ceil(sqrt(2 ** len(S) / num_sols) * 0.5)
    else:
        num_iter = 1
    orcl = oracle(S, T, old)
    qc = QuantumCircuit(orcl.num_qubits, orcl.num_clbits)
    qc.h(list(range(len(S))))

    qc.append(grover_operator(orcl, reflection_qubits=list(range(len(S)))).power(num_iter),
              list(range(orcl.num_qubits)), list(range(orcl.num_clbits)))

    for j in range(len(S)):
        qc.measure(j, j)

    simulator = AerSimulator(method='matrix_product_state')
    simulator.set_max_qubits(qc.num_qubits)
    circ = qiskit.transpile(qc, simulator)
    result = simulator.run(circ, shots=4096).result().get_counts()
    return pd.Series(result).sort_values(ascending=False)

def groversearch(S, T, num_sols, old=False):
    if num_sols > 0:
        num_iter = ceil(sqrt(len(S) / num_sols) * pi / 4) 
    else:
        num_iter = ceil(sqrt(len(S)) * pi / 4)
    orcl = oracle(S, T, old)
    qc = QuantumCircuit(orcl.num_qubits, orcl.num_clbits)
    qc.h(list(range(len(S))))

    qc.append(grover_operator(orcl, reflection_qubits=list(range(len(S)))).power(num_iter),
              list(range(orcl.num_qubits)), list(range(orcl.num_clbits)))
    return qc