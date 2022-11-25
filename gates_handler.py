import numpy as np

class FusedGate:
    def __init__(self,qubit:list):
        self.qubit = qubit
        self.absorbed_id = []
        self.is_control = False
        self.control_id = -1
        self.end_id = -1
        self.measure = False
        self.unitary = np.ndarray([])

    def get_qubits(self):
        return self.qubit

    def set_qubit_1(self,qubit):
        self.qubit.append(qubit)
        assert(len(self.qubit) == 1)
    
    def set_qubit_2(self,qubit):
        self.qubit.append(qubit)
        assert (len(self.qubit) == 2)

    def is_first(self,qubit):
        assert(len(self.qubit) >= 1)
        return self.qubit[0] == qubit
    
    def is_second(self,qubit):
        assert (len(self.qubit) == 2)
        return self.qubit[1] == qubit
    
    def current_size(self):
        return len(self.qubit)

    def is_exist(self,qubit):
        if qubit in self.qubit:
            return True
        else:
            return False

    def get_unitary(self):
        return self.unitary

    def set_unitary(self,u:np.ndarray):
        self.unitary = u

    def set_absorbed(self,id):
        self.absorbed_id.append(id)

    def get_absorbed(self):
        return self.absorbed_id

    def get_smallest_id(self):
        return min(self.absorbed_id)

    def set_control(self):
        self.is_control = True

    def get_control(self):
        return self.is_control

    def set_control_id(self,id):
        self.control_id = id

    def get_control_id(self):
        return self.control_id

    def set_end_id(self,id):
        self.end_id = id

    def get_end_id(self):
        return self.end_id

    def update_qubits(self,qubit):
        self.qubit = qubit

    def set_measure(self):
        self.measure = True

    def is_measure(self):
        return self.measure

class X:
    def __init__(self,gate):
        self.gate = gate

    def num_bits(self):
        return 1

    def matrix(self):
        return np.array([[0.+0.j, 1.+0.j],
                        [1.+0.j, 0.+0.j]])
    def name(self):
        return self.gate

class Y:
    def __init__(self,gate):
        self.gate = gate

    def num_bits(self):
        return 1

    def matrix(self):
        return np.array([[0.+0.j, 0.-1.j],
                        [0.+1.j, 0.+0.j]])
    def name(self):
        return self.gate

class Z:
    def __init__(self,gate):
        self.gate = gate

    def num_bits(self):
        return 1

    def matrix(self):
        return np.array([[ 1.+0.j,  0.+0.j],
                        [ 0.+0.j, -1.+0.j]])
    def name(self):
        return self.gate


class H:
    def __init__(self, gate):
        self.gate = gate

    def num_bits(self):
        return 1

    def matrix(self):
        return np.array([[ 1/np.sqrt(2)+0.j,  1/np.sqrt(2)+0.j],
                        [ 1/np.sqrt(2)+0.j, -1/(np.sqrt(2))+0.j]],dtype=complex)
    def name(self):
        return self.gate

class S:
    def __init__(self, gate):
        self.gate = gate

    def num_bits(self):
        return 1

    def matrix(self):
        return np.array([[1.+0.j, 0.+0.j],
                        [0.+0.j, 0.+1.j]],dtype=complex)
    def name(self):
        return self.gate

class SX:
    def __init__(self, gate):
        self.gate = gate

    def num_bits(self):
        return 1

    def matrix(self):
        return np.array([[0.5+0.5j, 0.5-0.5j],
                        [0.5-0.5j, 0.5+0.5j]],dtype=complex)
    def name(self):
        return self.gate

class SDG:
    def __init__(self, gate):
        self.gate = gate

    def num_bits(self):
        return 1

    def matrix(self):
        return np.array([[1.+0.j, 0.+0.j],
                        [0.+0.j, 0.-1.j]],dtype=complex)
    def name(self):
        return self.gate

class T:
    def __init__(self, gate):
        self.gate = gate

    def num_bits(self):
        return 1

    def matrix(self):
        return np.array([[1.+0.j, 0.+0.j],
        [0.+0.j, 0.70710678+0.70710678j]],dtype=complex)

    def name(self):
        return self.gate

class TDG:
    def __init__(self, gate):
        self.gate = gate

    def num_bits(self):
        return 1

    def matrix(self):
        return np.array([[1.+0.j, 0.+0.j],
        [0.+0.j, 0.+np.exp(-1j*np.pi/4)]],dtype=complex)

    def name(self):
        return self.gate

class RX:
    def __init__(self, gate, theta):
        self.gate = gate
        self.theta = theta

    def num_bits(self):
        return 1

    def matrix(self):
        return np.array([[np.cos(self.theta / 2) + 0.j, 0.-(np.sin(self.theta / 2) * 1j)],
        [0.-(np.sin(self.theta / 2) * 1j), np.cos(self.theta / 2) + 0.j]],dtype=complex)

    def name(self):
        return self.gate

class RY:
    def __init__(self, gate, theta):
        self.gate = gate
        self.theta = theta

    def num_bits(self):
        return 1

    def matrix(self):
        return np.array([[np.cos(self.theta/2)+0.j,-np.sin(self.theta/2)+0.j],
        [np.sin(self.theta/2)+0.j, np.cos(self.theta/2)+0.j]],dtype=complex)

    def name(self):
        return self.gate

class RZ:
    def __init__(self, gate, theta):
        self.gate = gate
        self.theta = theta

    def num_bits(self):
        return 1

    def matrix(self):
        return np.array([[0.+np.exp(-1.j*(self.theta / 2)), 0.+0.j],
        [0.+0.j, 0.+np.exp(1.j*(self.theta / 2))]])

    def name(self):
        return self.gate


class CX:

    def __init__(self,gate):
        self.gate = gate

    def num_bits(self):
        return 2

    #def matrix(self):
    #    return np.array([[1,0,0,0],
    #                     [0,1,0,0],
    #                     [0,0,0,1],
    #                     [0,0,1,0]],dtype=complex)
    def matrix(self):
        return np.array([[1,0,0,0],
                         [0,0,0,1],
                         [0,0,1,0],
                         [0,1,0,0]],dtype=complex)

    def name(self):
        return self.gate

class CNOT:

    def __init__(self,gate):
        self.gate = gate

    def num_bits(self):
        return 2

    #def matrix(self):
    #    return np.array([[1,0,0,0],
    #                     [0,1,0,0],
    #                     [0,0,0,1],
    #                     [0,0,1,0]],dtype=complex)
    # below is for matching qiskit
    def matrix(self):
        return np.array([[1,0,0,0],
                         [0,0,0,1],
                         [0,0,1,0],
                         [0,1,0,0]],dtype=complex)

    def name(self):
        return self.gate

class CNOT_RE:

    def __init__(self,gate):
        self.gate = gate

    def num_bits(self):
        return 2

    def matrix(self):
        return np.array([[1,0,0,0],
                         [0,1,0,0],
                         [0,0,0,1],
                         [0,0,1,0]],dtype=complex)
    # below is for matching qiskit
    #def matrix(self):
    #    return np.array([[1,0,0,0],
    #                     [0,0,0,1],
    #                     [0,0,1,0],
    #                     [0,1,0,0]],dtype=complex)

    def name(self):
        return self.gate

class CY:

    def __init__(self, gate):
        self.gate = gate

    def num_bits(self):
        return 2

    def matrix(self):
        return np.array([[1,0,0,0],
                         [0,1,0,0],
                         [0,0,0,-1.j],
                         [0,0,1.j,0]],dtype=complex)

    def name(self):
        return self.gate

class CZ:

    def __init__(self,gate):
        self.gate = gate

    def num_bits(self):
        return 2

    def matrix(self):
        return np.array([[1,0,0,0],
                         [0,1,0,0],
                         [0,0,1,0],
                         [0,0,0,-1]],dtype=complex)

    def name(self):
        return self.gate

class SWAP:

    def __init__(self,gate):
        self.gate = gate

    def num_bits(self):
        return 2

    def matrix(self):
        return np.array([[1,0,0,0],
                         [0,0,1,0],
                         [0,1,0,0],
                         [0,0,0,1]],dtype=complex)

    def name(self):
        return self.gate

class CR:

    def __init__(self,gate,theta):
        self.gate = gate
        self.theta = theta

    def num_bits(self):
        return 2

    def matrix(self):
        return np.array([[1,0,0,0],
                         [0,1,0,0],
                         [0,0,1,0],
                         [0,0,0,np.exp(self.theta*1j)]],dtype=complex)

    def name(self):
        return self.gate

class CH:

    def __init__(self,gate,theta):
        self.gate = gate
        self.theta = theta

    def num_bits(self):
        return 1

    def matrix(self):
        return np.array([[1,0,0,0],
                         [0,1,0,0],
                         [0,0,1/np.sqrt(2),1/np.sqrt(2)],
                         [0,0,1/np.sqrt(2), -1/np.sqrt(2)]],dtype=complex)

    def name(self):
        return self.gate

class U1:

    def __init__(self,gate,q_lambda):
        self.gate = gate
        self.q_lambda = q_lambda

    def num_bits(self):
        return 2

    def matrix(self):
        return np.array([[1,0],
                         [0,np.exp(1j*self.q_lambda)]])

    def name(self):
        return self.gate

class U2:

    def __init__(self,gate,phi,q_lambda):
        self.gate = gate
        self.phi = phi
        self.q_lambda = q_lambda

    def num_bits(self):
        return 1

    def matrix(self):
        return np.multiply(1/np.sqrt(2),np.array([[1,-1*np.exp(1j*self.q_lambda)],
                         [np.exp(1j*self.phi), np.exp(1j*(self.q_lambda+self.phi))]]))

    def name(self):
        return self.gate

class U3:

    def __init__(self, gate, theta, phi, q_lambda):
        self.gate = gate
        self.theta = theta
        self.phi = phi
        self.q_lambda = q_lambda

    def num_bits(self):
        return 1

    def matrix(self):
        return np.array([[np.cos(self.theta/2),-1*np.exp(1j*self.q_lambda)*np.sin(self.theta/2)],
        [np.exp(1j*self.phi)*np.sin(self.theta/2),np.exp(1j*(self.q_lambda + self.phi)) * np.cos(self.theta/2)]])

    def name(self):
        return self.gate


class ID:
    def __init__(self,gate):
        self.gate = gate

    def num_bits(self):
        return 1

    def matrix(self):
        return np.array([[1,0],
                         [0,1]],dtype=complex)

    def name(self):
        return self.gate


class TOFFOLI:

    def __init__(self,gate):
        self.gate = gate

    def num_bits(self):
        return 3

    def matrix(self):
        return np.array([[1,0,0,0,0,0,0,0],
                         [0,1,0,0,0,0,0,0],
                         [0,0,1,0,0,0,0,0],
                         [0,0,0,1,0,0,0,0],
                         [0,0,0,0,1,0,0,0],
                         [0,0,0,0,0,1,0,0],
                         [0,0,0,0,0,0,0,1],
                         [0,0,0,0,0,0,1,0]])

    def name(self):
        return self.gate

class CCZ:

    def __init__(self,gate):
        self.gate = gate

    def num_bits(self):
        return 3

    def matrix(self):
        return np.array([[1,0,0,0,0,0,0,0],
                         [0,1,0,0,0,0,0,0],
                         [0,0,1,0,0,0,0,0],
                         [0,0,0,1,0,0,0,0],
                         [0,0,0,0,1,0,0,0],
                         [0,0,0,0,0,1,0,0],
                         [0,0,0,0,0,0,1,0],
                         [0,0,0,0,0,0,0,-1]])

    def name(self):
        return self.gate

def get_gate_handler(gate:str, param:list = []):
    upper_gate = gate.upper()
    if upper_gate == "X":
        return X(upper_gate)
    elif upper_gate == "Y":
        return Y(upper_gate)
    elif upper_gate == "Z":
        return Z(upper_gate)
    elif upper_gate == "ID":
        return ID(upper_gate)
    elif upper_gate == "RX":
        return RX(upper_gate, param[0])
    elif upper_gate == "RY":
        return RY(upper_gate, param[0])
    elif upper_gate == "RZ":
        return RZ(upper_gate, param[0])
    elif upper_gate == "H":
        return H(upper_gate)
    elif upper_gate == "S":
        return S(upper_gate)
    elif upper_gate == "SX":
        return SX(upper_gate)
    elif upper_gate == "SDG":
        return SDG(upper_gate)
    elif upper_gate == "T":
        return T(upper_gate)
    elif upper_gate == "TDG":
        return TDG(upper_gate)
    elif upper_gate == "CX":
        return CX(upper_gate)
    elif upper_gate == "CNOT":
        return CNOT(upper_gate)
    elif upper_gate == "CNOT_RE":
        return CNOT_RE(upper_gate)
    elif upper_gate == "CZ":
        return CZ(upper_gate)
    elif upper_gate == "CH":
        return CH(upper_gate, param[0])
    elif upper_gate == "CR":
        return CR(upper_gate, param[0])
    elif upper_gate == "CY":
        return CY(upper_gate)
    elif upper_gate == "SWAP":
        return SWAP(upper_gate)
    elif upper_gate == "TOFFOLI":
        return TOFFOLI(upper_gate)
    elif upper_gate == "CCZ":
        return CCZ(upper_gate)
    elif upper_gate == "CR":
        return CR(upper_gate, param[0])
    elif upper_gate == "U1":
        return U1(upper_gate, param[0])
    elif upper_gate == "U2":
        return U2(upper_gate, param[0], param[1])
    elif upper_gate == "U3":
        return U3(upper_gate, param[0], param[1], param[2])
    else:
        raise ValueError

def expand_gate(u1:np.ndarray, u2:np.ndarray):
    return np.kron(u2,u1)

def muliply_by_one_qubit(u1:np.ndarray,u2:np.ndarray):
    return np.matmul(u2,u1)

def locate_fused_gate(qubits_boundary,qubit,index):
    for idx in range(len(qubits_boundary[qubit])):
        if qubits_boundary[qubit][idx] >= index:
            #if idx == len(qubits_boundary[qubit]):
            #   return idx-1
            return idx
    return -1
