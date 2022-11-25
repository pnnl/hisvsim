import re
import math
import collections
import copy
import networkx as nx
import os
import sys
from datetime import datetime
#import matplotlib.pyplot as plt
import svsimulator_py_wrapper as svsim
import time
import argparse
import mpi4py
import numpy as np
from mpi4py import MPI
import gates_handler
from collections import OrderedDict

gates = ['u3','u2','u1','id','x','y','z','h','s','sdg','t','tdg','rx','ry','rz','cz','cx','cy','ch','ccx','crz','cu1','cu3','reset','swap','rxx','ryy','rzz','cswap']

GATE_NOPARAM = ['I','X','Y','Z','H','S','T','SWAP','TDG','SDG']
GATE_ONEPARAM = ['RX','RY','RZ','RI','R1','U1','RXX','RZZ','RYY']
GATE_TWOPARAM = ['U2']
GATE_THREEPARAM = ['U3']
GATE_C1 = ['C1']
GATE_C2 = ['C2']
GATE_FRAC= ['RXFRAC','RYFRAC','RZFRAC','RIFRAC','R1FRAC']
NUM_MEASURE = 10

measure_op = {}
 
acc_timer = 0
start_timer = 0

class TargetGate:
    def __init__(self,index,func_name,gate,params,qubits):
        self.index = index
        self.func_name = func_name
        self.gate = gate
        self.params = params
        self.qubits = qubits

    def get_index(self):
        return self.index

    def get_func_name(self):
        return self.func_name

    def get_gate(self):
        return self.gate

    def get_params(self):
        return self.params

    def get_qubits(self):
        return self.qubits

    def update_qubits(self, qubits):
        self.qubits = qubits

    def get_index(self):
        return self.index

class QASM_parser:
    def __init__(self,filename):
        self.filename = filename
        self.line_index = collections.OrderedDict()
        self.main_start = 0
        self.ordered_gate_list = []
        self.func_names = []
        self.func_arguments = collections.OrderedDict()
        self.target_gate_profiler = collections.OrderedDict()
        self.func_lines = collections.OrderedDict()
        self.qubit_allocation = {}

    def identify_qubit_allocation(self):
        with open(self.filename) as qasm:
            lines = qasm.readlines()
            for line in lines:
                if "qreg" in line:
                    items = line.split(" ")
                    qubit_allocation = items[1]
                    # there should be [] in it
                    q_item = qubit_allocation.split("[")
                    #q_name = re.findall('\w+',qubit_allocation)
                    assert(len(q_item) >= 2)
                    name = q_item[0]
                    #q_num = re.findall('\d+',qubit_allocation)
                    #assert(len(q_num) >= 1)
                    q_num = q_item[1].split(']')
                    assert(len(q_num)>=1)
                    num = q_num[0]
                    self.qubit_allocation[name] = num

    def process_match(self,match,target_gate):
        if match:
            params = match.group(1)
            params = params.lstrip("(").rstrip(")")
            p_items = params.split(",")
            upper_gate = target_gate.upper()
            param = []
            # here only consider the gate that has params i.e. (xxx)
            if upper_gate == "RX":
                assert(len(p_items) == 1)
                param.append(convert_param(p_items[0]))
            elif upper_gate == "RY":
                assert (len(p_items) == 1)
                param.append(convert_param(p_items[0]))
            elif upper_gate == "RZ":
                assert (len(p_items) == 1)
                param.append(convert_param(p_items[0]))
            elif upper_gate == "RXX":
                assert (len(p_items) == 1)
                param.append(convert_param(p_items[0]))
            elif upper_gate == "RYY":
                assert (len(p_items) == 1)
                param.append(convert_param(p_items[0]))
            elif upper_gate == "RZZ":
                assert (len(p_items) == 1)
                param.append(convert_param(p_items[0]))
            elif upper_gate == "CH":
                assert (len(p_items) == 1)
                param.append(convert_param(p_items[0]))
            elif upper_gate == "CR":
                assert (len(p_items) == 1)
                param.append(convert_param(p_items[0]))
            elif upper_gate == "U1":
                assert (len(p_items) == 1)
                param.append(convert_param(p_items[0]))
            elif upper_gate == "U2":
                assert(len(p_items) == 2)
                param.append(convert_param(p_items[0]))
                param.append(convert_param(p_items[1]))
            elif upper_gate == "U3":
                assert (len(p_items) == 3)
                param.append(convert_param(p_items[0]))
                param.append(convert_param(p_items[1]))
                param.append(convert_param(p_items[2]))
            elif upper_gate == "CU1":
                assert (len(p_items) == 1)
                param.append(convert_param(p_items[0]))
            elif upper_gate == "CRZ":
                assert (len(p_items) == 1)
                param.append(convert_param(p_items[0]))
            elif upper_gate == "CU3":
                assert (len(p_items) == 3)
                param.append(convert_param(p_items[0]))
                param.append(convert_param(p_items[1]))
                param.append(convert_param(p_items[2]))
            else:
                print("the gate is not supported")
                raise ValueError
            return param
        else:
            print("the custom function match failed")
            raise ValueError

    def process_custom_gate(self,gate_function:dict):
        name = ""
        for index in gate_function:
            line = gate_function[index]
            line = line.rstrip("\n")
            if "gate" in line:
                function_def_list = line.split(" ")
                name = function_def_list[1]
                self.func_names.append(name)
                self.func_lines[name] = []
                self.func_arguments[name] = collections.OrderedDict()
                if len(function_def_list) >= 2:
                    for function_def_list_iter in function_def_list[2:]:
                        args = re.findall('\w+\d*',function_def_list_iter)
                        for arg in args:
                            self.func_arguments[name][arg] = ""
            elif "{" not in line and "}" not in line:
                gate = self.identify_gate_in_function(line)
                params = self.identify_param_in_function(line,gate)
                qubits = self.identify_qubit_in_function(line)
                gate_in_function = TargetGate(index,name,gate,params,qubits)
                self.func_lines[name].append(index)
                self.target_gate_profiler[index] = gate_in_function
            else:
                # do nothing
                pass

    def identify_custom_gate_pass(self):
        # need to run this first pass to identify the custom gate (i.e. user-defined function)
        with open(self.filename) as qasm:
            pattern_function = '^gate '
            pattern_function_open = '{'
            pattern_function_close = '}'
            lines = qasm.readlines()
            temp_func = {}
            flag = 0
            for index, line in enumerate(lines):
                # index all the lines
                self.line_index[index] = line
                match_start = re.search(pattern_function,line)
                if match_start:
                    flag = 1
                if flag == 1:
                    temp_func[index] = line
                match_close = re.search(pattern_function_close,line)
                if match_close:
                    flag = 0
                    self.process_custom_gate(temp_func)
                    self.main_start = index+1


    def order_gate_list(self):

        start_index = self.main_start
        index_list = list(self.line_index.keys())
        for item in index_list[start_index:]:
            if item not in self.target_gate_profiler:
                continue
            op = self.target_gate_profiler[item]
            if op.get_func_name() == 'main' and op.get_gate() in gates:
                self.ordered_gate_list.append(op)
            else:
                    if op.get_gate() == "measure":
                        self.ordered_gate_list.append(op)
                    elif op.get_gate() == "barrier":
                        pass
                    elif op.get_gate() in self.func_arguments:
                        user_defined_qubits = dict((self.func_arguments[op.get_gate()]))
                        qubit = op.get_qubits()
                        for index_quibit, qubit_gate in enumerate(list(user_defined_qubits)):
                            user_defined_qubits[qubit_gate] = qubit[index_quibit]
                        for t_line in self.func_lines[op.get_gate()]:
                            op_func = self.target_gate_profiler[t_line]
                            op_func_cp = copy.deepcopy(op_func)
                            qubits_t = op_func_cp.get_qubits()
                            updated_qubits = []
                            for q_t in qubits_t:
                                updated_qubits.append(user_defined_qubits[q_t])
                            op_func_cp.update_qubits(updated_qubits)
                            self.ordered_gate_list.append(op_func_cp)

                    else:
                        print(op.get_gate() + " not in function list!")
                        raise ValueError


        op_len = len(self.ordered_gate_list)
        is_done = False
        while (is_done == False):
            count = 0
            tmp_ordered_list = []
            for op in self.ordered_gate_list:
                if op.get_gate() in gates or op.get_gate() == 'measure':
                    count += 1
                    tmp_ordered_list.append(op)
                else:
                    if op.get_gate() in self.func_arguments:
                        user_defined_qubits = dict((self.func_arguments[op.get_gate()]))
                        qubit = op.get_qubits()
                        for index_quibit, qubit_gate in enumerate(list(user_defined_qubits)):
                            user_defined_qubits[qubit_gate] = qubit[index_quibit]
                        for t_line in self.func_lines[op.get_gate()]:
                            op_func = self.target_gate_profiler[t_line]
                            op_func_cp = copy.deepcopy(op_func)

                            # found the gates inside the function
                            # if target_gate == self.target_gate_profiler[t_line].get_func_name():
                            qubits_t = op_func_cp.get_qubits()
                            updated_qubits = []
                            for q_t in qubits_t:
                                updated_qubits.append(user_defined_qubits[q_t])
                            op_func_cp.update_qubits(updated_qubits)
                            tmp_ordered_list.append(op_func_cp)

                    else:
                        print(op.get_gate() + " not in function list!")
                        raise ValueError
            self.ordered_gate_list = tmp_ordered_list
            op_len_new = len(self.ordered_gate_list)
            if op_len == op_len_new and count == op_len:
                is_done = True
            else:
                op_len = op_len_new


    def expand_qubit(self):
        op_tmp = []
        for op in self.ordered_gate_list:
            qubits = op.get_qubits()
            if len(qubits) == 1 and "[" not in qubits[0]:
                # this means the gate is applied to all
                num = self.qubit_allocation[qubits[0]]
                for i in range(int(num)):
                    q = qubits[0]+"["+str(i)+"]"
                    op_cpy = copy.deepcopy(op)
                    op_cpy.update_qubits([q])
                    op_tmp.append(op_cpy)
            else:
                op_tmp.append(op)
        self.ordered_gate_list = op_tmp


    def identify_gate_in_function(self, gate_line):
        gate_line = gate_line.lstrip(" ")
        gate_line = gate_line.lstrip("\t")
        items = gate_line.split(" ")
        assert(len(items) >= 2)
        target_gate = items[0]
        if "(" in items[0]:
            target_gate = target_gate.split("(")[0]
        pattern_gate = '^' + target_gate
        # lets handle the qubit first
        match_gate = re.findall(pattern_gate, items[0])
        assert(len(match_gate) == 1)
        return match_gate[0]

    def identify_param_in_function(self, gate_line,target_gate):
        pattern_param = '\((.*?)\)'
        match_param = re.search(pattern_param,gate_line)
        if match_param:
            params = self.process_match(match_param,target_gate)
            return params
        else:
            return {}


    def identify_qubit_in_function(self,gate_line):
        gate_line = gate_line.lstrip(" ")
        gate_line = gate_line.lstrip("\t")
        gate_line = gate_line.split(";")
        assert(len(gate_line)>=1)
        gate_line = gate_line[0]
        items = gate_line.split(" ")
        assert(len(items) >= 2)
        qubits = items[len(items)-1]
        pattern_qubit = '\w+\d*'
        match = re.findall(pattern_qubit, qubits)
        assert (len(match) != 0)
        return match

    def handle_measure(self, line):
        line = line.split(";")[0]
        items = line.split(" ",2)
        assert(len(items) >=2)
        return items[1]


    def identify_gate_pass(self):
        with open(self.filename) as qasm:
            lines = qasm.readlines()
            pattern_param = '\((.*?)\)'
            ## we start from the "main"
            for index, line in enumerate(lines):
                if index < self.main_start or ";" not in line or "qreg"  in line or "creg"  in line or "OPENQASM" in line or "include" in line or line.startswith("//"):
                    continue
                line = line.split(";")[0]
                pattern_gate = '^'
                # ignore the extra spaces
                items = line.split(" ",1)
                assert(len(items) >= 2)
                # this means it has parameters
                target_gate = items[0]
                if "(" in items[0]:
                    target_gate = target_gate.split("(")[0]
                pattern_gate = pattern_gate+target_gate
                # lets handle the qubit first
                pattern_qubit = '\w+\[\d+\]'
                match = re.findall(pattern_qubit,items[1])
                # it may take q/a/b as a whole so lets see if that can match too
                if (len(match) == 0):
                    match = re.findall('\w+', items[1])
                    assert(len(match)>= 1)
                qubit = []
                for m in match:
                    #pos = re.findall('\d+',m)
                    #assert(len(pos) == 1)
                    qubit.append(m)
                # this means this is a custom gate, we need to handle the arguements
                '''
                if target_gate not in gates:
                    if target_gate in self.func_arguments:
                        user_defined_qubits= dict((self.func_arguments[target_gate]))
                        for index_quibit,qubit_gate in enumerate(list(user_defined_qubits)):
                            user_defined_qubits[qubit_gate] = qubit[index_quibit]
                        for t_line in self.func_lines[target_gate]:
                            # found the gates inside the function
                            #if target_gate == self.target_gate_profiler[t_line].get_func_name():
                            qubits_t = self.target_gate_profiler[t_line].get_qubits()
                            updated_qubits = []
                            for q_t in qubits_t:
                                updated_qubits.append(user_defined_qubits[q_t])
                            self.target_gate_profiler[t_line].update_qubits(updated_qubits)

                    else:
                        print(target_gate +" not in function list!")
                        raise ValueError
                '''
                match_gate = re.search(pattern_gate,line)
                if match_gate:
                    match_param = re.search(pattern_param,line)
                    if match_param:
                        params = self.process_match(match_param,target_gate)
                        if index not in self.target_gate_profiler:
                            gate_in_function = TargetGate(index,"main",target_gate,params,qubit)
                            self.target_gate_profiler[index] = gate_in_function
                    else:
                        params = []
                        if index not in self.target_gate_profiler:
                            gate_in_function = TargetGate(index,"main",target_gate,params,qubit)
                            self.target_gate_profiler[index] = gate_in_function


    def dump_qasm(self):
        sys.stdout = open(self.filename+"_exec","w")
        print("QUBIT ALLOCATION")
        for qubit in self.qubit_allocation:
            print(qubit)
            print("NUM: "+str(self.qubit_allocation[qubit]))
        for op in self.ordered_gate_list:
            print("------")
            print("gate: "+op.get_gate())
            print("parameters: ")
            print(*op.get_params(),sep=" ")
            print("on qubits")
            print(*op.get_qubits(),sep=" ")
        sys.stdout =  sys.__stdout__


    def dump_intel_gates(self):
         sys.stdout = open(self.filename+"_intel","w")
         for op in self.ordered_gate_list:
             if op.get_gate() == "measure":
                pass
             qubits = op.get_qubits()
             q_number = []
             for qubit in qubits:
                  items = re.findall('\[(\d+)]',qubit)
                  q_number.append(items[0])
             params = op.get_params()
             val_params = []
             for param in params:
                val_params.append(param)
             if op.get_gate() == "x":
                print("psi.ApplyPauliX"+"("+str(q_number[0])+");")
             elif op.get_gate() == "y":
                print("psi.ApplyPauliY"+"("+str(q_number[0])+");")
             elif op.get_gate() == "z":
                print("psi.ApplyPauliZ"+"("+str(q_number[0])+");")
             elif op.get_gate() == "h":
                print("psi.ApplyHadamard"+"("+str(q_number[0])+");")
             elif op.get_gate() == "rx":
                print("psi.ApplyRotationX"+"("+str(q_number[0])+","+str(val_params[0])+");")
             elif op.get_gate() == "ry":
                print("psi.ApplyRotationY"+"("+str(q_number[0])+","+str(val_params[0])+");")
             elif op.get_gate() == "rz":
                print("psi.ApplyRotationZ"+"("+str(q_number[0])+","+str(val_params[0])+");")
             elif op.get_gate() == "cx":
                print("psi.ApplyCPauliX"+"("+str(q_number[0])+","+str(q_number[1])+");")
             elif op.get_gate() == "cy":
                print("psi.ApplyCPauliY"+"("+str(q_number[0])+","+str(q_number[1])+");")
             elif op.get_gate() == "cz":
                print("psi.ApplyCPauliZ"+"("+str(q_number[0])+","+str(q_number[1])+");")
             elif op.get_gate() == "swap":
                print("psi.ApplySwap"+"("+str(q_number[0])+","+str(q_number[1])+");")
             elif op.get_gate() == "ccx":
                print("psi.ApplyToffoli"+"("+str(q_number[0])+","+str(q_number[1])+","+str(q_number[2])+");")
             elif op.get_gate() == "u1":
                 theta = val_params[0]
                 print("TM2x2<ComplexDP> G"+str(op.get_index())+";")
                 print("G"+str(op.get_index())+"(0,0) = {1.0, 0.0};")
                 print("G"+str(op.get_index())+"(0,1) = {0.0, 0.0};")
                 print("G"+str(op.get_index())+"(1,0) = {0.0, 0.0};")
                 u11 = np.exp(np.complex(0,theta))
                 print("G"+str(op.get_index())+"(1,1) = {"+str(np.real(u11))+","+str(np.imag(u11))+"};")
                 print("psi.Apply1QubitGate("+str(q_number[0])+",G"+str(op.get_index())+");")
             elif op.get_gate() == "u3":
                 theta = val_params[0]
                 phi = val_params[1]
                 lam = val_params[2]
                 print("TM2x2<ComplexDP> G"+str(op.get_index())+";")
                 u00 = np.cos(theta/2)
                 print("G"+str(op.get_index())+"(0,0) = {"+str(np.real(u00))+","+ str(np.imag(u00))+"};")
                 u01_theta = np.sin(theta/2)
                 u01_exp = -1*np.exp(np.complex(0,lam))
                 u01 = u01_exp*u01_theta
                 print("G"+str(op.get_index())+"(0,1) = {"+str(np.real(u01))+","+ str(np.imag(u01))+"};")
                 u10_theta = np.sin(theta/2)
                 u10_exp = np.exp(np.complex(0,phi))
                 u10 = u10_exp*u10_theta 
                 print("G"+str(op.get_index())+"(1,0) = {"+str(np.real(u10))+","+ str(np.imag(u10))+"};")
                 u11_theta = np.cos(theta/2)
                 u11_exp = np.exp(np.complex(0,theta+phi))
                 u11 = u11_exp*u11_theta
                 print("G"+str(op.get_index())+"(1,1) = {"+str(np.real(u11))+","+str(np.imag(u11))+"};")
                 print("psi.Apply1QubitGate("+str(q_number[0])+",G"+str(op.get_index())+");")
             elif op.get_gate() == 'ryy':
                 theta = val_params[0]
                 # (RXGate(np.pi / 2), [q[0]], []),
                 # (RXGate(np.pi / 2), [q[1]], []),
                 #(CXGate(), [q[0], q[1]], []),
                 #(RZGate(theta), [q[1]], []),
                 #(CXGate(), [q[0], q[1]], []),
                 #(RXGate(-np.pi / 2), [q[0]], []),
                 #(RXGate(-np.pi / 2), [q[1]], []),
                 print("psi.ApplyRotationX"+"("+str(q_number[0])+","+str(np.pi/2)+");")
                 print("psi.ApplyRotationX"+"("+str(q_number[1])+","+str(np.pi/2)+");")
                 print("psi.ApplyCPauliX"+"("+str(q_number[0])+","+str(q_number[1])+");")
                 print("psi.ApplyRotationZ"+"("+str(q_number[1])+","+str(theta)+");")
                 print("psi.ApplyCPauliX"+"("+str(q_number[0])+","+str(q_number[1])+");")
                 print("psi.ApplyRotationX"+"("+str(q_number[0])+","+str(-1*np.pi/2)+");")
                 print("psi.ApplyRotationX"+"("+str(q_number[1])+","+str(-1*np.pi/2)+");")
             elif op.get_gate() == 'rzz':
                 theta = val_params[0]
                 #(CXGate(), [q[0], q[1]], []),
                 #(RZGate(theta), [q[1]], []),
                 #(CXGate(), [q[0], q[1]], []),
                 print("psi.ApplyCPauliX"+"("+str(q_number[0])+","+str(q_number[1])+");")
                 print("psi.ApplyRotationZ"+"("+str(q_number[1])+","+str(theta)+");")
                 print("psi.ApplyCPauliX"+"("+str(q_number[0])+","+str(q_number[1])+");")
             elif op.get_gate() == 'cswap':
                #(CXGate(), [q[2], q[1]], []),
                #(CCXGate(), [q[0], q[1], q[2]], []),
                #(CXGate(), [q[2], q[1]], [])
                 print("psi.ApplyCPauliX"+"("+str(q_number[2])+","+str(q_number[1])+");")
                 print("psi.ApplyToffoli"+"("+str(q_number[0])+","+str(q_number[1])+","+str(q_number[2])+");")
                 print("psi.ApplyCPauliX"+"("+str(q_number[2])+","+str(q_number[1])+");") 
             else:
                print(op.get_gate()+" is not supported")

    def dump_quest_gates(self):
         sys.stdout = open(self.filename+"_quest","w")
         for op in self.ordered_gate_list:
             if op.get_gate() == "measure":
                pass
             qubits = op.get_qubits()
             q_number = []
             for qubit in qubits:
                  items = re.findall('\[(\d+)]',qubit)
                  q_number.append(items[0])
             params = op.get_params()
             val_params = []
             for param in params:
                val_params.append(param)
             if op.get_gate() == "x":
                print("psi.ApplyPauliX"+"("+str(q_number[0])+");")
             elif op.get_gate() == "y":
                print("psi.ApplyPauliY"+"("+str(q_number[0])+");")
             elif op.get_gate() == "z":
                print("psi.ApplyPauliZ"+"("+str(q_number[0])+");")
             elif op.get_gate() == "h":
                print("psi.ApplyHadamard"+"("+str(q_number[0])+");")
             elif op.get_gate() == "rx":
                print("psi.ApplyRotationX"+"("+str(q_number[0])+","+str(val_params[0])+");")
             elif op.get_gate() == "ry":
                print("psi.ApplyRotationY"+"("+str(q_number[0])+","+str(val_params[0])+");")
             elif op.get_gate() == "rz":
                print("psi.ApplyRotationZ"+"("+str(q_number[0])+","+str(val_params[0])+");")
             elif op.get_gate() == "cx":
                print("psi.ApplyCPauliX"+"("+str(q_number[0])+","+str(q_number[1])+");")
             elif op.get_gate() == "cy":
                print("psi.ApplyCPauliY"+"("+str(q_number[0])+","+str(q_number[1])+");")
             elif op.get_gate() == "cz":
                print("psi.ApplyCPauliZ"+"("+str(q_number[0])+","+str(q_number[1])+");")
             elif op.get_gate() == "swap":
                print("psi.ApplySwap"+"("+str(q_number[0])+","+str(q_number[1])+");")
             elif op.get_gate() == "ccx":
                print("psi.ApplyToffoli"+"("+str(q_number[0])+","+str(q_number[1])+","+str(q_number[2])+");")
             elif op.get_gate() == "u1":
                 theta = val_params[0]
                 print("TM2x2<ComplexDP> G"+str(op.get_index())+";")
                 print("G"+str(op.get_index())+"(0,0) = {1.0, 0.0};")
                 print("G"+str(op.get_index())+"(0,1) = {0.0, 0.0};")
                 print("G"+str(op.get_index())+"(1,0) = {0.0, 0.0};")
                 u11 = np.exp(np.complex(0,theta))
                 print("G"+str(op.get_index())+"(1,1) = {"+str(np.real(u11))+","+str(np.imag(u11))+"};")
                 print("psi.Apply1QubitGate("+str(q_number[0])+",G"+str(op.get_index())+");")
             elif op.get_gate() == "u3":
                 theta = val_params[0]
                 phi = val_params[1]
                 lam = val_params[2]
                 print("TM2x2<ComplexDP> G"+str(op.get_index())+";")
                 u00 = np.cos(theta/2)
                 print("G"+str(op.get_index())+"(0,0) = {"+str(np.real(u00))+","+ str(np.imag(u00))+"};")
                 u01_theta = np.sin(theta/2)
                 u01_exp = -1*np.exp(np.complex(0,lam))
                 u01 = u01_exp*u01_theta
                 print("G"+str(op.get_index())+"(0,1) = {"+str(np.real(u01))+","+ str(np.imag(u01))+"};")
                 u10_theta = np.sin(theta/2)
                 u10_exp = np.exp(np.complex(0,phi))
                 u10 = u10_exp*u10_theta 
                 print("G"+str(op.get_index())+"(1,0) = {"+str(np.real(u10))+","+ str(np.imag(u10))+"};")
                 u11_theta = np.cos(theta/2)
                 u11_exp = np.exp(np.complex(0,theta+phi))
                 u11 = u11_exp*u11_theta
                 print("G"+str(op.get_index())+"(1,1) = {"+str(np.real(u11))+","+str(np.imag(u11))+"};")
                 print("psi.Apply1QubitGate("+str(q_number[0])+",G"+str(op.get_index())+");")
             elif op.get_gate() == 'ryy':
                 theta = val_params[0]
                 # (RXGate(np.pi / 2), [q[0]], []),
                 # (RXGate(np.pi / 2), [q[1]], []),
                 #(CXGate(), [q[0], q[1]], []),
                 #(RZGate(theta), [q[1]], []),
                 #(CXGate(), [q[0], q[1]], []),
                 #(RXGate(-np.pi / 2), [q[0]], []),
                 #(RXGate(-np.pi / 2), [q[1]], []),
                 print("psi.ApplyRotationX"+"("+str(q_number[0])+","+str(np.pi/2)+");")
                 print("psi.ApplyRotationX"+"("+str(q_number[1])+","+str(np.pi/2)+");")
                 print("psi.ApplyCPauliX"+"("+str(q_number[0])+","+str(q_number[1])+");")
                 print("psi.ApplyRotationZ"+"("+str(q_number[1])+","+str(theta)+");")
                 print("psi.ApplyCPauliX"+"("+str(q_number[0])+","+str(q_number[1])+");")
                 print("psi.ApplyRotationX"+"("+str(q_number[0])+","+str(-1*np.pi/2)+");")
                 print("psi.ApplyRotationX"+"("+str(q_number[1])+","+str(-1*np.pi/2)+");")
             elif op.get_gate() == 'rzz':
                 theta = val_params[0]
                 #(CXGate(), [q[0], q[1]], []),
                 #(RZGate(theta), [q[1]], []),
                 #(CXGate(), [q[0], q[1]], []),
                 print("psi.ApplyCPauliX"+"("+str(q_number[0])+","+str(q_number[1])+");")
                 print("psi.ApplyRotationZ"+"("+str(q_number[1])+","+str(theta)+");")
                 print("psi.ApplyCPauliX"+"("+str(q_number[0])+","+str(q_number[1])+");")
             elif op.get_gate() == 'cswap':
                #(CXGate(), [q[2], q[1]], []),
                #(CCXGate(), [q[0], q[1], q[2]], []),
                #(CXGate(), [q[2], q[1]], [])
                 print("psi.ApplyCPauliX"+"("+str(q_number[2])+","+str(q_number[1])+");")
                 print("psi.ApplyToffoli"+"("+str(q_number[0])+","+str(q_number[1])+","+str(q_number[2])+");")
                 print("psi.ApplyCPauliX"+"("+str(q_number[2])+","+str(q_number[1])+");") 
             else:
                print(op.get_gate()+" is not supported")

    def regenerate_qasm_parts(self,parts_file):
        parts = self.parse_parts_file(parts_file)
        qubits_in_parts = self.collect_parts_qubits(parts_file)
        uniqubits = self.create_universal_qubit()
        op_counter = 0
        for part in parts:
            sys.stdout = open(self.filename+"_regen"+str(part),"w")
            print("OPENQASM 2.0;")
            print("include \"qelib1.inc\";")
            qubit_literal = "q"
            print("qreg "+qubit_literal+"["+str(len(qubits_in_parts[part]))+"]"+";")
            temp = qubits_in_parts[part]
            temp.sort()
            qubit_part_new_order = {}
            for t in temp:
                qubit_part_new_order[t] = len(qubit_part_new_order)
            for order_id in parts[part]:
                if order_id == 0:
                    continue
                elif (order_id - 1024) > len(self.ordered_gate_list):
                    continue
                else:
                    order_id = order_id - 1
                op = self.ordered_gate_list[order_id]
                inst = ""
                inst += op.get_gate()
                params = op.get_params()
                if len(params) != 0:
                    inst += " ("
                    for param in params:
                        inst += str(param)
                        inst += ","
                    inst = inst.rstrip(",")
                    inst += ")"
                qubits = op.get_qubits()
                inst += " "
                if len(qubits) != 0:
                    for q in qubits:
                        inst += qubit_literal+"["+str(qubit_part_new_order[uniqubits[q]])+"]"
                        inst += ","
                inst = inst.rstrip(",")
                inst += ";"
                print(inst)
                op_counter += 1
            sys.stdout = sys.__stdout__
        print(op_counter)

    def regenerate_qasm(self,parts_file):
        parts = self.parse_parts_file(parts_file)
        new_list = []
        for part in parts:
            for order_id in parts[part]:
                if order_id == 0:
                    continue
                elif (order_id - 1024) > len(self.ordered_gate_list):
                    continue
                else:
                    order_id = order_id - 1
                op = self.ordered_gate_list[order_id]
                new_list.append(op)
        sys.stdout = open(self.filename+"_reordered","w")
        print("OPENQASM 2.0;")
        print("include \"qelib1.inc\";")
        for qubit in self.qubit_allocation:
            print("qreg "+qubit+"["+str(self.qubit_allocation[qubit])+"]"+";")
        for op in new_list:
            inst = ""
            inst += op.get_gate()
            params = op.get_params()
            if len(params) != 0:
                inst += "("
                for param in params:
                    inst += str(param)
                    inst += ","
                inst = inst.rstrip(",")
                inst += ")"
            qubits = op.get_qubits()
            inst += " "
            if len(qubits) != 0:
                for q in qubits:
                    inst += q
                    inst += ","
            inst = inst.rstrip(",")
            inst += ";"
            print(inst)
        sys.stdout = sys.__stdout__


    def verify_compute(self):
        global acc_timer
        global start_timer
        global allocation_scheme
        uqubits = self.create_universal_qubit()
        inv_uqubits = self.reverse_qubit(uqubits)
        total_qubits = len(uqubits)
        s_before = datetime.now()
        sim = svsim.StateVector(total_qubits, allocation_scheme)
        sim_init = svsim.StateVector(total_qubits, allocation_scheme)
        s_after = datetime.now()
        s_diff = s_after - s_before
        start_timer = s_diff.seconds * 1000000 + s_diff.microseconds
        print("SVSIM start takes " + str(start_timer))
        for op in self.ordered_gate_list:
            gate = op.get_gate().upper()
            # if it is measure, handle it first
            if gate == "MEASURE":
                measure_qubit = []
                assert (len(op.get_qubits()) == 2)
                q_m = uqubits[op.get_qubits()[0]]
                measure_qubit.append(q_m)
                # results = sim.measure(measure_qubit,NUM_MEASURE)
                # print(results)
                continue
            if gate in GATE_NOPARAM:
                ag = svsim.AggregateGate_0(gate)
            elif gate in GATE_ONEPARAM:
                assert (len(op.get_params()) == 1)
                ag = svsim.AggregateGate_1(gate, op.get_params()[0])
            elif gate in GATE_TWOPARAM:
                assert (len(op.get_params()) == 2)
                ag = svsim.AggregateGate_2(gate, op.get_params()[0], op.get_params()[1])
            elif gate in GATE_THREEPARAM:
                assert (len(op.get_params()) == 3)
                ag = svsim.AggregateGate_3(gate, op.get_params()[0], op.get_params()[1], op.get_params()[2])
            elif gate in GATE_C1:
                assert (len(op.get_params()) == 4)
                ag = svsim.AggregateGate_C1(gate, op.get_params()[0], op.get_params()[1], op.get_params()[2],
                                            op.get_params()[3])
            elif gate in GATE_C2:
                assert (len(op.get_params()) == 16)
                ag = svsim.AggregateGate_C2(gate, op.get_params()[0], op.get_params()[1], op.get_params()[2],
                                            op.get_params()[3],
                                            op.get_params()[4], op.get_params()[5], op.get_params()[6],
                                            op.get_params()[7],
                                            op.get_params()[8], op.get_params()[9], op.get_params()[10],
                                            op.get_params()[11],
                                            op.get_params()[12], op.get_params()[13], op.get_params()[14],
                                            op.get_params()[15])
            elif gate in GATE_FRAC:
                assert (len(op.get_params()) == 2)
                ag = svsim.AggregateGate_Frac(gate, op.get_params()[0], op.get_params()[1])

            else:
                # this means the gate is probabaly not a basic gate. We need to handle them later
                ag = svsim.AggregateGate_0("I")
            ag_factory = svsim.AggregateGateFactory(ag)
            op_qubits = len(op.get_qubits())
            control_bits = []
            target_bits = []
            if op_qubits == 1:
                # this means the control qubit should be null
                target_bits.append(uqubits[op.get_qubits()[0]])
            if op_qubits == 2:
                if gate == 'CX':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'X'
                    ag = svsim.AggregateGate_0(gate_alt)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CY':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'Y'
                    ag = svsim.AggregateGate_0(gate_alt)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CZ':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'Z'
                    ag = svsim.AggregateGate_0(gate_alt)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CH':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'H'
                    ag = svsim.AggregateGate_0(gate_alt)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CRX':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'RX'
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CRY':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'RY'
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CRZ':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'RZ'
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'RXX':
                    t_bit = uqubits[op.get_qubits()[0]]
                    target_bits.append(t_bit)
                    t_bit = uqubits[op.get_qubits()[1]]
                    target_bits.append(t_bit)
                    gate_alt = 'RXX'
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'RYY':
                    t_bit = uqubits[op.get_qubits()[0]]
                    target_bits.append(t_bit)
                    t_bit = uqubits[op.get_qubits()[1]]
                    target_bits.append(t_bit)
                    gate_alt = 'RYY'
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'RZZ':
                    t_bit = uqubits[op.get_qubits()[0]]
                    target_bits.append(t_bit)
                    t_bit = uqubits[op.get_qubits()[1]]
                    target_bits.append(t_bit)
                    gate_alt = 'RZZ'
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CU1':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'U1'
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CU3':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'U3'
                    ag = svsim.AggregateGate_3(gate_alt, op.get_params()[0], op.get_params()[1], op.get_params()[2])
                    ag_factory = svsim.AggregateGateFactory(ag)
            if op_qubits == 3:
                if gate == 'CCX':
                    c_bit_1 = uqubits[op.get_qubits()[0]]
                    c_bit_2 = uqubits[op.get_qubits()[1]]
                    t_bit = uqubits[op.get_qubits()[2]]
                    control_bits.append(c_bit_1)
                    control_bits.append(c_bit_2)
                    target_bits.append(t_bit)
                    gate_alt = 'X'
                    ag = svsim.AggregateGate_0(gate_alt)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CSWAP':
                    c_bit_1 = uqubits[op.get_qubits()[0]]
                    t_bit_1 = uqubits[op.get_qubits()[1]]
                    t_bit_2 = uqubits[op.get_qubits()[2]]
                    control_bits.append(c_bit_1)
                    target_bits.append(t_bit_1)
                    target_bits.append(t_bit_2)
                    gate_alt = 'SWAP'
                    ag = svsim.AggregateGate_0(gate_alt)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'RCCX':
                    c_bit_1 = uqubits[op.get_qubits()[0]]
                    c_bit_2 = uqubits[op.get_qubits()[1]]
                    t_bit_1 = uqubits[op.get_qubits()[2]]
                    control_bits.append(c_bit_1)
                    control_bits.append(c_bit_2)
                    target_bits.append(t_bit_1)
                    gate_alt = 'RX'
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                    ag_factory = svsim.AggregateGateFactory(ag)
            if op_qubits == 4:
                if gate == 'RC3X':
                    c_bit_1 = uqubits[op.get_qubits()[0]]
                    c_bit_2 = uqubits[op.get_qubits()[1]]
                    c_bit_3 = uqubits[op.get_qubits()[2]]
                    t_bit_1 = uqubits[op.get_qubits()[3]]
                    control_bits.append(c_bit_1)
                    control_bits.append(c_bit_2)
                    control_bits.append(c_bit_3)
                    target_bits.append(t_bit_1)
                    gate_alt = 'RX'
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'C3X':
                    c_bit_1 = uqubits[op.get_qubits()[0]]
                    c_bit_2 = uqubits[op.get_qubits()[1]]
                    c_bit_3 = uqubits[op.get_qubits()[2]]
                    t_bit_1 = uqubits[op.get_qubits()[3]]
                    control_bits.append(c_bit_1)
                    control_bits.append(c_bit_2)
                    control_bits.append(c_bit_3)
                    target_bits.append(t_bit_1)
                    gate_alt = 'X'
                    ag = svsim.AggregateGate_0(gate_alt)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'C3XSQRTX':
                    pass
            if op_qubits == 5:
                if gate == 'C4X':
                    c_bit_1 = uqubits[op.get_qubits()[0]]
                    c_bit_2 = uqubits[op.get_qubits()[1]]
                    c_bit_3 = uqubits[op.get_qubits()[2]]
                    c_bit_4 = uqubits[op.get_qubits()[3]]
                    t_bit_1 = uqubits[op.get_qubits()[4]]
                    control_bits.append(c_bit_1)
                    control_bits.append(c_bit_2)
                    control_bits.append(c_bit_3)
                    control_bits.append(c_bit_4)
                    target_bits.append(t_bit_1)
                    gate_alt = 'X'
                    ag = svsim.AggregateGate_0(gate_alt)
                    ag_factory = svsim.AggregateGateFactory(ag)
            # print(control_bits)
            # print(target_bits)
            boundgate = svsim.BoundGate(ag_factory, control_bits, target_bits)
            now = datetime.now()
            boundgate.apply(sim)
            #sim.dump_state()
            later = datetime.now()
            diff = later - now
            acc_timer += diff.seconds * 1000000 + diff.microseconds
            print(gate + " has been applied.")
            print("time elapsed " + str(acc_timer))
        for op in reversed(self.ordered_gate_list):
            gate = op.get_gate().upper()
            # if it is measure, handle it first
            if gate == "MEASURE":
                measure_qubit = []
                assert (len(op.get_qubits()) == 2)
                q_m = uqubits[op.get_qubits()[0]]
                measure_qubit.append(q_m)
                # results = sim.measure(measure_qubit,NUM_MEASURE)
                # print(results)
                continue
            if gate in GATE_NOPARAM:
                ag = svsim.AggregateGate_0(gate)
            elif gate in GATE_ONEPARAM:
                assert (len(op.get_params()) == 1)
                if gate == 'U1':
                    ag = svsim.AggregateGate_1(gate, op.get_params()[0]*-1)
                else:
                    ag = svsim.AggregateGate_1(gate, op.get_params()[0])    
            elif gate in GATE_TWOPARAM:
                assert (len(op.get_params()) == 2)
                ag = svsim.AggregateGate_2(gate, op.get_params()[0], op.get_params()[1])
            elif gate in GATE_THREEPARAM:
                assert (len(op.get_params()) == 3)
                ag = svsim.AggregateGate_3(gate, op.get_params()[0], op.get_params()[1], op.get_params()[2])
            elif gate in GATE_C1:
                assert (len(op.get_params()) == 4)
                ag = svsim.AggregateGate_C1(gate, op.get_params()[0], op.get_params()[1], op.get_params()[2],
                                            op.get_params()[3])
            elif gate in GATE_C2:
                assert (len(op.get_params()) == 16)
                ag = svsim.AggregateGate_C2(gate, op.get_params()[0], op.get_params()[1], op.get_params()[2],
                                            op.get_params()[3],
                                            op.get_params()[4], op.get_params()[5], op.get_params()[6],
                                            op.get_params()[7],
                                            op.get_params()[8], op.get_params()[9], op.get_params()[10],
                                            op.get_params()[11],
                                            op.get_params()[12], op.get_params()[13], op.get_params()[14],
                                            op.get_params()[15])
            elif gate in GATE_FRAC:
                assert (len(op.get_params()) == 2)
                ag = svsim.AggregateGate_Frac(gate, op.get_params()[0], op.get_params()[1])

            else:
                # this means the gate is probabaly not a basic gate. We need to handle them later
                ag = svsim.AggregateGate_0("I")
            ag_factory = svsim.AggregateGateFactory(ag)
            op_qubits = len(op.get_qubits())
            control_bits = []
            target_bits = []
            if op_qubits == 1:
                # this means the control qubit should be null
                target_bits.append(uqubits[op.get_qubits()[0]])
            if op_qubits == 2:
                if gate == 'CX':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'X'
                    ag = svsim.AggregateGate_0(gate_alt)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CY':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'Y'
                    ag = svsim.AggregateGate_0(gate_alt)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CZ':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'Z'
                    ag = svsim.AggregateGate_0(gate_alt)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CH':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'H'
                    ag = svsim.AggregateGate_0(gate_alt)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CRX':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'RX'
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CRY':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'RY'
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CRZ':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'RZ'
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CU1':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'U1'
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0]*-1)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'RXX':
                    t_bit = uqubits[op.get_qubits()[0]]
                    target_bits.append(t_bit)
                    t_bit = uqubits[op.get_qubits()[1]]
                    target_bits.append(t_bit)
                    gate_alt = 'RXX'
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0]*-1)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'RYY':
                    t_bit = uqubits[op.get_qubits()[0]]
                    target_bits.append(t_bit)
                    t_bit = uqubits[op.get_qubits()[1]]
                    target_bits.append(t_bit)
                    gate_alt = 'RYY'
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0]*-1)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'RZZ':
                    t_bit = uqubits[op.get_qubits()[0]]
                    target_bits.append(t_bit)
                    t_bit = uqubits[op.get_qubits()[1]]
                    target_bits.append(t_bit)
                    gate_alt = 'RZZ'
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0]*-1)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CU3':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'U3'
                    ag = svsim.AggregateGate_3(gate_alt, op.get_params()[0], op.get_params()[1], op.get_params()[2])
                    ag_factory = svsim.AggregateGateFactory(ag)
            if op_qubits == 3:
                if gate == 'CCX':
                    c_bit_1 = uqubits[op.get_qubits()[0]]
                    c_bit_2 = uqubits[op.get_qubits()[1]]
                    t_bit = uqubits[op.get_qubits()[2]]
                    control_bits.append(c_bit_1)
                    control_bits.append(c_bit_2)
                    target_bits.append(t_bit)
                    gate_alt = 'X'
                    ag = svsim.AggregateGate_0(gate_alt)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CSWAP':
                    c_bit_1 = uqubits[op.get_qubits()[0]]
                    t_bit_1 = uqubits[op.get_qubits()[1]]
                    t_bit_2 = uqubits[op.get_qubits()[2]]
                    control_bits.append(c_bit_1)
                    target_bits.append(t_bit_1)
                    target_bits.append(t_bit_2)
                    gate_alt = 'SWAP'
                    ag = svsim.AggregateGate_0(gate_alt)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'RCCX':
                    c_bit_1 = uqubits[op.get_qubits()[0]]
                    c_bit_2 = uqubits[op.get_qubits()[1]]
                    t_bit_1 = uqubits[op.get_qubits()[2]]
                    control_bits.append(c_bit_1)
                    control_bits.append(c_bit_2)
                    target_bits.append(t_bit_1)
                    gate_alt = 'RX'
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                    ag_factory = svsim.AggregateGateFactory(ag)
            if op_qubits == 4:
                if gate == 'RC3X':
                    c_bit_1 = uqubits[op.get_qubits()[0]]
                    c_bit_2 = uqubits[op.get_qubits()[1]]
                    c_bit_3 = uqubits[op.get_qubits()[2]]
                    t_bit_1 = uqubits[op.get_qubits()[3]]
                    control_bits.append(c_bit_1)
                    control_bits.append(c_bit_2)
                    control_bits.append(c_bit_3)
                    target_bits.append(t_bit_1)
                    gate_alt = 'RX'
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'C3X':
                    c_bit_1 = uqubits[op.get_qubits()[0]]
                    c_bit_2 = uqubits[op.get_qubits()[1]]
                    c_bit_3 = uqubits[op.get_qubits()[2]]
                    t_bit_1 = uqubits[op.get_qubits()[3]]
                    control_bits.append(c_bit_1)
                    control_bits.append(c_bit_2)
                    control_bits.append(c_bit_3)
                    target_bits.append(t_bit_1)
                    gate_alt = 'X'
                    ag = svsim.AggregateGate_0(gate_alt)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'C3XSQRTX':
                    pass
            if op_qubits == 5:
                if gate == 'C4X':
                    c_bit_1 = uqubits[op.get_qubits()[0]]
                    c_bit_2 = uqubits[op.get_qubits()[1]]
                    c_bit_3 = uqubits[op.get_qubits()[2]]
                    c_bit_4 = uqubits[op.get_qubits()[3]]
                    t_bit_1 = uqubits[op.get_qubits()[4]]
                    control_bits.append(c_bit_1)
                    control_bits.append(c_bit_2)
                    control_bits.append(c_bit_3)
                    control_bits.append(c_bit_4)
                    target_bits.append(t_bit_1)
                    gate_alt = 'X'
                    ag = svsim.AggregateGate_0(gate_alt)
                    ag_factory = svsim.AggregateGateFactory(ag)
            # print(control_bits)
            # print(target_bits)
            boundgate = svsim.BoundGate(ag_factory, control_bits, target_bits)
            now = datetime.now()
            boundgate.apply(sim)
            #sim.dump_state()
            later = datetime.now()
            diff = later - now
            acc_timer += diff.seconds * 1000000 + diff.microseconds
            print(gate + " has been applied.")
            print("time elapsed " + str(acc_timer))
        if sim == sim_init:
            print("VERIFIED")
        else:
            print("NOT VERIFIED")

    def gate_fusion(self):
        #This gate fusion functionality is only for one and two qubits 
        op_blocks = []
        uqubits = self.create_universal_qubit()
        two_qubit_gates = []
        tmp_ops = []
        for op in self.ordered_gate_list:
            gate = op.get_gate().upper()
            if gate.startswith("C") or gate == "SWAP":
                if len(tmp_ops) != 0:
                    op_blocks.append(tmp_ops)
                    tmp_ops = []
                    tmp_ops.append(op)
                    two_qubit_gates.append(op)
                    op_blocks.append(tmp_ops)
                else:
                    tmp_ops.append(op)
                    two_qubit_gates.append(op)
                    op_blocks.append(tmp_ops)
                tmp_ops = []
            else:
                tmp_ops.append(op)
        if len(tmp_ops) != 0:
            op_blocks.append(tmp_ops)
        # first pass to absorb gates for control/swap gates
        fused_gate_dic_control = OrderedDict()
        for two_op in two_qubit_gates:
            qubits = two_op.get_qubits()
            op_gate = gates_handler.get_gate_handler(two_op.get_gate().upper(), two_op.get_params())
            if len(qubits) == 2:
                if qubits[0] not in fused_gate_dic_control and qubits[1] not in fused_gate_dic_control:
                    # this means we need to create a fused gate for this pair
                    fused_gate = gates_handler.FusedGate(qubits)
                    op_gate = gates_handler.get_gate_handler(two_op.get_gate().upper(), two_op.get_params())
                    fused_gate.set_unitary(op_gate.matrix())
                    fused_gate.set_control()
                    fused_gate.set_absorbed(two_op.get_index())
                    fused_gate.set_control_id(two_op.get_index())
                    fused_gate_dic_control[qubits[0]] = fused_gate
                    fused_gate_dic_control[qubits[1]] = fused_gate
                    fused_gate.set_end_id(op_blocks[-1][-1].get_index())
                else:
                    if qubits[0] in fused_gate_dic_control and qubits[1] not in fused_gate_dic_control:
                        fused_gate = fused_gate_dic_control[qubits[0]]
                        fused_gate.set_end_id(two_op.get_index())
                    if qubits[0] not in fused_gate_dic_control and qubits[1] in fused_gate_dic_control:
                        fused_gate = fused_gate_dic_control[qubits[1]]
                        fused_gate.set_end_id(two_op.get_index())
                    if qubits[0] in fused_gate_dic_control and qubits[1] in fused_gate_dic_control:
                        # this means this gate can be absorbed again
                        fused_gate = fused_gate_dic_control[qubits[0]]
                        fused_gate.set_unitary(gates_handler.muliply_by_one_qubit(op_gate.matrix(), fused_gate.get_unitary()))
        # to absorb the single qubit gates in this pass
        visited_qubit = []
        for qubit in fused_gate_dic_control:
            if qubit not in visited_qubit:
                visited_qubit.append(qubit)
                fused_gate = fused_gate_dic_control[qubit]
                op_blocks_after_absorb = []
                for ops in op_blocks:
                    tmp_ops = []
                    if ops[-1].get_index() < fused_gate.get_end_id():
                        tmp_ops = reversed(ops)
                    else:
                        tmp_ops = ops
                    for op in tmp_ops:
                        num_qubits = len(op.get_qubits())
                        gate = op.get_gate().upper()
                        op_gate = gates_handler.get_gate_handler(gate, op.get_params())
                        id = op.get_index()
                        if num_qubits == 1:
                            # add this qubit to the fused_qubits
                            qubit = op.get_qubits()[0]
                            if qubit not in fused_gate_dic_control:
                                # this means that this qubit is not related to the control gates.
                                # we do not process it in this pass
                                pass
                            else:
                                fused_gate = fused_gate_dic_control[qubit]
                                if fused_gate.get_control() == True:
                                    if fused_gate.is_first(qubit):
                                        new_tensored_gate = gates_handler.expand_gate(op_gate.matrix(),gates_handler.ID("I").matrix())
                                    if fused_gate.is_second(qubit):
                                        new_tensored_gate = gates_handler.expand_gate(gates_handler.ID("I").matrix(),op_gate.matrix())
                                    # the order is important here
                                    # it is different for before and after the control gate
                                    if id < fused_gate.get_control_id():
                                        fused_gate.set_unitary(gates_handler.muliply_by_one_qubit(fused_gate.get_unitary(),new_tensored_gate))
                                        fused_gate.set_absorbed(id)
                                    else:
                                        if id < fused_gate.get_end_id():
                                            fused_gate.set_unitary(gates_handler.muliply_by_one_qubit(new_tensored_gate,fused_gate.get_unitary()))
                                            fused_gate.set_absorbed(id)
        op_blocks_after_absorb = []
        for ops in op_blocks:
            tmp_ops = []
            for op in ops:
                is_absored = 0
                for qubit in fused_gate_dic_control:
                    fused_gate = fused_gate_dic_control[qubit]
                    if op.get_index() in fused_gate.absorbed_id:
                        is_absored = 1
                        break
                if is_absored == 0:
                    tmp_ops.append(op)
            op_blocks_after_absorb.append(tmp_ops)
        final_new_gates = []
        fused_gate_dic = OrderedDict()
        for ops in op_blocks_after_absorb:
            '''
            if len(ops) == 1:
                fused_gate = gates_handler.FusedGate(ops[0].get_qubits())
                op_gate = gates_handler.get_gate_handler(ops[0].get_gate().upper(),ops[0].get_params())
                fused_gate.set_unitary(op_gate.matrix())
                final_new_gates.append(fused_gate)
                continue
            '''
            fused_qubits_list = []
            # get the qubits in this block
            for op in ops:
                 for q in op.get_qubits():
                     if q not in fused_qubits_list:
                         fused_qubits_list.append(q)
            for op in ops:
                num_qubits = len(op.get_qubits())
                gate = op.get_gate().upper()
                op_gate = gates_handler.get_gate_handler(gate, op.get_params())
                if num_qubits == 1:
                    # add this qubit to the fused_qubits 
                    qubit = op.get_qubits()[0]
                    if qubit not in fused_gate_dic:
                        if len(fused_gate_dic) == 0:
                            fused_gate = gates_handler.FusedGate([qubit])
                            fused_gate_dic[qubit] = fused_gate
                            fused_gate.set_unitary(op_gate.matrix())
                        else:
                            qubit_flag = 0
                            for fused_q in fused_gate_dic:
                                fused_gate = fused_gate_dic[fused_q]
                                if fused_gate.current_size() != 2:
                                    if fused_gate.is_exist(qubit) == False:
                                        fused_gate.set_qubit_2(qubit)
                                        fused_gate_dic[qubit] = fused_gate
                                        fused_gate.set_unitary(gates_handler.expand_gate(fused_gate.get_unitary(), op_gate.matrix()))
                                        qubit_flag = 1
                            if qubit_flag == 0:
                                fused_gate = gates_handler.FusedGate([qubit])
                                fused_gate_dic[qubit] = fused_gate
                                fused_gate.set_unitary(op_gate.matrix())
                    else:
                        fused_gate = fused_gate_dic[qubit]
                        if fused_gate.is_first(qubit):
                            new_tensored_gate = gates_handler.expand_gate(op_gate.matrix(),gates_handler.ID("I").matrix())
                        if fused_gate.is_second(qubit):
                            new_tensored_gate = gates_handler.expand_gate(gates_handler.ID("I").matrix(),op_gate.matrix())
                        fused_gate.set_unitary(gates_handler.muliply_by_one_qubit(fused_gate.get_unitary(),new_tensored_gate))

                if num_qubits == 2:
                    # it seems that this case would not be encountered
                    pass
        free_qubit_list = []
        for qubit in fused_gate_dic:
            if qubit not in free_qubit_list:
                final_new_gates.append(fused_gate_dic[qubit])
                free_qubit_list.append(qubit)
                if len(fused_gate_dic[qubit].get_qubits()) >= 1:
                    free_qubit_list.append(fused_gate_dic[qubit].get_qubits()[1])
        for qubit in fused_gate_dic_control:
            if qubit not in fused_gate_dic_control:
                final_new_gates.append(fused_gate_dic_control[qubit])
                free_qubit_list.append(qubit)
                if len(fused_gate_dic_control[qubit].get_qubits()) >= 1:
                    free_qubit_list.append(fused_gate_dic_control[qubit].get_qubits()[1])
        print("Fusion complete")
                
                 

    def execute_qasm(self):
        global acc_timer
        global start_timer
        global allocation_scheme
        uqubits = self.create_universal_qubit()
        inv_uqubits = self.reverse_qubit(uqubits)
        total_qubits = len(uqubits)
        s_before = datetime.now()
        print(allocation_scheme)
        sim = svsim.StateVector(total_qubits,allocation_scheme)
        s_after = datetime.now()
        s_diff = s_after-s_before
        start_timer = s_diff.seconds*1000000+s_diff.microseconds
        print("SVSIM start takes "+str(start_timer))
        #exit()
        for op in self.ordered_gate_list:
            gate = op.get_gate().upper()
            # if it is measure, handle it first
            if gate == "MEASURE":
                measure_qubit = []
                assert(len(op.get_qubits()) == 2)
                q_m = uqubits[op.get_qubits()[0]]
                measure_qubit.append(q_m)
                #results = sim.measure(measure_qubit,NUM_MEASURE)
                #print(results)
                continue
            if gate in GATE_NOPARAM:
                ag = svsim.AggregateGate_0(gate)
            elif gate in GATE_ONEPARAM:
                assert(len(op.get_params()) == 1)
                ag = svsim.AggregateGate_1(gate,op.get_params()[0])
            elif gate in GATE_TWOPARAM:
                assert(len(op.get_params()) == 2)
                ag = svsim.AggregateGate_2(gate,op.get_params()[0],op.get_params()[1])
            elif gate in GATE_THREEPARAM:
                assert(len(op.get_params()) == 3)
                ag = svsim.AggregateGate_3(gate,op.get_params()[0],op.get_params()[1],op.get_params()[2])
            elif gate in GATE_C1:
                assert (len(op.get_params()) == 4)
                ag = svsim.AggregateGate_C1(gate, op.get_params()[0], op.get_params()[1], op.get_params()[2], op.get_params()[3])
            elif gate in GATE_C2:
                assert (len(op.get_params()) == 16)
                ag = svsim.AggregateGate_C2(gate,op.get_params()[0],op.get_params()[1],op.get_params()[2],op.get_params()[3],
                                                 op.get_params()[4],op.get_params()[5],op.get_params()[6],op.get_params()[7],
                                                 op.get_params()[8],op.get_params()[9],op.get_params()[10],op.get_params()[11],
                                                 op.get_params()[12],op.get_params()[13],op.get_params()[14],op.get_params()[15])
            elif gate in GATE_FRAC:
                assert(len(op.get_params()) == 2)
                ag = svsim.AggregateGate_Frac(gate,op.get_params()[0],op.get_params()[1])

            else:
                # this means the gate is probabaly not a basic gate. We need to handle them later
                ag = svsim.AggregateGate_0("I")
            ag_factory = svsim.AggregateGateFactory(ag)
            op_qubits = len(op.get_qubits())
            control_bits = []
            target_bits = []
            if op_qubits == 1:
                # this means the control qubit should be null
                target_bits.append(uqubits[op.get_qubits()[0]])
            if op_qubits == 2:
                if gate == 'CX':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'X'
                    ag = svsim.AggregateGate_0(gate_alt)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CY':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'Y'
                    ag = svsim.AggregateGate_0(gate_alt)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CZ':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'Z'
                    ag = svsim.AggregateGate_0(gate_alt)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CH':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'H'
                    ag = svsim.AggregateGate_0(gate_alt)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CRX':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'RX'
                    ag = svsim.AggregateGate_1(gate_alt,op.get_params()[0])
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CRY':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'RY'
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CRZ':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'RZ'
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'RXX':
                    t_bit = uqubits[op.get_qubits()[0]]
                    target_bits.append(t_bit)
                    t_bit = uqubits[op.get_qubits()[1]]
                    target_bits.append(t_bit)
                    gate_alt = 'RXX'
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'RYY':
                    t_bit = uqubits[op.get_qubits()[0]]
                    target_bits.append(t_bit)
                    t_bit = uqubits[op.get_qubits()[1]]
                    target_bits.append(t_bit)
                    gate_alt = 'RYY'
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'RZZ':
                    t_bit = uqubits[op.get_qubits()[0]]
                    target_bits.append(t_bit)
                    t_bit = uqubits[op.get_qubits()[1]]
                    target_bits.append(t_bit)
                    gate_alt = 'RZZ'
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CU1':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'U1'
                    print(op.get_params()[0])
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CU3':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'U3'
                    ag = svsim.AggregateGate_3(gate_alt, op.get_params()[0],op.get_params()[1],op.get_params()[2])
                    ag_factory = svsim.AggregateGateFactory(ag)
            if op_qubits == 3:
                if gate == 'CCX':
                    c_bit_1 = uqubits[op.get_qubits()[0]]
                    c_bit_2 = uqubits[op.get_qubits()[1]]
                    t_bit = uqubits[op.get_qubits()[2]]
                    control_bits.append(c_bit_1)
                    control_bits.append(c_bit_2)
                    target_bits.append(t_bit)
                    gate_alt = 'X'
                    ag = svsim.AggregateGate_0(gate_alt)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CSWAP':
                    c_bit_1 = uqubits[op.get_qubits()[0]]
                    t_bit_1 = uqubits[op.get_qubits()[1]]
                    t_bit_2 = uqubits[op.get_qubits()[2]]
                    control_bits.append(c_bit_1)
                    target_bits.append(t_bit_1)
                    target_bits.append(t_bit_2)
                    gate_alt = 'SWAP'
                    ag = svsim.AggregateGate_0(gate_alt)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'RCCX':
                    c_bit_1 = uqubits[op.get_qubits()[0]]
                    c_bit_2 = uqubits[op.get_qubits()[1]]
                    t_bit_1 = uqubits[op.get_qubits()[2]]
                    control_bits.append(c_bit_1)
                    control_bits.append(c_bit_2)
                    target_bits.append(t_bit_1)
                    gate_alt = 'RX'
                    ag = svsim.AggregateGate_1(gate_alt,op.get_params()[0])
                    ag_factory = svsim.AggregateGateFactory(ag)
            if op_qubits == 4:
                if gate == 'RC3X':
                    c_bit_1 = uqubits[op.get_qubits()[0]]
                    c_bit_2 = uqubits[op.get_qubits()[1]]
                    c_bit_3 = uqubits[op.get_qubits()[2]]
                    t_bit_1 = uqubits[op.get_qubits()[3]]
                    control_bits.append(c_bit_1)
                    control_bits.append(c_bit_2)
                    control_bits.append(c_bit_3)
                    target_bits.append(t_bit_1)
                    gate_alt = 'RX'
                    ag = svsim.AggregateGate_1(gate_alt,op.get_params()[0])
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'C3X':
                    c_bit_1 = uqubits[op.get_qubits()[0]]
                    c_bit_2 = uqubits[op.get_qubits()[1]]
                    c_bit_3 = uqubits[op.get_qubits()[2]]
                    t_bit_1 = uqubits[op.get_qubits()[3]]
                    control_bits.append(c_bit_1)
                    control_bits.append(c_bit_2)
                    control_bits.append(c_bit_3)
                    target_bits.append(t_bit_1)
                    gate_alt = 'X'
                    ag = svsim.AggregateGate_0(gate_alt)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'C3XSQRTX':
                    pass
            if op_qubits == 5:
                if gate == 'C4X':
                    c_bit_1 = uqubits[op.get_qubits()[0]]
                    c_bit_2 = uqubits[op.get_qubits()[1]]
                    c_bit_3 = uqubits[op.get_qubits()[2]]
                    c_bit_4 = uqubits[op.get_qubits()[3]]
                    t_bit_1 = uqubits[op.get_qubits()[4]]
                    control_bits.append(c_bit_1)
                    control_bits.append(c_bit_2)
                    control_bits.append(c_bit_3)
                    control_bits.append(c_bit_4)
                    target_bits.append(t_bit_1)
                    gate_alt = 'X'
                    ag = svsim.AggregateGate_0(gate_alt)
                    ag_factory = svsim.AggregateGateFactory(ag)
            #print(control_bits)
            #print(target_bits)
            boundgate = svsim.BoundGate(ag_factory,control_bits,target_bits)
            now = datetime.now()
            boundgate.apply(sim)
            later = datetime.now()
            diff = later-now
            acc_timer += diff.seconds*1000000+diff.microseconds
            print(gate+" has been applied.")
            print("time elapsed "+str(acc_timer))

    def collect_parts_qubits(self,parts_file):
        global measure_op
        global acc_timer
        global allocation_scheme
        parts = self.parse_parts_file(parts_file)
        uqubits = self.create_universal_qubit()
        parts_qubit_collection = []
        for part in parts:
            qubit_in_part = []
            for order_id in parts[part]:
                if order_id == 0:
                    continue
                elif (order_id - 1024) > len(self.ordered_gate_list):
                    continue
                else:
                    order_id = order_id - 1
                op = self.ordered_gate_list[order_id]
                gate = op.get_gate().upper()
                op_qubits = len(op.get_qubits())
                control_bits = []
                target_bits = []
                if op_qubits == 1:
                    # this means the control qubit should be null
                    target_bits.append(uqubits[op.get_qubits()[0]])
                if op_qubits == 2:
                    if gate == 'CX':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                    if gate == 'CY':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                    if gate == 'CZ':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                    if gate == 'CH':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                    if gate == 'CRX':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                    if gate == 'CRY':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                    if gate == 'CRZ':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                    if gate == 'RXX':
                        t_bit = uqubits[op.get_qubits()[0]]
                        target_bits.append(t_bit)
                        t_bit = uqubits[op.get_qubits()[1]]
                        target_bits.append(t_bit)
                    if gate == 'RYY':
                        t_bit = uqubits[op.get_qubits()[0]]
                        target_bits.append(t_bit)
                        t_bit = uqubits[op.get_qubits()[1]]
                        target_bits.append(t_bit)
                    if gate == 'RZZ':
                        t_bit = uqubits[op.get_qubits()[0]]
                        target_bits.append(t_bit)
                        t_bit = uqubits[op.get_qubits()[1]]
                        target_bits.append(t_bit)
                    if gate == 'CU1':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                    if gate == 'CU3':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                if op_qubits == 3:
                    if gate == 'CCX':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        c_bit_2 = uqubits[op.get_qubits()[1]]
                        t_bit = uqubits[op.get_qubits()[2]]
                        control_bits.append(c_bit_1)
                        control_bits.append(c_bit_2)
                        target_bits.append(t_bit)
                    if gate == 'CSWAP':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        t_bit_1 = uqubits[op.get_qubits()[1]]
                        t_bit_2 = uqubits[op.get_qubits()[2]]
                        control_bits.append(c_bit_1)
                        target_bits.append(t_bit_1)
                        target_bits.append(t_bit_2)
                    if gate == 'RCCX':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        c_bit_2 = uqubits[op.get_qubits()[1]]
                        t_bit_1 = uqubits[op.get_qubits()[2]]
                        control_bits.append(c_bit_1)
                        control_bits.append(c_bit_2)
                        target_bits.append(t_bit_1)
                if op_qubits == 4:
                    if gate == 'RC3X':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        c_bit_2 = uqubits[op.get_qubits()[1]]
                        c_bit_3 = uqubits[op.get_qubits()[2]]
                        t_bit_1 = uqubits[op.get_qubits()[3]]
                        control_bits.append(c_bit_1)
                        control_bits.append(c_bit_2)
                        control_bits.append(c_bit_3)
                        target_bits.append(t_bit_1)
                    if gate == 'C3X':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        c_bit_2 = uqubits[op.get_qubits()[1]]
                        c_bit_3 = uqubits[op.get_qubits()[2]]
                        t_bit_1 = uqubits[op.get_qubits()[3]]
                        control_bits.append(c_bit_1)
                        control_bits.append(c_bit_2)
                        control_bits.append(c_bit_3)
                        target_bits.append(t_bit_1)
                    if gate == 'C3XSQRTX':
                        pass
                if op_qubits == 5:
                    if gate == 'C4X':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        c_bit_2 = uqubits[op.get_qubits()[1]]
                        c_bit_3 = uqubits[op.get_qubits()[2]]
                        c_bit_4 = uqubits[op.get_qubits()[3]]
                        t_bit_1 = uqubits[op.get_qubits()[4]]
                        control_bits.append(c_bit_1)
                        control_bits.append(c_bit_2)
                        control_bits.append(c_bit_3)
                        control_bits.append(c_bit_4)
                        target_bits.append(t_bit_1)
                for cb in control_bits:
                    if cb not in qubit_in_part:
                        qubit_in_part.append(cb)
                for tb in target_bits:
                    if tb not in qubit_in_part:
                        qubit_in_part.append(tb)
            parts_qubit_collection.append(qubit_in_part)
        return parts_qubit_collection

    def execute_qasm_parts(self,parts_file):
        global measure_op
        global acc_timer
        global allocation_scheme
        parts = self.parse_parts_file(parts_file)
        uqubits = self.create_universal_qubit()
        inv_uqubits = self.reverse_qubit(uqubits)
        total_qubits = len(uqubits)
        s_before = datetime.now()
        print(allocation_scheme)
        sim = svsim.StateVector(total_qubits,allocation_scheme)
        s_after = datetime.now()
        s_diff = s_after-s_before
        start_timer = s_diff.seconds*1000000+s_diff.microseconds
        print("SVSIM start takes "+str(start_timer))
        for part in parts:
            boundgate_list = []
            for order_id in parts[part]:
                if order_id == 0:
                    continue
                elif (order_id - 1024) > len(self.ordered_gate_list):
                    continue
                else:
                    order_id = order_id - 1
                op = self.ordered_gate_list[order_id]
                gate = op.get_gate().upper()
                # if it is measure, handle it first
                if gate == "MEASURE":
                    measure_qubit = []
                    assert(len(op.get_qubits()) == 2)
                    q_m = uqubits[op.get_qubits()[0]]
                    measure_qubit.append(q_m)
                    #results = sim.measure(measure_qubit,NUM_MEASURE)
                    #print(results)
                    continue
                if gate in GATE_NOPARAM:
                    ag = svsim.AggregateGate_0(gate)
                elif gate in GATE_ONEPARAM:
                    assert(len(op.get_params()) == 1)
                    ag = svsim.AggregateGate_1(gate,op.get_params()[0])
                elif gate in GATE_TWOPARAM:
                    assert(len(op.get_params()) == 2)
                    ag = svsim.AggregateGate_2(gate,op.get_params()[0],op.get_params()[1])
                elif gate in GATE_THREEPARAM:
                    assert(len(op.get_params()) == 3)
                    ag = svsim.AggregateGate_3(gate,op.get_params()[0],op.get_params()[1],op.get_params()[2])
                elif gate in GATE_C1:
                    assert (len(op.get_params()) == 4)
                    ag = svsim.AggregateGate_C1(gate, op.get_params()[0], op.get_params()[1], op.get_params()[2], op.get_params()[3])
                elif gate in GATE_C2:
                    assert (len(op.get_params()) == 16)
                    ag = svsim.AggregateGate_C2(gate,op.get_params()[0],op.get_params()[1],op.get_params()[2],op.get_params()[3],
                                                     op.get_params()[4],op.get_params()[5],op.get_params()[6],op.get_params()[7],
                                                     op.get_params()[8],op.get_params()[9],op.get_params()[10],op.get_params()[11],
                                                     op.get_params()[12],op.get_params()[13],op.get_params()[14],op.get_params()[15])
                elif gate in GATE_FRAC:
                    assert(len(op.get_params()) == 2)
                    ag = svsim.AggregateGate_Frac(gate,op.get_params()[0],op.get_params()[1])
                else:
                    # this means the gate is probabaly not a basic gate. We need to handle them later
                    ag = svsim.AggregateGate_0("I")

                ag_factory = svsim.AggregateGateFactory(ag)
                op_qubits = len(op.get_qubits())
                control_bits = []
                target_bits = []
                if op_qubits == 1:
                    # this means the control qubit should be null
                    target_bits.append(uqubits[op.get_qubits()[0]])
                if op_qubits == 2:
                    if gate == 'CX':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'X'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CY':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'Y'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CZ':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'Z'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CH':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'H'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CRX':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'RX'
                        ag = svsim.AggregateGate_1(gate_alt,op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CRY':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'RY'
                        ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CRZ':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'RZ'
                        ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'RXX':
                        t_bit = uqubits[op.get_qubits()[0]]
                        target_bits.append(t_bit)
                        t_bit = uqubits[op.get_qubits()[1]]
                        target_bits.append(t_bit)
                        gate_alt = 'RXX'
                        ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'RYY':
                        t_bit = uqubits[op.get_qubits()[0]]
                        target_bits.append(t_bit)
                        t_bit = uqubits[op.get_qubits()[1]]
                        target_bits.append(t_bit)
                        gate_alt = 'RYY'
                        ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'RZZ':
                        t_bit = uqubits[op.get_qubits()[0]]
                        target_bits.append(t_bit)
                        t_bit = uqubits[op.get_qubits()[1]]
                        target_bits.append(t_bit)
                        gate_alt = 'RZZ'
                        ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CU1':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'U1'
                        ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CU3':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'U3'
                        ag = svsim.AggregateGate_3(gate_alt, op.get_params()[0],op.get_params()[1],op.get_params()[2])
                        ag_factory = svsim.AggregateGateFactory(ag)
                if op_qubits == 3:
                    if gate == 'CCX':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        c_bit_2 = uqubits[op.get_qubits()[1]]
                        t_bit = uqubits[op.get_qubits()[2]]
                        control_bits.append(c_bit_1)
                        control_bits.append(c_bit_2)
                        target_bits.append(t_bit)
                        gate_alt = 'X'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CSWAP':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        t_bit_1 = uqubits[op.get_qubits()[1]]
                        t_bit_2 = uqubits[op.get_qubits()[2]]
                        control_bits.append(c_bit_1)
                        target_bits.append(t_bit_1)
                        target_bits.append(t_bit_2)
                        gate_alt = 'SWAP'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'RCCX':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        c_bit_2 = uqubits[op.get_qubits()[1]]
                        t_bit_1 = uqubits[op.get_qubits()[2]]
                        control_bits.append(c_bit_1)
                        control_bits.append(c_bit_2)
                        target_bits.append(t_bit_1)
                        gate_alt = 'RX'
                        ag = svsim.AggregateGate_1(gate_alt,op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                if op_qubits == 4:
                    if gate == 'RC3X':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        c_bit_2 = uqubits[op.get_qubits()[1]]
                        c_bit_3 = uqubits[op.get_qubits()[2]]
                        t_bit_1 = uqubits[op.get_qubits()[3]]
                        control_bits.append(c_bit_1)
                        control_bits.append(c_bit_2)
                        control_bits.append(c_bit_3)
                        target_bits.append(t_bit_1)
                        gate_alt = 'RX'
                        ag = svsim.AggregateGate_1(gate_alt,op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'C3X':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        c_bit_2 = uqubits[op.get_qubits()[1]]
                        c_bit_3 = uqubits[op.get_qubits()[2]]
                        t_bit_1 = uqubits[op.get_qubits()[3]]
                        control_bits.append(c_bit_1)
                        control_bits.append(c_bit_2)
                        control_bits.append(c_bit_3)
                        target_bits.append(t_bit_1)
                        gate_alt = 'X'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'C3XSQRTX':
                        pass
                if op_qubits == 5:
                    if gate == 'C4X':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        c_bit_2 = uqubits[op.get_qubits()[1]]
                        c_bit_3 = uqubits[op.get_qubits()[2]]
                        c_bit_4 = uqubits[op.get_qubits()[3]]
                        t_bit_1 = uqubits[op.get_qubits()[4]]
                        control_bits.append(c_bit_1)
                        control_bits.append(c_bit_2)
                        control_bits.append(c_bit_3)
                        control_bits.append(c_bit_4)
                        target_bits.append(t_bit_1)
                        gate_alt = 'X'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                #print(control_bits)
                #print(target_bits)
                boundgate = svsim.BoundGate(ag_factory,control_bits,target_bits)
                boundgate_list.append(boundgate)
                #print(gate+" has been added to the gather list.")
            now = datetime.now()
            print("Total length of the boundgate list is "+str(len(boundgate_list)))
            svsim.gather_and_execute_on(sim,boundgate_list)
            print("Part "+str(part)+" has been executed")
            later = datetime.now()
            diff = later - now
            acc_timer += diff.seconds*1000000+diff.microseconds
        for part in measure_op:
            if part in parts_file:
                for op in measure_op[part]:
                    gate = op.get_gate().upper()
                    # if it is measure, handle it first
                    if gate == "MEASURE":
                        measure_qubit = []
                        assert (len(op.get_qubits()) == 2)
                        q_m = uqubits[op.get_qubits()[0]]
                        measure_qubit.append(q_m)
                        results = sim.measure(measure_qubit, NUM_MEASURE)
                        print(results)

    def execute_qasm_parts_mpi(self,parts_file,num_local_qubits,comm):
        global measure_op
        global acc_timer
        global allocation_scheme
        parts = self.parse_parts_file(parts_file)
        uqubits = self.create_universal_qubit()
        inv_uqubits = self.reverse_qubit(uqubits)
        total_qubits = len(uqubits)
        comm_size = comm.Get_size()
        num_remote_qubits = total_qubits - num_local_qubits
        if num_remote_qubits != math.log2(comm_size):
            print("remote qubits need to match log of number of processes")
            exit()
        parts_boundgate_list = []
        for part in parts:
            boundgate_list = []
            for order_id in parts[part]:
                if order_id == 0:
                    continue
                elif (order_id - 1024) > len(self.ordered_gate_list):
                    continue
                else:
                    order_id = order_id - 1
                op = self.ordered_gate_list[order_id]
                gate = op.get_gate().upper()
                # if it is measure, handle it first
                if gate == "MEASURE":
                    measure_qubit = []
                    assert(len(op.get_qubits()) == 2)
                    q_m = uqubits[op.get_qubits()[0]]
                    measure_qubit.append(q_m)
                    #results = sim.measure(measure_qubit,NUM_MEASURE)
                    #print(results)
                    continue
                if gate in GATE_NOPARAM:
                    ag = svsim.AggregateGate_0(gate)
                elif gate in GATE_ONEPARAM:
                    assert(len(op.get_params()) == 1)
                    ag = svsim.AggregateGate_1(gate,op.get_params()[0])
                elif gate in GATE_TWOPARAM:
                    assert(len(op.get_params()) == 2)
                    ag = svsim.AggregateGate_2(gate,op.get_params()[0],op.get_params()[1])
                elif gate in GATE_THREEPARAM:
                    assert(len(op.get_params()) == 3)
                    ag = svsim.AggregateGate_3(gate,op.get_params()[0],op.get_params()[1],op.get_params()[2])
                elif gate in GATE_C1:
                    assert (len(op.get_params()) == 4)
                    ag = svsim.AggregateGate_C1(gate, op.get_params()[0], op.get_params()[1], op.get_params()[2], op.get_params()[3])
                elif gate in GATE_C2:
                    assert (len(op.get_params()) == 16)
                    ag = svsim.AggregateGate_C2(gate,op.get_params()[0],op.get_params()[1],op.get_params()[2],op.get_params()[3],
                                                     op.get_params()[4],op.get_params()[5],op.get_params()[6],op.get_params()[7],
                                                     op.get_params()[8],op.get_params()[9],op.get_params()[10],op.get_params()[11],
                                                     op.get_params()[12],op.get_params()[13],op.get_params()[14],op.get_params()[15])
                elif gate in GATE_FRAC:
                    assert(len(op.get_params()) == 2)
                    ag = svsim.AggregateGate_Frac(gate,op.get_params()[0],op.get_params()[1])
                else:
                    # this means the gate is probabaly not a basic gate. We need to handle them later
                    ag = svsim.AggregateGate_0("I")

                ag_factory = svsim.AggregateGateFactory(ag)
                op_qubits = len(op.get_qubits())
                control_bits = []
                target_bits = []
                if op_qubits == 1:
                    # this means the control qubit should be null
                    target_bits.append(uqubits[op.get_qubits()[0]])
                if op_qubits == 2:
                    if gate == 'CX':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'X'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CY':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'Y'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CZ':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'Z'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CH':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'H'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CRX':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'RX'
                        ag = svsim.AggregateGate_1(gate_alt,op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CRY':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'RY'
                        ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CRZ':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'RZ'
                        ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'RXX':
                        t_bit = uqubits[op.get_qubits()[0]]
                        target_bits.append(t_bit)
                        t_bit = uqubits[op.get_qubits()[1]]
                        target_bits.append(t_bit)
                        gate_alt = 'RXX'
                        ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'RYY':
                        t_bit = uqubits[op.get_qubits()[0]]
                        target_bits.append(t_bit)
                        t_bit = uqubits[op.get_qubits()[1]]
                        target_bits.append(t_bit)
                        gate_alt = 'RYY'
                        ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'RZZ':
                        t_bit = uqubits[op.get_qubits()[0]]
                        target_bits.append(t_bit)
                        t_bit = uqubits[op.get_qubits()[1]]
                        target_bits.append(t_bit)
                        gate_alt = 'RZZ'
                        ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CU1':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'U1'
                        ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CU3':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'U3'
                        ag = svsim.AggregateGate_3(gate_alt, op.get_params()[0],op.get_params()[1],op.get_params()[2])
                        ag_factory = svsim.AggregateGateFactory(ag)
                if op_qubits == 3:
                    if gate == 'CCX':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        c_bit_2 = uqubits[op.get_qubits()[1]]
                        t_bit = uqubits[op.get_qubits()[2]]
                        control_bits.append(c_bit_1)
                        control_bits.append(c_bit_2)
                        target_bits.append(t_bit)
                        gate_alt = 'X'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CSWAP':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        t_bit_1 = uqubits[op.get_qubits()[1]]
                        t_bit_2 = uqubits[op.get_qubits()[2]]
                        control_bits.append(c_bit_1)
                        target_bits.append(t_bit_1)
                        target_bits.append(t_bit_2)
                        gate_alt = 'SWAP'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'RCCX':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        c_bit_2 = uqubits[op.get_qubits()[1]]
                        t_bit_1 = uqubits[op.get_qubits()[2]]
                        control_bits.append(c_bit_1)
                        control_bits.append(c_bit_2)
                        target_bits.append(t_bit_1)
                        gate_alt = 'RX'
                        ag = svsim.AggregateGate_1(gate_alt,op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                if op_qubits == 4:
                    if gate == 'RC3X':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        c_bit_2 = uqubits[op.get_qubits()[1]]
                        c_bit_3 = uqubits[op.get_qubits()[2]]
                        t_bit_1 = uqubits[op.get_qubits()[3]]
                        control_bits.append(c_bit_1)
                        control_bits.append(c_bit_2)
                        control_bits.append(c_bit_3)
                        target_bits.append(t_bit_1)
                        gate_alt = 'RX'
                        ag = svsim.AggregateGate_1(gate_alt,op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'C3X':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        c_bit_2 = uqubits[op.get_qubits()[1]]
                        c_bit_3 = uqubits[op.get_qubits()[2]]
                        t_bit_1 = uqubits[op.get_qubits()[3]]
                        control_bits.append(c_bit_1)
                        control_bits.append(c_bit_2)
                        control_bits.append(c_bit_3)
                        target_bits.append(t_bit_1)
                        gate_alt = 'X'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'C3XSQRTX':
                        pass
                if op_qubits == 5:
                    if gate == 'C4X':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        c_bit_2 = uqubits[op.get_qubits()[1]]
                        c_bit_3 = uqubits[op.get_qubits()[2]]
                        c_bit_4 = uqubits[op.get_qubits()[3]]
                        t_bit_1 = uqubits[op.get_qubits()[4]]
                        control_bits.append(c_bit_1)
                        control_bits.append(c_bit_2)
                        control_bits.append(c_bit_3)
                        control_bits.append(c_bit_4)
                        target_bits.append(t_bit_1)
                        gate_alt = 'X'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                #print(control_bits)
                #print(target_bits)
                boundgate = svsim.BoundGate(ag_factory,control_bits,target_bits)
                boundgate_list.append(boundgate)
                #print(gate+" has been added to the gather list.")
            parts_boundgate_list.append(boundgate_list)
            #print("Total length of the boundgate list is "+str(len(boundgate_list)))
        #now = datetime.now()
        MPI.COMM_WORLD.Barrier()    
        now = MPI.Wtime()
        svsim.gather_and_execute_on_mpi(total_qubits,num_local_qubits,parts_boundgate_list)
        #later = datetime.now()
        later = MPI.Wtime()
        diff = later - now
        rank = comm.Get_rank()
        if rank == 0:
           print("execution time measured from python "+str(diff))
        '''
        for part in measure_op:
            if part in parts_file:
                for op in measure_op[part]:
                    gate = op.get_gate().upper()
                    # if it is measure, handle it first
                    if gate == "MEASURE":
                        measure_qubit = []
                        assert (len(op.get_qubits()) == 2)
                        q_m = uqubits[op.get_qubits()[0]]
                        measure_qubit.append(q_m)
                        #results = sim.measure(measure_qubit, NUM_MEASURE)
                        #print(results)
        '''

    def execute_qasm_multiparts_mpi(self,parts_files:list,num_local_qubits,comm):
        global measure_op
        global acc_timer
        global allocation_scheme
        assert(len(parts_files) != 0)
        # verification
        # process the part files
        total_top_parts = 0
        total_second_parts = {}
        whole_boundgate_lists = []
        top_bg_lists = []
        second_bg_lists = []
        for idx, partfile in enumerate(parts_files):
          parts = self.parse_parts_file(partfile)
          uqubits = self.create_universal_qubit()
          inv_uqubits = self.reverse_qubit(uqubits)
          total_qubits = len(uqubits)
          comm_size = comm.Get_size()
          num_remote_qubits = total_qubits - num_local_qubits
          if num_remote_qubits != math.log2(comm_size):
              print("remote qubits need to match log of number of processes")
              exit()
          if idx == 0:
            total_top_parts = len(parts)
          else:
            part_id = idx-1
            total_second_parts[part_id] = len(parts)
          for idx_part, part in enumerate(parts):
              boundgate_list = []
              for order_id in parts[part]:
                  if order_id == 0:
                      continue
                  elif (order_id - 1024) > len(self.ordered_gate_list):
                      continue
                  else:
                      order_id = order_id - 1
                  op = self.ordered_gate_list[order_id]
                  gate = op.get_gate().upper()
                  # if it is measure, handle it first
                  if gate == "MEASURE":
                      measure_qubit = []
                      assert(len(op.get_qubits()) == 2)
                      q_m = uqubits[op.get_qubits()[0]]
                      measure_qubit.append(q_m)
                      #results = sim.measure(measure_qubit,NUM_MEASURE)
                      #print(results)
                      continue
                  if gate in GATE_NOPARAM:
                      ag = svsim.AggregateGate_0(gate)
                  elif gate in GATE_ONEPARAM:
                      assert(len(op.get_params()) == 1)
                      ag = svsim.AggregateGate_1(gate,op.get_params()[0])
                  elif gate in GATE_TWOPARAM:
                      assert(len(op.get_params()) == 2)
                      ag = svsim.AggregateGate_2(gate,op.get_params()[0],op.get_params()[1])
                  elif gate in GATE_THREEPARAM:
                      assert(len(op.get_params()) == 3)
                      ag = svsim.AggregateGate_3(gate,op.get_params()[0],op.get_params()[1],op.get_params()[2])
                  elif gate in GATE_C1:
                      assert (len(op.get_params()) == 4)
                      ag = svsim.AggregateGate_C1(gate, op.get_params()[0], op.get_params()[1], op.get_params()[2], op.get_params()[3])
                  elif gate in GATE_C2:
                      assert (len(op.get_params()) == 16)
                      ag = svsim.AggregateGate_C2(gate,op.get_params()[0],op.get_params()[1],op.get_params()[2],op.get_params()[3],
                                                       op.get_params()[4],op.get_params()[5],op.get_params()[6],op.get_params()[7],
                                                       op.get_params()[8],op.get_params()[9],op.get_params()[10],op.get_params()[11],
                                                       op.get_params()[12],op.get_params()[13],op.get_params()[14],op.get_params()[15])
                  elif gate in GATE_FRAC:
                      assert(len(op.get_params()) == 2)
                      ag = svsim.AggregateGate_Frac(gate,op.get_params()[0],op.get_params()[1])
                  else:
                      # this means the gate is probabaly not a basic gate. We need to handle them later
                      ag = svsim.AggregateGate_0("I")
  
                  ag_factory = svsim.AggregateGateFactory(ag)
                  op_qubits = len(op.get_qubits())
                  control_bits = []
                  target_bits = []
                  if op_qubits == 1:
                      # this means the control qubit should be null
                      target_bits.append(uqubits[op.get_qubits()[0]])
                  if op_qubits == 2:
                      if gate == 'CX':
                          c_bit = uqubits[op.get_qubits()[0]]
                          t_bit = uqubits[op.get_qubits()[1]]
                          control_bits.append(c_bit)
                          target_bits.append(t_bit)
                          gate_alt = 'X'
                          ag = svsim.AggregateGate_0(gate_alt)
                          ag_factory = svsim.AggregateGateFactory(ag)
                      if gate == 'CY':
                          c_bit = uqubits[op.get_qubits()[0]]
                          t_bit = uqubits[op.get_qubits()[1]]
                          control_bits.append(c_bit)
                          target_bits.append(t_bit)
                          gate_alt = 'Y'
                          ag = svsim.AggregateGate_0(gate_alt)
                          ag_factory = svsim.AggregateGateFactory(ag)
                      if gate == 'CZ':
                          c_bit = uqubits[op.get_qubits()[0]]
                          t_bit = uqubits[op.get_qubits()[1]]
                          control_bits.append(c_bit)
                          target_bits.append(t_bit)
                          gate_alt = 'Z'
                          ag = svsim.AggregateGate_0(gate_alt)
                          ag_factory = svsim.AggregateGateFactory(ag)
                      if gate == 'CH':
                          c_bit = uqubits[op.get_qubits()[0]]
                          t_bit = uqubits[op.get_qubits()[1]]
                          control_bits.append(c_bit)
                          target_bits.append(t_bit)
                          gate_alt = 'H'
                          ag = svsim.AggregateGate_0(gate_alt)
                          ag_factory = svsim.AggregateGateFactory(ag)
                      if gate == 'CRX':
                          c_bit = uqubits[op.get_qubits()[0]]
                          t_bit = uqubits[op.get_qubits()[1]]
                          control_bits.append(c_bit)
                          target_bits.append(t_bit)
                          gate_alt = 'RX'
                          ag = svsim.AggregateGate_1(gate_alt,op.get_params()[0])
                          ag_factory = svsim.AggregateGateFactory(ag)
                      if gate == 'CRY':
                          c_bit = uqubits[op.get_qubits()[0]]
                          t_bit = uqubits[op.get_qubits()[1]]
                          control_bits.append(c_bit)
                          target_bits.append(t_bit)
                          gate_alt = 'RY'
                          ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                          ag_factory = svsim.AggregateGateFactory(ag)
                      if gate == 'CRZ':
                          c_bit = uqubits[op.get_qubits()[0]]
                          t_bit = uqubits[op.get_qubits()[1]]
                          control_bits.append(c_bit)
                          target_bits.append(t_bit)
                          gate_alt = 'RZ'
                          ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                          ag_factory = svsim.AggregateGateFactory(ag)
                      if gate == 'RXX':
                          t_bit = uqubits[op.get_qubits()[0]]
                          target_bits.append(t_bit)
                          t_bit = uqubits[op.get_qubits()[1]]
                          target_bits.append(t_bit)
                          gate_alt = 'RXX'
                          ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                          ag_factory = svsim.AggregateGateFactory(ag)
                      if gate == 'RYY':
                          t_bit = uqubits[op.get_qubits()[0]]
                          target_bits.append(t_bit)
                          t_bit = uqubits[op.get_qubits()[1]]
                          target_bits.append(t_bit)
                          gate_alt = 'RYY'
                          ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                          ag_factory = svsim.AggregateGateFactory(ag)
                      if gate == 'RZZ':
                          t_bit = uqubits[op.get_qubits()[0]]
                          target_bits.append(t_bit)
                          t_bit = uqubits[op.get_qubits()[1]]
                          target_bits.append(t_bit)
                          gate_alt = 'RZZ'
                          ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                          ag_factory = svsim.AggregateGateFactory(ag)
                      if gate == 'CU1':
                          c_bit = uqubits[op.get_qubits()[0]]
                          t_bit = uqubits[op.get_qubits()[1]]
                          control_bits.append(c_bit)
                          target_bits.append(t_bit)
                          gate_alt = 'U1'
                          ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                          ag_factory = svsim.AggregateGateFactory(ag)
                      if gate == 'CU3':
                          c_bit = uqubits[op.get_qubits()[0]]
                          t_bit = uqubits[op.get_qubits()[1]]
                          control_bits.append(c_bit)
                          target_bits.append(t_bit)
                          gate_alt = 'U3'
                          ag = svsim.AggregateGate_3(gate_alt, op.get_params()[0],op.get_params()[1],op.get_params()[2])
                          ag_factory = svsim.AggregateGateFactory(ag)
                  if op_qubits == 3:
                      if gate == 'CCX':
                          c_bit_1 = uqubits[op.get_qubits()[0]]
                          c_bit_2 = uqubits[op.get_qubits()[1]]
                          t_bit = uqubits[op.get_qubits()[2]]
                          control_bits.append(c_bit_1)
                          control_bits.append(c_bit_2)
                          target_bits.append(t_bit)
                          gate_alt = 'X'
                          ag = svsim.AggregateGate_0(gate_alt)
                          ag_factory = svsim.AggregateGateFactory(ag)
                      if gate == 'CSWAP':
                          c_bit_1 = uqubits[op.get_qubits()[0]]
                          t_bit_1 = uqubits[op.get_qubits()[1]]
                          t_bit_2 = uqubits[op.get_qubits()[2]]
                          control_bits.append(c_bit_1)
                          target_bits.append(t_bit_1)
                          target_bits.append(t_bit_2)
                          gate_alt = 'SWAP'
                          ag = svsim.AggregateGate_0(gate_alt)
                          ag_factory = svsim.AggregateGateFactory(ag)
                      if gate == 'RCCX':
                          c_bit_1 = uqubits[op.get_qubits()[0]]
                          c_bit_2 = uqubits[op.get_qubits()[1]]
                          t_bit_1 = uqubits[op.get_qubits()[2]]
                          control_bits.append(c_bit_1)
                          control_bits.append(c_bit_2)
                          target_bits.append(t_bit_1)
                          gate_alt = 'RX'
                          ag = svsim.AggregateGate_1(gate_alt,op.get_params()[0])
                          ag_factory = svsim.AggregateGateFactory(ag)
                  if op_qubits == 4:
                      if gate == 'RC3X':
                          c_bit_1 = uqubits[op.get_qubits()[0]]
                          c_bit_2 = uqubits[op.get_qubits()[1]]
                          c_bit_3 = uqubits[op.get_qubits()[2]]
                          t_bit_1 = uqubits[op.get_qubits()[3]]
                          control_bits.append(c_bit_1)
                          control_bits.append(c_bit_2)
                          control_bits.append(c_bit_3)
                          target_bits.append(t_bit_1)
                          gate_alt = 'RX'
                          ag = svsim.AggregateGate_1(gate_alt,op.get_params()[0])
                          ag_factory = svsim.AggregateGateFactory(ag)
                      if gate == 'C3X':
                          c_bit_1 = uqubits[op.get_qubits()[0]]
                          c_bit_2 = uqubits[op.get_qubits()[1]]
                          c_bit_3 = uqubits[op.get_qubits()[2]]
                          t_bit_1 = uqubits[op.get_qubits()[3]]
                          control_bits.append(c_bit_1)
                          control_bits.append(c_bit_2)
                          control_bits.append(c_bit_3)
                          target_bits.append(t_bit_1)
                          gate_alt = 'X'
                          ag = svsim.AggregateGate_0(gate_alt)
                          ag_factory = svsim.AggregateGateFactory(ag)
                      if gate == 'C3XSQRTX':
                          pass
                  if op_qubits == 5:
                      if gate == 'C4X':
                          c_bit_1 = uqubits[op.get_qubits()[0]]
                          c_bit_2 = uqubits[op.get_qubits()[1]]
                          c_bit_3 = uqubits[op.get_qubits()[2]]
                          c_bit_4 = uqubits[op.get_qubits()[3]]
                          t_bit_1 = uqubits[op.get_qubits()[4]]
                          control_bits.append(c_bit_1)
                          control_bits.append(c_bit_2)
                          control_bits.append(c_bit_3)
                          control_bits.append(c_bit_4)
                          target_bits.append(t_bit_1)
                          gate_alt = 'X'
                          ag = svsim.AggregateGate_0(gate_alt)
                          ag_factory = svsim.AggregateGateFactory(ag)
                  #print(control_bits)
                  #print(target_bits)
                  boundgate = svsim.BoundGate(ag_factory,control_bits,target_bits)
                  boundgate_list.append(boundgate)
                  #print(gate+" has been added to the gather list.")
              if idx == 0:
                top_bg_lists.append(boundgate_list)
              else:
                second_bg_lists.append(boundgate_list)
              #print("Total length of the boundgate list is "+str(len(boundgate_list)))
          #now = datetime.now()
        # map the index to the second level list starting from 0
        start = 0
        for t_part_id,part in enumerate(top_bg_lists):
            part_tmp = []
            part_tmp.append(top_bg_lists[t_part_id])
            count = total_second_parts[t_part_id]+start
            part_tmp.extend(second_bg_lists[start:count])
            start = count
            whole_boundgate_lists.append(part_tmp)    
        MPI.COMM_WORLD.Barrier()    
        now = MPI.Wtime()
        rank = comm.Get_rank()
        #if rank == 0:
            #print(total_qubits)
            #print(num_local_qubits)
            #print(whole_boundgate_lists)
            #print(top_bg_lists)
            #print(second_bg_lists)
        svsim.gather_and_execute_multilevel_on_mpi(total_qubits,num_local_qubits,whole_boundgate_lists)
        #later = datetime.now()
        later = MPI.Wtime()
        diff = later - now
        #if rank == 0:
           #print("execution time measured from python "+str(diff))
        '''
        for part in measure_op:
            if part in parts_file:
                for op in measure_op[part]:
                    gate = op.get_gate().upper()
                    # if it is measure, handle it first
                    if gate == "MEASURE":
                        measure_qubit = []
                        assert (len(op.get_qubits()) == 2)
                        q_m = uqubits[op.get_qubits()[0]]
                        measure_qubit.append(q_m)
                        #results = sim.measure(measure_qubit, NUM_MEASURE)
                        #print(results)
        '''

    def execute_qasm_parts_with_verify(self,parts_file):
        global measure_op
        global acc_timer
        parts = self.parse_parts_file(parts_file)
        uqubits = self.create_universal_qubit()
        inv_uqubits = self.reverse_qubit(uqubits)
        total_qubits = len(uqubits)
        sim = svsim.StateVector(total_qubits,0)
        sim_init = svsim.StateVector(total_qubits,0)
        for part in parts:
            boundgate_list = []
            for order_id in parts[part]:
                if order_id == 0:
                    continue
                elif (order_id - 1024) > len(self.ordered_gate_list):
                    continue
                else:
                    order_id = order_id - 1
                op = self.ordered_gate_list[order_id]
                gate = op.get_gate().upper()
                # if it is measure, handle it first
                if gate == "MEASURE":
                    measure_qubit = []
                    assert(len(op.get_qubits()) == 2)
                    q_m = uqubits[op.get_qubits()[0]]
                    measure_qubit.append(q_m)
                    #results = sim.measure(measure_qubit,NUM_MEASURE)
                    #print(results)
                    continue
                if gate in GATE_NOPARAM:
                    ag = svsim.AggregateGate_0(gate)
                elif gate in GATE_ONEPARAM:
                    assert(len(op.get_params()) == 1)
                    ag = svsim.AggregateGate_1(gate,op.get_params()[0])
                elif gate in GATE_TWOPARAM:
                    assert(len(op.get_params()) == 2)
                    ag = svsim.AggregateGate_2(gate,op.get_params()[0],op.get_params()[1])
                elif gate in GATE_THREEPARAM:
                    assert(len(op.get_params()) == 3)
                    ag = svsim.AggregateGate_3(gate,op.get_params()[0],op.get_params()[1],op.get_params()[2])
                elif gate in GATE_C1:
                    assert (len(op.get_params()) == 4)
                    ag = svsim.AggregateGate_C1(gate, op.get_params()[0], op.get_params()[1], op.get_params()[2], op.get_params()[3])
                elif gate in GATE_C2:
                    assert (len(op.get_params()) == 16)
                    ag = svsim.AggregateGate_C2(gate,op.get_params()[0],op.get_params()[1],op.get_params()[2],op.get_params()[3],
                                                     op.get_params()[4],op.get_params()[5],op.get_params()[6],op.get_params()[7],
                                                     op.get_params()[8],op.get_params()[9],op.get_params()[10],op.get_params()[11],
                                                     op.get_params()[12],op.get_params()[13],op.get_params()[14],op.get_params()[15])
                elif gate in GATE_FRAC:
                    assert(len(op.get_params()) == 2)
                    ag = svsim.AggregateGate_Frac(gate,op.get_params()[0],op.get_params()[1])

                else:
                    # this means the gate is probabaly not a basic gate. We need to handle them later
                    ag = svsim.AggregateGate_0("I")
                ag_factory = svsim.AggregateGateFactory(ag)
                op_qubits = len(op.get_qubits())
                control_bits = []
                target_bits = []
                if op_qubits == 1:
                    # this means the control qubit should be null
                    target_bits.append(uqubits[op.get_qubits()[0]])
                if op_qubits == 2:
                    if gate == 'CX':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'X'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CY':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'Y'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CZ':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'Z'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CH':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'H'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CRX':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'RX'
                        ag = svsim.AggregateGate_1(gate_alt,op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CRY':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'RY'
                        ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CRZ':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'RZ'
                        ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'RXX':
                        t_bit = uqubits[op.get_qubits()[0]]
                        target_bits.append(t_bit)
                        t_bit = uqubits[op.get_qubits()[1]]
                        target_bits.append(t_bit)
                        gate_alt = 'RXX'
                        ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'RYY':
                        t_bit = uqubits[op.get_qubits()[0]]
                        target_bits.append(t_bit)
                        t_bit = uqubits[op.get_qubits()[1]]
                        target_bits.append(t_bit)
                        gate_alt = 'RYY'
                        ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'RZZ':
                        t_bit = uqubits[op.get_qubits()[0]]
                        target_bits.append(t_bit)
                        t_bit = uqubits[op.get_qubits()[1]]
                        target_bits.append(t_bit)
                        gate_alt = 'RZZ'
                        ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CU1':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'U1'
                        ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CU3':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'U3'
                        ag = svsim.AggregateGate_3(gate_alt, op.get_params()[0],op.get_params()[1],op.get_params()[2])
                        ag_factory = svsim.AggregateGateFactory(ag)
                if op_qubits == 3:
                    if gate == 'CCX':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        c_bit_2 = uqubits[op.get_qubits()[1]]
                        t_bit = uqubits[op.get_qubits()[2]]
                        control_bits.append(c_bit_1)
                        control_bits.append(c_bit_2)
                        target_bits.append(t_bit)
                        gate_alt = 'X'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CSWAP':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        t_bit_1 = uqubits[op.get_qubits()[1]]
                        t_bit_2 = uqubits[op.get_qubits()[2]]
                        control_bits.append(c_bit_1)
                        target_bits.append(t_bit_1)
                        target_bits.append(t_bit_2)
                        gate_alt = 'SWAP'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'RCCX':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        c_bit_2 = uqubits[op.get_qubits()[1]]
                        t_bit_1 = uqubits[op.get_qubits()[2]]
                        control_bits.append(c_bit_1)
                        control_bits.append(c_bit_2)
                        target_bits.append(t_bit_1)
                        gate_alt = 'RX'
                        ag = svsim.AggregateGate_1(gate_alt,op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                if op_qubits == 4:
                    if gate == 'RC3X':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        c_bit_2 = uqubits[op.get_qubits()[1]]
                        c_bit_3 = uqubits[op.get_qubits()[2]]
                        t_bit_1 = uqubits[op.get_qubits()[3]]
                        control_bits.append(c_bit_1)
                        control_bits.append(c_bit_2)
                        control_bits.append(c_bit_3)
                        target_bits.append(t_bit_1)
                        gate_alt = 'RX'
                        ag = svsim.AggregateGate_1(gate_alt,op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'C3X':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        c_bit_2 = uqubits[op.get_qubits()[1]]
                        c_bit_3 = uqubits[op.get_qubits()[2]]
                        t_bit_1 = uqubits[op.get_qubits()[3]]
                        control_bits.append(c_bit_1)
                        control_bits.append(c_bit_2)
                        control_bits.append(c_bit_3)
                        target_bits.append(t_bit_1)
                        gate_alt = 'X'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'C3XSQRTX':
                        pass
                if op_qubits == 5:
                    if gate == 'C4X':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        c_bit_2 = uqubits[op.get_qubits()[1]]
                        c_bit_3 = uqubits[op.get_qubits()[2]]
                        c_bit_4 = uqubits[op.get_qubits()[3]]
                        t_bit_1 = uqubits[op.get_qubits()[4]]
                        control_bits.append(c_bit_1)
                        control_bits.append(c_bit_2)
                        control_bits.append(c_bit_3)
                        control_bits.append(c_bit_4)
                        target_bits.append(t_bit_1)
                        gate_alt = 'X'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                #print(control_bits)
                #print(target_bits)
                boundgate = svsim.BoundGate(ag_factory,control_bits,target_bits)
                boundgate_list.append(boundgate)
                #print(gate+" has been added to the gather list.")
            now = datetime.now()
            print("Total length of the boundgate list is "+str(len(boundgate_list)))
            svsim.gather_and_execute_on(sim,boundgate_list)
            print("Part "+str(part)+" has been executed")
            later = datetime.now()
            diff = later - now
            acc_timer += diff.seconds*1000000+diff.microseconds
        for part in measure_op:
            if part in parts_file:
                for op in measure_op[part]:
                    gate = op.get_gate().upper()
                    # if it is measure, handle it first
                    if gate == "MEASURE":
                        measure_qubit = []
                        assert (len(op.get_qubits()) == 2)
                        q_m = uqubits[op.get_qubits()[0]]
                        measure_qubit.append(q_m)
                        results = sim.measure(measure_qubit, NUM_MEASURE)
                        print(results)
        for op in reversed(self.ordered_gate_list):
            gate = op.get_gate().upper()
            # if it is measure, handle it first
            if gate == "MEASURE":
                measure_qubit = []
                assert (len(op.get_qubits()) == 2)
                q_m = uqubits[op.get_qubits()[0]]
                measure_qubit.append(q_m)
                # results = sim.measure(measure_qubit,NUM_MEASURE)
                # print(results)
                continue
            if gate in GATE_NOPARAM:
                ag = svsim.AggregateGate_0(gate)
            elif gate in GATE_ONEPARAM:
                assert (len(op.get_params()) == 1)
                if gate == "U1":
                    ag = svsim.AggregateGate_1(gate, op.get_params()[0]*-1)
                else:
                    ag = svsim.AggregateGate_1(gate, op.get_params()[0])
            elif gate in GATE_TWOPARAM:
                assert (len(op.get_params()) == 2)
                ag = svsim.AggregateGate_2(gate, op.get_params()[0], op.get_params()[1])
            elif gate in GATE_THREEPARAM:
                assert (len(op.get_params()) == 3)
                ag = svsim.AggregateGate_3(gate, op.get_params()[0], op.get_params()[1], op.get_params()[2])
            elif gate in GATE_C1:
                assert (len(op.get_params()) == 4)
                ag = svsim.AggregateGate_C1(gate, op.get_params()[0], op.get_params()[1], op.get_params()[2],
                                            op.get_params()[3])
            elif gate in GATE_C2:
                assert (len(op.get_params()) == 16)
                ag = svsim.AggregateGate_C2(gate, op.get_params()[0], op.get_params()[1], op.get_params()[2],
                                            op.get_params()[3],
                                            op.get_params()[4], op.get_params()[5], op.get_params()[6],
                                            op.get_params()[7],
                                            op.get_params()[8], op.get_params()[9], op.get_params()[10],
                                            op.get_params()[11],
                                            op.get_params()[12], op.get_params()[13], op.get_params()[14],
                                            op.get_params()[15])
            elif gate in GATE_FRAC:
                assert (len(op.get_params()) == 2)
                ag = svsim.AggregateGate_Frac(gate, op.get_params()[0], op.get_params()[1])

            else:
                # this means the gate is probabaly not a basic gate. We need to handle them later
                ag = svsim.AggregateGate_0("I")
                ag_factory = svsim.AggregateGateFactory(ag)
            op_qubits = len(op.get_qubits())
            control_bits = []
            target_bits = []
            if op_qubits == 1:
                # this means the control qubit should be null
                target_bits.append(uqubits[op.get_qubits()[0]])
            if op_qubits == 2:
                if gate == 'CX':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'X'
                    ag = svsim.AggregateGate_0(gate_alt)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CY':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'Y'
                    ag = svsim.AggregateGate_0(gate_alt)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CZ':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'Z'
                    ag = svsim.AggregateGate_0(gate_alt)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CH':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'H'
                    ag = svsim.AggregateGate_0(gate_alt)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CRX':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'RX'
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CRY':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'RY'
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CRZ':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'RZ'
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'RXX':
                    t_bit = uqubits[op.get_qubits()[0]]
                    target_bits.append(t_bit)
                    t_bit = uqubits[op.get_qubits()[1]]
                    target_bits.append(t_bit)
                    gate_alt = 'RXX'
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0]*-1)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'RYY':
                    t_bit = uqubits[op.get_qubits()[0]]
                    target_bits.append(t_bit)
                    t_bit = uqubits[op.get_qubits()[1]]
                    target_bits.append(t_bit)
                    gate_alt = 'RYY'
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0]*-1)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'RZZ':
                    t_bit = uqubits[op.get_qubits()[0]]
                    target_bits.append(t_bit)
                    t_bit = uqubits[op.get_qubits()[1]]
                    target_bits.append(t_bit)
                    gate_alt = 'RZZ'
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0]*-1)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CU1':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'U1'
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0]*-1)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CU3':
                    c_bit = uqubits[op.get_qubits()[0]]
                    t_bit = uqubits[op.get_qubits()[1]]
                    control_bits.append(c_bit)
                    target_bits.append(t_bit)
                    gate_alt = 'U3'
                    ag = svsim.AggregateGate_3(gate_alt, op.get_params()[0], op.get_params()[1], op.get_params()[2])
                    ag_factory = svsim.AggregateGateFactory(ag)
            if op_qubits == 3:
                if gate == 'CCX':
                    c_bit_1 = uqubits[op.get_qubits()[0]]
                    c_bit_2 = uqubits[op.get_qubits()[1]]
                    t_bit = uqubits[op.get_qubits()[2]]
                    control_bits.append(c_bit_1)
                    control_bits.append(c_bit_2)
                    target_bits.append(t_bit)
                    gate_alt = 'X'
                    ag = svsim.AggregateGate_0(gate_alt)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'CSWAP':
                    c_bit_1 = uqubits[op.get_qubits()[0]]
                    t_bit_1 = uqubits[op.get_qubits()[1]]
                    t_bit_2 = uqubits[op.get_qubits()[2]]
                    control_bits.append(c_bit_1)
                    target_bits.append(t_bit_1)
                    target_bits.append(t_bit_2)
                    gate_alt = 'SWAP'
                    ag = svsim.AggregateGate_0(gate_alt)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'RCCX':
                    c_bit_1 = uqubits[op.get_qubits()[0]]
                    c_bit_2 = uqubits[op.get_qubits()[1]]
                    t_bit_1 = uqubits[op.get_qubits()[2]]
                    control_bits.append(c_bit_1)
                    control_bits.append(c_bit_2)
                    target_bits.append(t_bit_1)
                    gate_alt = 'RX'
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                    ag_factory = svsim.AggregateGateFactory(ag)
            if op_qubits == 4:
                if gate == 'RC3X':
                    c_bit_1 = uqubits[op.get_qubits()[0]]
                    c_bit_2 = uqubits[op.get_qubits()[1]]
                    c_bit_3 = uqubits[op.get_qubits()[2]]
                    t_bit_1 = uqubits[op.get_qubits()[3]]
                    control_bits.append(c_bit_1)
                    control_bits.append(c_bit_2)
                    control_bits.append(c_bit_3)
                    target_bits.append(t_bit_1)
                    gate_alt = 'RX'
                    ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'C3X':
                    c_bit_1 = uqubits[op.get_qubits()[0]]
                    c_bit_2 = uqubits[op.get_qubits()[1]]
                    c_bit_3 = uqubits[op.get_qubits()[2]]
                    t_bit_1 = uqubits[op.get_qubits()[3]]
                    control_bits.append(c_bit_1)
                    control_bits.append(c_bit_2)
                    control_bits.append(c_bit_3)
                    target_bits.append(t_bit_1)
                    gate_alt = 'X'
                    ag = svsim.AggregateGate_0(gate_alt)
                    ag_factory = svsim.AggregateGateFactory(ag)
                if gate == 'C3XSQRTX':
                    pass
            if op_qubits == 5:
                if gate == 'C4X':
                    c_bit_1 = uqubits[op.get_qubits()[0]]
                    c_bit_2 = uqubits[op.get_qubits()[1]]
                    c_bit_3 = uqubits[op.get_qubits()[2]]
                    c_bit_4 = uqubits[op.get_qubits()[3]]
                    t_bit_1 = uqubits[op.get_qubits()[4]]
                    control_bits.append(c_bit_1)
                    control_bits.append(c_bit_2)
                    control_bits.append(c_bit_3)
                    control_bits.append(c_bit_4)
                    target_bits.append(t_bit_1)
                    gate_alt = 'X'
                    ag = svsim.AggregateGate_0(gate_alt)
                    ag_factory = svsim.AggregateGateFactory(ag)
            # print(control_bits)
            # print(target_bits)
            boundgate = svsim.BoundGate(ag_factory, control_bits, target_bits)
            now = datetime.now()
            boundgate.apply(sim)
            later = datetime.now()
            diff = later - now
            acc_timer += diff.seconds * 1000000 + diff.microseconds
            print(gate + " has been applied.")
            print("time elapsed " + str(acc_timer))
        if sim == sim_init:
            print("VERIFIED")
        else:
            print("NOT VERIFIED")
    '''
    def execute_qasm_parts_apply(self,parts_file):
        global measure_op
        global acc_timer
        parts = self.parse_parts_file(parts_file)
        uqubits = self.create_universal_qubit()
        inv_uqubits = self.reverse_qubit(uqubits)
        total_qubits = len(uqubits)
        sim = svsim.StateVector(total_qubits,0)
        for part in parts:
            for order_id in parts[part]:
                if order_id == 0:
                    continue
                elif (order_id - 1024) > len(self.ordered_gate_list):
                    continue
                else:
                    order_id = order_id - 1
                op = self.ordered_gate_list[order_id]
                gate = op.get_gate().upper()
                # if it is measure, handle it first
                if gate == "MEASURE":
                    measure_qubit = []
                    assert(len(op.get_qubits()) == 2)
                    q_m = uqubits[op.get_qubits()[0]]
                    measure_qubit.append(q_m)
                    #results = sim.measure(measure_qubit,NUM_MEASURE)
                    #print(results)
                    continue
                if gate in GATE_NOPARAM:
                    ag = svsim.AggregateGate_0(gate)
                if gate in GATE_ONEPARAM:
                    assert(len(op.get_params()) == 1)
                    ag = svsim.AggregateGate_1(gate,op.get_params()[0])
                if gate in GATE_TWOPARAM:
                    assert(len(op.get_params()) == 2)
                    ag = svsim.AggregateGate_2(gate,op.get_params()[0],op.get_params()[1])
                if gate in GATE_THREEPARAM:
                    assert(len(op.get_params()) == 3)
                    ag = svsim.AggregateGate_3(gate,op.get_params()[0],op.get_params()[1],op.get_params()[2])
                if gate in GATE_C1:
                    assert (len(op.get_params()) == 4)
                    ag = svsim.AggregateGate_C1(gate, op.get_params()[0], op.get_params()[1], op.get_params()[2], op.get_params()[3])
                if gate in GATE_C2:
                    assert (len(op.get_params()) == 16)
                    ag = svsim.AggregateGate_C2(gate,op.get_params()[0],op.get_params()[1],op.get_params()[2],op.get_params()[3],
                                                     op.get_params()[4],op.get_params()[5],op.get_params()[6],op.get_params()[7],
                                                     op.get_params()[8],op.get_params()[9],op.get_params()[10],op.get_params()[11],
                                                     op.get_params()[12],op.get_params()[13],op.get_params()[14],op.get_params()[15])
                if gate in GATE_FRAC:
                    assert(len(op.get_params()) == 2)
                    ag = svsim.AggregateGate_Frac(gate,op.get_params()[0],op.get_params()[1])

                ag_factory = svsim.AggregateGateFactory(ag)
                op_qubits = len(op.get_qubits())
                control_bits = []
                target_bits = []
                if op_qubits == 1:
                    # this means the control qubit should be null
                    target_bits.append(uqubits[op.get_qubits()[0]])
                if op_qubits == 2:
                    if gate == 'CX':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'X'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CY':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'Y'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CZ':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'Z'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CH':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'H'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CRX':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'RX'
                        ag = svsim.AggregateGate_1(gate_alt,op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CRY':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'RY'
                        ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CRZ':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'RZ'
                        ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CU1':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'U1'
                        ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CU3':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'U3'
                        ag = svsim.AggregateGate_3(gate_alt, op.get_params()[0],op.get_params()[1],op.get_params()[2])
                        ag_factory = svsim.AggregateGateFactory(ag)
                if op_qubits == 3:
                    if gate == 'CCX':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        c_bit_2 = uqubits[op.get_qubits()[1]]
                        t_bit = uqubits[op.get_qubits()[2]]
                        control_bits.append(c_bit_1)
                        control_bits.append(c_bit_2)
                        target_bits.append(t_bit)
                        gate_alt = 'X'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CSWAP':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        t_bit_1 = uqubits[op.get_qubits()[1]]
                        t_bit_2 = uqubits[op.get_qubits()[2]]
                        control_bits.append(c_bit_1)
                        target_bits.append(t_bit_1)
                        target_bits.append(t_bit_2)
                        gate_alt = 'SWAP'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'RCCX':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        c_bit_2 = uqubits[op.get_qubits()[1]]
                        t_bit_1 = uqubits[op.get_qubits()[2]]
                        control_bits.append(c_bit_1)
                        control_bits.append(c_bit_2)
                        target_bits.append(t_bit_1)
                        gate_alt = 'RX'
                        ag = svsim.AggregateGate_1(gate_alt,op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                if op_qubits == 4:
                    if gate == 'RC3X':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        c_bit_2 = uqubits[op.get_qubits()[1]]
                        c_bit_3 = uqubits[op.get_qubits()[2]]
                        t_bit_1 = uqubits[op.get_qubits()[3]]
                        control_bits.append(c_bit_1)
                        control_bits.append(c_bit_2)
                        control_bits.append(c_bit_3)
                        target_bits.append(t_bit_1)
                        gate_alt = 'RX'
                        ag = svsim.AggregateGate_1(gate_alt,op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'C3X':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        c_bit_2 = uqubits[op.get_qubits()[1]]
                        c_bit_3 = uqubits[op.get_qubits()[2]]
                        t_bit_1 = uqubits[op.get_qubits()[3]]
                        control_bits.append(c_bit_1)
                        control_bits.append(c_bit_2)
                        control_bits.append(c_bit_3)
                        target_bits.append(t_bit_1)
                        gate_alt = 'X'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'C3XSQRTX':
                        pass
                if op_qubits == 5:
                    if gate == 'C4X':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        c_bit_2 = uqubits[op.get_qubits()[1]]
                        c_bit_3 = uqubits[op.get_qubits()[2]]
                        c_bit_4 = uqubits[op.get_qubits()[3]]
                        t_bit_1 = uqubits[op.get_qubits()[4]]
                        control_bits.append(c_bit_1)
                        control_bits.append(c_bit_2)
                        control_bits.append(c_bit_3)
                        control_bits.append(c_bit_4)
                        target_bits.append(t_bit_1)
                        gate_alt = 'X'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                #print(control_bits)
                #print(target_bits)
                boundgate = svsim.BoundGate(ag_factory,control_bits,target_bits)
                now = datetime.now()
                boundgate.apply(sim)
                print(gate+" has been executed directly.")
                later = datetime.now()
                diff = later - now
                acc_timer += diff.seconds*1000000+diff.microseconds
        for part in measure_op:
            if part in parts_file:
                for op in measure_op[part]:
                    gate = op.get_gate().upper()
                    # if it is measure, handle it first
                    if gate == "MEASURE":
                        measure_qubit = []
                        assert (len(op.get_qubits()) == 2)
                        q_m = uqubits[op.get_qubits()[0]]
                        measure_qubit.append(q_m)
                        results = sim.measure(measure_qubit, NUM_MEASURE)
                        print(results)

    def execute_qasm_part_direct(self,op_lists):
        global measure_op
        global acc_timer
        uqubits = self.create_universal_qubit()
        inv_uqubits = self.reverse_qubit(uqubits)
        total_qubits = len(uqubits)
        sim = svsim.StateVector(total_qubits,0)
        for part,oplist in enumerate(op_lists):
            boundgate_list = [] 
            for op in oplist:
                gate = op.get_gate().upper()
                # if it is measure, handle it first
                if gate == "MEASURE":
                    measure_qubit = []
                    assert(len(op.get_qubits()) == 2)
                    q_m = uqubits[op.get_qubits()[0]]
                    measure_qubit.append(q_m)
                    #results = sim.measure(measure_qubit,NUM_MEASURE)
                    #print(results)
                    continue
                if gate in GATE_NOPARAM:
                    ag = svsim.AggregateGate_0(gate)
                if gate in GATE_ONEPARAM:
                    assert(len(op.get_params()) == 1)
                    ag = svsim.AggregateGate_1(gate,op.get_params()[0])
                if gate in GATE_TWOPARAM:
                    assert(len(op.get_params()) == 2)
                    ag = svsim.AggregateGate_2(gate,op.get_params()[0],op.get_params()[1])
                if gate in GATE_THREEPARAM:
                    assert(len(op.get_params()) == 3)
                    ag = svsim.AggregateGate_3(gate,op.get_params()[0],op.get_params()[1],op.get_params()[2])
                if gate in GATE_C1:
                    assert (len(op.get_params()) == 4)
                    ag = svsim.AggregateGate_C1(gate, op.get_params()[0], op.get_params()[1], op.get_params()[2], op.get_params()[3])
                if gate in GATE_C2:
                    assert (len(op.get_params()) == 16)
                    ag = svsim.AggregateGate_C2(gate,op.get_params()[0],op.get_params()[1],op.get_params()[2],op.get_params()[3],
                                                     op.get_params()[4],op.get_params()[5],op.get_params()[6],op.get_params()[7],
                                                     op.get_params()[8],op.get_params()[9],op.get_params()[10],op.get_params()[11],
                                                     op.get_params()[12],op.get_params()[13],op.get_params()[14],op.get_params()[15])
                if gate in GATE_FRAC:
                    assert(len(op.get_params()) == 2)
                    ag = svsim.AggregateGate_Frac(gate,op.get_params()[0],op.get_params()[1])

                ag_factory = svsim.AggregateGateFactory(ag)
                op_qubits = len(op.get_qubits())
                control_bits = []
                target_bits = []
                if op_qubits == 1:
                    # this means the control qubit should be null
                    target_bits.append(uqubits[op.get_qubits()[0]])
                if op_qubits == 2:
                    if gate == 'CX':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'X'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CY':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'Y'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CZ':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'Z'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CH':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'H'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CRX':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'RX'
                        ag = svsim.AggregateGate_1(gate_alt,op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CRY':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'RY'
                        ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CRZ':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'RZ'
                        ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CU1':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'U1'
                        ag = svsim.AggregateGate_1(gate_alt, op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CU3':
                        c_bit = uqubits[op.get_qubits()[0]]
                        t_bit = uqubits[op.get_qubits()[1]]
                        control_bits.append(c_bit)
                        target_bits.append(t_bit)
                        gate_alt = 'U3'
                        ag = svsim.AggregateGate_3(gate_alt, op.get_params()[0],op.get_params()[1],op.get_params()[2])
                        ag_factory = svsim.AggregateGateFactory(ag)
                if op_qubits == 3:
                    if gate == 'CCX':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        c_bit_2 = uqubits[op.get_qubits()[1]]
                        t_bit = uqubits[op.get_qubits()[2]]
                        control_bits.append(c_bit_1)
                        control_bits.append(c_bit_2)
                        target_bits.append(t_bit)
                        gate_alt = 'X'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'CSWAP':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        t_bit_1 = uqubits[op.get_qubits()[1]]
                        t_bit_2 = uqubits[op.get_qubits()[2]]
                        control_bits.append(c_bit_1)
                        target_bits.append(t_bit_1)
                        target_bits.append(t_bit_2)
                        gate_alt = 'SWAP'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'RCCX':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        c_bit_2 = uqubits[op.get_qubits()[1]]
                        t_bit_1 = uqubits[op.get_qubits()[2]]
                        control_bits.append(c_bit_1)
                        control_bits.append(c_bit_2)
                        target_bits.append(t_bit_1)
                        gate_alt = 'RX'
                        ag = svsim.AggregateGate_1(gate_alt,op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                if op_qubits == 4:
                    if gate == 'RC3X':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        c_bit_2 = uqubits[op.get_qubits()[1]]
                        c_bit_3 = uqubits[op.get_qubits()[2]]
                        t_bit_1 = uqubits[op.get_qubits()[3]]
                        control_bits.append(c_bit_1)
                        control_bits.append(c_bit_2)
                        control_bits.append(c_bit_3)
                        target_bits.append(t_bit_1)
                        gate_alt = 'RX'
                        ag = svsim.AggregateGate_1(gate_alt,op.get_params()[0])
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'C3X':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        c_bit_2 = uqubits[op.get_qubits()[1]]
                        c_bit_3 = uqubits[op.get_qubits()[2]]
                        t_bit_1 = uqubits[op.get_qubits()[3]]
                        control_bits.append(c_bit_1)
                        control_bits.append(c_bit_2)
                        control_bits.append(c_bit_3)
                        target_bits.append(t_bit_1)
                        gate_alt = 'X'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                    if gate == 'C3XSQRTX':
                        pass
                if op_qubits == 5:
                    if gate == 'C4X':
                        c_bit_1 = uqubits[op.get_qubits()[0]]
                        c_bit_2 = uqubits[op.get_qubits()[1]]
                        c_bit_3 = uqubits[op.get_qubits()[2]]
                        c_bit_4 = uqubits[op.get_qubits()[3]]
                        t_bit_1 = uqubits[op.get_qubits()[4]]
                        control_bits.append(c_bit_1)
                        control_bits.append(c_bit_2)
                        control_bits.append(c_bit_3)
                        control_bits.append(c_bit_4)
                        target_bits.append(t_bit_1)
                        gate_alt = 'X'
                        ag = svsim.AggregateGate_0(gate_alt)
                        ag_factory = svsim.AggregateGateFactory(ag)
                #print(control_bits)
                #print(target_bits)
                boundgate = svsim.BoundGate(ag_factory,control_bits,target_bits)
                boundgate_list.append(boundgate)
                print(gate+" has been added to the gather list.")
            now = datetime.now()
            svsim.gather_and_execute_on(sim,boundgate_list)
            print("Part "+str(part)+" has been executed")
            later = datetime.now()
            diff = later - now
            acc_timer += diff.seconds*1000000+diff.microseconds
        
        for part in measure_op:
            if part in parts_file:
                for op in measure_op[part]:
                    gate = op.get_gate().upper()
                    # if it is measure, handle it first
                    if gate == "MEASURE":
                        measure_qubit = []
                        assert (len(op.get_qubits()) == 2)
                        q_m = uqubits[op.get_qubits()[0]]
                        measure_qubit.append(q_m)
                        results = sim.measure(measure_qubit, NUM_MEASURE)
                        print(results)
        

    '''

    def parse_parts_file(self,parts_file:str)->dict:
        parts = {}
        with open(parts_file,'r') as f:
            lines = f.readlines()
            # for now just use the line number as the order
            for idx, line in enumerate(lines):
                line = line.rstrip("\n")
                items = line.split(' ')
                assert(len(items) == 3)
                order_id = int(items[0])
                part_id = int(items[2])
                if part_id not in parts:
                    parts[part_id] = []
                    parts[part_id].append(order_id)
                else:
                    parts[part_id].append(order_id)
        #need to order the parts based on their part id
        parts = collections.OrderedDict(sorted(parts.items()))
        return parts

    def create_universal_qubit(self):
        uqubit = {}
        for q in self.qubit_allocation:
            for q_i in range(int(self.qubit_allocation[q])):
                uqubit[q+"["+str(q_i)+"]"] = len(uqubit)
        #print(uqubit)
        return uqubit

    def reverse_qubit(self,uqbuit):
        inv_map = {v: k for k, v in uqbuit.items()}
        return inv_map

def convert_param(val:str):
    if "pi" in val:
        return handle_pi(val)
    # seems no this case
    elif "e" in val:
        return 1.0
    else:
        return float(val)

def handle_pi(val:str):
    # most likely this is the case
    ret = 0.0
    if "/" in val:
        items = val.split("/")
        assert (len(items) == 2)
        if "pi" in items[0]:
            ret = math.pi / float(items[1])
        elif "pi" in items[1]:
            ret = float(items[0]) / math.pi
        else:
            print("no pi in handle_pi function")
            raise ValueError
    if "*" in val:
        items = val.split("*")
        assert (len(items) == 2)
        if "pi" in items[0]:
            ret = math.pi * float(items[1])
        elif "pi" in items[1]:
            ret = float(items[0]) * math.pi
        else:
            print("no pi in handle_pi function")
            raise ValueError
    if "+" in val:
        pass
    if "-pi" in val:
        ret = ret*(-1)
    return ret

def generate_void_vertex(qubit:str,counter:int):
    return qubit+"_exit_"+str(counter)

def get_unique_name(gate:str,counter:int):
    return gate+"_"+str(counter)

def add_input_node_pass(G):
    need_for_add = {}
    for node in G:
        if len(G.in_edges(node,keys=True)) == len(G.out_edges(node,keys=True)):
            continue
        else:
            #get out edge labels
            in_labels = []
            for u_in, v_in, key_in in G.in_edges(node,keys=True):
                in_labels.append(G[u_in][v_in][key_in]['label'])
            for u_out, v_out, key_out in G.out_edges(node,keys=True):
                if G[u_out][v_out][key_out]['label'] == 'exit':
                    continue
                if G[u_out][v_out][key_out]['label'] in in_labels:
                    continue
                else:
                    qubit = G[u_out][v_out][key_out]['label']
                    if node not in need_for_add:
                        need_for_add[node] = []
                        need_for_add[node].append(qubit)
                    else:
                        need_for_add[node].append(qubit)

    for node in need_for_add:
        for qubit in need_for_add[node]:
            G.add_node(qubit,label = qubit, order = 0)
            G.add_edge(qubit,node,label = qubit)

    # verify the graph property
    for node in G:
        if (G.in_degree(node) == 0 or G.out_degree(node) == 0 ):
            continue
        else:
            if G.in_degree(node) != G.out_degree(node):
                print(node)
                print("should not happen")
                raise ValueError
            in_labels = []
            for u_in, v_in, key_in in G.in_edges(node, keys=True):
                in_labels.append(G[u_in][v_in][key_in]['label'])
            out_labels = []
            for u_out, v_out, key_out in G.out_edges(node, keys=True):
                out_labels.append(G[u_out][v_out][key_out]['label'])
            if sorted(in_labels) != sorted(out_labels):
                print("node not match")
                raise ValueError

def reorder_files(files:list):
    ordered = []
    second = {}
    for f in files:
        if "indvpart_" not in f:
            ordered.append(f)
        else:
            tmp = re.findall('indvpart_\d+',f)
            assert(len(tmp) ==1)
            second[tmp[0]] = f
    od = collections.OrderedDict(sorted(second.items()))
    for k in od:
        ordered.append(od[k])
    return ordered
            

def produce_dot_file(op_list:list,qasm_name:str,path):
    global measure_op
    G = nx.MultiDiGraph()
    void_node_cache = []
    counter = 0
    part = 0
    qasm_name = qasm_name.replace("QASMBENCH",path)
    for order, op in enumerate(op_list):
        #if len(void_node_cache) % 100 == 0:
        #    print("Processed "+str(len(G.nodes))+" nodes "+str(len(void_node_cache)))
        tmp_cache = []
        gate = op.get_gate()
        qubits = op.get_qubits()
        gate_name = get_unique_name(gate,counter)
        if "measure" in gate_name:
            if len(G.nodes) == 0:
                if part not in measure_op:
                    measure_op[part] = []
                    measure_op[part].append(op)
                continue
            add_input_node_pass(G)
            print("Writing dot files")
            dotfile_before = datetime.now()
            nx.drawing.nx_pydot.write_dot(G, qasm_name + "_PART_" + str(part) + ".dot")
            dotfile_after = datetime.now()
            dotfile_diff = dotfile_after - dotfile_before
            dotfile_timer = dotfile_diff.seconds * 1000000 + dotfile_diff.microseconds
            print("SVSIM writing dot file takes " + str(dotfile_timer))
            part += 1
            G = nx.MultiDiGraph()
            void_node_cache = []
            continue
        counter += 1
        G.add_node(gate_name, label=gate,line=op.get_index(), order = order+1)
        qubit_stripped = []
        for qubit in qubits:
            qubit = qubit.replace('[',"")
            qubit = qubit.replace(']',"")
            qubit_stripped.append(qubit)
            void_node = generate_void_vertex(qubit,counter)
            counter += 1
            G.add_node(void_node,label='exit', order = len(op_list)+8192)
            tmp_cache.append(void_node)
            G.add_edge(gate_name,void_node,label=qubit)
        nodes_to_delete = []
        flag = len(qubits)
        for node in void_node_cache:
            for u,v,key in G.in_edges(node, keys=True):
                if G[u][v][key]['label'] in qubit_stripped:
                        # this means that the gate should be connected to u
                    G.add_edge(u,gate_name,label=G[u][v][key]['label'])
                    nodes_to_delete.append(node)
                    flag -= 1
            if flag == 0:
                break
        for node in nodes_to_delete:
            G.remove_node(node)
            void_node_cache.remove(node)
        void_node_cache.extend(tmp_cache)
    if (len(G.nodes) != 0):
        add_input_node_pass(G)
        print("Writing dot files")
        dotfile_before = datetime.now()
        nx.drawing.nx_pydot.write_dot(G, qasm_name + "_PART_" + str(part) + ".dot")
        dotfile_after = datetime.now()
        dotfile_diff = dotfile_after - dotfile_before
        dotfile_timer = dotfile_diff.seconds * 1000000 + dotfile_diff.microseconds
        print("SVSIM writing dot file takes " + str(dotfile_timer))
    '''
    pos = nx.spring_layout(G)
    node_labels = nx.get_node_attributes(G,'label')
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    edge_labels = nx.get_edge_attributes(G,'name')
    nx.draw_networkx_edge_labels(G, pos, labels=edge_labels)
    plt.savefig('test.png')
    plt.show()
    '''

comm = MPI.COMM_WORLD
argparser = argparse.ArgumentParser()
argparser.add_argument('--dot', default=False, action='store_true',help="to generate dot file")
argparser.add_argument('--intel', default=False, action='store_true',help="to generate intel gate file")  
argparser.add_argument('--parts', default=False, help="to run parts", action='store_true') 
argparser.add_argument('--app', type=str, required = True, help="to run the qasm file") 
argparser.add_argument('--n', type=int, required = True, help="to set up the number of omp threads") 
argparser.add_argument('--alloc', type=int, required = True, help="to set up the allocation scheme: 1 numa_lib, 0 first touch, 2 default") 
argparser.add_argument('--initn', type=int, required = True, help="to set up the number of omp threads init") 
argparser.add_argument('--partfile', type=str , help="to set up the number of omp threads")
argparser.add_argument('--slots', default=False, action='store_true',help="to automatically generate gathering slots")
argparser.add_argument('--reorder', default=False, action='store_true',help="to generate reordered qasm")
argparser.add_argument('--regen', default=False, action='store_true',help="to generate partitioned qasm files")
argparser.add_argument('--mpi', default=False, action='store_true',help="to enable mpi execution")
argparser.add_argument('--multi', default=False, action='store_true',help="to enable multi-level mpi execution")
argparser.add_argument('--nlocal', type=int, required = False, help="total number of local qubits for mpi")
argparser.add_argument('--fusion', default=False,action='store_true', help="enable fusion")  

args = argparser.parse_args()
qasmbench = os.listdir("QASMBench")
output_path = "qasmbench_dot"
allocation_scheme = args.alloc
'''
sim1 = svsim.StateVector(2,0)
control_bits = []
target_bits = [0]
gate_alt = 'X'
ag = svsim.AggregateGate_0(gate_alt)
ag_factory = svsim.AggregateGateFactory(ag)
boundgate = svsim.BoundGate(ag_factory,control_bits,target_bits)
boundgate.apply(sim1)
sim2 = svsim.StateVector(2,0)
if sim1 == sim2:
    print("Equal")
else:
    print("Not Equal")
exit()
'''
for d in qasmbench:
    #if "medium" in d or "small" in d or "large" in d:
    if "cluster" in d:
        sub_d = os.listdir(os.path.join("QASMBench",d))
        for ss_d in sub_d:
            if os.path.isfile(os.path.join("QASMBench",d,ss_d)):
                continue
            app = os.listdir(os.path.join("QASMBench",d,ss_d))
            for f in app:
                if f.endswith('.qasm') and args.app in f:
                    qasm_name = f
                    compile_before = datetime.now()
                    parser = QASM_parser(os.path.join("QASMBench",d,ss_d,qasm_name))
                    parser.identify_qubit_allocation()
                    parser.identify_custom_gate_pass()
                    parser.identify_gate_pass()
                    parser.order_gate_list()
                    parser.expand_qubit()
                    compile_after = datetime.now()
                    compile_diff = compile_after - compile_before
                    compile_timer = compile_diff.seconds * 1000000 + compile_diff.microseconds
                    if args.dot == True:
                        dot_before = datetime.now()
                        produce_dot_file(parser.ordered_gate_list,parser.filename,output_path)
                        dot_after = datetime.now()
                        dot_diff = dot_after - dot_before
                        dot_timer = dot_diff.seconds * 1000000 + dot_diff.microseconds
                        print("SVSIM dot takes " + str(dot_timer))
                    parser.dump_qasm()
                    if args.intel == True:
                        parser.dump_intel_gates()
                        exit()
                    svsim.set_num_threads(args.n)
                    svsim.set_init_num_threads(args.initn)
                    if args.parts == True:
                        if args.reorder == True:
                            parser.regenerate_qasm(os.path.join("QASMBench", d, ss_d, args.partfile))
                            exit()
                        if args.regen == True:
                            parser.regenerate_qasm_parts(os.path.join("QASMBench", d, ss_d, args.partfile))
                            exit()
                        if args.slots == True:
                            svsim.set_opt_slots(1)
                        if args.mpi == True and args.multi == False:
                            parser.execute_qasm_parts_mpi(os.path.join("QASMBench",d,ss_d,args.partfile),args.nlocal,comm)
                        elif args.mpi == True and args.multi == True:
                            path = os.path.join("QASMBench", d, ss_d,args.partfile)
                            partfiles = os.listdir(path)
                            # need to order the files
                            ordered_partfiles = reorder_files(partfiles)
                            #print(ordered_partfiles)
                            param_files = []
                            for pf in ordered_partfiles:
                                param_files.append(os.path.join("QASMBench", d, ss_d,args.partfile,pf))
                            #print(param_files)
                            parser.execute_qasm_multiparts_mpi(param_files,args.nlocal,comm)
                        elif args.mpi == False and args.multi == True:
                            print("mpi flag has to be true to enable multi")
                            exit()
                        else:
                            if args.fusion == True:
                                parser.gate_fusion()
                                exit()
                            parser.execute_qasm_parts(os.path.join("QASMBench",d,ss_d,args.partfile))
                        #parser.execute_qasm_part_direct([parser.ordered_gate_list])
                    else:
                        parser.execute_qasm()
                        #parser.verify_compute()
                    rank = comm.Get_rank()
                    if rank == 0:
                        print("SVSIM compile takes " + str(compile_timer))
                        print("elapsed time from python driver",acc_timer)
                        print("elapsed time in sim apply",svsim.obtain_apply_time())
                        print("# of basic gates applied",svsim.obtain_gate_counter())
                        print("# of move op",svsim.obtain_move_counter())
                        print("elapsed time in sim gather",svsim.obtain_gather_time())
                        print("========= "+qasm_name+" FINISHED")








