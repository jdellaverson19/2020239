import inspect

from pyquil import get_qc, Program
from pyquil.gates import H, X
from pyquil.latex import *
import numpy as np
import utils
from collections import defaultdict


#Take in mask s and a seed. Generate random 2-1 function as a bitmap. 
def create_simons_bitmap(s):


	



class Simon(object):

	def __init__(self):
		self.n_compQubs = 1
        self.mask = "001"

	def makeU_f(self, bitmap):
		#Speaking frankly, I have no idea how to do this -- this was 
		#Stolen from the rigetti code on their grove github (https://github.com/rigetti/grove/blob/master/grove/bernstein_vazirani/bernstein_vazirani.py)
		n_bits = len(list(bitmap.keys())[0])
		n_ancillas = 1
		ufunc = np.zeros(shape=(2 ** (n_bits + 1), 2 ** (n_bits + 1)))
		index_mapping_dct = defaultdict(dict)
		for b in range(2**n_ancillas):
			# padding according to ancilla state
			pad_str = np.binary_repr(b, width=1)
			for k, v in bitmap.items():
				# add mapping from initial state to the state in the ancilla system.
				# pad_str corresponds to the initial state of the ancilla system.
				index_mapping_dct[pad_str + k] = utils.xor(pad_str, v) + k
				# calculate matrix indices that correspond to the transition-matrix-element
				# of the oracle unitary
				i, j = int(pad_str+k, 2), int(utils.xor(pad_str, v) + k, 2)
				ufunc[i, j] = 1
		return ufunc



	def makeTheCircuit(self, bitmap):
		#Make U_f from what we've been given
		U_f = self.makeTheMatrix(bitmap)
		simCircuit = Program()

        #Create the U_f gate
		simCircuit.defgate("SIMON-ORACLE", U_f)

        #Apply Hadamard's to the computational (e.g. first n) qubits
		for i in range(self.n_compQubs):
			simCircuit+=H(i)
        
        #Apply U_f to the circuit (all qubits)
		oracle = self.makeU_f(bitmap)
		oracle_definition = DefGate("ORACLE", oracle)
		ORACLE = oracle_definition.get_constructor()
        simCircuit += oracle_definition
		simCircuit += ORACLE(*range(self.n_compQubs*2))

        #Apply a second Hadamard to the first n qubits (again)
        for i in range(self.n_compQubs):
            simCircuit+=H(i)

    return simCircuit

	def run(self, qc, bitmap):
		self.findIndependentVectors(bitmap)
        


	def get_mask(self):
		return self.mask




