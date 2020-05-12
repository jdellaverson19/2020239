import inspect

from pyquil import get_qc, Program
from pyquil.gates import H, X
from pyquil.latex import *
import numpy as np
import utils
from collections import defaultdict


#Current code structure: takes in an a and a b. I then create a dictionary that maps all inputs of length a to a value. 

def create_bv_bitmap(a,b):
	n_bits = len(a)
	bit_map = {}
	for bit_val in range(2 ** n_bits):
		binaryString = np.binary_repr(bit_val, width = n_bits)
		bit_map[binaryString] = str((int(utils.dot_product(binaryString, a)) + int(b, 2)) % 2)

	return bit_map



class BernsteinVazirani(object):

	def __init__(self):
		self.n_compQubs = 0
		self.n_ancillas = 1
		self.solution = None

	def makeTheMatrix(self, bitmap):
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
		#Make U_f from what we've been biven
		U_f = self.makeTheMatrix(bitmap)
		bvCircuit = Program()

		bvCircuit.defgate("BV-ORACLE", U_f)
		self.n_compQubs = len(list(bitmap.keys())[0])
		#set up/initialize the first hadamards
		for i in range(self.n_compQubs):
			bvCircuit+=H(i)
		#initialize the ancillary to be in the - state
		bvCircuit+=X(self.n_compQubs)
		bvCircuit+=H(self.n_compQubs)
		computational_qubits = list(range(self.n_compQubs))

		bvCircuit.inst(tuple(["BV-ORACLE"] + sorted(computational_qubits + [self.n_compQubs], reverse=True)))

		for i in range(self.n_compQubs):
			bvCircuit+=H(i)
		return bvCircuit




	def run(self, qc, bitmap):
		result = qc.run_and_measure(self.makeTheCircuit(bitmap), 1)
		solStr = ""
		for i in result:
			solStr+=str(result[i][0])
		self.solution = solStr
	def get_solution(self):
		return self.solution




