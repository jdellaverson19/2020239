import inspect

from pyquil import get_qc, Program
from pyquil.gates import H, X, MEASURE
from pyquil.latex import *
import numpy as np
import utils
from collections import defaultdict
from typing import Dict, Tuple
import numpy.random as rd
from pyquil.quil import DefGate




def create_1to1_bitmap(mask: str) -> Dict[str, str]:

	n_bits = len(mask)
	form_string = "{0:0" + str(n_bits) + "b}"
	bit_map_dct = {}
	for idx in range(2**n_bits):
		bit_string = form_string.format(idx)
		bit_map_dct[bit_string] = utils.xor(bit_string, mask)
	return bit_map_dct


def create_valid_2to1_bitmap(mask: str, random_seed: int = None) -> Dict[str, str]:

	if random_seed is not None:
		rd.seed(random_seed)
	bit_map = create_1to1_bitmap(mask)
	n_samples = int(len(bit_map.keys()) / 2)
	range_of_2to1_map = list(rd.choice(list(sorted(bit_map.keys())), replace=False, size=n_samples))

	list_of_bitstring_tuples = sorted([(k, v) for k, v in bit_map.items()], key=lambda x: x[0])

	bit_map_dct = {}
	for cnt in range(n_samples):
		bitstring_tup = list_of_bitstring_tuples[cnt]
		val = range_of_2to1_map[cnt]
		bit_map_dct[bitstring_tup[0]] = val
		bit_map_dct[bitstring_tup[1]] = val
	print (bit_map_dct)
	return bit_map_dct


#Take in mask s and a seed. Generate random 2-1 function as a bitmap. 
def create_simons_bitmap(s, random_seed = None):
	return create_valid_2to1_bitmap(s, random_seed)



	



class Simon(object):

	def __init__(self):
		self.n_compQubs = 1
		self.mask = "001"
		self.linIndepVectDict = {}

	def makeU_f(self, bitmap):
		#Speaking frankly, I have no idea how to do this -- this was 
		#Stolen from the rigetti code on their grove github (https://github.com/rigetti/grove/blob/master/grove/simona/Simon.py)
		n_bits = len(list(bitmap.keys())[0])
		ufunc = np.zeros(shape=(2 ** (2 * n_bits), 2 ** (2 * n_bits)))
		index_mapping_dct = defaultdict(dict)
		for b in range(2**n_bits):
			# padding according to ancilla state
			pad_str = np.binary_repr(b, n_bits)
			for k, v in bitmap.items():
				# add mapping from initial state to the state in the ancilla system.
				# pad_str corresponds to the initial state of the ancilla system.
				index_mapping_dct[pad_str + k] = utils.xor(pad_str, v) + k
				# calculate matrix indices that correspond to the transition-matrix-element
				# of the oracle unitary
				i, j = int(pad_str+k, 2), int(utils.xor(pad_str, v) + k, 2)
				ufunc[i, j] = 1
		print(ufunc)
		return ufunc, index_mapping_dct



	def makeTheCircuit(self, bitmap):
		#Make U_f from what we've been given
		simCircuit = Program()

		#Apply Hadamard's to the computational (e.g. first n) qubits
		for i in range(self.n_compQubs):
			simCircuit+=H(i)
		#Apply U_f to the circuit (all qubits)
		oracle = self.makeU_f(bitmap)[0]
		oracle_definition = DefGate("ORACLE", oracle)
		ORACLE = oracle_definition.get_constructor()
		simCircuit += oracle_definition
		simCircuit += ORACLE(*reversed(range(self.n_compQubs*2)))
		#Apply a second Hadamard to the first n qubits (again)
		for i in range(self.n_compQubs):
			simCircuit+=H(i)
		return simCircuit

	def run(self, qc, bitmap):
		self.n_compQubs = len(list(bitmap.keys())[0])
		self.findIndependentVectors(bitmap, qc)
		print("found indep vectors")
		self.findMaskFromEq(bitmap)

	def findIndependentVectors(self, bitmap, qc):
		while(len(self.linIndepVectDict) < self.n_compQubs -1):
			
			simCircuit = Program()
			simCircuitReadout = simCircuit.declare('ro', 'BIT', self.n_compQubs)
			print("Simcircuit declared")
			simCircuit += self.makeTheCircuit(bitmap)
			print("Circuit made")
			for i in range(self.n_compQubs):
				simCircuit+=MEASURE(i, simCircuitReadout[i])
			print(simCircuit)
			compiledCirq = qc.compile(simCircuit)
			print("Circuit compiled")
			sampled_bit_string = np.array(qc.run(compiledCirq)[0], dtype=int)
			self.checkIfSafeAddition(sampled_bit_string)
			print("Iteration of find indep over")
			print(sampled_bit_string)
	
	def checkIfSafeAddition(self, x):
		#Check if all 0's
		if (x == 0).all():
			print("it was all 0s")
			return None
		xMsb = utils.mSB(x)
		if (xMsb not in self.linIndepVectDict.keys()):
			self.linIndepVectDict[xMsb] = x
			print("About to add:")
			print(x)

	def findMaskFromEq(self, bitmap):
		print("shouldn't be here")


	def getMask(self):
		return self.mask


	#Do backsubstitution
	def backSub(self, firstMat, secondMat) -> np.ndarray:

		# iterate backwards, starting from second to last row for back-substitution
		m = np.copy(secondMat)
		n = len(secondMat)
		for row_num in range(n - 2, -1, -1):
			row = firstMat[row_num]
			for col_num in range(row_num + 1, n):
				if row[col_num] == 1:
					print("Are we in backsub?")
					m[row_num] = xor(secondMat[row_num], secondMat[col_num])

		return m[::-1]


