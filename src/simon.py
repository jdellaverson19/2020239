import inspect

from pyquil import get_qc, Program
from pyquil.gates import H, X, MEASURE
from pyquil.latex import *
import numpy as np
import utils
from collections import defaultdict
from typing import Dict, Tuple
import numpy.random as rd
from random import randint

from pyquil.quil import DefGate
from operator import xor




def binaryToInteger(binString):
	count = int(binString,2)
	return count

def intToBinString(theInt, N):
	listStr = ["{0:0","replace","b","}"]
	listStr[1] = str(N)
	out = "".join(listStr)
	output = out.format(theInt)
	return output
	#outStr = []
	#initialize to string of 0s
	#for i in range(N):
	#	outStr.append("0")
	#Could have been done in above, but this is conceptually simpler
	#for j in range(N):
	#	if theInt//2 > 1:
	#		theInt = theInt//2
	#		outStr[N-1-j] = "1"
	#output = "".join(outStr)
	#print("int2binstring,",output, theInt)
	#return output


def create_valid_2to1_bitmap2(mask):
	N = len(mask)
	domain = np.zeros(2**N)
	range = []
	#populate as 1-1 identity
	for x in np.arange(2**N):
		domain[x] = int(x)
		range.append(x)
	f = np.zeros(2**N)
	for j in np.arange(2**N):
		#Pick a random element in our range
		ran = range[randint(0,(len(range))-1)]
		f[j] = int(ran)
		#Set up the two-one ness of it
		#xor of j and s: utils.xor(intToBinString(j,N),s)
		f[int(binaryToInteger(utils.xor(intToBinString(j,N),mask)))] = int(ran)
		range.remove(ran)
	#Okay, so now we've made a mapping
	#Let's turn it into a map as we'd really like
	map = {}
	for k in np.arange(len(f)):
		map[intToBinString(int(k),N)] = intToBinString(int(f[k]),N)
	return map

#Take in mask s and a seed. Generate random 2-1 function as a bitmap. 
def create_simons_bitmap(s, random_seed = None):
	return create_valid_2to1_bitmap2(s)
	#Did not manage to figure out, grabbed Rigetti implementation for function encoding



	



class Simon(object):

	def __init__(self):
		self.n_compQubs = 1
		self.mask = "001"
		self.linIndepVectDict = {}

	def makeU_f(self, bitmap):
		#This was inspired by the rigetti code on their grove github (https://github.com/rigetti/grove/blob/master/grove/simona/Simon.py)
		n_bits = len(list(bitmap.keys())[0])
		U_f = np.zeros(shape=(2 ** (2 * n_bits), 2 ** (2 * n_bits)))
		indexmap = defaultdict(dict)
		for b in range(2**n_bits):
			pad_str = np.binary_repr(b, n_bits)
			for k, v in bitmap.items():
				indexmap[pad_str + k] = utils.xor(pad_str, v) + k
				i, j = int(pad_str+k, 2), int(utils.xor(pad_str, v) + k, 2)
				U_f[i, j] = 1
		return U_f, indexmap



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
		self.findMaskFromEq(bitmap)

	def findIndependentVectors(self, bitmap, qc):
		while(len(self.linIndepVectDict) < self.n_compQubs -1):
			
			simCircuit = Program()
			simCircuitReadout = simCircuit.declare('ro', 'BIT', self.n_compQubs)
			simCircuit += self.makeTheCircuit(bitmap)
			for i in range(self.n_compQubs):
				simCircuit+=MEASURE(i, simCircuitReadout[i])
			compiledCirq = qc.compile(simCircuit)
			simonbitstring = np.array(qc.run(compiledCirq)[0], dtype=int)
			self.checkIfSafeAddition(simonbitstring)
	
	def checkIfSafeAddition(self, x):
		#Check if all 0's
		if (x == 0).all():
			return None
	#otherwise find the most significant 
		xMsb = utils.mSB(x)
		if (xMsb not in self.linIndepVectDict.keys()):
			self.linIndepVectDict[xMsb] = x


	def findMaskFromEq(self, bitmap):

		#We got n-1 linearly independent equations, so now we need to add a last one to get n
		lasteq = self.lasteq()
		upper_triangular_matrix = np.asarray(
			[tup[1] for tup in sorted(zip(self.linIndepVectDict.keys(),
										  self.linIndepVectDict.values()),
									  key=lambda x: x[0])])

		msb_unit_vec = np.zeros(shape=(self.n_compQubs,), dtype=int)
		msb_unit_vec[lasteq] = 1
		self.mask = self.backSub(upper_triangular_matrix, msb_unit_vec).tolist()

	def lasteq(self):
		missing_msb = None
		for idx in range(self.n_compQubs):
			if idx not in self.linIndepVectDict.keys():
				missing_msb = idx

		if missing_msb is None:
			raise ValueError("Expected a missing provenance, but didn't find one.")

		augment_vec = np.zeros(shape=(self.n_compQubs,))
		augment_vec[missing_msb] = 1
		self.linIndepVectDict[missing_msb] = augment_vec.astype(int).tolist()
		return missing_msb

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
					m[row_num] = xor(secondMat[row_num], secondMat[col_num])

		return m[::-1]


