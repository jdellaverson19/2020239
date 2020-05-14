import inspect

from pyquil import get_qc, Program
from pyquil.gates import H, X, MEASURE
from pyquil.latex import *
import numpy as np
import utils
from collections import defaultdict
from typing import Dict, Tuple
import numpy.random as rd
from random import randint, shuffle

from pyquil.quil import DefGate
from operator import xor



#Helper function to turn binary strings into integers
def binaryToInteger(binString):
	count = int(binString,2)
	return count

#Helper function to turn integers into binary strings.
def intToBinString(theInt, N):
	listStr = ["{0:0","replace","b","}"]
	listStr[1] = str(N)
	out = "".join(listStr)
	output = out.format(theInt)
	return output


#Helper method that, given a mask, generates a random 2-1 function for Simon's. 
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
		#Convert to binary string so we can use our xor function
		f[int(binaryToInteger(utils.xor(intToBinString(j,N),mask)))] = int(ran)
		range.remove(ran)
	#Okay, so now we've made a mapping
	#Let's turn it into a map as we'd really like
	map = {}
	for k in np.arange(len(f)):
		map[intToBinString(int(k),N)] = intToBinString(int(f[k]),N)
	return map

#Creates a valid bitmap for the 1-1 case (s = 0)
def create_valid_1to1_bitmap(mask):
	N = len(mask)
	domain = np.zeros(2**N)
	range = []
	#populate as 1-1 identity
	for x in np.arange(2**N):
		domain[x] = int(x)
		range.append(x)
	#Shuffle it around so it's random
	shuffle(range)
	map = {}
	#Turn it into a string bitmap, so it plays nice with the rest of the code.
	for k in np.arange(len(range)):
		map[intToBinString(int(k),N)] = intToBinString(int(range[k]),N)
	return map


#Take in mask s and a seed. Generate random 2-1 function as a bitmap. 
#seed turned out to be irrelevant. Yay!
def create_simons_bitmap(s, random_seed = None):

	return create_valid_2to1_bitmap2(s)



class Simon(object):

	def __init__(self):
		self.n_compQubs = 1
		self.mask = "001"
		self.linIndepVectDict = {}

	#Given black box access (via a bitmap of inputs to outputs) constructs the appropriate U_f Matrix
	def makeU_f(self, bitmap):
		#This was inspired by the rigetti code on their grove github (https://github.com/rigetti/grove/blob/master/grove/simona/Simon.py)
		n_bits = self.n_compQubs
		U_f = np.zeros(shape=(2 ** (2 * n_bits), 2 ** (2 * n_bits)))
		indexmap = defaultdict(dict)
		for b in range(2**n_bits):
			pad_str = np.binary_repr(b, n_bits)
			for k, v in bitmap.items():
				indexmap[pad_str + k] = utils.xor(pad_str, v) + k
				i, j = int(pad_str+k, 2), int(utils.xor(pad_str, v) + k, 2)
				U_f[i, j] = 1
		return U_f


	#Make the quantum circuit for Simon's algorithm.
	def makeTheCircuit(self, bitmap):
		#Make U_f from what we've been given
		simCircuit = Program()

		#Apply Hadamard's to the computational (e.g. first n) qubits
		for i in range(self.n_compQubs):
			simCircuit+=H(i)
		#Apply U_f to the circuit (all qubits)
		oracle = self.makeU_f(bitmap)
		oracle_definition = DefGate("ORACLE", oracle)
		ORACLE = oracle_definition.get_constructor()
		simCircuit += oracle_definition
		simCircuit += ORACLE(*reversed(range(self.n_compQubs*2)))
		#Apply a second Hadamard to the first n qubits (again)
		for i in range(self.n_compQubs):
			simCircuit+=H(i)
		return simCircuit

	def run(self, qc, bitmap):
		#Initialize a class data member (just N, essentially)
		self.n_compQubs = len(list(bitmap.keys())[0])
		#Do the quantum part
		self.findIndepEqs(bitmap, qc)
		#Do the classical part
		self.findMaskFromEq(bitmap)

	#executes the quantum part of the algorithm. 
	def findIndepEqs(self, bitmap, qc):
		#While we haven't found the appropriate number of linearly independent equations...
		while(len(self.linIndepVectDict) < self.n_compQubs -1):
			#Set things up
			simCircuit = Program()
			simCircuitReadout = simCircuit.declare('ro', 'BIT', self.n_compQubs)
			simCircuit += self.makeTheCircuit(bitmap)
			for i in range(self.n_compQubs):
				simCircuit+=MEASURE(i, simCircuitReadout[i])
			#Compile our circuit
			compiledCirq = qc.compile(simCircuit)
			#get our output
			simonbitstring = np.array(qc.run(compiledCirq)[0], dtype=int)
			#See if it's an appropriate addition to our set of linearly indep equations. 
			self.checkIfSafeAddition(simonbitstring)


	def checkIfSafeAddition(self, x):
		#Check if all 0's. If it is, then it's not safe to add (not useful for equation solver at all)
		if (x == 0).all():
			return None
		#otherwise find the most significant bit. Inspired by Regetti's codebase, chose to index by MSB
		xMsb = utils.mSB(x)
		if (xMsb not in self.linIndepVectDict.keys()):
			self.linIndepVectDict[xMsb] = x


	def findMaskFromEq(self, bitmap):
		#Inspiration/help from rigetti grove code. 
		#We got n-1 linearly independent equations, so now we need to add a last one to get n
		lasteq = self.lasteq()
		#Turn it into a numpy array for convenience
		eqMatrix = np.asarray(
			[tup[1] for tup in sorted(zip(self.linIndepVectDict.keys(),
										  self.linIndepVectDict.values()),
									  key=lambda x: x[0])])
		msbUnitVec = np.zeros(shape=(self.n_compQubs,), dtype=int)
		msbUnitVec[lasteq] = 1
		self.mask = self.backSub(eqMatrix, msbUnitVec)


	def lasteq(self):
		missing_msb = None
		#Find our missing equation
		for s in range(self.n_compQubs):
			if s not in self.linIndepVectDict.keys():
				missing_msb = s
		#Create a 'dummy' equation that can fill in the gap. Add it to the dict. 
		#Naming clarification: QE is just the reverse of Eq (for equation). Doesn't mean anything in particular. 
		lastQE = []
		for i in range(0,self.n_compQubs):
			lastQE.append(0)
		self.linIndepVectDict[missing_msb] = lastQE
		return missing_msb


	def getMask(self):
		return self.mask


	#Do backsubstitution in GF2
	#Inspired by Regetti code (in grove repo)
	def backSub(self, firstMat, secondMat):
		m = np.copy(secondMat)
		n = len(secondMat)
		for row_num in range(n - 2, -1, -1):
			row = firstMat[row_num]
			for col_num in range(row_num + 1, n):
				if row[col_num] == 1:
					m[row_num] = xor(secondMat[row_num], secondMat[col_num])

		return m[::-1]


