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
        self.linIndepVectDict = {}

	def makeU_f(self, bitmap):
		#Speaking frankly, I have no idea how to do this -- this was 
		#Stolen from the rigetti code on their grove github (https://github.com/rigetti/grove/blob/master/grove/simona/Simon.py)
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
		self.findIndependentVectors(bitmap, qc)
        self.findMaskFromEq(bitmap)

    def findIndependentVectors(self, bitmap, qc):
        while(len(linIndepVectDict) < self.n_compQubs -1):
            simCircuit = Program()
            simCircuitReadout = prog.declare('ro', 'BIT', n_compQubs)
            simCircuit += makeTheCircuit(bitmap)
            for i in range(n_compQubs):
                prog+=MEASURE(i, i)
            compiledCirq = qc.compile(simCircuit)
            sampled_bit_string = np.array(qc.run(compiledCirq)[0], dtype=int)
            self.checkIfSafeAddition(sampled_bit_string)
    
    def checkIfSafeAddition(self, x):
        #Check if all 0's
        if (x = 0).all():
            return None
        xMsb = utils.mSB(x)


    def findMaskFromEq(self, ):



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


