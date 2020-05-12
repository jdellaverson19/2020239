import inspect
import math

from pyquil import get_qc, Program
from pyquil.gates import H, X, MEASURE
from pyquil.quil import DefGate
from pyquil.latex import *
import numpy as np

class Grover(object):
	"""
	Object that is solely used to construct Grover circuits for different oracles f.
	"""

	def __init__(self):
		pass

	# create phase_flipper gate
	def getFlipper(self, n):

		# create empty matrix
		flipper = np.zeros(shape=(2**n, 2**n))

		# flip amplitude signs for all inputs
		for i in range(2**n):
			flipper[i][i] = -1

		return flipper

	# create Z_0 gate
	def getHelper(self, n):
		
		# create empty matrix
		helper = np.zeros(shape=(2**n, 2**n))

		# flip amplitude sign for input 0^n
		helper[0][0] = -1

		# identity operator for all other inputs
		for i in range(1, 2**n):
			helper[i][i] = 1

		return helper

	# create Z_f gate
	def getOracle(self, f, n):

		# create empty oracle U_f
		oracle = np.zeros(shape=(2**n, 2**n))

		# flip amplitude for all inputs where f(x) = 1
		for i in range(2**n):
			oracle[i][i] = -1 if (f(i) == 1) else 1

		return oracle


	def run(self, f, n):
		# execute circuit on smallest possible qc available
		qc = get_qc(f'{n}q-qvm')
		qc.compiler.client.timeout = 300
		return self.execute(self.get_circuit(f, n), qc)


	def get_circuit(self, f, n_qubits):

		# make oracle gate Z_f
		oracle = self.getOracle(f, n_qubits)
		oracle_definition = DefGate('ORACLE', oracle)
		ORACLE = oracle_definition.get_constructor()

		# make helper gate Z_0
		helper = self.getHelper(n_qubits)
		helper_definition = DefGate('HELPER', helper)
		HELPER = helper_definition.get_constructor()

		# make flipper gate -I
		flipper = self.getFlipper(n_qubits)
		flipper_definition = DefGate('FLIPPER', flipper)
		FLIPPER = flipper_definition.get_constructor()

		# make circuit
		p = Program()

		# apply the first hadamard to all qubits
		for i in range(n_qubits):
			p += H(i)

		# add oracle and helper definitions
		p += oracle_definition
		p += helper_definition
		p += flipper_definition

		# apply oracle and diffuser designated amount of times
		NUM_ITERATIONS = int(math.sqrt(2**n_qubits) * math.pi / 4)
		for _ in range(NUM_ITERATIONS):

			# apply oracle
			p += ORACLE(*range(n_qubits))

			# apply diffuser
			for i in range(n_qubits):
				p += H(i)
			p += HELPER(*range(n_qubits))
			for i in range(n_qubits):
				p += H(i)
			p += FLIPPER(*range(n_qubits))


		return p


	def execute(self, p, qc):

		# run and measure circuit using 1 trial
		# print(to_latex(p))

		results = qc.run_and_measure(p, 1)
		# print('Results: ' + str(results))

		res = []
		for k in sorted(results):
			res += str(results[k][0])

		return int(''.join(res), base=2)