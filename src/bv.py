from pyquil import get_qc, Program
from pyquil.gates import H, X, MEASURE
from pyquil.quil import DefGate
from pyquil.latex import *
import numpy as np

class BernsteinVazirani(object):

	def __init__(self):
		pass


	def getOracle(self, f, n):

		# create empty oracle U_f
		oracle = np.zeros(shape=(2**(n+1), 2**(n+1)))

		# populate oracle according to f
		# basically,
			# when f(x)=0 apply Identity to ancilla qubit
			# when f(x)=1 apply NOT to ancilla qubit
		for i in range(2**n):
			if f(i) == 0:
				oracle[2*i][2*i] = 1
				oracle[2*i+1][2*i+1] = 1
			else:
				oracle[2*i+1][2*i] = 1
				oracle[2*i][2*i+1] = 1

		return oracle


	def run(self, f, n):
		# execute circuit on smallest possible qc available
		qc = get_qc(f'{n+1}q-qvm')
		qc.compiler.client.timeout = 600
		return self.execute(self.get_circuit(f, n), qc, f)


	def get_circuit(self, f, n_qubits):

		# construct oracle and define its gate constructor
		oracle = self.getOracle(f, n_qubits)
		oracle_definition = DefGate("ORACLE", oracle)
		ORACLE = oracle_definition.get_constructor()

		# make circuit
		p = Program()

		# apply the first hadamard to all qubits
		for i in range(n_qubits):
			p += H(i)

		# NOT ancilla qubit then apply hadamard
		p += X(n_qubits)
		p += H(n_qubits)

		# apply oracle to all qubits
		p += oracle_definition
		p += ORACLE(*range(n_qubits+1))

		# apply hadamard to computational qubits
		for i in range(n_qubits):
			p += H(i)

		return p


	def execute(self, p, qc, f):

		# print(to_latex(p))

		# run and measure circuit using 1 trial
		results = qc.run_and_measure(p, 1)
		
		# iterate through all but ancilla bit results
		res = []
		for k in sorted(results)[:-1]:
			res += str(results[k][0])
		return int(''.join(res), base=2), f(0)
