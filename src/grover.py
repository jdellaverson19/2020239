import inspect

from pyquil import get_qc, Program
from pyquil.gates import H, MEASURE
from pyquil.latex import *
import numpy as np

def quantum_grover(f):
	"""Determines whether the parameter oracle function is balanced or constant
	   via quantum computation (Deutsch-Jozsa Algorithm).

	Why it works:

	Parameters
	----------
	f : f : {0,1}^n -> {0,1}
		Takes an n-bit array and outputs 1 bit.

	Returns
	-------
	1 if there exists x in {0,1}^n such that f(x)=1
	0 otherwise
	"""
	n = len(inspect.signature(f).parameters)

	# construct U_f
	

	# prepare initial states
	qubits = list(range(n))

	p = Program()
	# apply hadamard to all qubits
	for i in range(n+1):
		p += H(i)