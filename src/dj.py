import inspect

from pyquil import get_qc, Program
from pyquil.gates import H, MEASURE
from pyquil.latex import *
import numpy as np

def quantum_DJ(f):
	"""Determines whether the parameter oracle function is balanced or constant
	   via quantum computation (Deutsch-Jozsa Algorithm).

	Why it works:

	Parameters
	----------
	f : f : {0,1}^n -> {0,1}
		Takes an n-bit array and outputs 1 bit.

	Returns
	-------
	1 if f is constant
	0 if f is balanced

	"""
	n = len(inspect.signature(f).parameters)

	# construct U_f

	# prepare initial states
	qubits = list(range(n+1))

	p = Program()
	# apply hadamard to all qubits
	for i in range(n+1):
		p += H(i)

	# apply U_f to all qubits
	p += U_f(*qubits)

	# apply hadamard to first n qubits
	for i in range(n):
		p += H(i)

	# prepare classical register
	ro = p.declare('ro', 'BIT', n)

	# measure first n qubits
	for i in range(n):
		p += MEASURE(i, ro[i])

	# get appropriate quantum computer
	# run program
		# if all 0 --> output 0
		# if nonzero --> output 1