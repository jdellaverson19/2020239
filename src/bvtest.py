from bv import BernsteinVazirani

import time
import sys
from random import seed
from random import randint
from random import randrange
import numpy as np
import matplotlib.pyplot as plt

def main():
	
	bvObject = BernsteinVazirani()

	# constants
	QUBIT_RANGE = 5
	ITERATIONS = 2


	worked = np.zeros(shape=(QUBIT_RANGE, ITERATIONS))
	timing = np.zeros(shape=(QUBIT_RANGE, ITERATIONS))

	print('Testing out Bernstein-Vazirani alorithm...')

	seed(3245234)
	for n in range(QUBIT_RANGE):
		print(f'Trying {n+3}-qubit machine...')
		for j in range(ITERATIONS):
			print(f'Iteration {j+1}...')

			# randomly decide f
			a = randrange(0,2**(n+1))
			b = randint(0,1)
			def f(x):
				y = (a&x)
				parity = 0
				while y:
					parity = ~parity
					y = y & (y-1)
				return parity^b

			start = time.perf_counter()
			result, _ = bvObject.run(f, n+2)
			end = time.perf_counter()

			timing[n][j] = (end - start)

	qubit_values = []
	for i in range(QUBIT_RANGE):
		qubit_values += [i+3]

	average_runtimes = []
	for i in range(QUBIT_RANGE):
		average_runtimes += [np.mean(timing[i])]

	plt.plot(qubit_values, average_runtimes)
	plt.ylabel('Runtime (s)')
	plt.xlabel('Number of Qubits')
	plt.xticks(qubit_values)
	plt.title('Quantum Simulation Scaling for Bernstein-Vazirani Algorithm')
	plt.show()

if __name__ == "__main__":
	main()
