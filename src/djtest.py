from dj import DeutschJozsa

import time
import sys
from random import seed
from random import randint
import numpy as np

def main():
	
	djObject = DeutschJozsa()

	# constants
	QUBIT_RANGE = 4
	ITERATIONS = 1


	worked = np.zeros(shape=(QUBIT_RANGE, ITERATIONS))
	timing = np.zeros(shape=(QUBIT_RANGE, ITERATIONS))

	seed(943856)
	for n in range(0,QUBIT_RANGE):
		for j in range(ITERATIONS):
			# randomly decide f
			const_val = randint(0,1)
			def f_constant(_):
				return const_val
			def f_balanced(x):
				return x%2

			constant = randint(0,1)

			f = f_constant if constant else f_balanced

			start = time.perf_counter()
			result = djObject.run(f, n+1)
			end = time.perf_counter()

			# print('worked' if result == constant else 'failed')
			timing[n][j] = (end - start)

	print('Timing: ')
	print(timing)

if __name__ == "__main__":
	main()