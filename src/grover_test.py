from grover import Grover

import time
import sys
from random import seed
from random import randrange
import numpy as np

def main():
	
	groverObject = Grover()

	# constants
	QUBIT_RANGE = 4
	ITERATIONS = 2


	worked = np.zeros(shape=(QUBIT_RANGE, ITERATIONS))
	timing = np.zeros(shape=(QUBIT_RANGE, ITERATIONS))

	seed(12345234)
	for n in range(0,QUBIT_RANGE):
		for j in range(ITERATIONS):
			# randomly decide target value
			target = randrange(0,2**(n+2))
			# print('Target value: ' + str(target))

			def f(x):
				return (x == target)

			start = time.perf_counter()
			result = groverObject.run(f, n+2)
			end = time.perf_counter()

			worked[n][j] = (result == target)
			timing[n][j] = (end - start)

	print('Timing: ')
	print(timing)
	print('Worked: ')
	print(worked)

if __name__ == "__main__":
	main()