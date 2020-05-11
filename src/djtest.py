from dj import DeutschJozsa

import time
import sys
from random import seed
from random import randint
import numpy as np

def main():
	
	djObject = DeutschJozsa()

	qubit_range = 4
	iterations = 1
	worked = np.zeros(shape=(qubit_range, iterations))
	timing = np.zeros(shape=(qubit_range, iterations))

	seed(943856)
	for n in range(0,qubit_range):
		for j in range(iterations):
			# randomly decide f
			const_val = randint(0,1)
			def f_constant(_):
				return const_val
			def f_balanced(x):
				return x%2

			constant = randint(0,1)

			f = f_constant if constant else f_balanced


			# if (constant):
			# 	print("Used constant function")
			start = time.perf_counter()
			result = djObject.run(f, n+1)
			end = time.perf_counter()

			# do something with result and time
			worked[n][j] = (result == constant)
			timing[n][j] = (end - start)

	print('Timing: ')
	print(timing)

if __name__ == "__main__":
	main()