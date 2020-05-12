import simon
from pyquil import Program, get_qc

import time
import sys
from random import seed
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import math

def main():
	

	# constants
	N = 2
	times = 3


	worked = np.zeros(shape=(N, times))
	timing = np.zeros(shape=(N, times))

	print('Testing out Simons alorithm...')

	seed(943856)
	for n in range(0,N):
		print(f'Trying 2*{N}-qubit machine...')
		for j in range(times):
			print(f'Iteration {j+1}...')

			# randomly decide f
			s = ""
			for i in range(0, N):
				s+=str(randint(0,1))
			bitmap = simon.create_simons_bitmap(s)
			
			strLen = N*2
			qcStr = str(strLen) + "q-qvm"
			qc = get_qc(qcStr)
			qc.compiler.client.timeout = 600
			simonObject = simon.Simon()

			start = time.perf_counter()
			result = simonObject.run(qc,bitmap)
			end = time.perf_counter()

			# print('worked' if result == constant else 'failed')
			timing[n][j] = (end - start)

			print("The calculated mask: " + simonObject.getMask())
			if(s!=simonObject.getMask()):
				print("FAILURE")
	qubit_values = []
	for i in range(N):
		qubit_values += [2*i]

	average_runtimes = []
	for i in range(N):
		average_runtimes += [np.mean(timing[i])]

	plt.plot(qubit_values, average_runtimes)
	plt.ylabel('Runtime (s)')
	plt.xlabel('Number of Qubits')
	plt.xticks(qubit_values)
	plt.title('Quantum Simulation Scaling for Simons Algorithm')
	plt.show()
	print("The string: " + s)
	
if __name__ == "__main__":
	main()