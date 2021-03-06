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
	N = 3
	times = 2


	worked = np.zeros(shape=(N, times))
	timing = np.zeros(shape=(N, times))

	print('Testing out Simons alorithm...')

	for n in range(0,N):
		print(f'Trying {(n+1)*2}-qubit machine...')
		for j in range(times):
			print(f'Iteration {j+1}...')

			# randomly decide f
			s = ""
			for i in range(0, n+1):
				s+=str(randint(0,1))
			print("s is: ", s)


			bitmap = simon.create_simons_bitmap(s, 42)
			qcStr = f"{(n+1)*2}q-qvm"
			qc = get_qc(qcStr)
			qc.compiler.client.timeout = 1500
			simonObject = simon.Simon()

			start = time.perf_counter()
			result = simonObject.run(qc,bitmap)
			end = time.perf_counter()
			print("over")
			# print('worked' if result == constant else 'failed')
			timing[n][j] = (end - start)

			print("The calculated mask: ", simonObject.getMask())
			calcS = ""
			for i in simonObject.getMask():
				calcS+=str(i)
			if(s!=calcS):
				print("FAILURE")
			else:
				print("SUCCESS")
			print(end-start, "time in seconds")
	qubit_values = []
	for i in range(N):
		qubit_values += [2*(i+1)]

	average_runtimes = []
	for i in range(N):
		average_runtimes += [np.mean(timing[i])]
	timing[0] = 0
	plt.plot(qubit_values, average_runtimes)
	plt.ylabel('Runtime (s)')
	plt.xlabel('Number of Qubits')
	plt.xticks(qubit_values)
	plt.title('Quantum Simulation Scaling for Simon\'s Algorithm')
	plt.show()
	print("The string: " + s)
	
if __name__ == "__main__":
	main()