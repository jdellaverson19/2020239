from pyquil import Program, get_qc
from pyquil.gates import *
import bv as bv
import time
import sys


#This main testing method invokes the bv class to solve the Berstein Vazirani problem. 
#it takes in two strings, a and b, which are the bitstring inputs to the problem. 
#f(x) = x dot a + b (wher b is addition mod 2 and the dot is the dot product)
def main(a , b ):
	#We must construct an oracle for the QC to do its magic. We do so by creating a mapping from 
	#all possible inputs to outputs
	bitmap = bv.create_bv_bitmap(a,b)

	#We use pyquil to instantiate an appropriately sized QVM
	strLen = len(a) + 1
	qcStr = str(strLen) + "q-qvm"
	qc = get_qc(qcStr)

	#We instantiate a Bernstein Vazirani object, so we can later run it. 
	bvObject = bv.BernsteinVazirani()

	#We go ahead and start a timer so we can get results
	start = time.perf_counter()
	#We run the BV object and get our solution/answer. Note that the BV object ONLY sees the oracle and the quantum computer, nothing else. 
	bvObject.run(qc, bitmap)
	print(bvObject.get_solution())

	end = time.perf_counter()

	#print our results
	diff = end-start
	print(str(diff) + " many seconds")

if __name__ == "__main__":

    main(str(sys.argv[1]), str(sys.argv[2]))