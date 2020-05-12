from operator import xor
import numpy as np


def dot_product(st1, st2):

	if len(st1) != len(st2):
		print("str lens not equal")
		exit()
	result = 0
	list1 = list(st1)
	list2 = list(st2)
	for i in range(len(st1)):
		result+= (int(list1[i]) + int(list2[i])) 
	result = result % 2
	return str(result)


def xor(st1, st2):
	try:
		if len(st1) != len(st2):
			print("str lens not equal")
			exit()
	except:
		st1 = st1.tolist()
		st2 = st2.tolist()
	n_bits = len(st1)
	result = []
	list1 = list(st1)
	list2 = list(st2)
	for i in range((n_bits)):
		result.append(str((int(list1[i]) + int(list2[i]))%2))
	strRes = ""
	for ele in result:
		strRes+=ele
	return strRes



def mSB(x):
	return np.argwhere(np.asarray(x) == 1)[0][0]

