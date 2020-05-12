# Use:

The following applies to each of the Deutsch-Jozsa, Bernstein-Vazirani, Simon's, and Grover's Algorithms:
- Create an an instance of the corresponding class. For example, `groverObject = Grover()`.
- Create your function `f` that takes an integer `x` as a parameter such that `log(x) < n`, where `n` is the number of input bits to your function. For example, instead of passing the binary integer `110` as input to `f`, one would pass the decimal integer `4`.
- Create and execute the circuit by calling `xxxObject.run(f,n)`, where `n` is number of bits that `f` takes as input and `xxxObject` is your class instance. The return values are as follows:
  - Deutsch-Jozsa: `1` if the input function `f` is constant and `0` if `f` is balanced.
  - Bernstein-Vazirani: the value `(a)`, where `f(x) = a*x + b`.
  - Simon's: the value `s` such that for any `x`, `y`, `f(x) = f(y)` iff `x XOR y = s`
  - Grover's: the integer value `x` such that `f(x) = 1`, where `f` is the input function.


Additional notes on Simon's: 
Simon's Algorithm, in order to ensure that the function remains a black-box, doesn't accept the function itself. Instead, it accepts a python dictionary that maps all possible inputs (as bitstrings [e.g. `0001`]) to their respective outputs (in the same format). 
Additionally, here are instructions to run the test file simontest.py:

- This file does not accept command-line arguments. Instead, to change the size of the mask (and thus the number of qubits) change the value of `N` at the head of the main function.
