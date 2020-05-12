# Use:

The following applies to each of the Deutsch-Jozsa, Bernstein-Vazirani, Simon's, and Grover's Algorithms:
- Create an an instance of the corresponding class. For example, `groverObject = Grover()`.
- Create your function `f` that takes an integer `x` as a parameter such that `log(x) < n`, where `n` is the number of input bits to your function. For example, instead of passing the binary integer `110` as input to `f`, one would pass the decimal integer `4`.
- Create and execute the circuit by calling `xxxObject.run(f,n)`, where `n` is number of bits that `f` takes as input and `xxxObject` is your class instance. The return values are as follows:
  - Deutsch-Jozsa: `1` if the input function `f` is constant and `0` if `f` is balanced.
  - Bernstein-Vazirani: a tuple `(a, b)`, where `f(x) = a*x + b`
  - Simon's:
  - Grover's: the integer value `x` such that `f(x) = 1`, where `f` is the input function.
