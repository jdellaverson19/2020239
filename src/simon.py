import inspect

from pyquil import get_qc, Program
from pyquil.gates import H, MEASURE
from pyquil.latex import *
import numpy as np

def quantum_DJ(f):