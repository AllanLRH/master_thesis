import platform
import sys
import os
if platform.system() == 'Darwin':
    pth = '/Users/allan/scriptsMount'
elif platform.system() == 'Linux':
    pth = '/lscr_paper/allan/scripts'
else:
    raise OSError('Could not identify system "{}"'.format(platform.system()))
if not os.path.isdir(pth):
    raise FileNotFoundError(f"Could not find the file {pth}")
sys.path.append(pth)

from speclib.graph import genAllMatrixPermutations
import numpy as np


a = np.arange(16).reshape((4, -1))
ac = np.zeros_like(a)
af = a.flatten()
b = np.random.randint(0, 10, (4, 4))
bc = np.zeros_like(b)
bf = b.flatten()

gena = genAllMatrixPermutations(a, ac)
print(sorted([(ac.flatten() * bf).sum() for pa in gena]))

genb = genAllMatrixPermutations(b, bc)
print(sorted([(bc.flatten() * af).sum() for pb in genb]))
