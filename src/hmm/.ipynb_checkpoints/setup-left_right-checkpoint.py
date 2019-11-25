from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(
  name = 'Baum Welch Code',
  ext_modules = cythonize("BaumWelchLR.pyx"),
  include_dirs = [numpy.get_include()]
)