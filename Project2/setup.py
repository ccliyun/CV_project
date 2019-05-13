from distutils.core import setup
from Cython.Build import cythonize
setup(
    name='pro2 pyx',
    ext_modules=cythonize('my_pro2.pyx')
)