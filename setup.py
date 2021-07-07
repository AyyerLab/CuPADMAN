from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import glob
import subprocess
import numpy as np
import h5py
out = subprocess.getoutput('h5cc -shlib -show')
hdf5_cflags = [[s for s in out.split() if s[:2] == '-I'][0]]
hdf5_libs = [[s for s in out.split() if s[:2] == '-L'][0], '-lhdf5']

mpi_cflags = subprocess.getoutput('mpicc --showme:compile').strip().split()
mpi_libs = subprocess.getoutput('mpicc --showme:link').strip().split()

compile_args = '-fopenmp -O3 -Wall'.split()
compile_args += '-Wno-cpp -Wno-unused-result -Wno-unused-function -Wno-format-overflow'.split() 
compile_args += hdf5_cflags + ['-I'+np.get_include()]
link_args = '-lm -fopenmp'.split() + hdf5_libs

# TODO: Fix undefined symbol problem requiring two rounds of compilation
ext_modules = [
    Extension(name='cupadman.detector', sources=['cupadman/detector.pyx'],
        depends=['cupadman/detector.h', 'cupadman/detector.pxd'],
        language='c', extra_compile_args=compile_args, extra_link_args=link_args),
    Extension(name='cupadman.emcfile', sources=['cupadman/emcfile.pyx', 'cupadman/src/emcfile.c'],
        depends=['cupadman/src/emcfile.h', 'cupadman/emcfile.pxd'],
        language='c', extra_compile_args=compile_args, extra_link_args=link_args),
    Extension(name='cupadman.quaternion', sources=['cupadman/quaternion.pyx', 'cupadman/src/quaternion.c'],
        depends=['cupadman/src/quaternion.h', 'cupadman/quaternion.pxd'],
        language='c', extra_compile_args=compile_args, extra_link_args=link_args),
]
py_packages = [
    'cupadman',
]
extensions = cythonize(ext_modules, language_level=3,
                       compiler_directives={'embedsignature': True,
                                            'boundscheck': False,
                                            'wraparound': False,
                                            'cdivision': True,
                                            'nonecheck': False})

setup(name='cupadman',
      packages=py_packages,
      ext_modules=extensions,
      entry_points={'console_scripts': [
        'emc_cu = cupadman.emc:main',
        'make_data_cu = cupadman.make_data:main',
        ],
      },
)
