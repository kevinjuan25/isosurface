from setuptools import setup, find_packages

setup(name='isosurface',\
	version='0.0.1',\
	author='Kevin Juan',\
	packages=find_packages(),\
	install_requires=["numpy", "scipy", "numba",\
	"MDAnalysis", "scikit-image", "networkx", "sklearn"])
