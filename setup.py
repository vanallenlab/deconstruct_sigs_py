from distutils.core import setup

from setuptools import find_packages
my_packages=find_packages()
setup(name='deconstructSigs',
      description='Implementation of DeconstructSigs algorithm for deducing cancer genome mutational signatures',
      author='Eric Kofman',
      author_email='ericrkofman@gmail.com',
      version='1.46',
      py_modules=my_packages,
      url='https://github.com/vanallenlab/deconstruct_sigs_py',
      download_url='https://github.com/vanallenlab/deconstruct_sigs_py/archive/v1.46.tar.gz'
      )