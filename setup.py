from distutils.core import setup, find_packages

setup(name='deconstructSigs',
      description='Implementation of DeconstructSigs algorithm for deducing cancer genome mutational signatures',
      author='Eric Kofman',
      author_email='ericrkofman@gmail.com',
      version='1.48',
	packages= find_packages(),
      url='https://github.com/vanallenlab/deconstruct_sigs_py',
      download_url='https://github.com/vanallenlab/deconstruct_sigs_py/archive/v1.47.tar.gz'
      )
