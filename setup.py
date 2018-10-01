from setuptools import setup, find_packages

setup (
	name             = 'deconstructSigs',
	version          = '1.48',
	description      = "Implementation of DeconstructSigs algorithm for deducing cancer genome mutational signatures",
	url              = "https://github.com/pwwang/deconstruct_sigs_py",
	author           = "Eric Kofman; pwwang",
	author_email     = "ericrkofman@gmail.com",
	packages         = find_packages(),
	data_files       = {'': ['deconstructSigs/data/about_cosmic_sigs.txt', 'deconstructSigs/data/signatures_probabilities.txt']},
	install_requires = [
		'pandas',
		'numpy',
		'matplotlib',
		'scipy'
	]
)
