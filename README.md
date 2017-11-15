# DeconstructSigs
Given a MAF or folder of MAF files, find the optimal linear combination of COSMIC mutational signatures that best
describe the sample's SNV trinucleotide context distribution.

A Python implementation of the DeconstructSigs algorithm described in
https://genomebiology.biomedcentral.com/articles/10.1186/s13059-016-0893-4. Modeled after the R implementation
coded by Rachel Rosenthal which can be found at https://github.com/raerose01/deconstructSigs.

From the GenomeBiology description:
    The deconstructSigs approach determines the linear combination of pre-defined signatures that most accurately
    reconstructs the mutational profile of a single tumor sample. It uses a multiple linear regression model with
    the caveat that any coefficient must be greater than 0, as negative contributions make no biological sense.

Installation
------------
Click the green Clone or Download button above, copy the presented web URL, and then paste it into your terminal, preceded by the command 'git clone'.
  `git clone https://github.com/vanallenlab/deconstruct_sigs_py.git`

Parameters
----------
The following parameters can be provided to an DeconstructSigs object:

* `mafs_folder`: Optional path to a folder containing multiple MAF file. If provided, analysis will be conducted on all MAF files within the given folder.

* `maf_file_path`: Optional path to a single MAF file. If provided, analysis will be conducted on this file.

* `context_counts`: Optional. This argument can be used to provide a dictionary of context counts rather than a MAF file. Keys are of the form 'A[C>A]A', 'A[C>A]C', etc., and values are integer counts.

* `cutoff`: Optional, default value of 0.06. The weights of all signatures calculated to be contributing less than the cutoff will be zeroed.

* `outfile_path`: Optional. If provided, calculated signature weights will be output here upon calling which_signatures().

* `analysis_handle`: Optional. If provided, analysis_handle will be used in the titles of all plots and figures generated.

* `hg19_fasta_path`: Optional. If provided, analysis will determine trinucleotide context by using samtools to search within provided fasta file for each SNP. Requires local installation of samtools, as samtools is run through subprocess. If not provided, DeconstructSigs assumes that the MAF file contains a ref_context column.


The which_signatures method takes a few parameters as well:

* `signatures_limit`: Optional, default None. If provided, number of signatures allowed to contribute to solution is capped at signatures_limit. Otherwise up to 30 COSMIC signatures could potentially be used.

* `associated`: Optional, default None, list of integer indices of COSMIC signatures in range 0-29. Useful when it is known that only a pre-determined subset of COSMIC signatures should be tried.

* `verbose`: Optional, default False. If True then logs describing weight updates on each iteration will be output to stdout.

Usage
-----
```
from deconstructSigs.deconstructSigs import DeconstructSigs


def main():
    fasta_path = '/path/to/Homo_sapiens_assembly19.fasta'
    ds = DeconstructSigs(maf_file_path='/path/to/snvs.maf',
                         hg19_fasta_path=fasta_path)

    weights = ds.which_signatures(verbose=True)
    ds.plot_signatures(weights, explanations=True)

if __name__ == '__main__':
    main()
```

Output
------
Output using plot_signatures(weights, explanations=True):
* Tumor profile
![Tumor Profile](deconstructSigs/example_plots/tumor_profile.png)
* Reconstructed tumor profile
![Reconstructed Tumor Profile](deconstructSigs/example_plots/reconstructed_tumor_profile.png)
* COSMIC signatures breakdown
![Cosmic Signature Pie Chart](deconstructSigs/example_plots/cosmic_signature_pie.png)

Data Sources
------------
* Cosmic Signature Probabilities (signatures_probabilities.txt): http://cancer.sanger.ac.uk/cosmic/signatures
* Signature Etiologies (about_cosmic_sigs.txt): http://cancer.sanger.ac.uk/cosmic/signatures, with updates added by Van Allen Lab