# DeconstructSigs
Given a MAF or folder of MAF files, find the optimal linear combination of COSMIC mutational signatures that best
describe the sample's SNV trinucleotide context distribution.

Installation
------------
Click the green Clone or Download button above, copy the presented web URL, and then paste it into your terminal, preceded by the command 'git clone'.
  `git clone https://github.com/vanallenlab/deconstruct_sigs_py.git`

Parameters
----------
The following parameters can be provided to an DeconstructSigs object:

mafs_folder=None, maf_file_path=None, context_counts=None, cutoff=0.06,
                 outfile_path=None, analysis_handle=None, hg19_fasta_path=None


* `mafs_folder`: Optional path to a folder containing multiple MAF file. If provided, analysis will be conducted on all MAF files within the given folder.

* `maf_file_path`: Optional path to a single MAF file. If provided, analysis will be conducted on this file.

* `context_counts`: Optional. This argument can be used to provide a dictionary of context counts rather than a MAF file. Keys are of the form 'A[C>A]A', 'A[C>A]C', etc., and values are integer counts.

* `cutoff`: Optional, default value of 0.06. The weights of all signatures calculated to be contributing less than the cutoff will be zeroed.

* `outfile_path`: Optional. If provided, calculated signature weights will be output here upon calling which_signatures().

* `analysis_handle`: Optional. If provided, analysis_handle will be used in the titles of all plots and figures generated.

* `hg19_fasta_path`: Optional. If provided, analysis will determine trinucleotide context by using samtools to search within provided fasta file for each SNP. Requires local installation of samtools, as samtools is run through subprocess. If not provided, DeconstructSigs assumes that the MAF file contains a ref_context column.

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


Data Sources
------------
Cosmic Signature Probabilities (signatures_probabilities.txt): http://cancer.sanger.ac.uk/cosmic/signatures
Signature Etiologies (about_cosmic_sigs.txt): http://cancer.sanger.ac.uk/cosmic/signatures, with updates added by Van Allen Lab