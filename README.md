# DeconstructSigs algorithm in python
See original repository: https://github.com/vanallenlab/deconstruct_sigs_py

# This fork:
1. Modified `setup.py` to allow installation via pip: `pip install git+https://github.com/pwwang/deconstruct_sigs_py`
2. Used `pyfaidx` to get context sequence of mutations from fasta file instead of `samtools` to speed up reading maf files.
3. Added `Settings` to set global settings:
   ```python
   from deconstructSigs import Settings, DeconstructSigs
   # font family used to plot
   Settings.font_family = 'Arial'
   # font weight used to plot
   Settings.font_weight = 'bold'
   # Print verbose information/logging
   Settings.verbose     = True
   # Cutoff below which calculated signatures will be discarded
   Settings.sig_cutoff  = 0.05
   # When the iteration stops
   Settings.err_thres   = 1e-3
   # logging format
   Settings.log_format  = '[%(asctime)s %(levelname)-.1s] %(message)s'
   Settings.log_time    = '%Y-%m-%d %H:%M:%S'
   ```
4. Combined `mafs_folder` and `maf_file_path` into a `maf` argument, which could be a directory containing maf files or a maf file itself.
5. Allowed gziped maf input files.
6. Allowed `'#'` as comment in maf files.
7. Formatted logging/verbose messages.
8. Made it compatible with python2 and python3.
