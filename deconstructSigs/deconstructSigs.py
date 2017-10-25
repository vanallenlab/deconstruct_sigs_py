import re
import scipy
from scipy.stats import ttest_ind

import numpy as np
import glob
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from datetime import datetime
from collections import OrderedDict, defaultdict
import itertools
import scipy.stats
import random
from matplotlib import ticker
import math
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import normalize


class DeconstructSigs:
    # base pairs dict
    pair = {
        'A': 'T',
        'C': 'G',
        'T': 'A',
        'G': 'C'
    }

    def __init__(self, mafs_folder=None, maf_file_path=None):
        self.num_samples = 0
        self.mafs_folder = mafs_folder
        self.maf_filepath = maf_file_path

        self.cosmic_signatures_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                       'data/signatures_probabilities.txt')

        # A dictionary to keep track of the SNVs and the context in which SNVs occurred in order to
        # build a mutational signature for the samples
        self.subs_dict = defaultdict(lambda: defaultdict(int))

        self.__load_cosmic_signatures()
        self.__load_mafs()

    def get_num_samples(self):
        return self.num_samples

    def __load_cosmic_signatures(self):
        self.cosmic_signatures = pd.read_csv('{}'.format(self.cosmic_signatures_filepath), sep='\t', engine='python')

    def __load_mafs(self):
        if self.mafs_folder:
            for filename in [n for n in os.listdir(self.mafs_folder) if n.endswith('maf')]:
                file_path = '{}/{}'.format(self.mafs_folder, filename)
                self.__load_maf(file_path)
        elif self.maf_filepath:
            self.__load_maf(self.maf_filepath)

    def __load_maf(self, file_path):
        df = pd.read_csv(file_path, sep='\t', engine='python')
        for (idx, row) in df.iterrows():
            ref_context = row.ref_context  # context is the ref flanked by 10 bp on both the 5' and 3' sides
            trinuc_context = self.__standardize_trinuc(ref_context[9:12])
            if len(ref_context) == 21:
                # Only consider SNPs (ignoring DNPs, and TNPs)
                substitution = self.__standardize_subs(row.Reference_Allele, row.Tumor_Seq_Allele2)
                assert (trinuc_context[1] == substitution[0])
                assert (substitution[0] in ['C', 'T'])
                assert (trinuc_context[1] in ['C', 'T'])
                self.subs_dict[substitution][trinuc_context] += 1
        self.num_samples += 1

    # A function that converts substitutions into their pyrimidine-based notation. Only C and T ref alleles.
    def __standardize_subs(self, ref, alt):
        if ref in ['G', 'A']:
            return '{}>{}'.format(DeconstructSigs.pair[ref], DeconstructSigs.pair[alt])
        else:
            return '{}>{}'.format(ref, alt)

    # A function that ensures trinucleotide contexts are centered around a pyrimidine, using complementary
    # sequence to achieve this if necessary.
    def __standardize_trinuc(self, trinuc):
        trinuc = trinuc.upper()
        if trinuc[1] in ['G', 'A']:
            return '{}{}{}'.format(DeconstructSigs.pair[trinuc[0]],
                                   DeconstructSigs.pair[trinuc[1]],
                                   DeconstructSigs.pair[trinuc[2]])
        else:
            return trinuc
