import re
import scipy
from scipy.stats import ttest_ind

import numpy as np
import glob
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
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

        self.__setup_subs_dict()
        self.__load_cosmic_signatures()
        self.__load_mafs()

    def __setup_subs_dict(self):
        """
        A dictionary to keep track of the SNVs and the trinucleotide context in which SNVs occurred in order to
        build a mutational signature for the samples
        """
        self.subs_dict = defaultdict(lambda: defaultdict(int))
        for sub in [self.__standardize_subs('C', 'A'),
                    self.__standardize_subs('C', 'G'),
                    self.__standardize_subs('C', 'T'),
                    self.__standardize_subs('T', 'A'),
                    self.__standardize_subs('T', 'C'),
                    self.__standardize_subs('T', 'G')]:
            ref = sub[0]
            for left_bp in ['A', 'C', 'T', 'G']:
                for right_bp in ['A', 'C', 'T', 'G']:
                    trinuc_context = '{}{}{}'.format(left_bp, ref, right_bp)
                    self.subs_dict[sub][trinuc_context] = 0

    def get_num_samples(self):
        return self.num_samples

    def plot_counts(self):
        # Set up several subplots
        fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(20, 5.5))
        fig.canvas.set_window_title('SNP Counts by Trinucleotide Context')

        # Set up some colors and markers to cycle through...
        colors = itertools.cycle(['#22bbff', 'k', 'r', '.6', '#88cc44', '#ffaaaa'])

        # Minor labels for trinucleotide bins and respective counts in flat arrays
        flat_bins = []
        flat_counts = []
        for subs, contexts in self.subs_dict.items():
            for context in sorted(contexts):
                flat_bins.append(context)
                flat_counts.append(contexts[context])

        graph = 0
        max_counts = max(flat_counts)+5
        for ax, data, color in zip(axes, [1, 2, 3, 4, 5, 6], colors):
            x = np.arange(16) - 10
            counts = flat_counts[0 + graph * 16:16 + graph * 16]
            ax.bar(x, counts, color=color, width=.5, align='edge')
            # Labels for the rectangles
            new_ticks = flat_bins[0 + graph * 16:16 + graph * 16]
            start, end = ax.get_xlim()
            ax.xaxis.set_ticks(np.arange(start, end, (end - start) / 16.5) + .85)
            ax.xaxis.set_major_formatter(ticker.FixedFormatter(new_ticks))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
            # Standardize y-axis ranges across graphs
            ax.set_ylim([0, max_counts])
            graph += 1

        # Set labels
        axes[0].set_ylabel('Count')
        labels = sorted(self.subs_dict.keys())
        for ax, label in zip(axes, labels):
            ax.set_xlabel(label)

        # Remove boundaries between subplots
        for ax in axes[1:]:
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.set_yticklabels([])
        axes[0].spines['right'].set_color('none')

        # Leave off last x-tick to reduce clutter.
        for ax in axes:
            xticks = ax.get_xticks()
            ax.set_xticks(xticks[0:-1])

        # Merge subplots together so that they look like one graph
        fig.subplots_adjust(wspace=-.03)

        plt.show()


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
        """Load a MAF file's trinucleotide counts."""
        df = pd.read_csv(file_path, sep='\t', engine='python')
        for (idx, row) in df.iterrows():
            ref_context = row.ref_context  # context is the ref flanked by 10 bp on both the 5' and 3' sides
            trinuc_context = self.__standardize_trinuc(ref_context[9:12])
            # Only consider SNPs (ignoring DNPs, and TNPs, which would have 22 and 23 context length respectively)
            if len(ref_context) == 21:
                substitution = self.__standardize_subs(row.Reference_Allele, row.Tumor_Seq_Allele2)
                assert (trinuc_context[1] == substitution[0])
                assert (substitution[0] in ['C', 'T'])
                assert (trinuc_context[1] in ['C', 'T'])
                self.subs_dict[substitution][trinuc_context] += 1
        self.num_samples += 1

    def __standardize_subs(self, ref, alt):
        """
        :param ref:
        A function that converts substitutions into their pyrimidine-based notation. Only C and T ref alleles."""
        if ref in ['G', 'A']:
            return '{}>{}'.format(DeconstructSigs.pair[ref], DeconstructSigs.pair[alt])
        else:
            return '{}>{}'.format(ref, alt)

    def __standardize_trinuc(self, trinuc):
        """
        :param trinuc:
        :return:
        A function that ensures trinucleotide contexts are centered around a pyrimidine, using complementary
        # sequence to achieve this if necessary.
        """
        trinuc = trinuc.upper()
        if trinuc[1] in ['G', 'A']:
            return '{}{}{}'.format(DeconstructSigs.pair[trinuc[0]],
                                   DeconstructSigs.pair[trinuc[1]],
                                   DeconstructSigs.pair[trinuc[2]])
        else:
            return trinuc
