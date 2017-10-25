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

    def __init__(self, mafs_folder=None, maf_file_path=None, verbose=False):
        self.num_samples = 0
        self.mafs_folder = mafs_folder
        self.maf_filepath = maf_file_path
        self.verbose = verbose

        self.cosmic_signatures_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                       'data/signatures_probabilities.txt')

        self.__setup_subs_dict()
        self.__load_cosmic_signatures()

        # Remove unnecessary columns from the cosmic signatures data and make the S matrix
        self.S = np.array(self.cosmic_signatures.select(
            lambda x: not re.search("(Substitution Type)|(Trinucleotide)|(Somatic Mutation Type)|(Unnamed)", x),
            axis=1))

        self.__load_mafs()

        print('subs dict', self.subs_dict['C>A'])
        self.signature_names = ['Signature 1', 'Signature 2', 'Signature 3', 'Signature 4', 'Signature 5',
                                'Signature 6', 'Signature 7', 'Signature 8', 'Signature 9', 'Signature 10',
                                'Signature 11', 'Signature 12', 'Signature 13', 'Signature 14', 'Signature 15',
                                'Signature 16', 'Signature 17', 'Signature 18', 'Signature 19', 'Signature 20',
                                'Signature 21', 'Signature 22', 'Signature 23', 'Signature 24', 'Signature 25',
                                'Signature 26', 'Signature 27', 'Signature 28', 'Signature 29', 'Signature 30']

    def which_signatures(self, signatures_limit=None):
        # If no signature limit is provided, simply set it to the number of signatures
        if signatures_limit is None:
            signatures_limit = len(self.S)

        iteration = 0
        _, flat_counts = self.__get_flat_bins_and_counts()
        print('flat counts', flat_counts[0:3])
        tumor = np.array(flat_counts)
        # Normalize the tumor data
        T = tumor / tumor.max(axis=0)
        w = self.__seed_weights(T, self.S)
        error_diff = math.inf
        error_threshold = 1e-3
        while error_diff > error_threshold:
            iteration = iteration + 1
            error_pre = self.__get_error(T, self.S, w)
            self.__status("Iter {}:\n\t Pre error: {}\n".format(iteration, error_pre))
            w = self.__updateW_GR(T, self.S, w, signatures_limit=signatures_limit)
            error_post = self.__get_error(T, self.S, w)
            self.__status("\t Post error: {}\n".format(error_post))
            error_diff = (error_pre - error_post) / error_pre

        normalized_weights = w / sum(w)
        for i, weight in enumerate(normalized_weights):
            if weight != 0:
                sys.stdout.write("{}: {}\n".format(self.signature_names[i], weight))

        flat_bins, _ = self.__get_flat_bins_and_counts()
        reconstructed_tumor_norm = self.__get_reconstructed_tumor(self.S, w)
        self.__plot_counts(flat_bins, reconstructed_tumor_norm*max(tumor), title='Reconstructed Tumor Profile')

    def get_num_samples(self):
        return self.num_samples

    def plot_sample_profile(self):
        # Minor labels for trinucleotide bins and respective counts in flat arrays
        flat_bins, flat_counts = self.__get_flat_bins_and_counts()
        self.__plot_counts(flat_bins, flat_counts, title='SNP Counts by Trinucleotide Context')

    def __plot_counts(self, flat_bins, flat_counts, title='Figure'):
        # Set up several subplots
        fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(20, 5.5))
        fig.canvas.set_window_title(title)

        # Set up some colors and markers to cycle through...
        colors = itertools.cycle(['#22bbff', 'k', 'r', '.6', '#88cc44', '#ffaaaa'])

        graph = 0
        max_counts = max(flat_counts)*1.05
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

    def __status(self, text):
        if self.verbose:
            sys.stdout.write(text)

    def __get_flat_bins_and_counts(self):
        flat_bins = []
        flat_counts = []
        for subs, contexts in self.subs_dict.items():
            for context in sorted(contexts):
                flat_bins.append(context)
                flat_counts.append(contexts[context])
        return flat_bins, flat_counts

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

    @staticmethod
    def __get_reconstructed_tumor(signatures, w):
        w_norm = w / sum(w)
        return w_norm.dot(np.transpose(signatures))

    def __get_error(self, T, signatures, w, verbose=False):
        """
        Calculate the SSE between the true tumor signature and the calculated linear combination of different signatures
        """
        reconstructed_tumor = self.__get_reconstructed_tumor(signatures, w)
        error = T - reconstructed_tumor
        squared_error_sum = np.sum(error.dot(np.transpose(error)))
        return squared_error_sum

    def __updateW_GR(self, tumor, signatures, w, signatures_limit, bound=100):
        # The number of signatures already being used in the current linear combination of signatures
        num_sigs_present = len([weight for weight in w if weight != 0])

        # The total number of signatures to choose from
        num_sigs = np.shape(signatures)[1]

        # The current sum of squares error given the present weights assigned for each signature
        error_old = self.__get_error(tumor, signatures, w)

        # Which weight indices to allow changes for
        if num_sigs_present < signatures_limit:
            # If we haven't reached the limit we can test adjusting all weights
            changeable_indices = range(num_sigs)
        else:
            # Work with the signatures already present if we have reached our maximum number
            # of contributing signatures allowed
            changeable_indices = np.nonzero(w)[0]

        # zero square matrix of num signatures dimensions
        v = np.zeros((num_sigs, num_sigs))

        # 1 * num signatures vector with values preset to infinity
        new_squared_errors = np.empty(num_sigs, )
        new_squared_errors.fill(math.inf)

        # Only consider adjusting the weights which are allowed to change
        for i in changeable_indices:
            # Find the weight x for the ith signature that minimizes the sum of squared error
            def to_minimize(x):
                # Initialize a temporary zero vector of length number of signatures
                tmp = np.zeros((1, num_sigs))
                tmp[0, i] = x
                return self.__get_error(tumor, signatures, w + tmp[0,])

            error_minimizer = minimize_scalar(to_minimize, bounds=(-w[i], bound), method="bounded").x
            v[i, i] = error_minimizer
            w_new = w + v[i]
            new_squared_errors[i] = self.__get_error(tumor, signatures, w_new)

        # Find which signature can be added to the weights vector to best reduce the error
        min_new_squared_error = min(new_squared_errors)
        index_of_min = np.argmin(new_squared_errors, axis=0)

        # Update that signature within the weights vector with the new value that best reduces the overall error
        if min_new_squared_error < error_old:
            w[index_of_min] = w[index_of_min] + v[index_of_min, index_of_min]

        return w

    def __seed_weights(self, tumor, signatures):
        ss_errors = np.empty(30, )
        ss_errors.fill(math.inf)
        for i in range(30):
            tmp_weights = np.zeros(30)
            tmp_weights[i] = 1
            error = self.__get_error(tumor, signatures, tmp_weights, verbose=True)
            ss_errors[i] = error
        # Seed index that minimizes sum of squared error metric
        seed_index = np.argmin(ss_errors, axis=0)
        print('tumor', tumor[0:3])
        print('ss errors', ss_errors)
        print('seed index', seed_index)
        final_weights = np.zeros(30)
        final_weights[seed_index] = 1
        return final_weights
