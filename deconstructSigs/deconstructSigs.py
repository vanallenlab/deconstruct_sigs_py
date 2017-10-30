import re
import numpy as np
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from collections import defaultdict
import itertools
from matplotlib import ticker
import math
from scipy.optimize import minimize_scalar


class DeconstructSigs:
    # base pairs dict
    pair = {
        'A': 'T',
        'C': 'G',
        'T': 'A',
        'G': 'C'
    }

    def __init__(self, mafs_folder=None, maf_file_path=None, context_counts=None, cutoff=0.06):
        self.num_samples = 0
        self.mafs_folder = mafs_folder
        self.maf_filepath = maf_file_path
        self.verbose = False
        self.signature_cutoff = cutoff

        self.cosmic_signatures_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                       'data/signatures_probabilities.txt')

        self.__setup_subs_dict()
        self.__load_cosmic_signatures()

        # Remove unnecessary columns from the cosmic signatures data and make the S matrix. Note: the substitution
        # contexts are in alphabetical order (A[C>A]A, A[C>A]C, A[C>A]G, A[C>A]T, A[C>G]A, A[C>G]C... etc.)
        self.S = np.array(self.cosmic_signatures.select(
            lambda x: not re.search("(Substitution Type)|(Trinucleotide)|(Somatic Mutation Type)|(Unnamed)", x),
            axis=1))

        if context_counts is None:
            self.__load_mafs()
        else:
            self.__add_context_counts_to_subs_dict(context_counts)

        self.signature_names = ['Signature 1', 'Signature 2', 'Signature 3', 'Signature 4', 'Signature 5',
                                'Signature 6', 'Signature 7', 'Signature 8', 'Signature 9', 'Signature 10',
                                'Signature 11', 'Signature 12', 'Signature 13', 'Signature 14', 'Signature 15',
                                'Signature 16', 'Signature 17', 'Signature 18', 'Signature 19', 'Signature 20',
                                'Signature 21', 'Signature 22', 'Signature 23', 'Signature 24', 'Signature 25',
                                'Signature 26', 'Signature 27', 'Signature 28', 'Signature 29', 'Signature 30']

    def which_signatures(self, signatures_limit=None, verbose=False):
        """Wrapper on __which_signatures function. Calls __which_signatures, then plots both the reconstructed
        tumor profile based on the calculated weights and the original tumor profile provided."""
        self.verbose = verbose
        w = self.__which_signatures(signatures_limit=signatures_limit)
        self.__plot_reconstructed_profile(w)
        self.plot_sample_profile()
        plt.show()
        self.verbose = False

    def get_num_samples(self):
        """Return the number of samples that has been loaded into this DeconstructSigs instance"""
        return self.num_samples

    def __add_context_counts_to_subs_dict(self, context_counts):
        """Contexts should be of the form C[A>G]T"""
        for context, count in context_counts.items():
            substitution = context[2:5]
            trinuc = '{}{}{}'.format(context[0], context[2], context[6])
            self.subs_dict[substitution][trinuc] = count

    def __calculate_ignorable_signatures(self):
        """
        Calculates which signatures can be ignored because they contain a peak for a context that is clearly not
        seen in the tumor data.
        :return: List of dicts of signatures to ignore along with contextual information for why.
        """
        somatic_mutation_counts = defaultdict(int)
        for subs, contexts in self.subs_dict.items():
            for context in sorted(contexts):
                count = self.subs_dict[subs][context]
                somatic_mutation_counts['{}[{}]{}'.format(context[0], subs, context[2])] = count
        total_counts = sum(somatic_mutation_counts.values())

        contexts_not_present_in_tumor = [sm for sm in somatic_mutation_counts
                                        if somatic_mutation_counts[sm]/total_counts < 0.01]

        signatures_to_ignore = []
        for i, signature_name in enumerate(self.signature_names):
            context_fractions = self.cosmic_signatures[signature_name]
            for j, cf in enumerate(context_fractions):
                if cf > 0.2:
                    somatic_mutation_type = self.cosmic_signatures['Somatic Mutation Type'][j]
                    if somatic_mutation_type in contexts_not_present_in_tumor:
                        signatures_to_ignore.append({'name': signature_name,
                                                     'index': i,
                                                     'outlier_context': somatic_mutation_type,
                                                     'context_fraction': cf})
                    break
        return signatures_to_ignore

    def __plot_reconstructed_profile(self, weights):
        """Given a set of weights for each signature plot the reconstructed tumor profile using the cosmic signatures"""
        reconstructed_tumor_profile = self.__get_reconstructed_tumor_profile(self.S, weights)

        # Reorder context counts, which were calculated using alphabetically sorted mutation contexts, to match the
        # format that the plotting function expects, where they are ordered alphabetically first by substitution type.
        alpha_flat_subs, _, alpha_flat_counts = self.__get_alphabetical_flat_bins_and_counts()
        total_tumor_substitutions = sum(np.array(alpha_flat_counts))
        reconstructed_counts_dict = defaultdict()
        for i, subs_type in enumerate(alpha_flat_subs):
            reconstructed_counts_dict[subs_type] = reconstructed_tumor_profile[i] * total_tumor_substitutions
        reconstructed_tumor_counts = []
        flat_subs, flat_bins, _ = self.__get_plottable_flat_bins_and_counts()
        for subs_type in flat_subs:
            reconstructed_tumor_counts.append(reconstructed_counts_dict[subs_type])
        self.__plot_counts(flat_bins, reconstructed_tumor_counts, title='Reconstructed Tumor Profile')

    def __which_signatures(self, signatures_limit=None):
        """Get the weights transformation vector"""
        # If no signature limit is provided, simply set it to the number of signatures
        if signatures_limit is None:
            signatures_limit = len(self.S)

        # Remove signatures from possibilities if they have a "strong" peak for a context that
        # is not seen in the tumor sample
        ignorable_signatures = self.__calculate_ignorable_signatures()
        self.__status('Signatures ignored because of outlying contexts:')
        for s in ignorable_signatures:
            self.__status('\t{} ignored because of outlying context {} with fraction {}'
                          .format(s.get('name'),
                                  s.get('outlier_context'),
                                  s.get('context_fraction')))

        ignorable_indices = [ig['index'] for ig in ignorable_signatures]
        iteration = 0
        _, _, flat_counts = self.__get_alphabetical_flat_bins_and_counts()

        # Normalize the tumor data
        T = np.array(flat_counts) / sum(flat_counts)

        w = self.__seed_weights(T, self.S)
        self.__status("Initial seed weights assigned:")
        self.__print_normalized_weights(w)

        error_diff = math.inf
        error_threshold = 1e-3
        while error_diff > error_threshold:
            iteration = iteration + 1
            error_pre = self.__get_error(T, self.S, w)
            if error_pre == 0:
                break
            self.__status("Iter {}:\n\tPre error: {}".format(iteration, error_pre))
            w = self.__update_weights(T, self.S, w,
                                      signatures_limit=signatures_limit,
                                      ignorable_signature_indices=ignorable_indices)
            error_post = self.__get_error(T, self.S, w)
            self.__status("\tPost error: {}".format(error_post))
            error_diff = (error_pre - error_post) / error_pre
            self.__status("\tNew normalized weights: ")
            self.__print_normalized_weights(w)

        normalized_weights = w/sum(w)

        # Filter out any weights less than 0.6
        np.place(normalized_weights, normalized_weights < self.signature_cutoff, 0)
        return normalized_weights

    def __print_normalized_weights(self, w):
        """A standard way to print normalized weights given a vector of potentially not yet normalized weights"""
        normalized_weights = w / sum(w)
        for i, weight in enumerate(normalized_weights):
            if weight != 0:
                self.__status("\t\t{}: {}".format(self.signature_names[i], weight))

    def plot_sample_profile(self):
        """Plot the substitution context profile for the original tumor sample given"""
        _, flat_bins, flat_counts = self.__get_plottable_flat_bins_and_counts()
        self.__plot_counts(flat_bins, flat_counts, title='SNP Counts by Trinucleotide Context')

    def __plot_counts(self, flat_bins, flat_counts, title='Figure'):
        """Plot subsitution counts per mutation context"""
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

    def __status(self, text):
        if self.verbose:
            sys.stdout.write('{}\n'.format(text))

    def __get_plottable_flat_bins_and_counts(self):
        """Return the tumor context counts in the order in which they are to be plotted."""
        flat_bins = []
        flat_counts = []
        flat_subs = []
        for subs, context_counts in self.subs_dict.items():
            for context in sorted(context_counts):
                flat_bins.append(context)
                flat_counts.append(context_counts[context])
                flat_subs.append('{}[{}]{}'.format(context[0], subs, context[2]))
        return flat_subs, flat_bins, flat_counts

    def __get_alphabetical_flat_bins_and_counts(self):
        """Return the tumor context counts in alphabetical order, as is used by the cosmic signatures file"""
        context_dict = defaultdict()
        for subs, context_counts in self.subs_dict.items():
            for context in context_counts:
                context_dict['{}[{}]{}'.format(context[0], subs, context[2])] = context_counts[context]

        flat_bins = []
        flat_counts = []
        for context in sorted(context_dict):
            flat_bins.append(context)
            flat_counts.append(context_dict.get(context))
        return sorted(context_dict), flat_bins, flat_counts

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
        """Load cosmic signatures file. Note that the mutation contexts are listed in alphabetical order:
        (A[C>A]A, A[C>A]C, A[C>A]G, A[C>A]T, A[C>G]A, A[C>G]C... etc.) """
        self.cosmic_signatures = pd.read_csv('{}'.format(self.cosmic_signatures_filepath), sep='\t', engine='python')

    def __load_mafs(self):
        """Load all *.maf files found in the directory provided"""
        if self.mafs_folder:
            for filename in [n for n in os.listdir(self.mafs_folder) if n.endswith('maf')]:
                file_path = '{}/{}'.format(self.mafs_folder, filename)
                self.__load_maf(file_path)
        elif self.maf_filepath:
            self.__load_maf(self.maf_filepath)

    def __load_maf(self, file_path):
        """Load a MAF file's trinucleotide counts for each type of substitution"""
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
        A function that converts substitutions into their pyrimidine-based notation. Only C and T ref alleles.
        :param ref: The reference allele
        :param alt: The alternate allele
        :return If reference allele is pyrimidine, returns string in format 'ref>alt.' Otherwise, returns string in
        format 'ref_complement_base>alt_complement>base' such that the ref is always a pyrimidine in the return value.
        """
        if ref in ['G', 'A']:
            return '{}>{}'.format(DeconstructSigs.pair[ref], DeconstructSigs.pair[alt])
        else:
            return '{}>{}'.format(ref, alt)

    def __standardize_trinuc(self, trinuc):
        """
        A function that ensures trinucleotide contexts are centered around a pyrimidine, using complementary
        sequence to achieve this if necessary.
        :param trinuc: A string representing a trinucleotide context, e.g. 'ACT' or 'GAT'
        :return: An uppercase representation of the given trinucleotide if the center base pair is a pyrimidine,
        otherwise an uppercase representation of the complementary sequence to the given trinucleotide.
        """
        trinuc = trinuc.upper()
        if trinuc[1] in ['G', 'A']:
            return '{}{}{}'.format(DeconstructSigs.pair[trinuc[0]],
                                   DeconstructSigs.pair[trinuc[1]],
                                   DeconstructSigs.pair[trinuc[2]])
        else:
            return trinuc

    @staticmethod
    def __get_reconstructed_tumor_profile(signatures, w):
        """Reconstruct a tumor profile given a set of signatures and a vector of signature weights"""
        w_norm = w/sum(w)
        return w_norm.dot(np.transpose(signatures))

    def __get_error(self, tumor, signatures, w):
        """
        Calculate the SSE between the true tumor signature and the calculated linear combination of different signatures
        :param tumor: normalized array of shape (1, 96) where each entry is a mutation context fraction for the tumor
        :param signatures: array of shape (96, num_signatures) where each row represents a mutation context and each
        column is a signature
        :param w: array of shape (num_signatures, 1) representing weight of each signature
        :return: sum of squares error between reconstructed tumor context fractions and actual tumor profile
        """
        tumor = tumor/sum(tumor)
        reconstructed_tumor_profile = self.__get_reconstructed_tumor_profile(signatures, w)
        error = tumor - reconstructed_tumor_profile
        squared_error_sum = np.sum(error.dot(np.transpose(error)))
        return squared_error_sum

    def __update_weights(self, tumor, signatures, w, signatures_limit, ignorable_signature_indices=None):
        """
        Given a set of initial weights, update the weights array with new values that shrink the sum of squares
        error metric.
        :param tumor: normalized array of shape (1, 96) where each entry is a mutation context fraction for the tumor
        :param signatures: signatures: array of shape (96, num_signatures) where each row represents a mutation context
        and each column is a signature
        :param w: array of shape (num_signatures, 1) representing weight of each signature
        :param signatures_limit: How many of the total signatures to consider when assigning weights
        :param ignorable_signature_indices: an array of indices into the signatures array indicating which to ignore
        :return: a new weights array, w.
        """
        if ignorable_signature_indices is None:
            ignorable_signature_indices = []

        # The number of signatures already being used in the current linear combination of signatures
        num_sigs_present = len([weight for weight in w if weight != 0])

        # The total number of signatures to choose from
        num_sigs = np.shape(signatures)[1]

        # The current sum of squares error given the present weights assigned for each signature
        error_old = self.__get_error(tumor, signatures, w)

        # Which weight indices to allow changes for; if we haven't reached the limit all weights are fair game
        if num_sigs_present < signatures_limit:
            changeable_indices = range(num_sigs)
        else:
            # Work with signatures already present if we have reached maximum number of contributing signatures allowed
            changeable_indices = np.nonzero(w)[0]
        changeable_indices = [i for i in changeable_indices if i not in ignorable_signature_indices]

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

            error_minimizer = minimize_scalar(to_minimize, bounds=(-w[i], 1), method="bounded").x
            v[i, i] = error_minimizer
            w_new = w + v[i]
            new_squared_errors[i] = self.__get_error(tumor, signatures, w_new)

        # Find which signature can be added to the weights vector to best reduce the error
        min_new_squared_error = min(new_squared_errors)

        # Update that signature within the weights vector with the new value that best reduces the overall error
        if min_new_squared_error < error_old:
            index_of_min = np.argmin(new_squared_errors, axis=0)
            w[index_of_min] = w[index_of_min] + v[index_of_min, index_of_min]

        return w

    def __seed_weights(self, tumor, signatures):
        """
        Find which of the cosmic signatures best approximates the tumor signature, and seed the weights such that that
        signature is assigned weight 1 and all other signatures are assigned weight zero. These are the seed weights
        upon which the algorithm will build as it tries to further reduce sum of squared error.
        :param tumor: normalized array of shape (1, 96) where each entry is a mutation context fraction
        :param signatures: array of shape (96, num_signatures) where each row represents a mutation context and each
        column is a signature
        :return: normalized array of shape (num_signatures, 1) representing weight of each signature
        """
        num_sigs = len(signatures[0])
        ss_errors = np.empty(num_sigs, )
        ss_errors.fill(math.inf)
        for i in range(num_sigs):
            tmp_weights = np.zeros((num_sigs,))
            tmp_weights[i] = 1
            error = self.__get_error(tumor, signatures, tmp_weights)
            ss_errors[i] = error
        # Seed index that minimizes sum of squared error metric
        seed_index = np.argmin(ss_errors, axis=0)
        final_weights = np.zeros(num_sigs)
        final_weights[seed_index] = 1
        return final_weights
