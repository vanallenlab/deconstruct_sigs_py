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

    def __init__(self, mafs_folder=None, maf_file_path=None, context_counts=None, verbose=False, cutoff=0.06):
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

    def which_signatures(self, signatures_limit=None):
        w = self.__which_signatures(signatures_limit=signatures_limit)
        self.__print_normalized_weights(w)

        flat_bins, flat_counts = self.__get_alphabetical_flat_bins_and_counts()
        T = np.array(flat_counts)
        reconstructed_tumor_profile = self.__get_reconstructed_tumor_profile(self.S, w)
        self.__plot_counts(flat_bins, reconstructed_tumor_profile*sum(T), title='Reconstructed Tumor Profile')
        self.plot_sample_profile()
        plt.show()

    def __which_signatures(self, signatures_limit=None):
        """Get the weights transformation vector"""
        # If no signature limit is provided, simply set it to the number of signatures
        if signatures_limit is None:
            signatures_limit = len(self.S)

        # Remove signatures from possibilities if they have a "strong" peak for a context that
        # is not seen in the tumor sample
        ignorable_signatures = self.__calculate_ignorable_signatures()
        self.__status('Signatures ignored because of outlying contexts: {}')
        for s in ignorable_signatures:
            self.__status('{} because of outlying context {} with fraction {}'.format(s.get('name'),
                                                                                      s.get('outlier_context'),
                                                                                      s.get('context_fraction')))

        ignorable_indices = [ig['index'] for ig in ignorable_signatures]
        iteration = 0
        _, flat_counts = self.__get_alphabetical_flat_bins_and_counts()

        T = np.array(flat_counts) / sum(flat_counts)
        # Normalize the tumor data
        w = self.__seed_weights(T, self.S)
        error_diff = math.inf
        error_threshold = 1e-3
        while error_diff > error_threshold:
            self.__print_normalized_weights(w)
            iteration = iteration + 1
            error_pre = self.__get_error(T, self.S, w)
            if error_pre == 0:
                break
            self.__status("Iter {}:\n\t Pre error: {}".format(iteration, error_pre))
            w = self.__updateW_GR(T, self.S, w,
                                  signatures_limit=signatures_limit,
                                  ignorable_signature_indices=ignorable_indices)
            error_post = self.__get_error(T, self.S, w)
            self.__status("\t Post error: {}".format(error_post))
            error_diff = (error_pre - error_post) / error_pre
        return w / sum(w)

    def __print_normalized_weights(self, w):
        normalized_weights = w / sum(w)
        for i, weight in enumerate(normalized_weights):
            if weight != 0:
                sys.stdout.write("{}: {}\n".format(self.signature_names[i], weight))

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

    def __status(self, text):
        if self.verbose:
            sys.stdout.write('{}\n'.format(text))

    def __get_flat_bins_and_counts(self):
        flat_bins = []
        flat_counts = []
        for subs, context_counts in self.subs_dict.items():
            for context in sorted(context_counts):
                flat_bins.append(context)
                flat_counts.append(context_counts[context])
        return flat_bins, flat_counts

    def __get_alphabetical_flat_bins_and_counts(self):
        context_dict = defaultdict()
        for subs, context_counts in self.subs_dict.items():
            for context in context_counts:
                context_dict['{}[{}]{}'.format(context[0], subs, context[2])] = context_counts[context]

        flat_bins = []
        flat_counts = []
        for context in sorted(context_dict):
            flat_bins.append(context)
            flat_counts.append(context_dict.get(context))
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
    def __get_reconstructed_tumor_profile(signatures, w):
        w_norm = w/sum(w)
        return w_norm.dot(np.transpose(signatures))

    def __get_error(self, T, signatures, w, verbose=False):
        """
        Calculate the SSE between the true tumor signature and the calculated linear combination of different signatures
        """
        T = T/sum(T)
        reconstructed_tumor_profile = self.__get_reconstructed_tumor_profile(signatures, w)
        error = T - reconstructed_tumor_profile
        squared_error_sum = np.sum(error.dot(np.transpose(error)))
        return squared_error_sum

    def __updateW_GR(self, tumor, signatures, w, signatures_limit, bound=100, ignorable_signature_indices=None):
        if ignorable_signature_indices is None:
            ignorable_signature_indices = []

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
            tmp_weights = np.zeros((30,))
            tmp_weights[i] = 1
            error = self.__get_error(tumor, signatures, tmp_weights, verbose=True)
            ss_errors[i] = error
        # Seed index that minimizes sum of squared error metric
        seed_index = np.argmin(ss_errors, axis=0)
        #seed_index = random.randrange(30)
        #print('tumor', tumor[0:3])
        #print('ss errors', ss_errors)
        #print('seed index', seed_index)
        final_weights = np.zeros(30)
        final_weights[seed_index] = 1
        return final_weights
