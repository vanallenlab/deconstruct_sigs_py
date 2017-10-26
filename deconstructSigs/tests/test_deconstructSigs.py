import unittest
import numpy as np
import pandas as pd
import os
import re
from collections import defaultdict
from deconstructSigs.deconstructSigs import DeconstructSigs

class TestDeconstructsig(unittest.TestCase):
    def test_one_signature(self):
        weights = np.zeros((30,))
        weights[0] = 1
        tumor_profile, context_counts = generate_tumor_profile(weights)
        ds = DeconstructSigs(context_counts=context_counts)
        for key in sorted(context_counts):
            print('{}: {}'.format(key, context_counts[key]))
        reconstructed_weights = ds._DeconstructSigs__which_signatures(1)
        pass

"""Helpers to generate fake tumor profiles from known linear combinations of cosmic signatures for testing"""


def generate_tumor_profile(w):
    """
    Given an array of 30 weights, one for each cosmic signature,
    admix the signatures to generate the tumor profile
    """
    cosmic_signatures_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/signatures_probabilities.txt')
    cosmic_signatures = pd.read_csv('{}'.format(cosmic_signatures_filepath), sep='\t', engine='python')

    # Remove unnecessary columns from the cosmic signatures data and make the S matrix
    S = np.array(cosmic_signatures.select(
        lambda x: not re.search("(Substitution Type)|(Trinucleotide)|(Somatic Mutation Type)|(Unnamed)", x),
        axis=1))
    # Compute dot product
    T = w.dot(np.transpose(S)).round(3)
    trinuc_counts = generate_trinuc_counts_for_tumor(T, cosmic_signatures)
    return T, trinuc_counts


def generate_trinuc_counts_for_tumor(tumor_profile, cosmic_signatures):
    multiplier = 10000
    counts = defaultdict()
    for i, context_fraction in enumerate(tumor_profile):
        mutation_type=cosmic_signatures['Somatic Mutation Type'][i]
        counts[mutation_type] = round(context_fraction * multiplier, 1)
    return counts


if __name__ == '__main__':
    unittest.main()
