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

        # Limit to one deconstructed signatures
        reconstructed_weights = ds._DeconstructSigs__which_signatures(1)
        self.assertAlmostEqual(reconstructed_weights[0], weights[0], places=2)

        # Don't impose a signature limit to test that algorithm correctly deduces only one are necessary
        reconstructed_weights = ds._DeconstructSigs__which_signatures()
        self.assertAlmostEqual(reconstructed_weights[0], weights[0], places=2)

    def test_two_signatures(self):
        weights = np.zeros((30,))
        weights[0] = .5
        weights[5] = .5
        tumor_profile, context_counts = generate_tumor_profile(weights)
        ds = DeconstructSigs(context_counts=context_counts)

        # Limit to two deconstructed signatures
        reconstructed_weights = ds._DeconstructSigs__which_signatures(2)
        self.assertAlmostEqual(reconstructed_weights[0], .5, places=2)
        self.assertAlmostEqual(reconstructed_weights[5], .5, places=2)

        # Don't impose a signature limit to test that algorithm correctly deduces only two are necessary
        reconstructed_weights = ds._DeconstructSigs__which_signatures()
        self.assertAlmostEqual(reconstructed_weights[0], .5, places=2)
        self.assertAlmostEqual(reconstructed_weights[5], .5, places=2)

    def test_three_signatures(self):
        weights = np.zeros((30,))
        weights[3] = .6
        weights[11] = .25
        weights[27] = .15
        tumor_profile, context_counts = generate_tumor_profile(weights)
        ds = DeconstructSigs(context_counts=context_counts)

        # Limit to two deconstructed signatures
        reconstructed_weights = ds._DeconstructSigs__which_signatures(3)
        self.assertAlmostEqual(reconstructed_weights[3], .6, places=2)
        self.assertAlmostEqual(reconstructed_weights[11], .25, places=2)
        self.assertAlmostEqual(reconstructed_weights[27], .15, places=2)

        # Don't impose a signature limit to test that algorithm correctly deduces only three are necessary
        reconstructed_weights = ds._DeconstructSigs__which_signatures()
        self.assertAlmostEqual(reconstructed_weights[3], .6, places=2)
        self.assertAlmostEqual(reconstructed_weights[11], .25, places=2)
        self.assertAlmostEqual(reconstructed_weights[27], .15, places=2)

    def test_four_signatures(self):
        weights = np.zeros((30,))
        weights[6] = .43
        weights[10] = .27
        weights[25] = .20
        weights[29] = .1
        tumor_profile, context_counts = generate_tumor_profile(weights)
        ds = DeconstructSigs(context_counts=context_counts)

        # Limit to four deconstructed signatures
        reconstructed_weights = ds._DeconstructSigs__which_signatures(4)
        self.assertAlmostEqual(reconstructed_weights[6], .43, places=2)
        self.assertAlmostEqual(reconstructed_weights[10], .27, places=2)
        self.assertAlmostEqual(reconstructed_weights[25], .20, places=2)
        self.assertAlmostEqual(reconstructed_weights[29], .1, places=2)

        # Don't impose a signature limit to test that algorithm correctly deduces only fiour are necessary
        reconstructed_weights = ds._DeconstructSigs__which_signatures()
        self.assertAlmostEqual(reconstructed_weights[6], .43, places=2)
        self.assertAlmostEqual(reconstructed_weights[10], .27, places=2)
        self.assertAlmostEqual(reconstructed_weights[25], .20, places=2)
        self.assertAlmostEqual(reconstructed_weights[29], .1, places=2)

    def test_five_signatures(self):
        weights = np.zeros((30,))
        weights[3] = .19
        weights[7] = .4
        weights[13] = .24
        weights[17] = .10
        weights[23] = .07
        tumor_profile, context_counts = generate_tumor_profile(weights)
        ds = DeconstructSigs(context_counts=context_counts)

        # Limit to five deconstructed signatures
        reconstructed_weights = ds._DeconstructSigs__which_signatures(5)
        self.assertAlmostEqual(reconstructed_weights[3], .19, places=2)
        self.assertAlmostEqual(reconstructed_weights[7], .4, places=2)
        self.assertAlmostEqual(reconstructed_weights[13], .24, places=2)
        self.assertAlmostEqual(reconstructed_weights[17], .1, places=2)
        self.assertAlmostEqual(reconstructed_weights[23], .07, places=2)

        # Don't impose a signature limit to test that algorithm correctly deduces only five are necessary
        reconstructed_weights = ds._DeconstructSigs__which_signatures()
        self.assertAlmostEqual(reconstructed_weights[3], .19, places=2)
        self.assertAlmostEqual(reconstructed_weights[7], .4, places=2)
        self.assertAlmostEqual(reconstructed_weights[13], .24, places=2)
        self.assertAlmostEqual(reconstructed_weights[17], .1, places=2)
        self.assertAlmostEqual(reconstructed_weights[23], .07, places=2)

    def test_five_signatures_discard_insignificant(self):
        weights = np.zeros((30,))
        weights[3] = .19
        weights[7] = .41
        weights[13] = .24
        weights[17] = .14
        weights[23] = .02
        tumor_profile, context_counts = generate_tumor_profile(weights)
        ds = DeconstructSigs(context_counts=context_counts)

        # Limit to five deconstructed signatures
        reconstructed_weights = ds._DeconstructSigs__which_signatures(5)
        self.assertAlmostEqual(reconstructed_weights[3], .19, places=2)
        self.assertAlmostEqual(reconstructed_weights[7], .41, places=2)
        self.assertAlmostEqual(reconstructed_weights[13], .24, places=2)
        self.assertAlmostEqual(reconstructed_weights[17], .14, places=2)
        self.assertAlmostEqual(reconstructed_weights[23], 0, places=2)

        # Don't impose a signature limit to test that algorithm correctly deduces only five necessary and significant
        reconstructed_weights = ds._DeconstructSigs__which_signatures()
        self.assertAlmostEqual(reconstructed_weights[3], .19, places=2)
        self.assertAlmostEqual(reconstructed_weights[7], .41, places=2)
        self.assertAlmostEqual(reconstructed_weights[13], .24, places=2)
        self.assertAlmostEqual(reconstructed_weights[17], .14, places=2)
        self.assertAlmostEqual(reconstructed_weights[23], 0, places=2)

    def test_three_signatures_perturbed(self):
        weights = np.zeros((30,))
        weights[3] = .6
        weights[11] = .3
        weights[27] = .1
        tumor_profile, context_counts = generate_tumor_profile(weights)

        # Perturb the counts by up to 6% more or less of value
        for k, v in context_counts.items():
            context_counts[k] = v*(1 + (np.random.rand()*0.12)-0.06)

        ds = DeconstructSigs(context_counts=context_counts)

        # Limit to two deconstructed signatures
        reconstructed_weights = ds._DeconstructSigs__which_signatures(3)
        self.assertAlmostEqual(reconstructed_weights[3], .6, places=1)
        self.assertAlmostEqual(reconstructed_weights[11], .3, places=1)
        self.assertAlmostEqual(reconstructed_weights[27], .1, places=1)

        # Don't impose a signature limit to test that algorithm correctly deduces only three are necessary
        reconstructed_weights = ds._DeconstructSigs__which_signatures()
        self.assertAlmostEqual(reconstructed_weights[3], .6, places=1)
        self.assertAlmostEqual(reconstructed_weights[11], .3, places=1)
        self.assertAlmostEqual(reconstructed_weights[27], .1, places=1)

    def test_four_signatures_perturbed(self):
        weights = np.zeros((30,))
        weights[6] = .4
        weights[10] = .3
        weights[25] = .2
        weights[29] = .1
        tumor_profile, context_counts = generate_tumor_profile(weights)

        # Perturb the counts by up to 6% more or less of value
        for k, v in context_counts.items():
            context_counts[k] = v*(1 + (np.random.rand()*0.12)-0.06)

        ds = DeconstructSigs(context_counts=context_counts)

        # Limit to four deconstructed signatures
        reconstructed_weights = ds._DeconstructSigs__which_signatures(4)
        self.assertAlmostEqual(reconstructed_weights[6], .4, places=1)
        self.assertAlmostEqual(reconstructed_weights[10], .3, places=1)
        self.assertAlmostEqual(reconstructed_weights[25], .2, places=1)
        self.assertAlmostEqual(reconstructed_weights[29], .1, places=1)

        # Don't impose a signature limit to test that algorithm correctly deduces only four are necessary
        reconstructed_weights = ds._DeconstructSigs__which_signatures()
        self.assertAlmostEqual(reconstructed_weights[6], .4, places=1)
        self.assertAlmostEqual(reconstructed_weights[10], .3, places=1)
        self.assertAlmostEqual(reconstructed_weights[25], .2, places=1)
        self.assertAlmostEqual(reconstructed_weights[29], .1, places=1)

    def test_five_signatures_perturbed(self):
        weights = np.zeros((30,))
        weights[3] = .4
        weights[7] = .2
        weights[13] = .2
        weights[17] = .1
        weights[23] = .1
        tumor_profile, context_counts = generate_tumor_profile(weights)

        # Perturb the counts by up to 6% more or less of value
        for k, v in context_counts.items():
            context_counts[k] = v * (1 + (np.random.rand() * 0.12) - 0.06)

        ds = DeconstructSigs(context_counts=context_counts)

        # Limit to two deconstructed signatures
        reconstructed_weights = ds._DeconstructSigs__which_signatures(5)
        self.assertAlmostEqual(reconstructed_weights[3], .4, places=1)
        self.assertAlmostEqual(reconstructed_weights[7], .2, places=1)
        self.assertAlmostEqual(reconstructed_weights[13], .2, places=1)
        self.assertAlmostEqual(reconstructed_weights[17], .1, places=1)
        self.assertAlmostEqual(reconstructed_weights[23], .1, places=1)

        # Don't impose a signature limit to test that algorithm correctly deduces only five are necessary
        reconstructed_weights = ds._DeconstructSigs__which_signatures()
        self.assertAlmostEqual(reconstructed_weights[3], .4, places=1)
        self.assertAlmostEqual(reconstructed_weights[7], .2, places=1)
        self.assertAlmostEqual(reconstructed_weights[13], .2, places=1)
        self.assertAlmostEqual(reconstructed_weights[17], .1, places=1)
        self.assertAlmostEqual(reconstructed_weights[23], .1, places=1)

"""Helpers to generate fake tumor profiles from known linear combinations of cosmic signatures for testing below"""


def generate_tumor_profile(w):
    """
    Given an array of 30 weights, one for each cosmic signature,
    admix the signatures to generate the tumor profile
    """
    cosmic_signatures_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                              '../data/signatures_probabilities.txt')
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
        mutation_type = cosmic_signatures['Somatic Mutation Type'][i]
        counts[mutation_type] = round(context_fraction * multiplier, 1)
    return counts


if __name__ == '__main__':
    unittest.main()
