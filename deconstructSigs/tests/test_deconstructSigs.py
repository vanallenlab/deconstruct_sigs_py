import unittest
import numpy as np
import pandas as pd
import os
import re
from collections import defaultdict
from deconstructSigs.deconstructSigs import DeconstructSigs


class TestDeconstructSigs(unittest.TestCase):
    def test_substitution_standardization(self):
        weights = np.zeros((30,))
        weights[0] = 1
        tumor_profile, context_counts = generate_tumor_profile(weights)
        ds = DeconstructSigs(context_counts=context_counts)

        # Standard form means that the ref is a pyrimidine
        standardized = ds._DeconstructSigs__standardize_subs('G', 'C')
        self.assertEqual(standardized, 'C>G')

        standardized = ds._DeconstructSigs__standardize_subs('G', 'A')
        self.assertEqual(standardized, 'C>T')

        standardized = ds._DeconstructSigs__standardize_subs('G', 'T')
        self.assertEqual(standardized, 'C>A')

        standardized = ds._DeconstructSigs__standardize_subs('A', 'C')
        self.assertEqual(standardized, 'T>G')

        standardized = ds._DeconstructSigs__standardize_subs('A', 'G')
        self.assertEqual(standardized, 'T>C')

        standardized = ds._DeconstructSigs__standardize_subs('A', 'T')
        self.assertEqual(standardized, 'T>A')

        standardized = ds._DeconstructSigs__standardize_subs('C', 'A')
        self.assertEqual(standardized, 'C>A')

        standardized = ds._DeconstructSigs__standardize_subs('C', 'G')
        self.assertEqual(standardized, 'C>G')

        standardized = ds._DeconstructSigs__standardize_subs('C', 'T')
        self.assertEqual(standardized, 'C>T')

        standardized = ds._DeconstructSigs__standardize_subs('T', 'A')
        self.assertEqual(standardized, 'T>A')

        standardized = ds._DeconstructSigs__standardize_subs('T', 'C')
        self.assertEqual(standardized, 'T>C')

        standardized = ds._DeconstructSigs__standardize_subs('T', 'G')
        self.assertEqual(standardized, 'T>G')

    def test_trinucleotide_standardization(self):
        weights = np.zeros((30,))
        weights[0] = 1
        tumor_profile, context_counts = generate_tumor_profile(weights)
        ds = DeconstructSigs(context_counts=context_counts)

        pair = {
            'A': 'T',
            'C': 'G',
            'T': 'A',
            'G': 'C'
        }

        # Standard form means that a pyrimidine is at the center position of the trinucleotide string
        already_standard_form = ['ACA', 'ACC', 'ACG', 'ACT',
                                 'CCA', 'CCC', 'CCG', 'CCT',
                                 'GCA', 'GCC', 'GCG', 'GCT',
                                 'TCA', 'TCC', 'TCG', 'TCT',
                                 'ATA', 'ATC', 'ATG', 'ATT',
                                 'CTA', 'CTC', 'CTG', 'CTT',
                                 'GTA', 'GTC', 'GTG', 'GTT',
                                 'TTA', 'TTC', 'TTG', 'TTT',
                                 ]

        for trinuc in already_standard_form:
            standardized_trinuc = ds._DeconstructSigs__standardize_trinuc(trinuc)
            self.assertEqual(trinuc, standardized_trinuc)

        # Non-standard form means that a purine is a the center position of the trinucleotide string
        non_standard_form = ['{}{}{}'.format(pair[trinuc[0]], pair[trinuc[1]], pair[trinuc[2]])
                             for trinuc in already_standard_form]

        for i, trinuc in enumerate(non_standard_form):
            standardized_trinuc = ds._DeconstructSigs__standardize_trinuc(trinuc)
            self.assertEqual(already_standard_form[i], standardized_trinuc)

        # Test that lower case trinucleotide strings get uppercased as well
        for trinuc in already_standard_form:
            standardized_trinuc = ds._DeconstructSigs__standardize_trinuc(trinuc.lower())
            self.assertEqual(trinuc, standardized_trinuc)
        for i, trinuc in enumerate(non_standard_form):
            standardized_trinuc = ds._DeconstructSigs__standardize_trinuc(trinuc.lower())
            self.assertEqual(already_standard_form[i], standardized_trinuc)

    def test_one_signature(self):
        weights = np.zeros((30,))
        weights[0] = 1
        tumor_profile, context_counts = generate_tumor_profile(weights)
        ds = DeconstructSigs(context_counts=context_counts)

        # Limit to one deconstructed signatures
        reconstructed_weights = ds.which_signatures(1)
        self.assertAlmostEqual(reconstructed_weights[0], weights[0], places=2)

        # Don't impose a signature limit to test that algorithm correctly deduces only one are necessary
        reconstructed_weights = ds.which_signatures()
        self.assertAlmostEqual(reconstructed_weights[0], weights[0], places=2)

    def test_two_signatures(self):
        weights = np.zeros((30,))
        weights[0] = .5
        weights[5] = .5
        tumor_profile, context_counts = generate_tumor_profile(weights)
        ds = DeconstructSigs(context_counts=context_counts)

        # Limit to two deconstructed signatures
        reconstructed_weights = ds.which_signatures(2)
        self.assertAlmostEqual(reconstructed_weights[0], .5, places=2)
        self.assertAlmostEqual(reconstructed_weights[5], .5, places=2)

        # Don't impose a signature limit to test that algorithm correctly deduces only two are necessary
        reconstructed_weights = ds.which_signatures()
        self.assertAlmostEqual(reconstructed_weights[0], .5, places=2)
        self.assertAlmostEqual(reconstructed_weights[5], .5, places=2)

    def test_two_signatures_with_associated(self):
        weights = np.zeros((30,))
        weights[0] = .5
        weights[5] = .5
        tumor_profile, context_counts = generate_tumor_profile(weights)
        ds = DeconstructSigs(context_counts=context_counts)

        # Limit to two deconstructed signatures
        reconstructed_weights = ds.which_signatures(2, associated=[5])
        self.assertAlmostEqual(reconstructed_weights[0], 0, places=2)
        self.assertAlmostEqual(reconstructed_weights[5], 1, places=2)

        # Don't impose a signature limit to test that algorithm correctly deduces only two are necessary
        reconstructed_weights = ds.which_signatures(associated=[5])
        self.assertAlmostEqual(reconstructed_weights[0], 0, places=2)
        self.assertAlmostEqual(reconstructed_weights[5], 1, places=2)

    def test_three_signatures(self):
        weights = np.zeros((30,))
        weights[3] = .6
        weights[11] = .25
        weights[27] = .15
        tumor_profile, context_counts = generate_tumor_profile(weights)
        ds = DeconstructSigs(context_counts=context_counts)

        # Limit to two deconstructed signatures
        reconstructed_weights = ds.which_signatures(3)
        self.assertAlmostEqual(reconstructed_weights[3], .6, places=2)
        self.assertAlmostEqual(reconstructed_weights[11], .25, places=2)
        self.assertAlmostEqual(reconstructed_weights[27], .15, places=2)

        # Don't impose a signature limit to test that algorithm correctly deduces only three are necessary
        reconstructed_weights = ds.which_signatures()
        self.assertAlmostEqual(reconstructed_weights[3], .6, places=2)
        self.assertAlmostEqual(reconstructed_weights[11], .25, places=2)
        self.assertAlmostEqual(reconstructed_weights[27], .15, places=2)

    def test_three_signatures_with_associated(self):
        weights = np.zeros((30,))
        weights[3] = .6
        weights[11] = .25
        weights[27] = .15
        tumor_profile, context_counts = generate_tumor_profile(weights)
        ds = DeconstructSigs(context_counts=context_counts)

        # Limit to two deconstructed signatures
        reconstructed_weights = ds.which_signatures(3, associated=[11, 27])
        self.assertAlmostEqual(reconstructed_weights[3], 0, places=2)

        # Don't impose a signature limit to test that algorithm correctly deduces only three are necessary
        reconstructed_weights = ds.which_signatures(associated=[11, 27])
        self.assertAlmostEqual(reconstructed_weights[3], 0, places=2)

    def test_four_signatures(self):
        weights = np.zeros((30,))
        weights[6] = .43
        weights[10] = .27
        weights[25] = .20
        weights[29] = .1
        tumor_profile, context_counts = generate_tumor_profile(weights)
        ds = DeconstructSigs(context_counts=context_counts)

        # Limit to four deconstructed signatures
        reconstructed_weights = ds.which_signatures(4)
        self.assertAlmostEqual(reconstructed_weights[6], .43, places=2)
        self.assertAlmostEqual(reconstructed_weights[10], .27, places=2)
        self.assertAlmostEqual(reconstructed_weights[25], .20, places=2)
        self.assertAlmostEqual(reconstructed_weights[29], .1, places=2)

        # Don't impose a signature limit to test that algorithm correctly deduces only fiour are necessary
        reconstructed_weights = ds.which_signatures()
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
        reconstructed_weights = ds.which_signatures(5)
        self.assertAlmostEqual(reconstructed_weights[3], .19, places=2)
        self.assertAlmostEqual(reconstructed_weights[7], .4, places=2)
        self.assertAlmostEqual(reconstructed_weights[13], .24, places=2)
        self.assertAlmostEqual(reconstructed_weights[17], .1, places=2)
        self.assertAlmostEqual(reconstructed_weights[23], .07, places=2)

        # Don't impose a signature limit to test that algorithm correctly deduces only five are necessary
        reconstructed_weights = ds.which_signatures()
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
        reconstructed_weights = ds.which_signatures(5)
        self.assertAlmostEqual(reconstructed_weights[3], .19, places=2)
        self.assertAlmostEqual(reconstructed_weights[7], .41, places=2)
        self.assertAlmostEqual(reconstructed_weights[13], .24, places=2)
        self.assertAlmostEqual(reconstructed_weights[17], .14, places=2)
        self.assertAlmostEqual(reconstructed_weights[23], 0, places=2)

        # Don't impose a signature limit to test that algorithm correctly deduces only five necessary and significant
        reconstructed_weights = ds.which_signatures()
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
        reconstructed_weights = ds.which_signatures(3)
        self.assertAlmostEqual(reconstructed_weights[3], .6, places=1)
        self.assertAlmostEqual(reconstructed_weights[11], .3, places=1)
        self.assertAlmostEqual(reconstructed_weights[27], .1, places=1)

        # Don't impose a signature limit to test that algorithm correctly deduces only three are necessary
        reconstructed_weights = ds.which_signatures()
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
        reconstructed_weights = ds.which_signatures(4)
        self.assertAlmostEqual(reconstructed_weights[6], .4, places=1)
        self.assertAlmostEqual(reconstructed_weights[10], .3, places=1)
        self.assertAlmostEqual(reconstructed_weights[25], .2, places=1)
        self.assertAlmostEqual(reconstructed_weights[29], .1, places=1)

        # Don't impose a signature limit to test that algorithm correctly deduces only four are necessary
        reconstructed_weights = ds.which_signatures()
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
        reconstructed_weights = ds.which_signatures(5)
        self.assertAlmostEqual(reconstructed_weights[3], .4, places=1)
        self.assertAlmostEqual(reconstructed_weights[7], .2, places=1)
        self.assertAlmostEqual(reconstructed_weights[13], .2, places=1)
        self.assertAlmostEqual(reconstructed_weights[17], .1, places=1)
        self.assertAlmostEqual(reconstructed_weights[23], .1, places=1)

        # Don't impose a signature limit to test that algorithm correctly deduces only five are necessary
        reconstructed_weights = ds.which_signatures()
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
