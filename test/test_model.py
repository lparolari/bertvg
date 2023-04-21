import unittest

import torch
from torch.nn.functional import cosine_similarity

from weakvg.model import TextualBranch, WordEmbedding, VisualBranch, ConceptBranch
from weakvg.wordvec import get_wordvec


class TestWordEmbedding(unittest.TestCase):
    def setUp(self):
        import pytorch_lightning as pl

        # seeding is required for reproducibility with BERT
        pl.seed_everything(42)

    def test_word_embedding_similarity(self):
        wordvec, vocab = get_wordvec()
        we = WordEmbedding(wordvec)

        index = torch.tensor([vocab["man"], vocab["woman"], vocab["person"]])

        emb = we(index)

        man = emb[0]
        woman = emb[1]
        person = emb[2]

        self.assertTensorAlmostEqual(cosine_similarity(man, person, dim=0), 0.6443)
        self.assertTensorAlmostEqual(cosine_similarity(woman, person, dim=0), 0.6171)
        self.assertTensorAlmostEqual(cosine_similarity(man, woman, dim=0), 0.6999)

    def test_word_embedding_similarity_bert(self):
        wordvec, vocab = get_wordvec("bert")
        we = WordEmbedding(wordvec)

        index = torch.tensor([vocab["man"], vocab["woman"], vocab["person"]])
        attn = torch.tensor([1, 1, 1]).bool()

        emb = we(index, attn)

        man = emb[0]
        woman = emb[1]
        person = emb[2]

        self.assertTensorAlmostEqual(cosine_similarity(man, person, dim=0), 0.9381)
        self.assertTensorAlmostEqual(cosine_similarity(woman, person, dim=0), 0.9927)
        self.assertTensorAlmostEqual(cosine_similarity(man, woman, dim=0), 0.9454)

    def test_word_embedding_similarity_bert_on_single_words(self):
        wordvec, vocab = get_wordvec("bert")
        we = WordEmbedding(wordvec)
        sim = cosine_similarity

        index = torch.tensor([vocab["man"], vocab["woman"], vocab["person"]])
        attn = torch.tensor([1, 1, 1]).bool()

        emb = we(index, attn)
        emb_man = we(index[0:1], attn[0:1])
        emb_woman = we(index[1:2], attn[1:2])
        emb_person = we(index[2:3], attn[2:3])

        man = emb[0]
        woman = emb[1]
        person = emb[2]

        man_nodep = emb_man[0]
        woman_nodep = emb_woman[0]
        person_nodep = emb_person[0]

        self.assertTensorAlmostEqual(sim(man_nodep, person_nodep, dim=0), 0.4169)
        self.assertTensorAlmostEqual(sim(woman_nodep, person_nodep, dim=0), 0.4421)
        self.assertTensorAlmostEqual(sim(man_nodep, woman_nodep, dim=0), 0.4858)

        self.assertTensorAlmostEqual(sim(man, man_nodep, dim=0), 0.5119)
        self.assertTensorAlmostEqual(sim(woman, woman_nodep, dim=0), 0.3374)
        self.assertTensorAlmostEqual(sim(person, person_nodep, dim=0), 0.3712)

    def assertTensorAlmostEqual(self, t, x, places=3):
        self.assertAlmostEqual(t.item(), x, places=places)


class TestTextualBranch(unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.wordvec, self.vocab = get_wordvec()
        self.we = WordEmbedding(self.wordvec)

    def test_lstm(self):
        queries_idx = [
            self.vocab.lookup_indices("the quick brown fox".split()),
            self.vocab.lookup_indices("the lazy dog <unk>".split()),
        ]
        queries = torch.tensor([queries_idx])

        textual_branch = TextualBranch(word_embedding=self.we)

        output = textual_branch.forward({"queries": queries})

        self.assertEqual(output.shape, (1, 2, 300))


class TestVisualBranch(unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.wordvec, self.vocab = get_wordvec()
        self.we = WordEmbedding(self.wordvec)

    def test_spatial(self):
        proposals = torch.tensor([[[20, 20, 80, 100]]])  # [b, p, 4]

        visual_branch = VisualBranch(word_embedding=self.we)

        spat = visual_branch.spatial({"proposals": proposals})

        self.assertEqual(spat.shape, (1, 1, 5))

        self.assertAlmostEqual(spat[0, 0, 0].item(), 20)
        self.assertAlmostEqual(spat[0, 0, 1].item(), 20)
        self.assertAlmostEqual(spat[0, 0, 2].item(), 80)
        self.assertAlmostEqual(spat[0, 0, 3].item(), 100)
        self.assertAlmostEqual(spat[0, 0, 4].item(), ((80 - 20) * (100 - 20)))

    def test_project(self):
        # please note that VisualBranch requires the `v` dimension to be 2048,
        # while last dimension of `spat` need to be 5
        proposals_feat = torch.rand(2, 6, 2048)  # [b, p, v]
        spat = torch.rand(2, 6, 5).mul(100).long()  # [b, p, 5]

        visual_branch = VisualBranch(word_embedding=self.we)

        proj = visual_branch.project(proposals_feat, spat)

        # the network projects visual features to 300 in order to match
        # the word embedding dimension
        self.assertEqual(proj.shape, (2, 6, 300))


class TestConceptBranch(unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.wordvec, self.vocab = get_wordvec()
        self.we = WordEmbedding(self.wordvec)

    def test_forward(self):
        heads = torch.tensor(
            [
                self.vocab.lookup_indices("man <unk>".split()),
                self.vocab.lookup_indices("dog <unk>".split()),
            ]
        ).unsqueeze(
            0
        )  # [b, q, h] = [1, 2, 2]

        unk = self.vocab["<unk>"]  # or, self.vocab.get_default_index()

        labels = torch.tensor(
            [
                self.vocab["person"],
                self.vocab["dog"],
                self.vocab["sky"],
                self.vocab["person"],
                self.vocab["bird"],
                unk,  # pad
                unk,  # pad
                unk,  # pad
            ]
        ).unsqueeze(
            0
        )  # [b, p] = [1, 8]

        concept_branch = ConceptBranch(word_embedding=self.we)

        output = concept_branch.forward({"heads": heads, "labels": labels})

        self.assertEqual(output.shape, (1, 2, 1, 8))

        q1 = (0, 0, 0)
        q2 = (0, 1, 0)

        # label 1 and 3 have both "person" label which is close to "man", but due
        # to argmax impl the first one is chosen
        self.assertEqual(output[q1].argmax().item(), 0)

        self.assertEqual(output[q2].argmax().item(), 1)
