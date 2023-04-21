import unittest

import torch

from weakvg.wordvec import (
    WordvecBuilder,
    BertBuilder,
    get_wordvec,
    get_objects_vocab,
    get_tokenizer,
)


class TestWordvec(unittest.TestCase):
    def test_get_wordvec_glove(self):
        wordvec, vocab = get_wordvec(config={"dim": 50}, custom_labels=[])

        self.assertEqual(len(vocab), 400000 + 2)  # +2 for <pad>, <unk>
        self.assertEqual(len(wordvec), 400000 + 2)

        self.assertEqual(wordvec.vectors.shape, (400000 + 2, 50))

    def test_get_wordvec_bert(self):
        # BERT based models do not have vectors, thus we test whether the model
        # can encode a sentence

        from weakvg.masking import get_queries_mask

        wv, vocab = get_wordvec("bert")
        tokenizer = get_tokenizer("bert")

        sentence = "the quick brown fox jumps over the lazy dog"
        n_words = len(sentence.split())

        tokens = tokenizer(
            sentence,
            padding="max_length",
            max_length=12,
            truncation=True,
        )
        ids = vocab(tokens)
        
        batch = torch.tensor(ids).unsqueeze(0)
        is_word, _ = get_queries_mask(batch)

        out, _ = wv(batch, attention_mask=is_word, return_dict=False)

        self.assertEqual(out.shape, (1, 12, 768))

        # bert adds a [CLS] token at the beginning and a [SEP] token at the end of
        # the sentence, thus the number of tokens increase by 2
        self.assertEqual(n_words + 2, is_word.sum())

        # the last token is [PAD], however bert produces a non-zero vector
        self.assertFalse(torch.equal(out[0, -1], torch.zeros(768)))


    def test_get_objects_vocab(self):
        self.assertEqual(len(get_objects_vocab()), 1600)


class TestWordvecBuilder(unittest.TestCase):
    dim = 50
    wv_kwargs = {"name": "6B", "dim": dim, "cache": "data/glove"}

    def test_with_glove(self):
        d = self.dim
        wv, _ = WordvecBuilder().with_glove(**self.wv_kwargs).build()
        vecs = wv.vectors

        self.assertEqual(vecs.shape, (400000, d))

    def test_with_vocab(self):
        d = self.dim
        b = WordvecBuilder().with_glove(**self.wv_kwargs).with_vocab()

        wv, vocab = b.build()
        vecs = wv.vectors

        self.assertEqual(vecs.shape, (400002, d))

        self.assertEqual(len(wv), 400002)
        self.assertEqual(len(vocab), 400002)

        self.assertTrue(torch.equal(wv["<pad>"], torch.zeros(d)))
        self.assertTrue(torch.equal(wv["<pad>"], vecs[vocab["<pad>"]]))

        self.assertTrue(torch.equal(wv["<unk>"], torch.zeros(d)))
        self.assertTrue(torch.equal(wv["<unk>"], vecs[vocab["<unk>"]]))

        self.assertTrue(torch.equal(wv["the"], vecs[vocab["the"]]))

        self.assertEqual(wv.stoi["<pad>"], 0)
        self.assertEqual(wv.stoi["<unk>"], 1)
        self.assertEqual(wv.stoi["the"], 2)

        self.assertEqual(vocab.get_default_index(), 1)

        self.assertEqual(vocab["<pad>"], 0)

        self.assertEqual(vocab.lookup_token(0), "<pad>")
        self.assertEqual(vocab.lookup_token(1), "<unk>")
        self.assertEqual(vocab.lookup_token(2), "the")

    def test_with_vocab__raises_whether_wordvec_not_build(self):
        with self.assertRaises(AssertionError):
            WordvecBuilder().with_vocab()

    def test_with_custom_labels__known(self):
        b = (
            WordvecBuilder()
            .with_glove(**self.wv_kwargs)
            .with_vocab()
            .with_custom_labels(["microwave"])
        )

        wv, vocab = b.build()

        self.assertEqual(len(wv), 400002)
        self.assertEqual(len(vocab), 400002)

    def test_with_custom_labels__unknown(self):
        sim = torch.cosine_similarity

        b = (
            WordvecBuilder()
            .with_glove(**self.wv_kwargs)
            .with_vocab()
            .with_custom_labels(["microwave oven"])
        )

        wv, vocab = b.build()
        vecs = wv.vectors

        self.assertEqual(len(wv), 400003)  # +2 pad and unk, +1 added
        self.assertEqual(len(vocab), 400003)

        self.assertEqual(
            vocab["microwave oven"], 400002
        )  # vocab should be updated accordingly

        self.assertTrue(
            torch.equal(wv["microwave oven"], vecs[-1])
        )  # should be last added

        # "microwave oven" is build by averaging "microwave" and "oven", thus
        # we perform a sanity check through cosine similarity
        self.assertAlmostEqual(
            sim(wv["microwave oven"], wv["microwave"], dim=0).item(), 0.9121, places=4
        )
        self.assertAlmostEqual(
            sim(wv["microwave oven"], wv["oven"], dim=0).item(), 0.9370, places=4
        )

    def test_with_custom_labels__from_vocab(self):
        d = self.dim
        b = (
            WordvecBuilder()
            .with_glove(**self.wv_kwargs)
            .with_vocab()
            .with_custom_labels(get_objects_vocab())
        )

        wv, vocab = b.build()
        vecs = wv.vectors

        self.assertEqual(
            vecs.shape, (400000 + 2 + 299, d)
        )  # +2 pad and oov, +299 added

        self.assertTrue(torch.equal(wv["the"], vecs[vocab["the"]]))
        self.assertTrue(
            torch.equal(
                wv["microwave,microwave oven"], vecs[vocab["microwave,microwave oven"]]
            )
        )

    def test_with_custom_labels__not_breaking_prev_structure(self):
        d = self.dim
        b = (
            WordvecBuilder()
            .with_glove(**self.wv_kwargs)
            .with_vocab()
            .with_custom_labels(["microwave oven"])
        )

        wv, _ = b.build()
        vecs = wv.vectors

        self.assertTrue(torch.equal(vecs[0], torch.zeros(d)))

    def test_with_custom_tokens__known(self):
        d = self.dim
        b = (
            WordvecBuilder()
            .with_glove(**self.wv_kwargs)
            .with_vocab()
            .with_custom_tokens(["dog"])
        )

        wv, vocab = b.build()

        self.assertEqual(len(wv), 400000 + 2)  # +2 pad and oov
        self.assertEqual(len(vocab), 400000 + 2)

    def test_with_custom_tokens__unknown(self):
        d = self.dim
        b = (
            WordvecBuilder()
            .with_glove(**self.wv_kwargs)
            .with_vocab()
            .with_custom_tokens(["blabla"])
        )

        wv, vocab = b.build()

        self.assertEqual(len(wv), 400000 + 2 + 1)  # +2 pad and oov, +1 added
        self.assertEqual(len(vocab), 400000 + 2 + 1)
        self.assertTrue("blabla" in vocab)
        self.assertTrue(~torch.equal(wv["blabla"], torch.zeros(d)))

    def test_with_custom_tokens__not_breaking_prev_structure(self):
        d = self.dim
        b = (
            WordvecBuilder()
            .with_glove(**self.wv_kwargs)
            .with_vocab()
            .with_custom_labels(["microwave oven"])
        )

        wv, _ = b.build()
        vecs = wv.vectors

        self.assertTrue(torch.equal(vecs[0], torch.zeros(d)))
        self.assertTrue(torch.equal(vecs[1], torch.zeros(d)))


class TestBertBuilder(unittest.TestCase):
    def test_with_bert(self):
        b = BertBuilder().with_bert()

        wv, _ = b.build()

        from transformers import BertModel

        self.assertEqual(type(wv), BertModel)

        self.assertEqual(
            vars(wv.config),
            vars(wv.config) |
            # bert-base-uncased configs
            {
                "_name_or_path": "bert-base-uncased",
                "architectures": ["BertForMaskedLM"],
                "attention_probs_dropout_prob": 0.1,
                "classifier_dropout": None,
                "gradient_checkpointing": False,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.1,
                "hidden_size": 768,
                "initializer_range": 0.02,
                "intermediate_size": 3072,
                "layer_norm_eps": 1e-12,
                "max_position_embeddings": 512,
                "model_type": "bert",
                "num_attention_heads": 12,
                "num_hidden_layers": 12,
                "pad_token_id": 0,
                "position_embedding_type": "absolute",
                "vocab_size": 30522,
            },
            vars(wv.config),
        )

    def test_with_vocab(self):
        b = BertBuilder().with_vocab()

        _, vocab = b.build()

        self.assertEqual(len(vocab), 30522)

        self.assertEqual(vocab["[PAD]"], 0)
        self.assertEqual(vocab["[UNK]"], 100)
        self.assertEqual(vocab["[CLS]"], 101)
        self.assertEqual(vocab["[SEP]"], 102)
        self.assertEqual(vocab["[MASK]"], 103)
        self.assertEqual(vocab["the"], 1996)
        self.assertEqual(vocab["pippofranco"], vocab["[UNK]"])
