import unittest

import torch

from weakvg.wordvec import (
    BertBuilder,
    get_wordvec,
    get_objects_vocab,
    get_tokenizer,
)


class TestWordvec(unittest.TestCase):
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
