import logging
from typing import List

import torch
import torch.nn as nn
import torchtext

WV_GLOVE = "glove"
WV_BERT = "bert"
WV_TYPES = [WV_GLOVE, WV_BERT]


def get_wordvec(
    wv_type=WV_GLOVE,
    config={},
    *,
    custom_labels: List[str] = [],
    custom_tokens: List[str] = [],
    return_vocab=True,
):
    if wv_type not in WV_TYPES:
        raise ValueError(f"Wordvec type '{wv_type}' not supported, allowed types are {WV_TYPES}")

    if wv_type == WV_BERT:
        model = "bert-base-uncased"

        builder = (
            BertBuilder(model)
            .with_vocab()
            .with_bert()
        )

    if wv_type == WV_GLOVE:
        model = "6B"
        dim = config.get("dim", 300)

        builder = (
            WordvecBuilder()
            .with_glove(model, dim, cache="data/glove")
            .with_vocab()
            .with_custom_labels(custom_labels)
            .with_custom_tokens(custom_tokens)
        )

        logging.info(f"Added {builder.n_added_tokens} custom tokens")

    wordvec, vocab = builder.build()

    if return_vocab:
        return wordvec, vocab

    return wordvec


def get_objects_vocab(path="data/objects_vocab.txt"):
    with open(path, "r") as f:
        labels = f.readlines()
        labels = [label.strip() for label in labels]
    return labels


def get_nlp():
    import spacy

    nlp = spacy.load("en_core_web_sm")
    return nlp


def get_tokenizer(wv_type=WV_GLOVE):
    if wv_type not in WV_TYPES:
        raise ValueError(f"Wordvec type '{wv_type}' not supported, allowed types are {WV_TYPES}")

    if wv_type == WV_BERT:
        from transformers import BertTokenizer

        bert_model = "bert-base-uncased"

        tokenizer = BertTokenizer.from_pretrained(bert_model)

        # We delegate to the tokenizer the padding and truncation logic
        # to make the collate_fn simpler and agnostic to the wordvec type.
        # To make this happen, we need to specify in the dataset class the
        # sentence and queries lengths.

        def wrapper(text, **kwargs):
            kwargs_default = {
                "padding": "max_length",
                "max_length": 12,
                "truncation": True,
            }

            input_ids = tokenizer(text, **(kwargs_default | kwargs))["input_ids"]

            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            
            return tokens

    if wv_type == WV_GLOVE:
        tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

        def wrapper(text, **kwargs):
            return tokenizer(text)
    
    return wrapper


class WordvecBuilder:
    wordvec = None
    vocab = None

    n_added_tokens = 0

    from spellchecker import SpellChecker

    spell = SpellChecker()

    def build(self):
        return self.wordvec, self.vocab

    def with_glove(self, name="840B", dim=300, **kwargs):
        from torchtext.vocab import GloVe

        self.wordvec = GloVe(name, dim, **kwargs)

        return self

    def with_vocab(self, pad_token="<pad>", unk_token="<unk>"):
        """
        Build a vocab using GloVe tokens
        """
        from torchtext.vocab import vocab as make_vocab

        assert self.wordvec is not None, "wordvec must be built first"

        # min_freq=0 is required to include the first token in `wordvec.stoi` which has index 0
        # the `make_vocab` function requires an ordered dict as input and discards entries
        # whose value is less than `min_freq`
        self.vocab = make_vocab(
            self.wordvec.stoi,
            specials=[pad_token, unk_token],
            special_first=True,
            min_freq=0,
        )

        self.vocab.set_default_index(1)

        # add the unk embedding (zeros tensor) at the beginning of wordvec
        self.wordvec.vectors = torch.cat(
            [
                torch.zeros(1, self.wordvec.dim),
                torch.zeros(1, self.wordvec.dim),
                self.wordvec.vectors,
            ],
            dim=0,
        )
        # and update accordingly itos and stoi
        self.wordvec.itos.insert(0, pad_token)
        self.wordvec.itos.insert(1, unk_token)
        self.wordvec.stoi = {word: i for i, word in enumerate(self.wordvec.itos)}

        return self

    def with_custom_labels(self, labels):
        for label in labels:
            self._add_label(label)

        return self

    def with_custom_tokens(self, tokens):
        for token in tokens:
            self._add_token(token)

        return self

    def _add_label(self, label):
        if label in self.vocab:
            return

        parts = label.split(",")
        parts = [part.split(":")[-1] for part in parts]  # remove index encoding, if any

        embedding = self._find_embedding_for_parts(parts)

        if embedding is None:
            return  # no embedding found, keep `label` oov

        self._append(label, embedding)

    def _add_token(self, token):
        if token in self.vocab:
            return  # token already in vocab

        empty = torch.empty(1, self.wordvec.dim)  # xavier_normal_ requires 2 dim
        embedding = nn.init.xavier_normal_(empty).reshape(-1)

        self._append(token, embedding)

    def _find_embedding_for_parts(self, parts):
        for part in parts:
            if part in self.vocab:
                return self.wordvec[part]

            else:
                # part may be a compound word like "stop sign", so we need to compute
                # the average embedding over words

                part_emb = torch.zeros(self.wordvec.dim)
                words_found = 0.0

                for word in part.split():
                    if word in self.vocab:
                        part_emb += self.wordvec[word]
                        words_found += 1.0
                    else:
                        corrected = self.spell.correction(word)

                        if corrected in self.vocab:
                            part_emb += self.wordvec[corrected]
                            words_found += 1.0

                if words_found > 0:
                    part_emb = part_emb / words_found

                    return part_emb

        return None

    def _append(self, token, emb):
        self.vocab.append_token(token)

        self.wordvec.vectors = torch.cat(
            [self.wordvec.vectors, emb.unsqueeze(0)], dim=0
        )
        self.wordvec.itos.append(token)
        self.wordvec.stoi[token] = len(self.wordvec.itos) - 1

        self.n_added_tokens += 1


class BertBuilder:
    vocab = None
    wordvec = None

    def __init__(self, model="bert-base-uncased"):
        self.model = model

    def build(self):
        return self.wordvec, self.vocab

    def with_vocab(self, *init_inputs, **kwargs):
        from torchtext.vocab import vocab as make_vocab
        from transformers import BertTokenizer

        tokenizer = BertTokenizer.from_pretrained(self.model, *init_inputs, **kwargs)

        self.vocab = make_vocab(tokenizer.vocab, min_freq=0)
        self.vocab.set_default_index(self.vocab["[UNK]"])

        return self

    def with_bert(self):
        from transformers import BertModel

        self.wordvec = BertModel.from_pretrained(self.model)

        return self
