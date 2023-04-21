from torchtext.vocab import vocab as make_vocab
from transformers import BertTokenizer


def get_wordvec(*, return_vocab=True):
    model = "bert-base-uncased"

    builder = BertBuilder(model).with_vocab().with_bert()

    wordvec, vocab = builder.build()

    if return_vocab:
        return wordvec, vocab

    return wordvec


def get_tokenizer():
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

    return wrapper


def get_nlp():
    import spacy

    nlp = spacy.load("en_core_web_sm")
    return nlp


class BertBuilder:
    vocab = None
    wordvec = None

    def __init__(self, model="bert-base-uncased"):
        self.model = model

    def build(self):
        return self.wordvec, self.vocab

    def with_vocab(self, *init_inputs, **kwargs):
        tokenizer = BertTokenizer.from_pretrained(self.model, *init_inputs, **kwargs)

        self.vocab = make_vocab(tokenizer.vocab, min_freq=0)
        self.vocab.set_default_index(self.vocab["[UNK]"])

        return self

    def with_bert(self):
        from transformers import BertModel

        self.wordvec = BertModel.from_pretrained(self.model)

        return self
