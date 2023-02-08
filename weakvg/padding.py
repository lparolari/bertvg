from typing import List

import torch
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence

Sentence = List[int]
Query = List[int]
Queries = List[Query]
Proposal = List[int]  # [int, int, int, int]
Proposals = List[Proposal]


def pad_sentence(b: List[Sentence], max_sentence_length: int, *, padding_value=0):
    b = [torch.tensor(s) for s in b]
    b = pad_sequence(b, batch_first=True, padding_value=padding_value)
    b = b[..., :max_sentence_length]
    return b


def pad_queries(b: List[Queries], max_query_length: int, *, val=0):
    to_tensor = lambda qs: [torch.tensor(q) for q in qs]
    cap_length = lambda qs: [q[..., :max_query_length] for q in qs]
    pad_smaller = lambda qs: [
        pad(q, (0, max_query_length - q.shape[0]), value=val) for q in qs
    ]

    b = [to_tensor(qs) for qs in b]
    b = [cap_length(qs) for qs in b]
    b = [pad_smaller(qs) for qs in b]
    b = [torch.stack(qs) for qs in b]
    b = pad_sequence(b, batch_first=True, padding_value=val)

    return b


def pad_proposals(b: List[Proposals], max_proposal_length, *, val=0):
    b = [torch.tensor(p) for p in b]
    b = pad_sequence(b, batch_first=True, padding_value=val)
    b = b[..., :max_proposal_length, :]
    return b


def pad_targets(b, *, val=0):
    to_tensor = lambda ts: [torch.tensor(t) for t in ts]

    b = [to_tensor(qs) for qs in b]
    b = [torch.stack(qs) for qs in b]
    b = pad_sequence(b, batch_first=True, padding_value=val)

    return b