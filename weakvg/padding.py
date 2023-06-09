from typing import List

import torch
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence

Sentence = List[int]
Query = List[int]
Queries = List[Query]
Proposal = List[int]  # [int, int, int, int]
Proposals = List[Proposal]
LabelsAlternatives = List[int]
LabelsSyn = List[LabelsAlternatives]
Labels = List[int]


def pad_sentence(b: List[Sentence], max_sentence_length: int, *, padding_value=0):
    b = [torch.tensor(s) for s in b]
    b = pad_sequence(b, batch_first=True, padding_value=padding_value)
    b = b[..., :max_sentence_length]
    return b


def pad_queries(b: List[Queries], max_query_num=None, max_query_length=None, *, val=0):
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
    b = b[..., :max_query_num, :max_query_length]

    return b


def pad_proposals(b: List[Proposals], max_proposal_length, *, val=0):
    b = [torch.tensor(p) for p in b]
    b = pad_sequence(b, batch_first=True, padding_value=val)
    b = b[..., :max_proposal_length, :]
    return b


def pad_labels(b: List[Labels], max_labels_length, *, val=0):
    b = [torch.tensor(p) for p in b]
    b = pad_sequence(b, batch_first=True, padding_value=val)
    b = b[..., :max_labels_length]
    return b


def pad_labels_syn(
    b: List[LabelsSyn], max_labels_length, max_alternatives_length, *, val=0
):
    to_tensor = lambda ls: [torch.tensor(l) for l in ls]
    cap_alternatives = lambda ls: [l[..., :max_alternatives_length] for l in ls]
    cap_labels = lambda ls: ls[:max_labels_length]
    pad_alternatives = lambda ls: [
        pad(l, (0, max_alternatives_length - l.shape[0]), value=val) for l in ls
    ]
    pad_labels = lambda ls, n_max_labels: [
        pad(l, (0, 0, 0, n_max_labels - l.shape[0]), value=val) for l in ls
    ]

    b = [to_tensor(ps) for ps in b]
    b = [cap_alternatives(ps) for ps in b]
    b = [cap_labels(ps) for ps in b]
    b = [pad_alternatives(ps) for ps in b]
    b = [torch.stack(ps) for ps in b]
    n_max_labels = max([len(ps) for ps in b])
    b = pad_labels(b, n_max_labels)
    b = torch.stack(b)

    return b


def pad_targets(b, max_targets_num=None, *, val=0):
    to_tensor = lambda ts: [torch.tensor(t) for t in ts]

    b = [to_tensor(qs) for qs in b]
    b = [torch.stack(qs) for qs in b]
    b = pad_sequence(b, batch_first=True, padding_value=val)
    b = b[..., :max_targets_num, :]

    return b


def pad_locations(b, max_locations_num=None, *, val=0):
    b = [torch.tensor(p) for p in b]
    b = pad_sequence(b, batch_first=True, padding_value=val)
    b = b[..., :max_locations_num, :]
    return b
