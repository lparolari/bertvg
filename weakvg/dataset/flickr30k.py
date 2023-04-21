import logging
import os
import re
from typing import Any, Dict, List

import numpy as np
from torch.utils.data import Dataset

from weakvg.dataset.mixin import UpperboundAccuracyMixin, QueryNumberMixin
from weakvg.repo import (
    HeadsRepository,
    ImagesSizeRepository,
    LabelsRepository,
    ObjectsDetectionRepository,
    ObjectsFeatureRepository,
)
from weakvg.dataset.shared import get_locations, get_relations

Box = List[int]


class Flickr30kDatum:
    def __init__(
        self,
        identifier: int,
        *,
        data_dir: str,
        precomputed: Dict[str, Dict[int, Any]],
        nlp=None,
    ):
        self.identifier = identifier
        self.data_dir = data_dir
        self.precomputed = precomputed
        self.nlp = nlp

        self._sentences_ann = None
        self._targets_ann = None

        self._load_sentences_ann()
        self._load_targets_ann()

    def get_sentences_ann(self) -> List[str]:
        return self._sentences_ann

    def get_sentences_ids(self) -> List[int]:
        return list(range(len(self.get_sentences_ann())))

    def get_sentence(self, sentence_id, *, return_ann=False) -> str:
        sentence_ann = self._sentences_ann[sentence_id]
        sentence = self._remove_sentence_ann(sentence_ann)
        sentence = self._process_text(sentence)

        if return_ann:
            return sentence, sentence_ann

        return sentence

    def get_queries_ids(self, sentence_id) -> List[int]:
        return list(range(len(self.get_queries(sentence_id))))

    def get_queries(self, sentence_id, query_id=None, *, return_ann=False) -> List[str]:
        _, sentence_ann = self.get_sentence(sentence_id, return_ann=True)

        queries_ann = self._extract_queries_ann(sentence_ann)

        # important: we need to filter out queries without targets in order to
        #            produce aligned data
        queries_ann = [
            query_ann for query_ann in queries_ann if self.has_target_for(query_ann)
        ]

        queries = [self._extract_phrase(query_ann) for query_ann in queries_ann]
        queries = [self._process_text(query) for query in queries]

        a_slice = slice(query_id, query_id and query_id + 1)

        if return_ann:
            return queries[a_slice], queries_ann[a_slice]

        return queries[a_slice]

    def get_heads(self, sentence_id, query_id=None) -> List[str]:
        a_slice = slice(query_id, query_id and query_id + 1)

        if self._has_precomputed("heads"):
            return self.precomputed["heads"].get_heads(self.identifier, sentence_id)[
                a_slice
            ]

        queries = self.get_queries(sentence_id, query_id)

        if not self.nlp:
            # as fallback we return the full query, which, instead
            # of being a specific part of the query is the query itself
            return queries

        def get_head(cs) -> str:
            heads = [c.root.text for c in cs]
            head = " ".join(heads)
            return head

        docs = [self.nlp(query) for query in queries]
        chunks = [doc.noun_chunks for doc in docs]
        heads = [get_head(cs) for cs in chunks]

        fallbacks = [doc[-1].text for doc in docs]

        heads = [head if head else fallback for head, fallback in zip(heads, fallbacks)]

        return heads[a_slice]

    def get_targets(self, sentence_id) -> List[List[int]]:
        targets_ann = self._targets_ann
        _, queries_ann = self.get_queries(sentence_id, return_ann=True)

        return [
            targets_ann[self._get_entity_id(query_ann)] for query_ann in queries_ann
        ]

    def get_image_w(self) -> int:
        if "images_size" not in self.precomputed:
            raise NotImplementedError(
                "Extracting image width is not supported, please provide precomputed data"
            )
        return self.precomputed["images_size"].get_width(self.identifier)

    def get_image_h(self) -> int:
        return self.precomputed["images_size"].get_height(self.identifier)

    def get_proposals(self) -> List[List[int]]:
        return self.precomputed["objects_detection"].get_boxes(self.identifier)

    def get_labels(self) -> List[str]:
        return self.precomputed["objects_detection"].get_classes(self.identifier)

    def get_labels_raw(self) -> List[str]:
        if not self._has_precomputed("labels"):
            return self.get_labels()

        return [
            self.precomputed["labels"].get_raw(label) for label in self.get_labels()
        ]

    def get_labels_syn(self) -> List[List[str]]:
        if not self._has_precomputed("labels"):
            return [[label] for label in self.get_labels()]

        get_alternatives = self.precomputed["labels"].get_alternatives

        return [get_alternatives(label) for label in self.get_labels()]

    def get_attrs(self) -> List[str]:
        return self.precomputed["objects_detection"].get_attrs(self.identifier)

    def get_proposals_feat(self) -> np.array:
        return self.precomputed["objects_feature"].get_feature(
            self.identifier
        )  # [x, 2048]

    def get_locations(self, sentence_id, query_id=None) -> List[List[int]]:
        queries = self.get_queries(sentence_id, query_id)
        a_slice = slice(query_id, query_id and query_id + 1)
        return get_locations(queries)[a_slice]

    def get_relations(self):
        proposals = self.get_proposals()
        labels = self.get_labels()

        return get_relations(proposals, labels)

    def has_queries_for(self, sentence_id) -> bool:
        return len(self.get_queries(sentence_id)) > 0

    def has_target_for(self, query_ann: str) -> bool:
        return self._get_entity_id(query_ann) in self._targets_ann

    def __iter__(self):
        sentence_ids = self.get_sentences_ids()

        for sentence_id in sentence_ids:
            if not self.has_queries_for(sentence_id):
                continue

            yield {
                "identifier": self.identifier,
                # text
                "sentence": self.get_sentence(sentence_id),
                "queries": self.get_queries(sentence_id),
                "heads": self.get_heads(sentence_id),
                # image
                "image_w": self.get_image_w(),
                "image_h": self.get_image_h(),
                # box
                "proposals": self.get_proposals(),
                "labels": self.get_labels(),
                "attrs": self.get_attrs(),
                "labels_syn": self.get_labels_syn(),
                "labels_raw": self.get_labels_raw(),
                # feats
                "proposals_feat": self.get_proposals_feat(),
                # targets
                "targets": self.get_targets(sentence_id),
                # relations
                "locations": self.get_locations(sentence_id),
                "relations": self.get_relations(),
            }

    def _get_sentences_file(self) -> str:
        return os.path.join(
            self.data_dir, "Flickr30kEntities", "Sentences", f"{self.identifier}.txt"
        )

    def _load_sentences_ann(self) -> List[str]:
        with open(self._get_sentences_file(), "r") as f:
            sentences = [x.strip() for x in f]

        self._sentences_ann = sentences

    def _load_targets_ann(self):
        from xml.etree.ElementTree import parse

        targets_ann = {}

        annotation_file = os.path.join(
            self.data_dir, "Flickr30kEntities", "Annotations", f"{self.identifier}.xml"
        )

        root = parse(annotation_file).getroot()
        elements = root.findall("./object")

        for element in elements:
            bndbox = element.find("bndbox")

            if bndbox is None or len(bndbox) == 0:
                continue

            left = int(element.findtext("./bndbox/xmin"))
            top = int(element.findtext("./bndbox/ymin"))
            right = int(element.findtext("./bndbox/xmax"))
            bottom = int(element.findtext("./bndbox/ymax"))

            for name in element.findall("name"):
                entity_id = int(name.text)

                if not entity_id in targets_ann.keys():
                    targets_ann[entity_id] = []
                targets_ann[entity_id].append([left, top, right, bottom])

        self._targets_ann = targets_ann

    def _has_precomputed(self, key: str) -> bool:
        return key in self.precomputed and self.precomputed[key] is not None

    @staticmethod
    def _remove_sentence_ann(sentence: str) -> str:
        return re.sub(r"\[[^ ]+ ", "", sentence).replace("]", "")

    @staticmethod
    def _extract_queries_ann(sentence_ann: str) -> List[str]:
        query_pattern = r"\[(.*?)\]"
        queries_ann = re.findall(query_pattern, sentence_ann)
        return queries_ann

    @staticmethod
    def _extract_phrase(query_ann: str) -> str:
        """
        Extracts the phrase from the query annotation

        Example:
          '/EN#283585/people A young white boy' --> 'A young white boy'
        """
        return query_ann.split(" ", 1)[1]

    @staticmethod
    def _extract_entity(query_ann: str) -> str:
        """
        Extracts the entity from the query annotation

        Example:
          '/EN#283585/people A young white boy' --> '/EN#283585/people'
        """
        return query_ann.split(" ", 1)[0]

    @staticmethod
    def _get_entity_id(query_ann: str) -> int:
        """
        Extracts the entity id from the query annotation

        Example:
          '/EN#283585/people A young white boy' --> 283585
        """
        entity = Flickr30kDatum._extract_entity(query_ann)

        entity_id_pattern = r"\/EN\#(\d+)"

        matches = re.findall(entity_id_pattern, entity)
        first_match = matches[0]

        return int(first_match)

    @staticmethod
    def _process_text(text: str) -> str:
        return text.lower()


class Flickr30kDataset(Dataset, UpperboundAccuracyMixin, QueryNumberMixin):
    def __init__(self, split, data_dir, tokenizer, vocab, nlp=None, transform=None):
        self.data_dir = data_dir
        self.split = split
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.nlp = nlp
        self.transform = transform

        self.identifiers = None
        """A list identifiers for each image-sentence pair"""

        self.data = None
        """A list of image-sentence pair datum"""

        self.samples = None
        """A list of batch-able samples for the model"""

        self.load()
        self.preflight_check()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        meta = [idx, sample["identifier"]]
        sentence = self._prepare_sentence(sample["sentence"])
        queries = self._prepare_queries(sample["queries"])
        heads = self._prepare_heads(sample["heads"])
        image_w = sample["image_w"]
        image_h = sample["image_h"]
        proposals = sample["proposals"]
        labels = self._prepare_labels(sample["labels"])
        attrs = self._prepare_labels(sample["attrs"])
        labels_raw = self._prepare_labels(sample["labels_raw"])
        labels_syn = self._prepare_labels_syn(sample["labels_syn"])
        proposals_feat = sample["proposals_feat"]
        targets = self._prepare_targets(sample["targets"])
        locations = sample["locations"]
        relations = sample["relations"]

        assert len(queries) == len(
            targets
        ), f"Expected length of `targets` to be {len(queries)}, got {len(targets)}"
        assert len(proposals) == len(
            labels
        ), f"Expected length of `labels` to be {len(proposals)}, got {len(labels)}"
        assert len(proposals) == len(
            attrs
        ), f"Expected length of `attrs` to be {len(proposals)}, got {len(attrs)}"
        assert len(proposals) == len(
            proposals_feat
        ), f"Expected length of `proposals_feat` to be {len(proposals)}, got {len(proposals_feat)}"

        item = {
            "meta": meta,
            "sentence": sentence,
            "queries": queries,
            "heads": heads,
            "image_w": image_w,
            "image_h": image_h,
            "proposals": proposals,
            "labels": labels,
            "attrs": attrs,
            "labels_raw": labels_raw,
            "labels_syn": labels_syn,
            "proposals_feat": proposals_feat,
            "targets": targets,
            "locations": locations,
            "relations": relations,
        }

        if self.transform:
            item = self.transform(item)

        return item

    def load(self):
        self._load_identifiers()

        images_size_repo = self._open_images_size()
        objects_detection_repo = self._open_objects_detection()
        objects_feature_repo = self._open_objects_feature()
        heads_repo = self._open_heads()
        labels_repo = self._open_labels_repo()

        precomputed = {
            "images_size": images_size_repo,
            "objects_detection": objects_detection_repo,
            "objects_feature": objects_feature_repo,
            "heads": heads_repo,
            "labels": labels_repo,
        }

        samples = []
        data = []

        logging.info(f"Loading {len(self.identifiers)} image-sentence pairs")

        for identifier in self.identifiers:
            datum = Flickr30kDatum(
                identifier,
                data_dir=self.data_dir,
                precomputed=precomputed,
                nlp=self.nlp,
            )

            data.append(datum)

            for sample in datum:
                samples.append(sample)

        self.samples = samples
        self.data = data

    def preflight_check(self):
        if len(self.identifiers) == 0:
            raise RuntimeError("Empty dataset, please check the identifiers file")

        if len(self.samples) == 0:
            raise RuntimeError("Cannot create dataset with 0 samples")

    def get_split(self):
        return self.split

    def print_statistics(self):
        print(f"Flickr30k ({self.split})")
        print(f"Number of images-sentences pairs: {len(self.data)}")
        print(f"Number of samples: {len(self)}")
        print(f"Upperbound accuracy: {self.get_upperbound_accuracy() * 100:.2f}%")
        print(f"Min number of queries: {self.get_min_query_number()}")
        print(f"Max number of queries: {self.get_max_query_number()}")
        print(f"Avg number of queries: {self.get_avg_query_number():.2f}")

    def get_image_path(self, image_id):
        return f"{self.data_dir}/flickr30k_images/{image_id}.jpg"

    def _prepare_sentence(self, sentence: str) -> List[int]:
        sentence = self.tokenizer(sentence, max_length=32)
        sentence = self.vocab(sentence)
        return sentence

    def _prepare_queries(self, queries: List[str]) -> List[List[int]]:
        queries = [self.tokenizer(query, max_length=12) for query in queries]
        queries = [self.vocab(query) for query in queries]
        return queries

    def _prepare_heads(self, heads: List[str]) -> List[List[int]]:
        # we use a custom tokenizer for heads that splits on spaces ignoring
        # rules on sub-words (like in BERT)
        heads = [head.split() for head in heads]
        heads = [self.vocab(head) for head in heads]
        return heads

    def _prepare_labels(self, labels: List[str]) -> List[int]:
        return self.vocab(labels)

    def _prepare_labels_syn(self, labels: List[List[str]]) -> List[List[int]]:
        return [self.vocab(alternatives) for alternatives in labels]

    def _prepare_targets(self, targets: List[List[Box]]) -> List[Box]:
        return [union_box(target) for target in targets]

    def _load_identifiers(self):
        identifier_file = os.path.join(
            self.data_dir, "Flickr30kEntities", self.split + ".txt"
        )

        with open(identifier_file, "r") as f:
            identifiers = f.readlines()

        identifiers = [int(identifier.strip()) for identifier in identifiers]

        self.identifiers = identifiers

    def _open_images_size(self):
        images_size_file = os.path.join(self.data_dir, self.split + "_images_size.json")
        return ImagesSizeRepository(images_size_file)

    def _open_objects_detection(self):
        objects_detection_file = os.path.join(
            self.data_dir, self.split + "_detection_dict.json"
        )
        return ObjectsDetectionRepository(objects_detection_file)

    def _open_objects_feature(self):
        id2idx_path = os.path.join(self.data_dir, self.split + "_imgid2idx.pkl")
        objects_feature_file = os.path.join(
            self.data_dir, self.split + "_features_compress.hdf5"
        )
        return ObjectsFeatureRepository(id2idx_path, objects_feature_file)

    def _open_heads(self):
        heads_file = os.path.join(self.data_dir, self.split + "_heads.json")

        if not os.path.exists(heads_file):
            return None

        return HeadsRepository(heads_file)

    def _open_labels_repo(self):
        labels_file = os.path.join(self.data_dir, "objects_vocab.txt")
        alternatives_file = os.path.join(self.data_dir, "objects_vocab_merged.txt")

        return LabelsRepository.from_vocab(labels_file, alternatives_file)


def union_box(boxes: List[Box]):
    x1 = min([box[0] for box in boxes])
    y1 = min([box[1] for box in boxes])
    x2 = max([box[2] for box in boxes])
    y2 = max([box[3] for box in boxes])
    return [x1, y1, x2, y2]


if __name__ == "__main__":
    # Run with: python -m weakvg.dataset

    import argparse
    import logging

    from weakvg.wordvec import get_nlp, get_objects_vocab, get_tokenizer, get_wordvec

    logging.basicConfig(level=logging.DEBUG)

    args = argparse.ArgumentParser(
        description="Load Flickr30k Dataset and print its statistics"
    )
    args.add_argument("--split", type=str, default="val")
    args.add_argument("--load_objects_vocab", action="store_true", default=False)

    args = args.parse_args()

    load_objects_vocab = args.load_objects_vocab
    split = args.split

    tokenizer = get_tokenizer()
    nlp = get_nlp()
    wordvec, vocab = get_wordvec(
        custom_labels=get_objects_vocab() if load_objects_vocab else []
    )

    ds = Flickr30kDataset(
        split=split,
        data_dir="data/flickr30k",
        tokenizer=tokenizer,
        vocab=vocab,
        nlp=nlp,
    )

    ds.print_statistics()
