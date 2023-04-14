import pytorch_lightning as pl
import torch
import torch.nn as nn

from weakvg.loss import Loss
from weakvg.masking import (
    get_concepts_mask_,
    get_mask_,
    get_proposals_mask,
    get_proposals_mask_,
    get_queries_count,
    get_queries_mask,
    get_queries_mask_,
    get_relations_mask_,
)
from weakvg.utils import ext_textual, ext_visual, iou, mask_softmax, tlbr2ctwh


class WeakvgModel(pl.LightningModule):
    def __init__(
        self,
        wordvec,
        vocab,
        omega=0.5,
        lr=1e-5,
        neg_selection="random",
        use_relations=False,
    ) -> None:
        super().__init__()
        self.use_relations = use_relations
        self.lr = lr
        # TODO: temporarily freezed, revert back to False
        # we = WordEmbedding(wordvec, freeze=True)
        we_freezed = WordEmbedding(wordvec, freeze=True)
        self.concept_branch = ConceptBranch(word_embedding=we_freezed)
        self.visual_branch = VisualBranch(word_embedding=we_freezed)
        self.textual_branch = TextualBranch(word_embedding=we_freezed)
        self.prediction_module = SimilarityPredictionModule(omega=omega)
        self.loss = Loss(word_embedding=we_freezed, neg_selection=neg_selection)

        self.save_hyperparameters(ignore=["wordvec", "vocab"])

    def forward(self, x):
        concepts_pred = self.concept_branch(x)  # [b, q, b, p]

        visual_feat = self.visual_branch(x)  # [b, p, d], [b, p, 1]

        textual_feat = self.textual_branch(x)  # [b, q, d], [b, q, 1]

        visual_mask = get_proposals_mask_(x)  # [b, p]
        textual_mask = get_queries_mask_(x)[1]  # [b, q]
        concepts_mask = get_concepts_mask_(x)  # [b, q, b, p]

        if self.use_relations:
            relations_mask = get_relations_mask_(x)  # [b, q, b, p]
            concepts_mask = concepts_mask.logical_and(relations_mask)  # [b, q, b, p]

        scores, (multimodal_scores, concepts_scores) = self.prediction_module(
            (visual_feat, visual_mask),
            (textual_feat, textual_mask),
            (concepts_pred, concepts_mask),
        )  # [b, q, b, p], ([b, q, b, p], [b, q, b, p])

        scores = scores.masked_fill(~get_mask_(x), 0)

        return scores, (multimodal_scores, concepts_scores)

    def step(self, batch, batch_idx):
        """
        :return: A tuple `(loss, metrics)`, where metrics is a dict with `acc`, `point_it`
        """
        queries = batch["queries"]  # [b, q, w]
        proposals = batch["proposals"]  # [b, p, 4]
        targets = batch["targets"]  # [b, q, 4]

        scores, _ = self.forward(batch)  # [b, q, b, p]

        loss = self.loss({**batch, "scores": scores})

        candidates, _ = self.predict_candidates(scores, proposals)  # [b, q, 4]

        acc = self.accuracy(
            candidates, targets, queries
        )  # TODO: refactor -> avoid passing queries whenever possible

        point_it = self.point_it(candidates, targets, queries)

        metrics = {
            "acc": acc,
            "point_it": point_it,
        }

        return loss, metrics

    def training_step(self, batch, batch_idx):
        loss, metrics = self.step(batch, batch_idx)

        acc = metrics["acc"]
        point_it = metrics["point_it"]

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_pointit", point_it, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self.step(batch, batch_idx)

        acc = metrics["acc"]
        point_it = metrics["point_it"]

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_pointit", point_it, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, metrics = self.step(batch, batch_idx)

        acc = metrics["acc"]
        point_it = metrics["point_it"]

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_pointit", point_it, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def predict_candidates(self, scores, proposals):
        """
        Predict a candidate bounding box for each query

        :param scores: A tensor of shape [b, q, b, p] with the scores
        :param proposals: A tensor of shape [b, p, 4] with the proposals
        :return: A tensor of shape [b, q, 4] with the predicted candidates
        """
        b = scores.shape[0]
        q = scores.shape[1]
        p = scores.shape[3]

        index = torch.arange(b).to(self.device)  # [b]
        index = index.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [b, 1, 1, 1]
        index = index.repeat(1, q, 1, p)  # [b, q, 1, p]

        # select only positive scores
        scores = scores.gather(-2, index).squeeze(-2)  # [b, q, p]

        # find best best proposal index
        best_idx = scores.argmax(-1)  # [b, q]

        select_idx = best_idx.unsqueeze(-1).repeat(1, 1, 4)  # [b, q, 4]

        candidates = proposals.gather(-2, select_idx)  # [b, q, 4]

        return candidates, best_idx

    def accuracy(self, candidates, targets, queries):
        # scores: [b, q, b, p]
        # candidates: [b, p, 4]
        # targets: [b, q, 4]
        # queries: [b, q, w]

        thresh = 0.5

        scores = iou(candidates, targets)  # [b, q]

        matches = scores >= thresh  # [b, q]

        # mask padding queries
        _, is_query = get_queries_mask(queries)  # [b, q, w], [b, q]
        _, n_queries = get_queries_count(queries)  # [b, q], [b]

        matches = matches.masked_fill(~is_query, False)  # [b, q]

        acc = matches.sum() / n_queries.sum()

        return acc

    def point_it(self, candidates, targets, queries):
        # candidates: [b, q, 4]
        # targets: [b, q, 4]
        # queries: [b, q, w]

        centers = tlbr2ctwh(candidates)[..., :2]  # [b, q, 2]

        topleft = targets[..., :2]  # [b, q, 2]
        bottomright = targets[..., 2:]  # [b, q, 2]

        # count a match whether center is inside the target
        matches = (centers >= topleft) & (centers <= bottomright)  # [b, q, 2]
        matches = matches.all(-1)  # [b, q]

        # mask padding queries
        _, is_query = get_queries_mask(queries)  # [b, q, w], [b, q]
        _, n_queries = get_queries_count(queries)  # [b, q], [b]

        matches = matches.masked_fill(~is_query, False)  # [b, q]

        point_it = matches.sum() / n_queries.sum()

        return point_it


class SimilarityPredictionModule(nn.Module):
    def __init__(self, omega):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.omega = omega
        # self.

    def forward(self, visual, textual, concepts):
        """
        :param visual: A tuple `(visual_feat, visual_mask)`, where `visual_feat` is a
            tensor of shape `[b, p, d]` and `visual_mask` is a tensor of shape `[b, p]`

        :param textual: A tuple `(textual_feat, textual_mask)`, where `textual_feat`
            is a tensor of shape `[b, q, d]` and `textual_mask` is a tensor of
            shape `[b, q]`

        :param concepts: A tuple `(concepts_pred, concepts_mask)`, where `concepts_pred`
            is a tensor of shape `[b, q, b, p]` and `concepts_mask` is a tensor of
            shape `[b, q, b, p]`

        :return: A tensor of shape `[b, q, b, p], ([b, q, b, p], [b, q, b, p])` with the
            similarity scores and the two predictions of the underlying models: the
            multimodal prediction and the concepts prediction
        """
        multimodal_pred, multimodal_mask = self.predict_multimodal(
            visual, textual
        )  # [b, q, b, p], [b, q, b, p]
        concepts_pred, concepts_mask = self.predict_concepts(
            concepts
        )  # [b, q, b, p], [b, q, b, p]

        scores = self.apply_prior(multimodal_pred, concepts_pred)  # [b, q, b, p]

        mask = multimodal_mask & concepts_mask  # [b, q, b, p]

        scores = scores.masked_fill(~mask, 0)  # [b, q, b, p]

        return scores, (multimodal_pred, concepts_pred)

    def predict_multimodal(self, visual, textual):
        visual_feat, visual_mask = visual  # [b, q, d], [b, q]
        textual_feat, textual_mask = textual  # [b, p, d], [b, p]

        b = textual_feat.shape[0]
        q = textual_feat.shape[1]
        p = visual_feat.shape[1]

        visual_feat, visual_mask = ext_visual(visual_feat, visual_mask, b, q, p)
        textual_feat, textual_mask = ext_textual(textual_feat, textual_mask, b, q, p)

        multimodal_mask = visual_mask & textual_mask  # [b, q, b, p]

        multimodal_pred = torch.cosine_similarity(
            visual_feat, textual_feat, dim=-1
        )  # [b, q, b, p]
        multimodal_pred = mask_softmax(multimodal_pred, multimodal_mask)  # [b, p, b, p]
        multimodal_pred = self.softmax(multimodal_pred)  # [b, p, b, p]

        return multimodal_pred, multimodal_mask

    def predict_concepts(self, concepts):
        concepts_pred, concepts_mask = concepts  # [b, q, b, p], [b, q, b, p]

        concepts_pred = mask_softmax(concepts_pred, concepts_mask)  # [b, q, b, p]
        concepts_pred = self.softmax(concepts_pred)  # [b, q, b, p]

        return concepts_pred, concepts_mask

    def apply_prior(self, predictions, prior):
        w = self.omega

        return w * predictions + (1 - w) * prior  # [b, q, b, p]


class WordEmbedding(nn.Module):
    def __init__(self, wordvec, *, freeze=True):
        super().__init__()

        from torchtext.vocab import Vectors
        from transformers import BertModel

        if issubclass(type(wordvec), Vectors):
            self.factory = "vectors"
            self.emb = nn.Embedding.from_pretrained(
                wordvec.vectors, freeze=freeze, padding_idx=0
            )

        if issubclass(type(wordvec), BertModel):
            self.factory = "bert"
            self.bert = wordvec
            self.lin = nn.Linear(768, 300)

            for param in self.bert.parameters():
                param.requires_grad = not freeze

            nn.init.xavier_uniform_(self.lin.weight)
            nn.init.zeros_(self.lin.bias)

    def forward(self, x, *args, **kwargs):
        if self.factory == "bert":
            mask = args[0]
            return self.forward_bert(x, mask, **kwargs)

        if self.factory == "vectors":
            return self.emb(x)

        raise NotImplementedError(
            f"WordEmbedding does not implement '{self.factory}' factory"
        )

    def forward_bert(self, x, mask, **kwargs):
        kwargs = {"return_dict": False} | kwargs

        sh = x.shape

        input_ids = x.reshape(-1, sh[-1])
        attention_mask = mask.reshape(-1, sh[-1]).long()

        out, _ = self.bert(input_ids, attention_mask, **kwargs)

        out = out.reshape(*sh, -1)

        out = out.masked_fill(~mask.unsqueeze(-1), 0)
        out = self.lin(out)
        out = out.masked_fill(~mask.unsqueeze(-1), 0)

        return out


class ConceptBranch(nn.Module):
    def __init__(self, word_embedding):
        super().__init__()
        self.we = word_embedding
        self.sim_fn = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, x):
        heads = x["heads"]  # [b, q, h]
        labels = x["labels"]  # [b, p]

        heads_mask = heads != 0  # [b, q, h]
        n_heads = heads_mask.sum(-1).unsqueeze(-1)  # [b, q, 1]

        label_mask = labels != 0

        heads_e = self.we(heads, heads_mask)  # [b, q, h, d]
        heads_e = heads_e.masked_fill(~heads_mask.unsqueeze(-1), 0.)
        heads_e = heads_e.sum(-2) / n_heads.clamp(1)  # [b, q, d]  - clamp is required to avoid div by 0
        heads_e = heads_e.unsqueeze(-2).unsqueeze(-2)  # [b, q, 1, 1, d]

        labels_e = self.we(labels, label_mask)  # [b, p, d]
        labels_e = labels_e.unsqueeze(0).unsqueeze(0)  # [1, 1, b, p, d]

        scores = self.sim_fn(heads_e, labels_e)  # [b, q, b, p]

        scores = scores.masked_fill(~get_concepts_mask_(x), 0)

        return scores


class VisualBranch(nn.Module):
    def __init__(self, word_embedding):
        super().__init__()

        self.we = word_embedding

        self.fc = nn.Linear(2048 + 5, 300)
        self.act = nn.LeakyReLU()

        self._init_weights()

    def forward(self, x):
        proposals = x["proposals"]  # [b, p, 4]
        proposals_feat = x["proposals_feat"]  # [b, p, v]
        labels = x["labels"]  # [b, p]

        mask = get_proposals_mask(proposals)  # [b, p]

        labels_e = self.we(labels, mask)  # [b, p, d]
        spat = self.spatial(x)  # [b, p, 5]

        proj = self.project(proposals_feat, spat)  # [b, p, d]
        fusion = proj + labels_e  # [b, p, d]

        fusion = fusion.masked_fill(~mask.unsqueeze(-1), 0)

        return fusion

    def spatial(self, x):
        """
        Compute spatial features for each proposals as [x1, y1, x2, y2, area] assuming
        that coordinates are already normalized to [0, 1].
        """
        proposals = x["proposals"]  # [b, p, 4]

        x1, y1, x2, y2 = proposals.unbind(-1)  # [b, p], [b, p], [b, p], [b, p]

        area = (x2 - x1) * (y2 - y1)  # [b, p]

        spat = torch.stack([x1, y1, x2, y2, area], dim=-1)

        return spat

    def project(self, proposals_feat, spat):
        viz = torch.cat([proposals_feat, spat], dim=-1)  # [b, p, v + 5]

        proj = self.fc(viz)  # [b, p, d]
        proj = self.act(proj)

        return proj

    def _init_weights(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)


class TextualBranch(nn.Module):
    def __init__(self, word_embedding):
        super().__init__()
        self.we = word_embedding
        self.lstm = nn.LSTM(300, 300)

    def forward(self, x):
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

        queries = x["queries"]

        b = queries.shape[0]
        q = queries.shape[1]
        w = queries.shape[2]

        is_word, is_query = get_queries_mask(queries)
        n_words, _ = get_queries_count(queries)

        queries_e = self.we(queries, is_word)  # [b, q, w, d]

        d = queries_e.shape[-1]

        queries_e = queries_e.view(-1, w, d)
        queries_e = queries_e.permute(1, 0, 2).contiguous()

        # required on CPU by pytorch, see
        # https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
        lengths = n_words.view(-1).cpu()
        lengths = lengths.clamp(min=1)  # elements with 0 are not accepted

        queries_packed = pack_padded_sequence(queries_e, lengths, enforce_sorted=False)

        output, hidden = self.lstm(queries_packed)

        queries_x, lengths = pad_packed_sequence(output)
        # queries_x is a tensor with shape [l, b * q, d], where l is the max length
        # of the non-padded sequence

        lengths = lengths.to(queries.device)  # back to original device

        # we need to gather the representation of the last word, so we can build an
        # index based on lengths. for example, if we have a query with length 4, its
        # index will be 3

        index = lengths - 1  # [b * q]
        index = index.unsqueeze(0).unsqueeze(-1)  # [1, b * q, 1]
        index = index.repeat(1, 1, d)  # [1, b * q, d]

        queries_x = queries_x.gather(0, index)  # [1, b * q, d]

        queries_x = queries_x.permute(1, 0, 2).contiguous()  # [b * q, 1, d]
        queries_x = queries_x.squeeze(1)  # [b * q, d]
        queries_x = queries_x.view(b, q, d)  # [b, q, d]

        mask = is_query.unsqueeze(-1)

        queries_x = queries_x.masked_fill(~mask, 0)

        return queries_x
