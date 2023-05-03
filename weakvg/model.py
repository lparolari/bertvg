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
    get_multimodal_mask,
)
from weakvg.utils import ext_textual, ext_visual, iou, mask_softmax, tlbr2ctwh


class WeakvgModel(pl.LightningModule):
    def __init__(
        self,
        wordvec,
        vocab,
        lr=1e-5,
        omega=0.5,
        hidden_size=300,
        neg_selection="random",
        use_relations=False,
    ) -> None:
        super().__init__()
        self.vocab = vocab  # for debugging purposes - decode queries indexes

        # hyper parameters
        self.use_relations = use_relations
        self.lr = lr
        self.hidden_size = hidden_size

        # modules
        self.we = WordEmbedding(wordvec, hid_size=hidden_size, freeze=True)
        # self.concept_branch = ConceptBranch()
        self.visual_branch = VisualBranch(hid_size=hidden_size)
        # self.prediction_module = SimilarityPredictionModule(omega=omega)
        self.loss = Loss(neg_selection=neg_selection)
        self.heads_encoder = HeadsEncoder()
        self.labels_encoder = LabelsEncoder(word_embedding=self.we)

        self.save_hyperparameters(ignore=["wordvec", "vocab"])

    def forward(self, x):
        queries = x["queries"]
        proposals = x["proposals"]

        is_word, _ = get_queries_mask_(x)

        queries_e, queries_pooler = self.we(
            queries, is_word, return_pooler=True
        )  # [b, q, w, d], [b, q, d]

        x |= {"queries_e": queries_e}

        heads_e = self.heads_encoder(x)  # [b, q, d]
        labels_e = self.labels_encoder(x)  # [b, p, q, d]

        x |= {"heads_e": heads_e, "labels_e": labels_e}

        b = queries.shape[0]
        q = queries.shape[1]
        p = proposals.shape[1]

        visual_feat = self.visual_branch(x)  # [b, p, q, d]
        visual_feat = (
            visual_feat.transpose(0, 2).reshape(1, q, b, p, -1).repeat(b, 1, 1, 1, 1)
        )  # [b, q, b, p, d]
        textual_feat = queries_pooler.reshape(b, q, 1, 1, -1).repeat(
            1, 1, b, p, 1
        )  # [b, q, b, p, d]

        multimodal_mask = get_multimodal_mask(queries, proposals)
        logits = torch.cosine_similarity(
            visual_feat, textual_feat, dim=-1
        )  # [b, q, b, p]
        logits = mask_softmax(logits, multimodal_mask)
        scores = torch.softmax(logits, dim=-1)  # [b, q, b, p]
        scores = scores.masked_fill(~multimodal_mask, 0)

        # TODO: concepts branch temporarily disabled
        return scores, (None, None)

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


class HeadsEncoder(nn.Module):
    def forward(self, x):
        assert "queries_e" in x, "Queries embedding is required"

        heads = x["heads"]  # [b, q, h]
        queries = x["queries"]  # [b, q, w]
        queries_e = x["queries_e"]  # [b, q, w, d]

        b = queries_e.shape[0]
        q = queries_e.shape[1]
        d = queries_e.shape[3]

        is_query = get_queries_mask(queries)[1].reshape(b, q, 1, 1)  # [b, q, 1, 1]

        heads_i = HeadsEncoder.get_heads_index(queries, heads)  # [b, q]

        heads_i = heads_i.reshape(b, q, 1, 1).repeat(1, 1, 1, d)  # [b, q, 1, d]
        heads_e = queries_e.gather(-2, heads_i)  # [b, q, 1, d]
        heads_e = heads_e.masked_fill(~is_query, 0)  # [b, q, 1, d]
        heads_e = heads_e.squeeze(-2)  # [b, q, d]

        return heads_e

    @staticmethod
    def get_heads_index(queries, heads):
        # queries: [b, q, w]
        # heads: [b, q, h]

        is_head = HeadsEncoder.get_heads_mask(queries, heads)  # [b, q, w]

        queries = queries * is_head  # [b, q, w]
        index = queries.argmax(-1)  # [b, q]
        return index

    @staticmethod
    def get_heads_mask(queries, heads):
        # queries: [b, q, w]
        # head: [b, q]

        is_query = get_queries_mask(queries)[1].unsqueeze(-1)  # [b, q, 1]

        head = HeadsEncoder._select_head(heads)  # [b, q]
        head = head.unsqueeze(-1)  # [b, q, 1]

        is_head = queries == head  # [b, q, w]
        is_head = is_head & is_query  # discard padding

        return is_head

    @staticmethod
    def _select_head(heads):
        # for sake of simplicity, we assume that there will be only
        # one head per query
        head = heads[..., 0]  # [b, q]
        return head


class LabelsEncoder(nn.Module):
    def __init__(self, word_embedding):
        super().__init__()
        self.we = word_embedding

    def forward(self, x):
        queries = x["queries"]  # [b, q, w]
        labels = x["labels"]  # [b, p]
        heads = x["heads"]  # [b, q, h]
        proposals = x["proposals"]  # [b, p, 4]

        b = queries.shape[0]
        q = queries.shape[1]
        w = queries.shape[2]
        p = labels.shape[1]

        head_mask = HeadsEncoder.get_heads_mask(queries, heads)  # [b, q, w]
        head_mask = head_mask.reshape(b, 1, q, w).repeat(1, p, 1, 1)  # [b, p, q, w]
        queries_ext = queries.reshape(b, 1, q, w).repeat(1, p, 1, 1)  # [b, p, q, w]
        labels_ext = labels.reshape(b, p, 1, 1).repeat(1, 1, q, w)  # [b, p, q, w]

        queries_l = self.replace_heads_with_labels(
            queries_ext, labels_ext, head_mask
        )  # [b, p, q, w]

        is_word = (
            get_queries_mask(queries)[0].reshape(b, 1, q, w).repeat(1, p, 1, 1)
        )  # [b, p, q, w]
        queries_e = self.we(queries_l, is_word)  # [b, p, q, w, d]

        d = queries_e.shape[-1]

        label_i = HeadsEncoder.get_heads_index(queries, heads)  # [b, q]
        label_i = label_i.reshape(b, 1, q, 1, 1).repeat(
            1, p, 1, 1, d
        )  # [b, p, q, 1, d]

        labels_e = queries_e.gather(-2, label_i)  # [b, p, q, 1, d]
        labels_e = labels_e.squeeze(-2)  # [b, p, q, d]

        is_query = (
            get_queries_mask(queries)[1].reshape(b, 1, q, 1).repeat(1, p, 1, 1)
        )  # [b, p, q, 1]
        is_proposal = (
            get_proposals_mask(proposals).reshape(b, p, 1, 1).repeat(1, 1, q, 1)
        )  # [b, p, q, 1]
        labels_e = labels_e.masked_fill(~(is_query & is_proposal), 0)  # [b, p, q, d]

        return labels_e

    def replace_heads_with_labels(self, queries, labels, where):
        where = where.long()
        return queries * (1 - where) + labels * where


class WordEmbedding(nn.Module):
    def __init__(self, wordvec, hid_size=768, *, freeze=True):
        super().__init__()
        self.bert = wordvec

        for param in self.bert.parameters():
            param.requires_grad = not freeze

        self.hid_size = hid_size
        self.wordemb_size = self.bert.config.hidden_size

        if self.wordemb_size != self.hid_size:
            self.lin = nn.Linear(self.wordemb_size, self.hid_size)

            nn.init.xavier_uniform_(self.lin.weight)
            nn.init.zeros_(self.lin.bias)

    def forward(self, x, mask, **kwargs):
        """
        :param x: A tensor of shape `[*, w]` with the input ids
        :param mask: A tensor of shape `[*, w]` with the attention mask
        :return: A tensor of shape `[*, w, d]` with the word embeddings
            (d = hid_size) and a tensor of shape `[*, d]` with the pooler
            output if `return_pooler=True`
        """
        return_pooler = kwargs.pop("return_pooler", False)
        return_dict = kwargs.pop("return_dict", False)
        kwargs = {"return_dict": False} | kwargs

        shape = x.shape

        input_ids = x.reshape(
            -1, shape[-1]
        )  # [s, w], s = x1 * ... * xn-1 given * = [x1, ..., xn]
        attention_mask = mask.reshape(-1, shape[-1]).long()

        out, pooler = self.bert(
            input_ids, attention_mask, **kwargs
        )  # [s, w, d], [s, d]

        out = out.reshape(*shape, -1)  # [*, w, d]
        out = out.masked_fill(~mask.unsqueeze(-1), 0)

        pooler = pooler.reshape(*shape[:-1], -1)  # [*, d]

        if self.wordemb_size != self.hid_size:
            out = self.lin(out)  # [*, w, d']
            out = out.masked_fill(~mask.unsqueeze(-1), 0)

            pooler = self.lin(pooler)  # [*, d']

        if return_pooler:
            return out, pooler

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
        heads_e = heads_e.masked_fill(~heads_mask.unsqueeze(-1), 0.0)
        heads_e = heads_e.sum(-2) / n_heads.clamp(
            1
        )  # [b, q, d]  - clamp is required to avoid div by 0
        heads_e = heads_e.unsqueeze(-2).unsqueeze(-2)  # [b, q, 1, 1, d]

        labels_e = self.we(labels, label_mask)  # [b, p, d]
        labels_e = labels_e.unsqueeze(0).unsqueeze(0)  # [1, 1, b, p, d]

        scores = self.sim_fn(heads_e, labels_e)  # [b, q, b, p]

        scores = scores.masked_fill(~get_concepts_mask_(x), 0)

        return scores


class VisualBranch(nn.Module):
    def __init__(self, hid_size):
        super().__init__()

        self.fc = nn.Linear(2048 + 5, hid_size)
        self.act = nn.LeakyReLU()

        self._init_weights()

    def forward(self, x):
        proposals = x["proposals"]  # [b, p, 4]
        proposals_feat = x["proposals_feat"]  # [b, p, v]
        labels_e = x["labels_e"]  # [b, p, q, d]

        q = labels_e.shape[-2]

        # TODO: update the following code because labels_e are now query dependent

        mask = get_proposals_mask(proposals)  # [b, p]

        spat = self.spatial(x)  # [b, p, 5]

        proj = self.project(proposals_feat, spat)  # [b, p, d]
        proj = proj.unsqueeze(-2).expand(-1, -1, q, -1)  # [b, p, q, d]
        fusion = proj + labels_e  # [b, p, d]

        mask = mask.unsqueeze(-1).expand(-1, -1, q)  # [b, p, q]
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
