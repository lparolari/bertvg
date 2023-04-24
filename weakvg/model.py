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
        self.prediction_module = SimilarityPredictionModule(omega=omega)
        self.loss = Loss(neg_selection=neg_selection)

        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

        self.save_hyperparameters(ignore=["wordvec", "vocab"])

    def forward(self, x):
        queries = x["queries"]
        is_word, is_query = get_queries_mask_(x)

        queries_e, queries_pooler = self.we(
            queries, is_word, return_pooler=True
        )  # [b, q, w, d], [b, q, d]
        heads_e = None  # TODO: get the heads embedding from queries_e through index
        labels_e = None  # TODO: follow the paper

        # heads

        heads = x["heads"]  # [b, q, h]
        heads = heads[..., 0]  # [b, q] - select the first head
        is_head = (queries == heads.unsqueeze(-1)) & is_query.unsqueeze(-1)  # [b, q, w]
        queries_h = queries.masked_fill(~is_head, 0)  # [b, q, w]
        index = queries_h.argmax(-1)  # [b, q]
        index = index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, queries_e.size(-1))  # [b, q, 1, d]
        heads_e = queries_e.gather(-2, index).squeeze(-2)  # [b, q, d]
        heads_e.masked_fill(~is_query.unsqueeze(-1), 0)

        # labels

        queries = x["queries"]  # [b, q, w]
        labels = x["labels"]  # [b, p]
        q = queries.shape[1]
        w = queries.shape[-1]
        p = labels.shape[-1]
        labels = labels.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, q, w)  # [b, p, q, w]
        queries = queries.unsqueeze(1).repeat(1, p, 1, 1)  # [b, p, q, w]
        # calcolo maschera head
        heads = x["heads"]  # [b, q, h]
        heads = heads[..., 0]  # [b, q] - select the first head
        is_head = (x["queries"] == heads.unsqueeze(-1)) & is_query.unsqueeze(-1)  # [b, q, w]
        is_head = is_head.unsqueeze(1).repeat(1, p, 1, 1)  # [b, p, q, w]

        is_head = is_head.long()

        # azzero gli indici dove c'è la head e sommo l'indice della label (0 + label)
        new_q = queries * (1 - is_head) + labels * is_head

        is_head = is_head.bool()
    
        queries_ext_e = self.we(new_q, is_head)  # [b, p, q, w, d]

        # ottengo l'indice della head head per estrarre l'embedding della label
        queries_h = queries.masked_fill(~is_head, 0)  # [b, p, q, w]
        index = queries_h.argmax(-1)  # [b, p, q]
        index = index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 1, queries_e.size(-1))  # [b, p, q, 1, d]
        labels_e = queries_ext_e.gather(-2, index).squeeze(-2)  # [b, p, q, d]

        x |= {"queries_e": queries_e, "heads_e": heads_e, "labels_e": labels_e}

        # TODO: concepts branch temporarily disabled
        # concepts_pred = self.concept_branch(x)  # [b, q, b, p]
        b = x["proposals"].shape[0]
        visual_feat = self.visual_branch(x)
        visual_feat = visual_feat.transpose(1, 2).unsqueeze(2).repeat(1, 1, b, 1, 1)  # [b, q, b, q, d]
        textual_feat = queries_pooler  # [b, q, d]
        textual_feat = textual_feat.unsqueeze(2).unsqueeze(3).repeat(1, 1, b, p, 1)  # [b, q, b, p, d]

        visual_mask = get_proposals_mask_(x).unsqueeze(-2).unsqueeze(-2).repeat(1, q, b, 1)
        textual_mask = get_queries_mask_(x)[1].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, b, p)
        mask = visual_mask & textual_mask
        mask = mask.unsqueeze(-1)

        # scores, (multimodal_scores, concepts_scores) = self.prediction_module(
        #     (visual_feat, visual_mask),
        #     (textual_feat, textual_mask),
        #     # (concepts_pred, concepts_mask),
        # )  # [b, q, b, p], ([b, q, b, p], [b, q, b, p])

        inp = torch.cat([visual_feat, textual_feat], dim=-1)
        inp = inp.masked_fill(~mask, 0)

        inp = self.fc1(inp)
        inp = self.act(inp)
        scores = self.fc2(inp)

        scores = scores.masked_fill(~mask, 0)

        return scores.squeeze(-1), (None, None)

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
        # concepts_pred, concepts_mask = self.predict_concepts(
        #     concepts
        # )  # [b, q, b, p], [b, q, b, p]

        # scores = self.apply_prior(multimodal_pred, concepts_pred)  # [b, q, b, p]
        scores = multimodal_pred

        # mask = multimodal_mask & concepts_mask  # [b, q, b, p]
        mask = multimodal_mask

        scores = scores.masked_fill(~mask, 0)  # [b, q, b, p]

        # return scores, (multimodal_pred, concepts_pred)
        return scores, (multimodal_pred, torch.zeros_like(multimodal_pred))

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
