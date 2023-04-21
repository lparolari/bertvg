import argparse
import logging
from math import ceil

import cv2
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from weakvg.dataset import Flickr30kDataset, ReferitDataset
from weakvg.datamodule import WeakvgDataModule, NormalizeCoord, collate_fn
from weakvg.model import WeakvgModel
from weakvg.wordvec import get_nlp, get_objects_vocab, get_tokenizer, get_wordvec

vocab = None
font_size = 14
text_props = dict(facecolor="blue", alpha=0.5)

mpl.rcParams["figure.dpi"] = 100


def main():
    global vocab

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["flickr30k", "referit"])
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--wv_type", type=str, default="glove")
    parser.add_argument("--search", type=str, default=None)
    parser.add_argument("--show_proposals", action="store_true", default=False)

    args = parser.parse_args()

    dataset_name = args.dataset
    checkpoint = args.checkpoint
    search = args.search
    split = args.split
    wv_type = args.wv_type
    show_proposals = args.show_proposals

    pl.seed_everything(42, workers=True)

    tokenizer = get_tokenizer()
    wordvec, vocab = get_wordvec(wv_type, custom_labels=get_objects_vocab())
    nlp = get_nlp()

    dataset_cls = WeakvgDataModule.get_dataset_cls(dataset_name)
    data_dir = WeakvgDataModule.get_data_dir(dataset_name)

    dataset = dataset_cls(
        split=split,
        data_dir=data_dir,
        tokenizer=tokenizer,
        vocab=vocab,
        nlp=nlp,
        transform=NormalizeCoord(),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    model = WeakvgModel.load_from_checkpoint(checkpoint, wordvec=wordvec, vocab=vocab, strict=False)

    for batch in dataloader:
        # forward the model
        scores, (multimodal_scores, concepts_scores) = model(batch)

        # compute candidates box for the full, multimodal and concepts model
        model_candidates, model_candidates_idx = model.predict_candidates(
            scores, batch["proposals"]
        )

        multimodal_candidates, multimodal_candidates_idx = model.predict_candidates(
            multimodal_scores, batch["proposals"]
        )

        concepts_candidates, concepts_candidates_idx = model.predict_candidates(
            concepts_scores, batch["proposals"]
        )

        coverage_candidates, coverage_candidates_idx = model.predict_candidates(
            coverage_scores(batch["proposals"], batch["targets"]), batch["proposals"]
        )

        # "unbatch" data -> remove batch dimension (getting the first element)
        # and convert to numpy arrays
        meta = unbatch(batch["meta"])
        targets = unbatch(batch["targets"])
        queries = unbatch(batch["queries"])
        image_w = unbatch(batch["image_w"])
        image_h = unbatch(batch["image_h"])
        sentence = unbatch(batch["sentence"])
        labels = unbatch(batch["labels"])
        proposals = unbatch(batch["proposals"])

        identifier = meta[1]  # 0: idx, 1: identifier

        model_candidates, model_candidates_idx = unbatch(model_candidates), unbatch(
            model_candidates_idx
        )
        multimodal_candidates, multimodal_candidates_idx = unbatch(
            multimodal_candidates
        ), unbatch(multimodal_candidates_idx)
        concepts_candidates, concepts_candidates_idx = unbatch(
            concepts_candidates
        ), unbatch(concepts_candidates_idx)
        coverage_candidates, coverage_candidates_idx = unbatch(
            coverage_candidates
        ), unbatch(coverage_candidates_idx)

        # get the labels for each candidate
        model_candidates_label = np.take(labels, model_candidates_idx, axis=0)
        multimodal_candidates_label = np.take(labels, multimodal_candidates_idx, axis=0)
        concepts_candidates_label = np.take(labels, concepts_candidates_idx, axis=0)
        coverage_candidates_label = np.take(labels, coverage_candidates_idx, axis=0)

        # get the image to show on plot
        image = get_image(dataset.get_image_path(identifier))

        gt_color = np.array([255, 102, 102]) / 255  # #ff6666 (red)
        model_color = np.array([153, 255, 153]) / 255  # #99ff99 (green)
        concept_color = np.array([69, 205, 255]) / 255  # #45cdff (blue)
        mm_color = np.array([255, 255, 153]) / 255  # #ffff99 (yellow)
        cov_label = np.array([255, 153, 255]) / 255  # #ff99ff (pink)

        sentence_txt = txt(sentence)
        model_candidates_label_txt = txt(model_candidates_label).split()
        multimodal_candidates_label_txt = txt(multimodal_candidates_label).split()
        concepts_candidates_label_txt = txt(concepts_candidates_label).split()
        coverage_candidates_label_txt = txt(coverage_candidates_label).split()

        if search is not None and search not in sentence_txt:
            continue

        # x1, y1, x2, y2
        img_size = [
            image_w,
            image_h,
            image_w,
            image_h,
        ]

        for i in range(len(queries)):
            query = queries[i]

            query_txt = txt(query)
            candidate_lab_txt = model_candidates_label_txt[i]
            concepts_lab_txt = concepts_candidates_label_txt[i]
            mm_lab_txt = multimodal_candidates_label_txt[i]
            cov_lab_txt = coverage_candidates_label_txt[i]

            # rescale boxes to image size
            target_box = targets[i] * img_size
            candidate_box = model_candidates[i] * img_size
            mm_box = multimodal_candidates[i] * img_size
            concepts_box = concepts_candidates[i] * img_size
            coverage_box = coverage_candidates[i] * img_size
            proposals_box = proposals * img_size

            iou_score = iou(target_box, candidate_box)

            cols = 3
            rows = ceil(len(queries) / cols)

            plt.subplot(rows, min(cols, len(queries)), i + 1)
            plt.imshow(image)
            plt.title(
                f"{query_txt} ({round(iou_score, ndigits=2)})",
                fontdict={"fontsize": font_size},
                y=-0.1,
            )
            plt.suptitle(sentence_txt, fontsize=14)

            ax = plt.gca()

            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)

            draw_box(ax, target_box, gt_color)

            draw_box(
                ax,
                coverage_box,
                cov_label,
                label=cov_lab_txt + show_iou(coverage_box, target_box),
            )

            draw_box(
                ax, mm_box, mm_color, label=mm_lab_txt + show_iou(target_box, mm_box)
            )

            draw_box(
                ax,
                concepts_box,
                concept_color,
                label=concepts_lab_txt + show_iou(target_box, concepts_box),
            )

            draw_box(
                ax,
                candidate_box,
                model_color,
                label=candidate_lab_txt + show_iou(candidate_box, target_box),
                dashed=True,
            )

            if show_proposals:
                for p_box in proposals_box:
                    draw_box(ax, p_box, color=[0, 0, 0], dashed=True)

        gt_patch = patches.Patch(color=gt_color, label="ground truth")
        model_patch = patches.Patch(color=model_color, label="model")
        mm_patch = patches.Patch(color=mm_color, label="multimodal")
        concept_patch = patches.Patch(color=concept_color, label="concepts")
        cov_patch = patches.Patch(color=cov_label, label="coverage")

        plt.figlegend(
            handles=[gt_patch, model_patch, mm_patch, concept_patch, cov_patch]
        )

        plt.tight_layout()
        plt.show()


def unbatch(x):
    return x[0].cpu().numpy()


def get_image(path):
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    return im


def txt(tokens):
    return " ".join(vocab.lookup_tokens([t for t in tokens if t != 0]))


def draw_box(ax, box, color, label=None, dashed=False):
    x1, y1 = box[0], box[1]
    x2, y2 = box[2], box[3]
    xy = (x1, y1)
    width, height = x2 - x1, y2 - y1

    rect = patches.Rectangle(
        xy,
        width,
        height,
        linewidth=2,
        edgecolor=[*color, 1.0],
        facecolor=[*color, 0.2],
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(rect)

    if label is not None:
        ax.text(
            x1,
            y1,
            label,
            bbox={"facecolor": [*color, 1.0], "edgecolor": [*color, 0.0], "pad": 1.0},
            fontsize=font_size,
            color="black",
        )


def iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union


def show_iou(box_a, box_b):
    return f" ({round(iou(box_a, box_b), ndigits=2)})"


def coverage_scores(proposals, targets):
    """
    Compute the coverage scores between proposals and targets.

    :param proposals: A tensor of shape `[b, p, 4]`
    :param targets: A tensor of shape `[b, q, 4]`
    :return: A tensor of shape `[b, p, b, q]` with scores for each proposal-target pair.
    """
    from torch import arange
    from torchvision.ops import box_iou

    b = proposals.shape[0]
    p = proposals.shape[1]
    q = targets.shape[1]

    proposals = proposals.reshape(1, 1, b, p, 4).repeat(b, q, 1, 1, 1)
    targets = targets.reshape(b, q, 1, 1, 4).repeat(1, 1, b, p, 1)

    scores = box_iou(
        proposals.view(-1, 4), targets.view(-1, 4)
    )  #  [b * q * b * p, b * q * b * p]

    index = (
        arange(b * q * b * p).to(proposals.device).unsqueeze(-1)
    )  # [b * q * b * p, 1]

    scores = scores.gather(-1, index)  # [b * q * b * p, 1]
    scores = scores.view(b, q, b, p)  # [b, q, b, p]

    return scores


if __name__ == "__main__":
    main()
