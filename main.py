import logging

import pytorch_lightning as pl
from torchtext.data.utils import get_tokenizer

from weakvg.dataset import Flickr30kDataModule
from weakvg.model import MyModel
from weakvg.wordvec import get_wordvec, get_objects_vocab
from weakvg.cli import get_args, get_logger


def main():
    logging.basicConfig(level=logging.INFO)

    pl.seed_everything(42, workers=True)

    args = get_args()
    logger = get_logger(args)

    tokenizer = get_tokenizer("basic_english")
    wordvec, vocab = get_wordvec(custom_tokens=get_objects_vocab())

    dm = Flickr30kDataModule(
        data_dir="data/flickr30k",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_fraction=args.train_fraction,
        dev=args.dev,
        tokenizer=tokenizer,
        vocab=vocab,
    )

    model = MyModel(wordvec, vocab)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        logger=logger,
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
