import json
import logging

import pytorch_lightning as pl

from weakvg.cli import get_args, get_callbacks, get_logger
from weakvg.datamodule import WeakvgDataModule
from weakvg.model import WeakvgModel
from weakvg.wordvec import get_nlp, get_tokenizer, get_wordvec


def main():
    args = get_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    logging.info("Args: " + json.dumps(vars(args), indent=4))

    pl.seed_everything(42, workers=True)

    tokenizer = get_tokenizer()
    wordvec, vocab = get_wordvec()
    nlp = get_nlp()

    dm = WeakvgDataModule(
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_fraction=args.train_fraction,
        dev=args.dev,
        tokenizer=tokenizer,
        vocab=vocab,
        nlp=nlp,
    )

    if args.checkpoint:
        model = WeakvgModel.load_from_checkpoint(
            args.checkpoint, strict=False, wordvec=wordvec, vocab=vocab,
        )
        logging.info(f"Loaded model at {args.checkpoint}")
    else:
        model = WeakvgModel(
            wordvec,
            vocab,
            omega=args.omega,
            lr=args.lr,
            neg_selection=args.neg_selection,
            use_relations=args.use_relations,
        )

    logging.info(f"Model hparams: " + json.dumps(model.hparams_initial, indent=4))

    logger = get_logger(args, model)
    callbacks = get_callbacks(args)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=callbacks,
    )

    if args.mode == "train":
        trainer.fit(model, dm)
        trainer.test(model, dm, ckpt_path="best")

    if args.mode == "test":
        trainer.test(model, dm)


if __name__ == "__main__":
    main()
