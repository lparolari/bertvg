import argparse

import pytorch_lightning as pl


def get_args():
    parser = argparse.ArgumentParser()

    exp_group = parser.add_argument_group("experiment arguments")
    exp_group.add_argument(
        "--exp_id",
        type=str,
        default=None,
        help="Experiment identifier. Default: random",
    )
    exp_group.add_argument(
        "--exp_notes",
        type=str,
        default=None,
        help="Experiment description. Default: None",
    )
    exp_group.add_argument(
        "--exp_tags",
        type=str,
        nargs="*",
        default=None,
        help="Experiment tags. Default: None",
    )
    exp_group.add_argument(
        "--exp_group",
        type=str,
        default=None,
        help="Experiment group. Default: None",
    )
    exp_group.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Program mode. Default: train",
    )
    exp_group.add_argument(
        "--max_epochs",
        type=int,
        default=25,
        help="Maximum number of epochs. Default: 25",
    )
    exp_group.add_argument(
        "--devices", type=int, default=1, help="Number of devices to use. Default: 1"
    )
    exp_group.add_argument(
        "--accelerator",
        type=str,
        choices=["cpu", "gpu"],
        default="gpu",
        help="Accelerator to use. Default: gpu",
    )
    exp_group.add_argument(
        "--logger",
        type=str,
        choices=["wandb", "tensorboard"],
        default="wandb",
        help="Logger to use for experiment tracking. Default: wandb",
    )
    exp_group.add_argument(
        "--dev",
        default=False,
        action="store_true",
        help="Dev mode uses validation set instead of training set and do not load custom tokens. Default: false",
    )
    exp_group.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Verbose mode. Default: false",
    )

    # dataset params
    dataset_group = parser.add_argument_group("dataset arguments")
    dataset_group.add_argument(
        "--dataset",
        type=str,
        choices=["flickr30k", "referit"],
        required=True,
    )
    dataset_group.add_argument(
        "--batch_size", type=int, default=32, help="Batch size. Default: 32"
    )
    dataset_group.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers. Default: 4"
    )
    dataset_group.add_argument(
        "--train_fraction",
        type=float,
        default=1.0,
        help="Fraction of training set to use. Default: 1.0",
    )

    # model params
    model_group = parser.add_argument_group("model arguments")
    model_group.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint to load. Default: None"
    )
    model_group.add_argument(
        "--omega",
        type=float,
        default=0.5,
        help="Weight for the network prediction. Default: 0.5",
    )
    model_group.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate. Default: 1e-5",
    )
    model_group.add_argument(
        "--neg_selection",
        type=str,
        default="random",
        choices=["random", "textual_sim_max"],
        help="Strategy for negative example selection. Default: random",
    )
    model_group.add_argument(
        "--use_relations",
        default=False,
        action="store_true",
        help="Enable the attention mask based on relations. Default: false",
    )
    model_group.add_argument(
        "--hidden_size",
        type=int,
        default=300,
        help="Size of the visual and textual representation embeddings. Default: 300",
    )

    args = parser.parse_args()

    if args.dev:
        # force verbose logging in dev mode
        args.verbose = True

    return args


def get_logger(args, model=None):
    if args.logger == "tensorboard":
        import os

        logger = pl.loggers.tensorboard.TensorBoardLogger(
            name=args.exp_id, save_dir=os.path.join(os.getcwd(), "tb_logs")
        )

        args.exp_id = args.exp_id or f"version_{logger.version}"

        return logger

    if args.logger == "wandb":
        import copy
        import wandb

        logger = pl.loggers.WandbLogger(
            project="bertvg",
            entity="weakly_guys",
            log_model=False if args.dev else True,
            name=args.exp_id,
            notes=args.exp_notes,
            tags=args.exp_tags,
            group=args.exp_group,
            settings=wandb.Settings(start_method="fork"),
        )

        # we use the `logger.version` to be coherent with wandb naming.
        # please note that
        #
        #  * `logger.version` is the experiment id in the wandb logger
        #    (`logger.experiment.id`), i.e. a random string like `9n17vs3z`.
        #
        #  * the wandb experiment name like `frosty-grass-69` can be found
        #    at `logger.experiment.name` instead.
        args.exp_id = args.exp_id or f"{logger.version}"

        args_to_log = copy.deepcopy(args)

        # remove args that do not belong to experiment config
        to_remove = [
            "exp_id",
            "exp_notes",
            "exp_tags",
            "exp_group",
            "verbose",
            "logger",
            "dev",
        ]

        for key in to_remove:
            delattr(args_to_log, key)

        # then update experiment config
        logger.experiment.config.update(vars(args_to_log))

        if model is not None:
            logger.watch(model, log="all")

        return logger

    return None


def get_callbacks(args):
    model_checkpoint_clbk = pl.callbacks.model_checkpoint.ModelCheckpoint(
        dirpath=f"output/{args.exp_id}",
        filename="{epoch}-{step}-{val_acc:.2f}",
        monitor="val_acc",
        mode="max",
        save_last=True,
    )

    callbacks = [
        model_checkpoint_clbk,
    ]

    return callbacks
