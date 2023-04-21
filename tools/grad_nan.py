"""
Gradient to linear layer was unexpectedly NaN. We used this script to reproduce the
issue. We finally found that gradient was NaN due to division by zero done by the line

```
query_emb = word_emb.sum(-2) / n_words.unsqueeze(-1)
```

We used to mask out "inf" values but this fixes only the forward pass. In the backward 
pass the gradient is computed as the derivative of the division of the input over the
output:

d   f(x)   g(x) f'(x) - f(x) g'(x)
--  ---- = -----------------------
dx  g(x)         g(x)²

Note that g(x) is the output, i.e. the tensor with "inf" values.
"""

import pytorch_lightning as pl

import torch
import torch.nn as nn


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self):
        # 4 examples, each example has 2 query, every query has 2 words
        # words are represented by their index in the vocabulary
        # 0 is reserved for padding
        self.x = torch.tensor(
            [
                [
                    [1, 0],
                    [0, 0],
                ],
                [
                    [0, 0],
                    [0, 0],
                ],
                [
                    [1, 0],
                    [2, 3],
                ],
                [
                    [4, 5],
                    [6, 7],
                ],
            ]
        )  # [n, q, w]
        self.x.requires_grad = False

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return self.x[idx]


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 8)  # given the index, compute an 8d embedding
        self.lin2 = nn.Linear(8, 8)  # given the index, compute an 8d embedding

    def forward(self, x):
        return self.lin(x.float().unsqueeze(-1))

    def training_step(self, batch, batch_idx):
        x = batch  # [b, q, w]

        mask = x != 0  # [b, q, w]
        n_words = mask.sum(-1)  # [b, q]

        # compute the fake embedding through a linear layer
        word_emb = self(x)  # [b, q, w, d]

        # compute the query embedding, i.e., for each query compute tha average
        # of its words
        query_emb = word_emb.sum(-2) / n_words.unsqueeze(-1)  # [b, q, d]

        # in some cases, queries may not have any words: 
        # we need to mask them out
        query_emb = query_emb.masked_fill(~n_words.bool().unsqueeze(-1), 0.)
        # query_emb = query_emb.nan_to_num(nan=0., posinf=0., neginf=0.)
        
        loss = query_emb.sum()  # dummy loss

        self.log("train_loss", loss)
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def on_after_backward(self):
        super().on_after_backward()
        print(torch.isnan(self.lin.weight.grad).any())


def main():
    loader = torch.utils.data.DataLoader(
        dataset=FakeDataset(),
        batch_size=1,
        num_workers=1,
    )

    model = Model()

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=None,
        limit_val_batches=0.0,
    )

    trainer.fit(model, loader, loader)


if __name__ == "__main__":
    main()
