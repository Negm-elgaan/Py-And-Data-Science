import torch.nn as nn
import lightning.pytorch as pl

# Define the model class
class LightModel(pl.LightningModule):
  	# Define a linear layer to transform your input
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(16 , 10)
    def forward(self, x):
        return self.layer(x)
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        return loss