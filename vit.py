# import sys
# sys.path.insert(0, '/home/moibhattacha/project_aorta')

import torch
import monai
import pl_bolts
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl
from monai.utils import ensure_tuple_rep

from dataset import *

class swin_model(nn.Module):
    def __init__(self,):
        super(swin_model, self).__init__()
        patch_size = ensure_tuple_rep(2, 3)
        self.vit = monai.networks.nets.UNETR(
            in_channels=4,
            out_channels=1,
            img_size=(96, 96, 96),
            feature_size=12,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            # num_layers=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            conv_block=True,
            dropout_rate=0.0,
        )

        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(884736, 2),
        )

    def forward(self, x):
        x = x.contiguous()
        x = self.vit(x)
        x = self.classifier_head(x)
        return x

class swin_baseline(pl.LightningModule):
  def __init__(self, batch_size=1, learning_rate=1e-2):
    super(swin_baseline, self).__init__()

    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.num_workers = 8
    
    # _unetr = monai.networks.nets.UNETR(
    #         in_channels=4,
    #         out_channels=3,
    #         img_size=(96, 96, 96),
    #         feature_size=12,
    #         hidden_size=768,
    #         mlp_dim=3072,
    #         num_heads=12,
    #         # num_layers=12,
    #         pos_embed="perceptron",
    #         norm_name="instance",
    #         res_block=True,
    #         conv_block=True,
    #         dropout_rate=0.0,
    #     )

    self.model = swin_model()
    # nn.Sequential(
    #   _unetr,
    #   nn.Flatten(),
    #   nn.Linear(2654208, 2),
    # )

    self.classification_loss = torch.nn.CrossEntropyLoss()

    self.train_accuracy = torchmetrics.Accuracy()
    self.train_f1 = torchmetrics.F1Score()
    self.train_precision = torchmetrics.Precision()
    self.train_recall = torchmetrics.Recall()

    self.val_accuracy = torchmetrics.Accuracy()
    self.val_f1 = torchmetrics.F1Score()
    self.val_precision = torchmetrics.Precision()
    self.val_recall = torchmetrics.Recall()

  def forward(self,x):
    return self.model(x)

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-1)
    scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=50, max_epochs=5000)
    return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
  def training_step(self, batch, batch_idx):
    inputs = batch['image']
    labels = batch['label_classification']
    outputs = self.forward(inputs)
    loss = self.classification_loss(outputs, labels)
    self.train_accuracy(outputs, labels)
    self.train_f1(outputs, labels)
    self.train_precision(outputs, labels)
    self.train_recall(outputs, labels)
    return {"loss": loss, "number": len(outputs)}

  def training_epoch_end(self, outputs):
    train_loss, train_num_items = 0, 0
    for output in outputs:
        train_loss += output["loss"].sum().item()
        train_num_items += output["number"]
    mean_train_accuracy = self.train_accuracy.compute()
    mean_train_f1 = self.train_f1.compute()
    mean_train_precision = self.train_precision.compute()
    mean_train_recall = self.train_recall.compute()
    self.train_accuracy.reset()
    self.train_f1.reset()
    self.train_precision.reset()
    self.train_recall.reset()
    mean_train_loss = torch.tensor(train_loss / train_num_items)
    self.log('train_loss', mean_train_loss, on_step=False, on_epoch=True, prog_bar=True)
    self.log('train_acc', mean_train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
    self.log('train_f1', mean_train_f1, on_step=False, on_epoch=True, prog_bar=True)
    self.log('train_pr', mean_train_precision, on_step=False, on_epoch=True, prog_bar=True)
    self.log('train_re', mean_train_recall, on_step=False, on_epoch=True, prog_bar=True)

  def validation_step(self, batch, batch_idx):
    inputs = batch['image']
    labels = batch['label_classification']
    outputs = self.forward(inputs)
    loss = self.classification_loss(outputs, labels)
    self.val_accuracy(outputs, labels)
    self.val_f1(outputs, labels)
    self.val_precision(outputs, labels)
    self.val_recall(outputs, labels)
    return {"val_loss": loss, "val_number": len(outputs)}

  def validation_epoch_end(self, outputs):
    val_loss, num_items = 0, 0
    for output in outputs:
        val_loss += output["val_loss"].sum().item()
        num_items += output["val_number"]
    mean_val_accuracy = self.val_accuracy.compute()
    mean_val_f1 = self.val_f1.compute()
    mean_val_precision = self.val_precision.compute()
    mean_val_recall = self.val_recall.compute()
    self.val_accuracy.reset()
    self.val_f1.reset()
    self.val_precision.reset()
    self.val_recall.reset()
    mean_val_loss = torch.tensor(val_loss / num_items)
    self.log('val_loss', mean_val_loss, on_step=False, on_epoch=True, prog_bar=True)
    self.log('val_acc', mean_val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
    self.log('val_f1', mean_val_f1, on_step=False, on_epoch=True, prog_bar=True)
    self.log('val_pr', mean_val_precision, on_step=False, on_epoch=True, prog_bar=True)
    self.log('val_re', mean_val_recall, on_step=False, on_epoch=True, prog_bar=True)

if __name__ == "__main__":    
    flag = 0

    _unetr = monai.networks.nets.UNETR(
            in_channels=4,
            out_channels=1,
            img_size=(96, 96, 96),
            feature_size=12,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            # num_layers=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            conv_block=True,
            dropout_rate=0.0,
        )

    model = nn.Sequential(
      _unetr,
      nn.Flatten(),
      nn.Linear(884736, 2),
    )
    inputs = torch.randn(1, 4, 96, 96, 96)
    outputs = model(inputs)
    print(outputs.shape)