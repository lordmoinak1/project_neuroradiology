import torch
import monai
import pl_bolts
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl

from dataset import *

class swin_model(nn.Module):
    def __init__(self,):
        super(swin_model, self).__init__()

        self.model = monai.networks.nets.SwinUNETR(img_size=128,
                      in_channels=4,
                      out_channels=3,
                      feature_size=48,
                      drop_rate=0.0,
                      attn_drop_rate=0.0,
                      dropout_path_rate=0.0,
                      use_checkpoint=True,
                      )
        model_dict = torch.load('/path/to/weights/fold0_f48_ep300_4gpu_dice0_8854/model.pt')["state_dict"]
        self.model.load_state_dict(model_dict)
        self.model.eval()
        
        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6291456, 2),
        )

    def forward(self, x):
        x = x.contiguous()
        x = self.model(x)
        x = self.classifier_head(x)
        return x

class swin_baseline(pl.LightningModule):
  def __init__(self, batch_size=1, learning_rate=1e-2):
    super(swin_baseline, self).__init__()

    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.num_workers = 8
    
    self._swin = swin_model()

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
    return self._swin(x)

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self._swin.parameters(), lr=self.learning_rate, weight_decay=1e-1)
    scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=50, max_epochs=5000)
    return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
  def training_step(self, batch, batch_idx):
    inputs = batch['image']
    labels = batch['label_classification']
    labels = torch.argmax(labels)
    labels = torch.unsqueeze(labels, dim=-1)
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
    labels = torch.argmax(labels)
    labels = torch.unsqueeze(labels, dim=-1)
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
