from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification import (
    Accuracy,
    AUROC,
    Specificity,
    Recall,
)

torch.autograd.set_detect_anomaly(True)

class IMLENetLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()

        num_classes = self.net.classes

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        logits = self.forward(x)
        
        loss = self.criterion(logits, y)
        preds = torch.sigmoid(logits)
        preds = torch.argmax(logits, dim=1)
        y = torch.argmax(y, dim=1)
        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

# ! ===============================================
# from typing import Any, Dict, Tuple

# import torch
# from lightning import LightningModule
# from torchmetrics import MaxMetric, MeanMetric
# from torchmetrics.classification import (
#     Accuracy,
#     AUROC,
#     Specificity,
#     Recall,
# )


# class IMLENetLitModule(LightningModule):
#     def __init__(
#         self,
#         net: torch.nn.Module,
#         optimizer: torch.optim.Optimizer,
#         scheduler: torch.optim.lr_scheduler,
#         compile: bool,
#     ) -> None:
#         super().__init__()

#         self.save_hyperparameters(logger=False)
#         self.net = net
#         self.criterion = torch.nn.CrossEntropyLoss()
#         num_classes = self.net.classes

#         # Metrics for each stage
#         self.train_metrics = self._create_metrics(num_classes)
#         self.val_metrics = self._create_metrics(num_classes)
#         self.test_metrics = self._create_metrics(num_classes)

#         # for tracking best so far validation accuracy
#         self.val_acc_best = MaxMetric()

#     def _create_metrics(self, num_classes):
#         return torch.nn.ModuleDict({
#             "acc": Accuracy(task="multiclass", num_classes=num_classes),
#             "auc": AUROC(task="multiclass", num_classes=num_classes),
#             "specificity": Specificity(task="multiclass", num_classes=num_classes),
#             "sensitivity": Recall(task="multiclass", num_classes=num_classes),
#             "loss": MeanMetric(),
#         })

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.net(x)

#     def on_train_start(self) -> None:
#         self.val_acc_best.reset()

#     def model_step(
#         self, batch: Tuple[torch.Tensor, torch.Tensor]
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         x, y = batch
#         logits = self.forward(x)
#         loss = self.criterion(logits, y)
#         preds = torch.argmax(logits, dim=1)
#         # y = torch.argmax(y, dim=1)
#         return loss, logits, preds, y

#     def _update_metrics(self, stage, loss, logits, preds, targets):
#         metrics = getattr(self, f"{stage}_metrics")
#         metrics["loss"](loss)
#         metrics["acc"](preds, targets)
#         metrics["auc"](logits, targets)
#         metrics["specificity"](preds, targets)
#         metrics["sensitivity"](preds, targets)

#         for metric_name, metric in metrics.items():
#             self.log(f"{stage}/{metric_name}", metric, on_step=True, on_epoch=True, prog_bar=True)

#     def training_step(
#         self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
#     ) -> torch.Tensor:
#         loss, logits, preds, targets = self.model_step(batch)
#         self._update_metrics("train", loss, logits, preds, targets)
#         return loss

#     def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
#         loss, logits, preds, targets = self.model_step(batch)
#         self._update_metrics("val", loss, logits, preds, targets)

#     def on_validation_epoch_end(self) -> None:
#         acc = self.val_metrics["acc"].compute()
#         self.val_acc_best(acc)
#         self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

#     def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
#         loss, logits, preds, targets = self.model_step(batch)
#         self._update_metrics("test", loss, logits, preds, targets)

#     def setup(self, stage: str) -> None:
#         if self.hparams.compile and stage == "fit":
#             self.net = torch.compile(self.net)

#     def configure_optimizers(self) -> Dict[str, Any]:
#         optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
#         if self.hparams.scheduler is not None:
#             scheduler = self.hparams.scheduler(optimizer=optimizer)
#             return {
#                 "optimizer": optimizer,
#                 "lr_scheduler": {
#                     "scheduler": scheduler,
#                     "monitor": "val/loss",
#                     "interval": "epoch",
#                     "frequency": 1,
#                 },
#             }
#         return {"optimizer": optimizer}