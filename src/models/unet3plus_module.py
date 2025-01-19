from typing import Any, Dict, Tuple

import torch
import wandb
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from src.utils.utils import draw_segmentation_timeline
from src.utils.metric.segment import *


class Unet3PlusLitModule(LightningModule):

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        focal_loss,
        alpha: float,
        beta:float,
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
        # self.criterion = 
        self.loss_func_seg = focal_loss
        self.loss_func_cls = torch.nn.CrossEntropyLoss()


        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.train_loss_cls = MeanMetric()
        self.train_loss_seg = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_loss_seg = MeanMetric()
        self.val_loss_cls = MeanMetric()
        
        self.test_loss = MeanMetric()

        self.alpha = alpha
        self.beta = beta

        self.wave_order = ['P_onset', 'P_offset', 'QRS_onset', 'QRS_offset', 'T_onset', 'T_offset']
        
        # for val/test process

        # Initialize metrics for P wave
        self.total_P_onset_tp = 0
        self.total_P_onset_fp = 0
        self.total_P_onset_fn = 0
        self.total_P_offset_tp = 0
        self.total_P_offset_fp = 0
        self.total_P_offset_fn = 0

        # Initialize metrics for QRS complex
        self.total_QRS_onset_tp = 0
        self.total_QRS_onset_fp = 0
        self.total_QRS_onset_fn = 0
        self.total_QRS_offset_tp = 0
        self.total_QRS_offset_fp = 0
        self.total_QRS_offset_fn = 0

        # Initialize metrics for T wave
        self.total_T_onset_tp = 0
        self.total_T_onset_fp = 0
        self.total_T_onset_fn = 0
        self.total_T_offset_tp = 0
        self.total_T_offset_fp = 0
        self.total_T_offset_fn = 0
        
        
        
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
        self.val_loss_seg.reset()
        self.val_loss_cls.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, seg_target, cls_target = batch
        seg_output, cls_output = self.forward(x)
        loss_cls = self.loss_func_cls(cls_output, cls_target)
        loss_seg = self.loss_func_seg(seg_output, torch.argmax(seg_target, dim=1))
        
        loss = self.alpha * loss_seg + self.beta * loss_cls
        return loss, loss_seg, loss_cls, seg_target, seg_output

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, loss_seg, loss_cls, _, _ = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_loss_seg(loss_seg)
        self.train_loss_cls(loss_cls)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_seg", self.train_loss_seg, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_cls", self.train_loss_cls, on_step=False, on_epoch=True, prog_bar=True)
        

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, seg_loss, cls_loss, seg_target, seg_output = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_loss_seg(seg_loss)
        self.val_loss_cls(cls_loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss_seg", self.val_loss_seg, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss_cls", self.val_loss_cls, on_step=False, on_epoch=True, prog_bar=True)
        
        seg_output_np = seg_output.cpu().numpy()[:, :, 500:3500]
        seg_target_np = seg_target.cpu().numpy()[:, :, 500:3500]
        
        metrics = calculate_wave_metrics(seg_target_np, seg_output_np, tolerance=75)
        
        for component in self.wave_order:
            values = metrics[component]
            
            setattr(self, f"total_{component}_tp", getattr(self, f"total_{component}_tp") + values['true_positives'])
            setattr(self, f"total_{component}_fp", getattr(self, f"total_{component}_fp") + values['false_positives'])
            setattr(self, f"total_{component}_fn", getattr(self, f"total_{component}_fn") + values['false_negatives'])
    
        
        if batch_idx == 1:
            x, seg_tg, cls_tg = batch
             
            seg_pred, cls_pred = self.forward(x)
            
            pred_img = draw_segmentation_timeline(ecg_signal=x.cpu().numpy()[0], ecg_segment=seg_pred.cpu().numpy()[0], length=5000)
            tg_img = draw_segmentation_timeline(ecg_signal=x.cpu().numpy()[0], ecg_segment=seg_tg.cpu().numpy()[0], length=5000, is_gt=True)
            
            wandb.log({"pred": wandb.Image(pred_img)})
            wandb.log({"gt": wandb.Image(tg_img)})
            
        
    def on_validation_epoch_end(self) -> None:
        
        
# Calculate metrics for each wave component
        result_P_onset = calculate_metrics(
            tp=self.total_P_onset_tp,
            fn=self.total_P_onset_fn,
            fp=self.total_P_onset_fp
        )  
    
        result_P_offset = calculate_metrics(
            tp=self.total_P_offset_tp,
            fn=self.total_P_offset_fn,
            fp=self.total_P_offset_fp
        )

        result_QRS_onset = calculate_metrics(
            tp=self.total_QRS_onset_tp,
            fn=self.total_QRS_onset_fn,
            fp=self.total_QRS_onset_fp
        )

        result_QRS_offset = calculate_metrics(
            tp=self.total_QRS_offset_tp,
            fn=self.total_QRS_offset_fn,
            fp=self.total_QRS_offset_fp
        )

        result_T_onset = calculate_metrics(
            tp=self.total_T_onset_tp,
            fn=self.total_T_onset_fn,
            fp=self.total_T_onset_fp
        )

        result_T_offset = calculate_metrics(
            tp=self.total_T_offset_tp,
            fn=self.total_T_offset_fn,
            fp=self.total_T_offset_fp
        )
        
        
        wandb.log({
            "val/F1_P_onset": result_P_onset["f1_score"],
            "val/F1_P_offset": result_P_offset["f1_score"],
            "val/F1_QRS_onset": result_QRS_onset["f1_score"],
            "val/F1_QRS_offset": result_QRS_offset["f1_score"],
            "val/F1_T_onset": result_T_onset["f1_score"],
            "val/F1_T_offset": result_T_offset["f1_score"],
        })
        
        attributes_to_reset = [
            'total_P_onset_tp', 'total_P_onset_fp', 'total_P_onset_fn',
            'total_P_offset_tp', 'total_P_offset_fp', 'total_P_offset_fn',
            'total_QRS_onset_tp', 'total_QRS_onset_fp', 'total_QRS_onset_fn',
            'total_QRS_offset_tp', 'total_QRS_offset_fp', 'total_QRS_offset_fn',
            'total_T_onset_tp', 'total_T_onset_fp', 'total_T_onset_fn',
            'total_T_offset_tp', 'total_T_offset_fp', 'total_T_offset_fn',
        ]
        for attr in attributes_to_reset:
            setattr(self, attr, 0)

        pass
    

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, _, _, _, _ = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
 

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


if __name__ == "__main__":
    _ = Unet3PlusLitModule(None, None, None, None, None, None, None)
