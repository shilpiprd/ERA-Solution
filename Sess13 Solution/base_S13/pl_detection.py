import torch
import config
from model import YOLOv3
from loss import YoloLoss
from lightning import LightningModule
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)
from torch.optim.lr_scheduler import OneCycleLR


loss_fn = YoloLoss()



class LitYOLOv3(LightningModule):

    def __init__(self):
        super().__init__()
        self.model = YOLOv3(num_classes=config.NUM_CLASSES)
        self.save_hyperparameters()
        #self.lr = config.LEARNING_RATE

    def forward(self, imgs):
        detections = self.model(imgs)
        return detections

    def criterion(self, out, y):
        y0, y1, y2 = (y[0], y[1], y[2])
        scaled_anchors = (
            torch.tensor(config.ANCHORS)
            * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1,3,2)
        ).to(self.device)
        loss = (
                    loss_fn(out[0], y0, scaled_anchors[0])
                    + loss_fn(out[1], y1, scaled_anchors[1])
                    + loss_fn(out[2], y2, scaled_anchors[2])
                )
        return loss

    def training_step(self, batch, batch_id):
        x,y = batch
        out = self(x)

        loss = self.criterion(out,y)

        self.log("training loss", loss, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)

        return loss

    # def validation_step(self, batch, batch_id):
    #     x,y = batch
    #     out = self(x)

    #     loss = self.criterion(out,y)

    #     self.log("validation loss", loss, prog_bar=True)

    #     return loss


    # def on_train_epoch_end(self):
    #     check_class_accuracy(self.model, self.val_dataloader(), threshold=config.CONF_THRESHOLD)

    def on_train_end(self) -> None:
        scaled_anchors = (
            torch.tensor(config.ANCHORS)
            * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1,3,2)
        ).to(self.device)

        plot_couple_examples(self.model, self.test_dataloader(), 0.6, 0.5, scaled_anchors)
        #print("Best mAP = {:.3f}, best mAP50 = {:.3f}".format(self.ap50_95, self.ap50))

        check_class_accuracy(self.model, self.train_dataloader(), threshold=config.CONF_THRESHOLD)
        pred_boxes, true_boxes = get_evaluation_bboxes(
            self.test_dataloader(),
            self.model,
            iou_threshold=config.NMS_IOU_THRESH,
            anchors=config.ANCHORS,
            threshold=config.CONF_THRESHOLD,
        )
        mapval = mean_average_precision(
            pred_boxes,
            true_boxes,
            iou_threshold=config.MAP_IOU_THRESH,
            box_format="midpoint",
            num_classes=config.NUM_CLASSES,
        )
        print(f"MAP: {mapval.item()}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config.LEARNING_RATE,
            # momentum=0.9,
            # weight_decay=5e-4,
        )
        EPOCHS = config.NUM_EPOCHS * 2 // 5

        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                max_lr=1E-3,
                steps_per_epoch=len(self.train_dataloader()),
                epochs=EPOCHS,
                pct_start=5/EPOCHS,
                div_factor=100,
                three_phase=False,
                final_div_factor=100,
                anneal_strategy='linear'
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
    
    def setup(self, stage=None):
        self.train_loader, self.test_loader, self.train_eval_loader = get_loaders(
            train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv"
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.train_eval_loader

    def test_dataloader(self):
        return self.test_loader