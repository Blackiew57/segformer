import os
from PIL import Image
import numpy as np
import argparse
import warnings
import torch
import wandb
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.pytorch.accelerator import find_usable_cuda_devices
from transformers import SegformerImageProcessor
# from torchmetrics.segmentation import MeanIoU
from src.dataset import SemanticSegmentationDataset
from src.model import create_model
import segmentation_models_pytorch as smp
# from segmentation_models_pytorch.metrics import dice_score
# from src.utils import add_batch, compute, apalette
warnings.filterwarnings('ignore')

pl.seed_everything(63, workers=True)
image_processor = SegformerImageProcessor(do_resize=False, reduce_labels=False)


class DataModule(pl.LightningDataModule):
    def __init__(self,
        root_dir,
        crop_size,
        image_processor,
        batch_size,
        num_workers):
        super().__init__()
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.image_processor = image_processor
        self.batch_size = batch_size
        self.num_workers = num_workers


    def setup(self, stage: str):
        if stage == "fit":
            self.trainset = SemanticSegmentationDataset(root_dir=self.root_dir, image_processor=self.image_processor, crop_size=self.crop_size)
            self.validset = SemanticSegmentationDataset(root_dir=self.root_dir, image_processor=self.image_processor, crop_size=self.crop_size, train='val')
        if stage == 'predict':
            self.predset = SemanticSegmentationDataset(root_dir=self.root_dir, image_processor=self.image_processor, crop_size=self.crop_size, train='predict')

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True, pin_memory=True)
        
    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=1, num_workers=self.num_workers, shuffle=False, drop_last=True, pin_memory=True)
    
    def predict_dataloader(self):
        return DataLoader(self.predset, batch_size=1, num_workers=self.num_workers, shuffle=False, drop_last=True, pin_memory=True)


class TrainingModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.train_miou_smp = list()
        self.train_f1_smp = list()
        self.train_acc_smp = list()
        # self.train_dice_smp = list()
        self.val_miou_smp = list()
        self.val_f1_smp = list()
        self.val_acc_smp = list()
        # self.val_dice_smp = list()

    def forward(self, images, targets):
        return self.model(images, targets)

    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=6e-5)
        # return optim.Adam(self.model.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx):
        batch, _ = batch
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        outputs = self.model(pixel_values=pixel_values, labels=labels)
        loss, logits = outputs.loss, outputs.logits

        preds = nn.functional.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        # if batch_idx % 80 == 0:
        #     sample_images = pixel_values.permute(0,2,3,1).detach().cpu().numpy()
        #     label_images = torch.squeeze(labels,dim=1).detach().cpu().numpy()
        #     pred_images = torch.squeeze(preds,dim=1).detach().cpu().numpy()
        #     columns = ['sample_images', 'label_images', 'pred_images']
        #     data = [[wandb.Image(sample_images[i]), wandb.Image(label_images[i]), wandb.Image(pred_images[i])] for i in range(len(sample_images))]
        #     wandb_logger.log_table(key='train_tumor', columns=columns, data=data)

        tp, fp, fn, tn = smp.metrics.get_stats(preds, labels, mode='binary')
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn).mean()
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn).mean()
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn).mean()
        # dice = smp.metrics.dice_score(tp, fp, fn, tn).mean()
        self.train_miou_smp.append(iou_score)
        self.train_f1_smp.append(f1_score)
        self.train_acc_smp.append(accuracy)
        # self.train_dice_smp.append(dice)
        self.log(
            'train_loss', loss.item(), on_step=True, on_epoch=False, prog_bar=True
        )
        
        return loss
    
    def on_train_epoch_end(self):
        train_miou_smp = sum(self.train_miou_smp) / len(self.train_miou_smp)
        train_f1_smp = sum(self.train_f1_smp) / len(self.train_f1_smp)
        train_acc_smp = sum(self.train_acc_smp) / len(self.train_acc_smp)
        # train_dice_smp = sum(self.train_dice_smp) / len(self.train_dice_smp)
        self.log(
            'train_epoch_miou', train_miou_smp, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            'train_epoch_f1', train_f1_smp, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            'train_epoch_accracy', train_acc_smp, on_step=False, on_epoch=True, prog_bar=True
        )
        # self.log(
        #     'train_epoch_dice_score', train_dice_smp, on_step=False, on_epoch=True, prog_bar=True
        # )
        self.train_miou_smp.clear()
        self.train_f1_smp.clear()
        self.train_acc_smp.clear()
        # self.train_dice_smp.clear()

    def validation_step(self, batch, batch_idx):
        batch, _ = batch
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        outputs = self.model(pixel_values=pixel_values, labels=labels)
        loss, logits = outputs.loss, outputs.logits

        preds = nn.functional.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)
        # if batch_idx % 5 == 0:
        #     sample_images = pixel_values.permute(0,2,3,1).detach().cpu().numpy()
        #     label_images = torch.squeeze(labels,dim=1).detach().cpu().numpy()
        #     pred_images = torch.squeeze(preds,dim=1).detach().cpu().numpy()
        #     columns = ['sample_images', 'label_images', 'pred_images']
        #     data = [[wandb.Image(sample_images[i]), wandb.Image(label_images[i]), wandb.Image(pred_images[i])] for i in range(len(sample_images))]
        #     wandb_logger.log_table(key='val_tumor', columns=columns, data=data)
        
        tp, fp, fn, tn = smp.metrics.get_stats(preds, labels, mode='binary')
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn).mean()
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn).mean()
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn).mean()
        # dice = smp.metrics.dice_score(tp, fp, fn, tn).mean()
        self.val_miou_smp.append(iou_score)
        self.val_f1_smp.append(f1_score)
        self.val_acc_smp.append(accuracy)
        # self.val_dice_smp.append(dice)
        
        self.log(
            'val_loss', loss.item(), on_step=True, on_epoch=False, prog_bar=True
        )
        return loss
    
    def on_validation_epoch_end(self):
        val_miou_smp = sum(self.val_miou_smp) / len(self.val_miou_smp)
        val_f1_smp = sum(self.val_f1_smp) / len(self.val_f1_smp)
        val_acc_smp = sum(self.val_acc_smp) / len(self.val_acc_smp)
        # val_dice_smp = sum(self.val_dice_smp) / len(self.val_dice_smp)
        self.log(
            'val_epoch_miou', val_miou_smp, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            'val_epoch_f1', val_f1_smp, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            'val_epoch_accracy', val_acc_smp, on_step=False, on_epoch=True, prog_bar=True
        )
        # self.log(
        #     'val_epoch_dice_score', val_dice_smp, on_step=False, on_epoch=True, prog_bar=True
        # )
        self.val_miou_smp.clear()
        self.val_f1_smp.clear()
        self.val_acc_smp.clear()
        # self.val_dice_smp.clear()

    def predict_step(self, batch, batch_idx):
        batch, image_path = batch
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits

        preds = nn.functional.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_images = torch.squeeze(preds,dim=1).detach().cpu().numpy()
        save_dir = f"{image_path[0].split('/')[0]}/{image_path[0].split('/')[1]}/preds/new/"
        image_name = image_path[0].split('/')[-1].replace('.jpg', '.png')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        palette = [0,0,0,255,255,255]
        pred = Image.fromarray(pred_images[0].astype(np.int8)).convert('P')
        pred.putpalette(palette)
        pred.save(f'{save_dir}{image_name}')


def run(args):
    data_module = DataModule(args.root,
                             args.crop_size,
                             image_processor,
                             args.batch_size,
                             args.num_workers)

    if len(args.devices[0]) == 1:
        devices = [int(args.devices[0])]
    else:
        devices = list(map(int, args.devices[0].split(',')))

    model = create_model()
    if args.mode == 'train':
        model = TrainingModule(model)
        ckpt_path = args.weight_path
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        
        checkpoint_callback = ModelCheckpoint(
            monitor='val_epoch_miou',
            dirpath=ckpt_path,
            filename='{epoch}-{val_epoch_miou:.4f}',
            save_top_k=1,
            mode='max',
            save_weights_only=True
        )
        
        early_stopping_callback = EarlyStopping(
            monitor='val_epoch_miou',
            patience=10,
            mode='max'
        )
        
        trainer = pl.Trainer(
            check_val_every_n_epoch=1,
            log_every_n_steps=1,
            logger=wandb_logger,
            max_epochs=args.max_epochs,
            accelerator='gpu',
            devices=devices,
            precision='16-mixed',
            callbacks=[checkpoint_callback]
            # callbacks=[checkpoint_callback, early_stopping_callback]
        )

        trainer.fit(model, data_module)
    
    if args.mode == 'predict':
        # 체크포인트 파일을 로드합니다.
        checkpoint = torch.load(args.checkpoint)

        # state_dict에서 "model." 접두사를 제거합니다.
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[len('model.'):]] = v
            else:
                new_state_dict[k] = v

        # 수정된 state_dict를 사용하여 모델에 로드합니다.
        model.load_state_dict(new_state_dict)
        # model = model.load_state_dict(torch.load((args.checkpoint)))
        model = TrainingModule(model)
        trainer = pl.Trainer(
            accelerator='cuda',
            strategy='ddp',
            devices=devices,
            precision='16-mixed',
            
        )
        
        trainer.predict(model, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segformer')
    parser.add_argument('--root', type=str, default='./dataset/')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--crop_size', type=int, default=1024)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--project', type=str, default='segformer-YSGIST')
    parser.add_argument('--weight_path', type=str, default='weights/')
    parser.add_argument('--devices', type=str, nargs='+', default='0, 1, 2, 3')
    parser.add_argument('--checkpoint', type=str, default='weights/')
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()
    if args.mode == 'train':
        wandb_logger = WandbLogger(project=args.project)
    run(args)