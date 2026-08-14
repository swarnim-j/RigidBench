import argparse
import os
import time
from functools import partial
from pathlib import Path

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy
from peft import LoraConfig, inject_adapter_in_model
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from rigidbench.train.backbones import BackboneAdapter, get_adapter
from rigidbench.train.data.dataset import EmbeddingDataset


class StepTimer(L.Callback):
    """Lightning callback that logs wall-clock seconds per training step."""

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._t = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        pl_module.log("train/step_time_s", time.perf_counter() - self._t)


class DiffusionModule(L.LightningModule):
    """Lightning module wrapping a BackboneAdapter for flow-matching fine-tuning."""

    def __init__(self, cfg: dict, adapter: BackboneAdapter):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.adapter = adapter
        self.dit = None
        if cfg.get("lora"):
            self.strict_loading = False

    def setup(self, stage: str):
        """Lazy backbone load, optional LoRA injection, optional conditioning-pathway freeze."""
        if self.dit is not None:
            return

        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        self.adapter.load(device=self.device, mode="train", rank=self.global_rank, world_size=world_size)
        self.dit = self.adapter.get_trainable_module()

        lora_cfg = self.cfg.get("lora")
        if lora_cfg:
            target_modules = lora_cfg.get("target_modules") or self.adapter.default_lora_target_modules()
            inject_adapter_in_model(
                LoraConfig(
                    r=lora_cfg["rank"],
                    lora_alpha=lora_cfg.get("alpha", lora_cfg["rank"]),
                    lora_dropout=lora_cfg.get("dropout", 0.0),
                    target_modules=target_modules,
                ),
                self.dit,
            )
            for n, p in self.dit.named_parameters():
                if "lora_" not in n:
                    p.requires_grad = False

        if self.cfg.get("freeze_conditioning"):
            prefixes = self.adapter.conditioning_param_prefixes()
            for n, p in self.dit.named_parameters():
                if n.startswith(prefixes):
                    p.requires_grad = False

    def on_save_checkpoint(self, checkpoint):
        """Keep only LoRA tensors in saved state dict when training a LoRA."""
        if self.cfg.get("lora"):
            checkpoint["state_dict"] = self.adapter.filter_lora_state_dict(checkpoint["state_dict"])

    def training_step(self, batch, batch_idx):
        """Delegate loss computation to the backbone adapter."""
        loss, logs = self.adapter.training_step(
            batch=batch,
            device=self.device,
            use_gradient_checkpointing=self.cfg["grad_checkpoint"],
        )
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        for key, value in logs.items():
            self.log(f"train/{key}", value, sync_dist=True)
        return loss

    def configure_optimizers(self):
        params = [p for p in self.dit.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=float(self.cfg["lr"]),
            weight_decay=float(self.cfg["weight_decay"]),
            eps=float(self.cfg["adam_eps"]),
            fused=True,
        )
        max_steps = self.cfg["max_steps"]
        warmup_steps = int(max_steps * self.cfg.get("warmup_ratio", 0.05))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        """FSDP-aware gradient clipping that logs the resulting norm."""
        if isinstance(self.trainer.model, FSDP):
            norm = self.trainer.model.clip_grad_norm_(gradient_clip_val)
        else:
            norm = torch.nn.utils.clip_grad_norm_(self.dit.parameters(), gradient_clip_val)
        self.log("train/grad_norm", norm, sync_dist=True)


def load_config(
    path: str,
    data_path: str | None = None,
    output_dir: str | None = None,
    wandb_project: str | None = None,
) -> dict:
    """Load a training config and resolve its local paths."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["data_path"] = data_path or os.environ.get("DATA_PATH") or cfg.get("data_path")
    cfg["output_dir"] = output_dir or os.environ.get("OUTPUT_DIR") or cfg.get("output_dir")
    cfg["wandb_project"] = wandb_project or os.environ.get("WANDB_PROJECT") or cfg.get("wandb_project", "rigidbench")
    if not cfg["data_path"] or not cfg["output_dir"]:
        raise ValueError("Training requires --data and --output (or DATA_PATH and OUTPUT_DIR).")
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Wan on preprocessed RigidBench clips.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data", "--data-path", dest="data_path")
    parser.add_argument("--output", "--output-dir", dest="output_dir")
    parser.add_argument("--wandb-project")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    try:
        cfg = load_config(args.config, args.data_path, args.output_dir, args.wandb_project)
    except ValueError as error:
        parser.error(str(error))
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = get_adapter(cfg.get("backbone", "wan"), cfg)
    resume_path = args.resume

    L.seed_everything(cfg["seed"])

    train_loader = DataLoader(
        EmbeddingDataset(cfg["data_path"]),
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    run_id = os.environ.get("WANDB_RUN_ID")
    wandb_logger = WandbLogger(
        project=cfg["wandb_project"],
        name=run_id,
        id=run_id,
        resume="allow" if run_id else None,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=str(output_dir),
            filename="ckpt-{step:06d}",
            auto_insert_metric_name=False,
            every_n_train_steps=cfg["ckpt_every_steps"],
            save_top_k=-1,
            save_last=False,
        ),
        LearningRateMonitor(),
        StepTimer(),
    ]

    if torch.cuda.device_count() <= 1:
        strategy = "auto"
    elif cfg.get("strategy", "fsdp") == "ddp":
        strategy = DDPStrategy()
    else:
        strategy = FSDPStrategy(
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=adapter.fsdp_wrap_classes(),
            ),
            sharding_strategy="FULL_SHARD",
            state_dict_type="full",
        )

    trainer = L.Trainer(
        accelerator="gpu",
        devices="auto",
        strategy=strategy,
        precision="bf16-mixed",
        max_steps=cfg["max_steps"],
        accumulate_grad_batches=cfg["gradient_accumulation_steps"],
        gradient_clip_val=cfg["grad_clip"],
        log_every_n_steps=cfg["log_every_n_steps"],
        callbacks=callbacks,
        logger=wandb_logger,
    )

    trainer.fit(DiffusionModule(cfg, adapter), train_loader, ckpt_path=resume_path)


if __name__ == "__main__":
    main()
