"""Standalone fully supervised DeepLabV3+ training for semantic segmentation."""

import argparse
import json
import logging
import os
from pathlib import Path
import pprint
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.classes import CLASSES
from util.dist_helper import setup_distributed
from util.utils import AverageMeter, count_params, init_log, intersectionAndUnion

try:
    import wandb
except Exception:  # pragma: no cover - optional dependency
    wandb = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fully supervised DeepLabV3+ training"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--labeled-id-path", required=True)
    parser.add_argument("--unlabeled-id-path", default=None)
    parser.add_argument("--pretrained-path", default=None)
    parser.add_argument("--allow-weight-download", action="store_true")
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--local_rank", "--local-rank", default=0, type=int)
    parser.add_argument("--port", default=None, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="supervised-segmentation")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-tags", default=None)
    return parser.parse_args()


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.enabled = True
    cudnn.deterministic = deterministic
    cudnn.benchmark = not deterministic


def evaluate(model, loader, cfg, device):
    model.eval()
    nclass = cfg["nclass"]
    totals = torch.zeros(3, nclass, dtype=torch.float64, device=device)

    with torch.inference_mode():
        for image, mask, _ in loader:
            image = image.to(device, non_blocking=True)
            prediction = model(image).argmax(dim=1).cpu().numpy()
            intersection, union, target = intersectionAndUnion(
                prediction,
                mask.numpy(),
                nclass,
                cfg["criterion"]["kwargs"].get("ignore_index", 255),
            )
            totals[0] += torch.as_tensor(intersection, dtype=torch.float64, device=device)
            totals[1] += torch.as_tensor(union, dtype=torch.float64, device=device)
            totals[2] += torch.as_tensor(target, dtype=torch.float64, device=device)

    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    intersection, union, target = totals.cpu().numpy()
    iou_class = intersection / (union + 1e-10) * 100.0
    prediction_area = union + intersection - target
    dice_class = 2.0 * intersection / (prediction_area + target + 1e-10) * 100.0
    return float(np.mean(iou_class)), float(np.mean(dice_class)), iou_class, dice_class


def save_metrics(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main():
    args = parse_args()
    with open(args.config, "r") as handle:
        cfg = yaml.safe_load(handle)

    rank, world_size = setup_distributed(port=args.port)
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    set_seed(args.seed + rank, args.deterministic)

    logger = init_log("global", logging.INFO)
    logger.propagate = False
    save_path = Path(args.save_path)
    if rank == 0:
        save_path.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    latest_path = save_path / "latest.pth"
    if latest_path.exists() and not args.resume:
        raise FileExistsError(
            f"{latest_path} already exists. Use a new save path or pass --resume."
        )

    pretrained_path = args.pretrained_path or cfg.get("pretrained_path")
    model = DeepLabV3Plus(
        nclass=cfg["nclass"],
        output_stride=cfg.get("output_stride", 16),
        pretrained_path=pretrained_path,
        allow_weight_download=args.allow_weight_download,
    )
    if cfg.get("sync_bn", world_size > 1):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)

    encoder_lr = float(cfg["lr"])
    decoder_lr = encoder_lr * float(cfg.get("lr_multi", 10.0))
    model_for_groups = model
    optimizer = SGD(
        [
            {"params": list(model_for_groups.encoder_parameters()), "lr": encoder_lr},
            {"params": list(model_for_groups.decoder_parameters()), "lr": decoder_lr},
        ],
        lr=encoder_lr,
        momentum=float(cfg.get("momentum", 0.9)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        nesterov=bool(cfg.get("nesterov", True)),
    )
    initial_lrs = [encoder_lr, decoder_lr]

    start_epoch = 0
    best_miou = -1.0
    best_dice = -1.0
    best_epoch = -1
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.get("amp", True)))
    if args.resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"Cannot resume; checkpoint not found: {latest_path}")
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_miou = float(checkpoint.get("best_miou", checkpoint.get("previous_best", -1.0)))
        best_dice = float(checkpoint.get("best_dice", -1.0))
        best_epoch = int(checkpoint.get("best_epoch", checkpoint["epoch"]))

    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
    )

    ignore_index = cfg["criterion"]["kwargs"].get("ignore_index", 255)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index).to(device)

    samples_per_epoch = cfg.get("samples_per_epoch")
    trainset = SemiDataset(
        cfg["dataset"],
        cfg["data_root"],
        "train_l",
        cfg["crop_size"],
        args.labeled_id_path,
        nsample=samples_per_epoch,
    )
    valset = SemiDataset(cfg["dataset"], cfg["data_root"], "val")
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=True, seed=args.seed
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        valset, shuffle=False
    )
    trainloader = DataLoader(
        trainset,
        batch_size=cfg["batch_size"],
        sampler=train_sampler,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
        persistent_workers=bool(cfg.get("num_workers", 4)),
    )
    valloader = DataLoader(
        valset,
        batch_size=1,
        sampler=val_sampler,
        num_workers=int(cfg.get("val_num_workers", 2)),
        pin_memory=True,
        drop_last=False,
        persistent_workers=bool(cfg.get("val_num_workers", 2)),
    )

    epochs = int(cfg["epochs"])
    total_iters = max(1, len(trainloader) * epochs)
    poly_power = float(cfg.get("poly_power", 0.9))
    eval_every = int(cfg.get("eval_every", 1))
    use_amp = bool(cfg.get("amp", True))
    global_step = start_epoch * len(trainloader)

    writer = None
    wandb_run = None
    if rank == 0:
        all_args = {**cfg, **vars(args), "world_size": world_size}
        logger.info("%s\n", pprint.pformat(all_args))
        logger.info(
            "Model: DeepLabV3+ ResNet-50 OS%d, parameters: %.1fM",
            cfg.get("output_stride", 16),
            count_params(model.module),
        )
        logger.info(
            "Training images in split: %d; samples per epoch: %d; "
            "batches per epoch/rank: %d",
            len(Path(args.labeled_id_path).read_text().splitlines()),
            len(trainset),
            len(trainloader),
        )
        writer = SummaryWriter(str(save_path))
        if args.wandb:
            if wandb is None:
                logger.warning("wandb is unavailable; continuing without it")
            else:
                tags = [tag for tag in (args.wandb_tags or "").split(",") if tag]
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=args.wandb_name,
                    tags=tags or None,
                    config=all_args,
                    dir=str(save_path),
                )

    for epoch in range(start_epoch, epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        loss_meter = AverageMeter()

        for iteration, (image, mask) in enumerate(trainloader):
            progress = min(global_step / total_iters, 1.0)
            lr_scale = (1.0 - progress) ** poly_power
            for group, initial_lr in zip(optimizer.param_groups, initial_lrs):
                group["lr"] = initial_lr * lr_scale

            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(image)
                loss = criterion(logits, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), image.shape[0])
            if rank == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr_encoder", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("train/lr_decoder", optimizer.param_groups[1]["lr"], global_step)
                if wandb_run is not None:
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/lr_encoder": optimizer.param_groups[0]["lr"],
                            "train/lr_decoder": optimizer.param_groups[1]["lr"],
                        },
                        step=global_step,
                    )
            global_step += 1

            log_every = max(1, len(trainloader) // 8)
            if rank == 0 and iteration % log_every == 0:
                logger.info(
                    "Epoch %d/%d, iter %d/%d, loss %.4f, encoder LR %.6g",
                    epoch + 1,
                    epochs,
                    iteration + 1,
                    len(trainloader),
                    loss_meter.avg,
                    optimizer.param_groups[0]["lr"],
                )

        should_evaluate = (epoch + 1) % eval_every == 0 or epoch + 1 == epochs
        if not should_evaluate:
            continue

        miou, mean_dice, iou_class, dice_class = evaluate(model, valloader, cfg, device)
        is_best = miou > best_miou
        if is_best:
            best_miou = miou
            best_dice = mean_dice
            best_epoch = epoch

        if rank == 0:
            for class_index, (iou, dice) in enumerate(zip(iou_class, dice_class)):
                class_name = CLASSES[cfg["dataset"]][class_index]
                logger.info(
                    "Evaluation class [%d %s]: IoU %.2f, Dice %.2f",
                    class_index,
                    class_name,
                    iou,
                    dice,
                )
            logger.info(
                "Evaluation epoch %d: MeanIoU %.2f, MeanDice %.2f; "
                "BEST mIoU %.2f, Dice %.2f at epoch %d",
                epoch,
                miou,
                mean_dice,
                best_miou,
                best_dice,
                best_epoch,
            )
            writer.add_scalar("eval/mIoU", miou, epoch)
            writer.add_scalar("eval/mean_dice", mean_dice, epoch)
            writer.add_scalar("eval/best_mIoU", best_miou, epoch)
            if wandb_run is not None:
                wandb.log(
                    {
                        "eval/mIoU": miou,
                        "eval/mean_dice": mean_dice,
                        "eval/best_mIoU": best_miou,
                        "eval/best_dice": best_dice,
                    },
                    step=global_step,
                )

            checkpoint = {
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_miou": best_miou,
                "best_dice": best_dice,
                "config": cfg,
                "args": vars(args),
            }
            torch.save(checkpoint, latest_path)
            if is_best:
                torch.save(checkpoint, save_path / "best.pth")
            save_metrics(
                save_path / "metrics.json",
                {
                    "best": {
                        "epoch": best_epoch,
                        "miou": best_miou,
                        "dice": best_dice,
                    },
                    "latest": {
                        "epoch": epoch,
                        "miou": miou,
                        "dice": mean_dice,
                    },
                    "seed": args.seed,
                    "labeled_id_path": args.labeled_id_path,
                    "pretrained_path": pretrained_path,
                },
            )

    if rank == 0:
        logger.info(
            "Training complete. BEST mIoU %.2f, Dice %.2f at epoch %d",
            best_miou,
            best_dice,
            best_epoch,
        )
        writer.close()
        if wandb_run is not None:
            wandb_run.finish()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
