import argparse
import random
from copy import deepcopy
import logging
import os
import pprint

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiDataset
from model.semseg.dpt import DPT
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
from calibrator import MedSigLIPCalibrator

try:
    import wandb
except Exception:  # pragma: no cover - optional dependency
    wandb = None


parser = argparse.ArgumentParser(description='TiSage: Semi-Supervised Tissue Segmentation with Multi-Scale Semantic Guidance')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--wandb', action='store_true', help='enable Weights & Biases logging')
parser.add_argument('--wandb-project', type=str, default='tisage')
parser.add_argument('--wandb-entity', type=str, default=None)
parser.add_argument('--wandb-name', type=str, default=None)
parser.add_argument('--wandb-tags', type=str, default=None, help='comma-separated tags')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--deterministic', action='store_true', help='enable deterministic training (slower)')
parser.add_argument('--medsiglip', action='store_true', help='enable MedSigLIP pseudo-label calibration')
parser.add_argument('--medsiglip-alpha', type=float, default=None,
                    help='calibration strength (overrides config if set)')
parser.add_argument('--medsiglip-n-segments', type=int, default=None,
                    help='SLIC n_segments (overrides config if set)')
parser.add_argument('--medsiglip-context-margin', type=int, default=None,
                    help='crop context margin in pixels (overrides config if set)')
parser.add_argument('--medsiglip-recompute-every', type=int, default=None,
                    help='recompute MedSigLIP prior every N iters (overrides config if set)')
parser.add_argument('--medsiglip-gate-conf', type=float, default=None,
                    help='apply MedSigLIP only where teacher conf < this (0 = no gating)')
parser.add_argument('--medsiglip-classifier-path', type=str, default=None,
                    help='optional checkpoint path for MedSigLIP embedding classifier prior')
parser.add_argument('--medsiglip-soft-label', action='store_true',
                    help='use soft KL distillation with calibrated probs (requires --medsiglip)')
parser.add_argument('--medsiglip-soft-beta', type=float, default=None,
                    help='blend (1-beta)*CE_hard + beta*KL_soft; 1.0=pure soft (default from cfg or 0.5)')
parser.add_argument('--medsiglip-use-multiscale', action='store_true',
                    help='enable multiscale prior fusion (coarse+fine SLIC) in calibrator')
parser.add_argument('--medsiglip-coarse-n-segments', type=int, default=None,
                    help='coarse SLIC n_segments for multiscale prior')
parser.add_argument('--medsiglip-coarse-min-size', type=int, default=None,
                    help='coarse SLIC min region size for multiscale prior')
parser.add_argument('--medsiglip-fine-n-segments', type=int, default=None,
                    help='fine SLIC n_segments for multiscale prior')
parser.add_argument('--medsiglip-fine-min-size', type=int, default=None,
                    help='fine SLIC min region size for multiscale prior')
parser.add_argument('--medsiglip-prior-beta', type=float, default=None,
                    help='multiscale prior fusion weight for fine logits')
parser.add_argument('--medsiglip-adaptive-alpha', action='store_true',
                    help='enable pixel-adaptive alpha fusion based on teacher confidence')
parser.add_argument('--medsiglip-conf-weighted-kl', action='store_true',
                    help='use entropy-weighted KL on unlabeled data (no hard conf threshold by default)')
parser.add_argument('--medsiglip-use-hard-conf-mask-for-kl', action='store_true',
                    help='when conf-weighted KL is enabled, also keep hard confidence mask')


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)
    wandb_run = None
    
    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        logger.info(f'world_size={world_size}, rank={rank}')
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)
        
        if args.wandb:
            if wandb is None:
                logger.warning('wandb not installed; proceeding without wandb logging')
            else:
                tags = [t for t in (args.wandb_tags or '').split(',') if t]
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=args.wandb_name,
                    dir=args.save_path,
                    config=all_args,
                    tags=tags if tags else None,
                )

    cudnn.enabled = True
    if args.deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True

    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]], 'nclass': cfg['nclass']})
    state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth')
    model.backbone.load_state_dict(state_dict)
        
    if cfg['lock_backbone']:
        model.lock_backbone()
    
    optimizer = AdamW(
        [
            {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': cfg['lr']},
            {'params': [param for name, param in model.named_parameters() if 'backbone' not in name], 'lr': cfg['lr'] * cfg['lr_multi']}
        ], 
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01
    )
    
    if rank == 0:
        logger.info('Total params: {:.1f}M'.format(count_params(model)))
        logger.info('Encoder params: {:.1f}M'.format(count_params(model.backbone)))
        logger.info('Decoder params: {:.1f}M\n'.format(count_params(model.head)))
    
    local_rank = int(os.environ["LOCAL_RANK"])
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=False, output_device=local_rank, find_unused_parameters=True
    )
    
    model_ema = deepcopy(model)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False
    
    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)

    # MedSigLIP calibrator (frozen; zero-shot or embedding-classifier prior)
    use_medsiglip = args.medsiglip
    calibrator = None
    medsiglip_alpha = 0.0
    recompute_every = 1
    gate_conf = 0.0
    use_soft_label = False
    soft_beta = 0.5
    use_multiscale_prior = False
    coarse_n_segments = None
    coarse_min_size = None
    fine_n_segments = None
    fine_min_size = None
    prior_beta = 0.5
    adaptive_alpha = False
    conf_weighted_kl = False
    use_hard_conf_mask_for_kl = False
    if use_medsiglip:
        medsiglip_alpha = args.medsiglip_alpha if args.medsiglip_alpha is not None else cfg.get('medsiglip_alpha', 0.3)
        n_segments = args.medsiglip_n_segments if args.medsiglip_n_segments is not None else cfg.get('medsiglip_n_segments', 64)
        context_margin = args.medsiglip_context_margin if args.medsiglip_context_margin is not None else cfg.get('medsiglip_context_margin', 2)
        recompute_every = args.medsiglip_recompute_every if args.medsiglip_recompute_every is not None else cfg.get('medsiglip_recompute_every', 5)
        gate_conf = args.medsiglip_gate_conf if args.medsiglip_gate_conf is not None else cfg.get('medsiglip_gate_conf', 0.0)
        classifier_path = args.medsiglip_classifier_path if args.medsiglip_classifier_path else cfg.get('medsiglip_classifier_path', None)
        embed_batch_size = cfg.get('medsiglip_embed_batch_size', 32)
        use_multiscale_prior = args.medsiglip_use_multiscale or cfg.get('medsiglip_use_multiscale', False)
        coarse_n_segments = args.medsiglip_coarse_n_segments if args.medsiglip_coarse_n_segments is not None else cfg.get('medsiglip_coarse_n_segments', 32)
        coarse_min_size = args.medsiglip_coarse_min_size if args.medsiglip_coarse_min_size is not None else cfg.get('medsiglip_coarse_min_size', 80)
        fine_n_segments = args.medsiglip_fine_n_segments if args.medsiglip_fine_n_segments is not None else cfg.get('medsiglip_fine_n_segments', 128)
        fine_min_size = args.medsiglip_fine_min_size if args.medsiglip_fine_min_size is not None else cfg.get('medsiglip_fine_min_size', 40)
        prior_beta = args.medsiglip_prior_beta if args.medsiglip_prior_beta is not None else cfg.get('medsiglip_prior_beta', 0.5)
        adaptive_alpha = args.medsiglip_adaptive_alpha or cfg.get('medsiglip_adaptive_alpha', False)
        conf_weighted_kl = args.medsiglip_conf_weighted_kl or cfg.get('medsiglip_conf_weighted_kl', False)
        use_hard_conf_mask_for_kl = args.medsiglip_use_hard_conf_mask_for_kl or cfg.get('medsiglip_use_hard_conf_mask_for_kl', False)
        class_prompts = cfg.get('medsiglip_text_prompts', None)
        if class_prompts is not None and not isinstance(class_prompts, list):
            raise ValueError("cfg['medsiglip_text_prompts'] must be a list when provided.")
        calibrator = MedSigLIPCalibrator(
            device=torch.device('cuda', local_rank),
            n_segments=n_segments,
            context_margin=context_margin,
            slic_seed=args.seed,
            classifier_path=classifier_path,
            embed_batch_size=embed_batch_size,
            use_multiscale=use_multiscale_prior,
            coarse_n_segments=coarse_n_segments,
            coarse_min_size=coarse_min_size,
            fine_n_segments=fine_n_segments,
            fine_min_size=fine_min_size,
            prior_beta=prior_beta,
            dataset_name=cfg['dataset'],
            num_classes=cfg['nclass'],
            class_prompts=class_prompts,
        )
        use_soft_label = args.medsiglip_soft_label
        soft_beta = args.medsiglip_soft_beta if args.medsiglip_soft_beta is not None else cfg.get('medsiglip_soft_beta', 0.5)
        if rank == 0:
            prior_src = f'classifier={classifier_path}' if classifier_path else 'zero-shot text prompts'
            logger.info(f'MedSigLIP calibrator enabled: alpha={medsiglip_alpha}, '
                        f'n_segments={n_segments}, context_margin={context_margin}, '
                        f'recompute_every={recompute_every}, gate_conf={gate_conf}, '
                        f'embed_batch_size={embed_batch_size}, prior_source={prior_src}')
            if use_multiscale_prior:
                logger.info(f'MedSigLIP multiscale prior: coarse={coarse_n_segments}/{coarse_min_size}, '
                            f'fine={fine_n_segments}/{fine_min_size}, beta={prior_beta}')
            if adaptive_alpha:
                logger.info('MedSigLIP adaptive alpha enabled')
            if use_soft_label:
                logger.info(f'MedSigLIP soft-label KL distillation: beta={soft_beta} '
                            f'(loss = (1-beta)*CE_hard + beta*KL_soft)')
            if conf_weighted_kl:
                logger.info(f'MedSigLIP confidence-weighted KL enabled '
                            f'(hard_conf_mask_for_kl={use_hard_conf_mask_for_kl})')
            if class_prompts is not None:
                logger.info(f'MedSigLIP custom text prompts enabled (n={len(class_prompts)})')

    trainset_u = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_u', cfg['crop_size'], args.unlabeled_id_path
    )
    trainset_l = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids)
    )
    valset = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'val'
    )
    
    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(
        trainset_l, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_l
    )
    
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(
        trainset_u, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_u
    )
    
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
        valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, sampler=valsampler
    )
    
    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best, previous_best_ema = 0.0, 0.0
    best_epoch, best_epoch_ema = 0, 0
    epoch = -1
    
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model_ema.load_state_dict(checkpoint['model_ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        previous_best_ema = checkpoint['previous_best_ema']
        best_epoch = checkpoint['best_epoch']
        best_epoch_ema = checkpoint['best_epoch_ema']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    cached_medsiglip_prior = None
    total_prior_recomputes = 0
    total_pixels_calibrated = 0
    total_pixels_seen_unlabeled = 0

    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, Previous best: {:.2f} @epoch-{:}, '
                        'EMA: {:.2f} @epoch-{:}'.format(epoch, previous_best, best_epoch, previous_best_ema, best_epoch_ema))
        
        total_loss  = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_kl = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)
        
        # Reset MedSigLIP cache each epoch (dataloader reshuffles batches)
        cached_medsiglip_prior = None

        model.train()

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2)) in enumerate(loader):
            
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s1, img_u_s2 = img_u_w.cuda(), img_u_s1.cuda(), img_u_s2.cuda()
            ignore_mask, cutmix_box1, cutmix_box2 = ignore_mask.cuda(), cutmix_box1.cuda(), cutmix_box2.cuda()
            
            with torch.no_grad():
                pred_u_w = model_ema(img_u_w).detach()
                conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w.argmax(dim=1)

            # MedSigLIP calibration: reweight teacher pseudo-labels before CutMix
            soft_u_w = None
            if calibrator is not None and medsiglip_alpha > 0:
                with torch.no_grad():
                    teacher_probs = pred_u_w.softmax(dim=1)  # (B, C, H, W)
                    B, C, H, W = teacher_probs.shape
                    total_pixels_seen_unlabeled += B * H * W
                    if cached_medsiglip_prior is None or i % recompute_every == 0:
                        cached_medsiglip_prior = calibrator.compute_pixel_prior(img_u_w)
                        total_prior_recomputes += 1

                    if adaptive_alpha:
                        if gate_conf > 0:
                            alpha_x = medsiglip_alpha * torch.clamp(
                                (gate_conf - conf_u_w) / gate_conf, min=0.0, max=1.0
                            )
                            alpha_x = alpha_x.unsqueeze(1)  # (B, 1, H, W)
                            total_pixels_calibrated += (alpha_x > 0).sum().item()
                        else:
                            alpha_x = torch.full_like(teacher_probs[:, :1], fill_value=medsiglip_alpha)
                            total_pixels_calibrated += B * H * W

                        z_teacher = torch.log(torch.clamp(teacher_probs, min=1e-8))
                        z_prior = torch.log(torch.clamp(cached_medsiglip_prior, min=1e-8))
                        z_fused = (1.0 - alpha_x) * z_teacher + alpha_x * z_prior
                        calibrated = F.softmax(z_fused, dim=1)
                    else:
                        blended = teacher_probs * (1 - medsiglip_alpha + medsiglip_alpha * cached_medsiglip_prior)
                        blended = blended / blended.sum(dim=1, keepdim=True)  # renormalize
                        if gate_conf > 0:
                            mask_low = (conf_u_w < gate_conf).unsqueeze(1)  # (B, 1, H, W)
                            total_pixels_calibrated += mask_low.sum().item()
                            calibrated = torch.where(mask_low, blended, teacher_probs)
                        else:
                            total_pixels_calibrated += B * H * W
                            calibrated = blended
                    conf_u_w = calibrated.max(dim=1)[0]
                    mask_u_w = calibrated.argmax(dim=1)
                    if use_soft_label:
                        soft_u_w = calibrated  # (B, C, H, W) soft target for KL distillation

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = img_u_s1.flip(0)[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = img_u_s2.flip(0)[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
            
            pred_x = model(img_x)
            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2)), comp_drop=True).chunk(2)
            
            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w.flip(0)[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w.flip(0)[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask.flip(0)[cutmix_box1 == 1]
            
            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w.flip(0)[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w.flip(0)[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask.flip(0)[cutmix_box2 == 1]
            
            loss_x = criterion_l(pred_x, mask_x)

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()
            
            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

            # Soft-label KL distillation: blend (1-beta)*CE_hard + beta*KL_soft
            loss_kl_val = 0.0
            conf_weight_mean_val = 0.0
            if soft_u_w is not None:
                # CutMix the soft distributions (same boxes as hard labels)
                soft_u_w_cm1 = soft_u_w.clone()
                soft_u_w_cm2 = soft_u_w.clone()
                soft_u_w_cm1[cutmix_box1.unsqueeze(1).expand_as(soft_u_w) == 1] = \
                    soft_u_w.flip(0)[cutmix_box1.unsqueeze(1).expand_as(soft_u_w) == 1]
                soft_u_w_cm2[cutmix_box2.unsqueeze(1).expand_as(soft_u_w) == 1] = \
                    soft_u_w.flip(0)[cutmix_box2.unsqueeze(1).expand_as(soft_u_w) == 1]

                kl_map_s1 = F.kl_div(
                    F.log_softmax(pred_u_s1, dim=1), soft_u_w_cm1,
                    reduction='none', log_target=False,
                ).sum(dim=1)  # (B, H, W) per-pixel KL
                kl_map_s2 = F.kl_div(
                    F.log_softmax(pred_u_s2, dim=1), soft_u_w_cm2,
                    reduction='none', log_target=False,
                ).sum(dim=1)

                if conf_weighted_kl:
                    max_entropy = float(np.log(C))
                    entropy_s1 = -(soft_u_w_cm1 * torch.log(torch.clamp(soft_u_w_cm1, min=1e-8))).sum(dim=1)
                    entropy_s2 = -(soft_u_w_cm2 * torch.log(torch.clamp(soft_u_w_cm2, min=1e-8))).sum(dim=1)
                    conf_weight_s1 = (1.0 - entropy_s1 / max_entropy).clamp(0.0, 1.0)
                    conf_weight_s2 = (1.0 - entropy_s2 / max_entropy).clamp(0.0, 1.0)

                    base_mask_s1 = (ignore_mask_cutmixed1 != 255).float()
                    base_mask_s2 = (ignore_mask_cutmixed2 != 255).float()
                    if use_hard_conf_mask_for_kl:
                        base_mask_s1 = base_mask_s1 * (conf_u_w_cutmixed1 >= cfg['conf_thresh']).float()
                        base_mask_s2 = base_mask_s2 * (conf_u_w_cutmixed2 >= cfg['conf_thresh']).float()

                    denom_s1 = max(base_mask_s1.sum().item(), 1.0)
                    denom_s2 = max(base_mask_s2.sum().item(), 1.0)
                    kl_s1 = (kl_map_s1 * conf_weight_s1 * base_mask_s1).sum() / denom_s1
                    kl_s2 = (kl_map_s2 * conf_weight_s2 * base_mask_s2).sum() / denom_s2
                    conf_weight_mean_val = (
                        (conf_weight_s1 * base_mask_s1).sum() / denom_s1 +
                        (conf_weight_s2 * base_mask_s2).sum() / denom_s2
                    ).item() / 2.0
                else:
                    valid1 = ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255)).float()
                    valid2 = ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255)).float()
                    kl_s1 = (kl_map_s1 * valid1).sum() / max((ignore_mask_cutmixed1 != 255).sum().item(), 1.0)
                    kl_s2 = (kl_map_s2 * valid2).sum() / max((ignore_mask_cutmixed2 != 255).sum().item(), 1.0)

                loss_kl_val = ((kl_s1 + kl_s2) / 2.0).item()
                loss_u_s1 = (1 - soft_beta) * loss_u_s1 + soft_beta * kl_s1
                loss_u_s2 = (1 - soft_beta) * loss_u_s2 + soft_beta * kl_s2

            loss_u_s = (loss_u_s1 + loss_u_s2) / 2.0
            
            loss = (loss_x + loss_u_s) / 2.0
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())
            if soft_u_w is not None:
                total_loss_kl.update(loss_kl_val)
            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            ema_ratio = min(1 - 1 / (iters + 1), 0.996)
            
            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
            for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', loss_u_s.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)
                log_dict = {
                    'train/loss_all': loss.item(),
                    'train/loss_x': loss_x.item(),
                    'train/loss_s': loss_u_s.item(),
                    'train/mask_ratio': mask_ratio,
                    'train/lr': optimizer.param_groups[0]['lr'],
                }
                if soft_u_w is not None:
                    writer.add_scalar('train/loss_kl', loss_kl_val, iters)
                    log_dict['train/loss_kl'] = loss_kl_val
                    if conf_weighted_kl:
                        writer.add_scalar('train/conf_weight_mean', conf_weight_mean_val, iters)
                        log_dict['train/conf_weight_mean'] = conf_weight_mean_val
                if wandb_run is not None:
                    wandb.log(log_dict, step=iters)

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, optimizer.param_groups[0]['lr'], total_loss.avg, total_loss_x.avg, 
                                            total_loss_s.avg, total_mask_ratio.avg))
        
        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg, multiplier=14)
        mIoU_ema, iou_class_ema = evaluate(model_ema, valloader, eval_mode, cfg, multiplier=14)
        
        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] IoU: {:.2f}, '
                            'EMA: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou, iou_class_ema[cls_idx]))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}, EMA: {:.2f}\n'.format(eval_mode, mIoU, mIoU_ema))
            
            writer.add_scalar('eval/mIoU', mIoU, epoch)
            writer.add_scalar('eval/mIoU_ema', mIoU_ema, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)
                writer.add_scalar('eval/%s_IoU_ema' % (CLASSES[cfg['dataset']][i]), iou_class_ema[i], epoch)
            if wandb_run is not None:
                wandb.log(
                    {
                        'eval/mIoU': mIoU,
                        'eval/mIoU_ema': mIoU_ema,
                    },
                    step=iters,
                )

        is_best = mIoU >= previous_best
        
        previous_best = max(mIoU, previous_best)
        previous_best_ema = max(mIoU_ema, previous_best_ema)
        if mIoU == previous_best:
            best_epoch = epoch
        if mIoU_ema == previous_best_ema:
            best_epoch_ema = epoch

        if rank == 0 and wandb_run is not None:
            wandb.log(
                {
                    'eval/best_mIoU': previous_best,
                    'eval/best_mIoU_ema': previous_best_ema,
                },
                step=iters,
            )
        
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'model_ema': model_ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
                'previous_best_ema': previous_best_ema,
                'best_epoch': best_epoch,
                'best_epoch_ema': best_epoch_ema
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))

    # MedSigLIP calibration summary (global counts across all ranks)
    if calibrator is not None and medsiglip_alpha > 0:
        import torch.distributed as dist

        t_recomputes = torch.tensor([total_prior_recomputes], dtype=torch.long, device=torch.device('cuda', local_rank))
        t_pixels_cal = torch.tensor([total_pixels_calibrated], dtype=torch.long, device=torch.device('cuda', local_rank))
        t_pixels_seen = torch.tensor([total_pixels_seen_unlabeled], dtype=torch.long, device=torch.device('cuda', local_rank))
        dist.all_reduce(t_recomputes, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_pixels_cal, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_pixels_seen, op=dist.ReduceOp.SUM)

        if rank == 0:
            pct = 100.0 * t_pixels_cal.item() / max(1, t_pixels_seen.item())
            logger.info('***** MedSigLIP calibration summary ***** prior_recomputes={}, '
                        'pixels_calibrated={} / {} ({:.2f}%)'.format(
                        t_recomputes.item(), t_pixels_cal.item(), t_pixels_seen.item(), pct))

    if rank == 0 and wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    main()
