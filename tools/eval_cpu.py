# ------------------------------------------------------------------------------
# CPU-only evaluation script for PIDNet / PIDNet-FFC variants.
# Based on tools/eval.py — all CUDA dependencies removed.
# Usage:
#   python tools/eval_cpu.py --cfg configs/cityscapes/pidnet_small_cityscapes.yaml \
#       --model pidnet                          # or pidnet_ffc_34
#       TEST.MODEL_FILE output/cityscapes/pidnet_small_cityscapes/best.pt
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import logging
import timeit

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

import _init_paths
import datasets
from configs import config
from configs import update_config
from utils.utils import create_logger


# ── helpers ─────────────────────────────────────────────────────────────────────

def get_confusion_matrix_cpu(label, pred, size, num_class, ignore=-1):
    """Same as get_confusion_matrix but runs entirely on CPU."""
    seg_pred = pred.argmax(dim=1)               # (B, H, W)
    seg_gt = label[:, :size[-2], :size[-1]]      # (B, H, W)

    mask = seg_gt != ignore
    seg_gt = seg_gt[mask]
    seg_pred = seg_pred[mask]

    index = (seg_gt * num_class + seg_pred).long()
    cm_flat = torch.bincount(index, minlength=num_class * num_class)
    return cm_flat.reshape(num_class, num_class).numpy()


def testval_cpu(config, test_dataset, testloader, model, sv_dir='./', sv_pred=False):
    """Validation on the val split, fully on CPU."""
    model.eval()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))

    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, _, name = batch
            size = label.size()
            label = label.long()        # keep on CPU
            # inference — image stays on CPU
            pred = test_dataset.single_scale_inference(config, model, image)

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear',
                    align_corners=config.MODEL.ALIGN_CORNERS
                )

            confusion_matrix += get_confusion_matrix_cpu(
                label, pred, size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'val_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % mean_IoU)

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum() / pos.sum()
    mean_acc = (tp / np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc


def test_cpu(config, test_dataset, testloader, model, sv_dir='./'):
    """Test-set inference (no labels), fully on CPU."""
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.single_scale_inference(config, model, image)

            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear',
                    align_corners=config.MODEL.ALIGN_CORNERS
                )

            sv_path = os.path.join(sv_dir, 'test_results')
            if not os.path.exists(sv_path):
                os.mkdir(sv_path)
            test_dataset.save_pred(pred, sv_path, name)


# ── CLI ─────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='PIDNet CPU evaluation')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='configs/cityscapes/pidnet_small_cityscapes.yaml',
                        type=str)
    parser.add_argument('--model',
                        help='model module to use: pidnet | pidnet_ffc_34 | pidnet_ffc_45 | pidnet_ffc_345',
                        default='pidnet',
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)
    return args


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    device = torch.device('cpu')
    logger.info('Running evaluation on CPU')

    # ── build model ────────────────────────────────────────────────────────
    import importlib
    model_module = importlib.import_module('models.' + args.model)
    model = model_module.get_seg_model(config, imgnet_pretrained=False)

    # ── load checkpoint ────────────────────────────────────────────────────
    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir, 'best.pt')

    logger.info('=> loading model from {}'.format(model_state_file))

    pretrained_dict = torch.load(model_state_file, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']

    model_dict = model.state_dict()

    # Handle keys saved with 'model.' prefix (from FullModel wrapper)
    cleaned = {}
    for k, v in pretrained_dict.items():
        new_key = k
        if k.startswith('model.'):
            new_key = k[6:]
        if new_key in model_dict and v.shape == model_dict[new_key].shape:
            cleaned[new_key] = v

    logger.info('=> loaded {} / {} parameters'.format(len(cleaned), len(model_dict)))
    model_dict.update(cleaned)
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()

    # ── dataset / dataloader ───────────────────────────────────────────────
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])

    # Temporarily make .cuda() a no-op so Cityscapes.__init__ doesn't crash
    _real_cuda = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *a, **kw: self
    test_dataset = eval('datasets.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TEST_SET,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE,
        crop_size=test_size)
    torch.Tensor.cuda = _real_cuda  # restore

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,       # CPU-friendly: avoid heavy multi-processing
        pin_memory=False)

    # ── run ─────────────────────────────────────────────────────────────────
    start = timeit.default_timer()

    if ('test' in config.DATASET.TEST_SET) and ('city' in config.DATASET.DATASET):
        test_cpu(config, test_dataset, testloader, model,
                 sv_dir=final_output_dir)
    else:
        mean_IoU, IoU_array, pixel_acc, mean_acc = testval_cpu(
            config, test_dataset, testloader, model)

        msg = 'MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f}, ' \
              'Mean_Acc: {: 4.4f}, Class IoU:'.format(
                  mean_IoU, pixel_acc, mean_acc)
        logging.info(msg)
        logging.info(IoU_array)

    end = timeit.default_timer()
    logger.info('Mins: %d' % int((end - start) / 60))
    logger.info('Done')


if __name__ == '__main__':
    main()
