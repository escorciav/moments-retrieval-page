import glob
import json
import logging
import random
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from tensorboardX import SummaryWriter
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataloader import default_collate

PATH_VARS = {'h5_path_nis', 'test_list', 'val_list', 'train_list'}


def load_args_from_snapshot(args, path_vars=PATH_VARS):
    "Update arguments with those from snapshot JSON"
    if not args.snapshot.exists():
        return False
    elif len(args.snapshot.name) == 0:
        return True
    # Protect the lede
    snapshot, logfile = args.snapshot, args.logfile
    with open(args.snapshot, 'r') as fid:
        hyper_prm = json.load(fid)
    for key, value in hyper_prm.items():
        # PR welcome if you know a more elegant way to do this
        if value and key in path_vars:
            value = Path(value)
        setattr(args, key, value)
    args.snapshot, args.logfile = snapshot, logfile
    return True


def setup_hyperparameters(args):
    "Update Namescope with random hyper-parameters according to a YAML-file"
    if not args.hps:
        return
    filename = args.logfile.parent / 'hps.yml'
    if not filename.exists():
        logging.error(f'Ignoring HPS. Not found {filename}')
        return
    with open(filename, 'r') as fid:
        config = yaml.load(fid)
    logging.info('Proceeding to perform random HPS')
    args_dview = vars(args)

    # Random search over single parameter of tied variables
    slack_tied = {'w_intra': 'w_inter',
                  'c_intra': 'c_inter'}
    for slack, tied in slack_tied.items():
        if tied in config:
            if isinstance(config.get(slack), list):
                logging.warning(f'Ignoring {tied}')
                del config[tied]

    for k, v in config.items():
        if not isinstance(v, list):
            args_dview[k] = v
            continue
        random.shuffle(v)
        args_dview[k] = v[0]
        if k == slack_tied:
            args_dview[slack_tied[k]] = 1 - v[0]

    # Note: only available in YAML
    if args.clip_loss and args.only_clip_loss:
        args_dview['w_intra'] = 0.0
        args_dview['w_inter'] = 0.0


def setup_logging(args):
    "Setup logging to dump progress into file or print it"
    log_prm = dict(format='%(asctime)s:%(levelname)s:%(message)s',
                   level=logging.DEBUG)
    if len(args.logfile.name) >= 1:
        log_prm['filename'] = args.logfile.with_suffix('.log')
        log_prm['filemode'] = 'x'
    logging.basicConfig(**log_prm)
    args.writer = None
    if args.enable_tb:
        # This should be a module variable in case we don't want tensorboard
        args.writer = SummaryWriter(args.logfile.with_suffix(''))


def setup_metrics(args, topks, iou_thresholds, topks_didemo):
    "Update args with metrics"
    args.topk = torch.tensor(topks)
    args.iou_thresholds = iou_thresholds
    args.topk_ = topks_didemo


def setup_rng(args):
    "Init random number generators from seed in Namespace"
    if args.seed < 1:
        args.seed = random.randint(0, 2**16)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def save_checkpoint(args, state, record=False):
    "Serialize model into pth"
    if len(args.logfile.name) == 0 or not args.serialize:
        return
    filename = args.logfile.with_suffix('.pth.tar')
    if record:
        epoch = args.epochs
        name = args.logfile.stem
        filename = args.logfile.with_name(f'{name}-{epoch}.pth.tar')
    torch.save(state, filename)


def ship_to(x, device):
    # TODO: clean like default_collate :S
    y = []
    for i in x:
        if isinstance(i, dict):
            y.append({k: v.to(device) for k, v in i.items()})
        elif isinstance(i, torch.Tensor):
            y.append(i.to(device))
        else:
            y.append(i)
    return y


def unique2d_perserve_order(x):
    """Return unique along rows in x

    Note: It assumes the same range of numbers for each row in x
    """
    assert x.ndim == 2
    y = []
    for i in range(len(x)):
        _, ind = np.unique(x[i, :], return_index=True)
        y.append(x[i, np.sort(ind)])
    return np.row_stack(y)


class AverageMeter(object):
    """Computes and stores the average and current value

    Credits: pytorch-imagenet-example
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Tracker(object):
    "Keep track of torch tensors or numpy scalar things"

    def __init__(self, keys):
        self.data = {i: [] for i in keys}

    def append(self, *args):
        "Add values to track"
        for i, key in enumerate(self.data):
            self.data[key].append(args[i])

    def freeze(self, cpu=True):
        "Make everything a tensor and move to cpu"
        for key, value in self.data.items():
            if not isinstance(value[0], torch.Tensor):
                self.data[key] = torch.tensor(value)
                continue
            elif value[0].dim() > 1 or value[0].shape[0] > 1:
                self.data[key] = torch.stack(value)
            else:
                self.data[key] = torch.cat(value)

            if cpu:
                self.data[key] = self.data[key].to('cpu')
            del (value)


class Multimeter(object):
    "Keep multiple AverageMeter"

    def __init__(self, keys=None):
        self.metrics = keys
        self.meters = [AverageMeter() for i in keys]

    def reset(self):
        for i, _ in enumerate(self.metrics):
            self.meters.reset()

    def update(self, vals, n=1):
        assert len(vals) == len(self.metrics)
        for i, v in enumerate(self.meters):
            v.update(vals[i], n)

    def report(self):
        msg = ''
        for i, v in enumerate(self.metrics):
            msg += f'{v}: {self.meters[i].avg:.4f}\t'
        return msg[:-1]

    def dump(self):
        return {v: self.meters[i].avg for i, v in enumerate(self.metrics)}


def get_git_revision_hash():
    "credits: https://stackoverflow.com/a/21901260"
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                   universal_newlines=True).strip()


def jsons_to_dataframe(wilcard):
    "Read multiple json files and stack them into a DataFrame"
    data = []
    for filename in glob.glob(wilcard):
        with open(filename) as f:
            data.append(json.load(f))
    df = pd.DataFrame(data)
    return df


if __name__ == '__main__':
    aja = Multimeter(['hi', 'vi', 'tor'])
    aja.update([1, 2, 3])
    aja.update([3, 2, 1])
    print(f'{aja.report()}')
