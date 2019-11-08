"""Frequency prior baseline for single video moment retrieval

Huge thanks to @ModarTensai for a helpful discussion that elucidated the
procedure for the KDE approach.

TODO:
    Implement sample with replacement, possibly applying NMS, in between.
    Motivation: sample with "diversity" instead of returning the sorted
    segments.
"""
import argparse
import json
import logging
from itertools import product
from pathlib import Path

import numpy as np
import torch
from scipy.stats import gaussian_kde

import dataset_untrimmed
import proposals
from evaluation import single_moment_retrieval, didemo_evaluation
from np_segments_ops import non_maxima_suppresion
from utils import setup_logging, setup_metrics
from utils import Multimeter

TOPK, IOU_THRESHOLDS = (1, 5), (0.5, 0.7)
TOPK_DIDEMO = torch.tensor([1, 5]).float()
METRICS = [f'r@{k},{iou}' for iou, k in product(IOU_THRESHOLDS, TOPK)]
METRICS_OLD = ['iou', 'r@1', 'r@5']


def main(args):
    "'train'&evaluate the moment frequency prior of the dataset"
    setup_logging(args)
    setup_metrics(args, TOPK, IOU_THRESHOLDS, TOPK_DIDEMO)
    logging.info('Moment frequency prior')
    logging.info(args)
    logging.info('Setting-up datasets')
    train_dataset, test_dataset = setup_dataset(args)
    moment_freq_prior = setup_model(args)

    logging.info('Estimating prior')
    for i, data in enumerate(train_dataset):
        gt_moments = data[-2]
        duration_i = video_duration(train_dataset, i)
        moment_freq_prior.update(gt_moments, duration_i)
    logging.info('Model fitting')
    moment_freq_prior.fit()

    logging.info(f'* Evaluation')
    meters, meters_old = Multimeter(keys=METRICS), None
    if args.proposal_interface == 'DidemoICCV17SS':
        meters_old = Multimeter(keys=METRICS_OLD)
        meters_old_ = Multimeter(keys=METRICS_OLD)
        # Details are provided in help
        args.nms_threshold = 1.0
    for i, data in enumerate(test_dataset):
        duration_i = video_duration(test_dataset, i)
        gt_moments = torch.from_numpy(data[-2])
        proposals_i = data[-1]
        prob = moment_freq_prior.predict(proposals_i, duration_i)

        if args.nms_threshold < 1:
            ind = non_maxima_suppresion(
                proposals_i, prob, args.nms_threshold)
        else:
            ind = prob.argsort()[::-1]

        sorted_proposals = proposals_i[ind, :]
        sorted_proposals = torch.from_numpy(sorted_proposals)
        hit_k_iou = single_moment_retrieval(
            gt_moments, sorted_proposals, topk=args.topk)

        meters.update([i.item() for i in hit_k_iou])
        if meters_old:
            iou_r_at_ks = didemo_evaluation(
                gt_moments, sorted_proposals, args.topk_)
            meters_old.update([i.item() for i in iou_r_at_ks])

    logging.info(f'{meters.report()}')
    if meters_old:
        logging.info(f'{meters_old.report()}')

    logging.info('Dumping model and parameters')
    dumping_arguments_and_model(args, moment_freq_prior, meters)


def dumping_arguments_and_model(args, model, meters):
    "Save model and serialized parameters into JSON"
    if len(args.logfile.name) == 0:
        return
    file_model = args.logfile.with_suffix('')
    model.save(file_model)

    file_args = args.logfile.with_suffix('.json')
    if hasattr(args, 'topk_'):
        delattr(args, 'topk_')
    args.topk = args.topk.tolist()
    args.logfile = str(args.logfile)
    args.train_list = str(args.train_list) if args.train_list.exists() else None
    args.test_list = str(args.test_list) if args.test_list.exists() else None
    args_dict = vars(args)
    for key, value in meters.dump().items():
        args_dict[key] = value
    with open(file_args, 'x') as fid:
        json.dump(args_dict, fid, skipkeys=True, indent=1, sort_keys=True)


def video_duration(dataset, moment_index):
    "Return duration of video of a given moment"
    video_id = dataset.metadata[moment_index]['video']
    return dataset.metadata_per_video[video_id]['duration']


def setup_dataset(args):
    "Setup dataset"
    proposals_interface = proposals.__dict__[args.proposal_interface](
        args.min_length, args.scales, args.stride)
    lastname = 'kde' if args.kde else 'discrete'
    cues = {f'mfp-{lastname}': None}
    subset_files = [('train', args.train_list), ('test', args.test_list)]
    datasets = []
    for i, (subset, filename) in enumerate(subset_files):
        datasets.append(
            dataset_untrimmed.UntrimmedMCN(
                filename, proposals_interface=proposals_interface,
                # we don't care about language or visual info, only the
                # location of moments, thus `debug` is fine. `eval` is also
                # fine as we don't need the sampling scheme. Finally
                # `clip-length` may not be needed, but we didn't have time to
                # check nor update the code, sorry for that.
                eval=True, debug=True, cues=cues, no_visual=True, loc=True,
                clip_length=args.clip_length)
        )
    return datasets


def setup_model(args):
    "Return MFP-model"
    if args.kde:
        model = KDEFrequencyPrior()
    else:
        if args.proposal_interface == 'DidemoICCV17SS':
            logging.info('Ignoring bins argument')
            args.ts_edges = np.arange(0, 31, 5.0) / 30
            args.te_edges = np.arange(5, 36, 5.0) / 30
        if args.ts_edges is not None:
            args.bins = [args.ts_edges, args.te_edges]
        model = DiscretizedFrequencyPrior(args.bins)
    return model


class BaseFrequencyPrior():
    "Compute the frequency prior of segments in dataset"

    def __init__(self):
        self.table = []
        self.model = None

    def load(self, filename):
        raise NotImplementedError('Subclass and implement')

    def update(self, gt_segments, duration):
        "Count how often a given segment appears"
        self.table.append(gt_segments / duration)

    def save(self, filename):
        raise NotImplementedError('Subclass and implement')


class DiscretizedFrequencyPrior(BaseFrequencyPrior):
    "Estimate frequency prior of segments in dataset via discretization"

    def __init__(self, bins):
        "Edges have priority over bins"
        super(DiscretizedFrequencyPrior, self).__init__()
        self._x_edges, self._y_edges = None, None
        self.bins = bins

    def fit(self):
        "Make 2D histogram"
        table_np = np.row_stack(self.table)
        self.model, self._x_edges, self._y_edges = np.histogram2d(
            table_np[:, 0], table_np[:, 1], bins=self.bins)

    def load(self, filename):
        data = np.load(filename)
        self.model = data['model']
        self._x_edges = data['x_edges']
        self._y_edges = data['y_edges']

    def predict(self, pred_segments=None, duration=None):
        "Return prob that a proposal belongs to the dataset"
        assert self.model is not None
        normalized_proposals = pred_segments / duration
        ind_x = np.digitize(normalized_proposals[:, 0], self._x_edges, True)
        ind_y = np.digitize(normalized_proposals[:, 1], self._y_edges, True)
        ind_x = np.clip(ind_x, 0, self.model.shape[0] - 1)
        ind_y = np.clip(ind_y, 0, self.model.shape[1] - 1)
        return self.model[ind_x, ind_y]

    def save(self, filename):
        np.savez(filename, model=self.model, x_edges=self._x_edges,
                 y_edges=self._y_edges)


class KDEFrequencyPrior(BaseFrequencyPrior):
    "Estimate frequency prior of segments in dataset via KDE"

    def fit(self):
        "Fit PDF with KDE and default bandwidth selection rule"
        all_segments = np.row_stack(self.table).T
        self.model = gaussian_kde(all_segments)

    def predict(self, pred_segments=None, duration=None):
        "Return prob that a proposal belongs to the dataset"
        assert self.model is not None
        normalized_proposal = (pred_segments / duration).T
        return self.model.pdf(normalized_proposal)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MFP for single video moment retrieval')
    parser.add_argument(
        '--train-list', type=Path, required=True,
        help='JSON-file with training instances')
    parser.add_argument(
        '--test-list', type=Path, required=True,
        help='JSON-file with training instances')
    parser.add_argument(
        '--clip-length', type=float, required=True,
        help='Clip length in seconds')
    # Freq prior parameters
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--kde', action='store_true',
        help='Perform continous analysis')
    group.add_argument(
        '--bins', type=int,
        help=('Number of bins for discretization. Please provide something '
              'for DiDeMo, but would be ignored in favor of the ICCV-2017 '
              ' bin edges.'))
    group_edges = group.add_argument_group(
        description='Bins for discretization')
    group_edges.add_argument(
        '--ts-edges', type=float, nargs='+',
        help='Bin edges for t-start')
    group_edges.add_argument(
        '--te-edges', type=float, nargs='+',
        help='Bin edges for t-end')
    # Hyper-parameters to explore search space (inference)
    parser.add_argument(
        '--proposal-interface', default='SlidingWindowMSFS',
        choices=proposals.PROPOSAL_SCHEMES,
        help='Type of proposals spanning search space')
    parser.add_argument(
        '--min-length', type=float, default=3,
        help='Minimum length of slidding windows (seconds)')
    parser.add_argument(
        '--scales', type=int, default=None, nargs='+',
        help='Number of scales in a multi-scale linear slidding window')
    parser.add_argument(
        '--stride', type=float, default=0.3,
        help=('Relative stride for sliding windows [0, 1]. Check'
              'SlidingWindowMSRSS details'))
    parser.add_argument(
        '--nms-threshold', type=float, default=0.6,
        help=('Threshold used to remove overlapped predictions. We use 1.0'
              'in DiDeMo for fair comparsion and because the evaluation code '
              'also assumes that.'))
    # Logging
    parser.add_argument(
        '--logfile', type=Path, default='', help='Logging file')
    parser.add_argument(
        '--enable-tb', action='store_true', help='Log to tensorboard')

    args = parser.parse_args()
    main(args)
