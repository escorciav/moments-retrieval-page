"""Quick&Dirty approach to get Two-Stage retrieval system

!!! This program will not run !!!
We are providing it to showcase the evaluation protocol
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

import corpus
import dataset_untrimmed
import model
import proposals
from evaluation import CorpusVideoMomentRetrievalEval
from utils import setup_logging, get_git_revision_hash

NMS_THRESHOLD = 1.0
parser = argparse.ArgumentParser(
    description='Corpus Retrieval 2nd Stage Evaluation',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Data
parser.add_argument('--test-list', type=Path, required=True,
                    help='JSON-file with corpus instances')
parser.add_argument('--h5-path', type=Path, default='non-existent',
                    help='HDF5-file with features')
# Architecture
parser.add_argument('--corpus-setup',
                    choices=['LoopOverKMoments',
                            #  'TwoStageClipPlusGeneric',
                            #  'TwoStageClipPlusCAL'],
                            ]
                    default='LoopOverKMoments',
                    help='Kind of two-stage retrieval approach')
parser.add_argument('--snapshot', type=Path, required=True, nargs='+',
                    help=('JSON-file of model. Only pass two models for '
                          'LateFusion of Chamfer and only-TEF.'))
parser.add_argument('--h5-1ststage', type=Path,
                    help='HDF5-file of 1st stage results')
parser.add_argument('--snapshot-1ststage', type=Path,
                    help='JSON-file of model')
parser.add_argument('--k-first', type=int, required=True,
                    help='K first retrieved resuslts')
# Evaluation parameters
parser.add_argument('--topk', nargs='+', type=int,
                    default=[1, 10, 100],
                    help='top-k values to compute in ascending order.')
# Dump results and logs
parser.add_argument('--dump', action='store_true',
                    help='Save log in text file and json')
parser.add_argument('--logfile', type=Path, default='',
                    help='Logging file')
parser.add_argument('--output-prefix', type=Path, default='',
                    help="")
parser.add_argument('--n-display', type=float, default=0.2,
                    help='logging rate during epoch')
parser.add_argument('--disable-tqdm', action='store_true',
                    help='Disable progress-bar')
parser.add_argument('--enable-tb', action='store_true',
                    help='Log to tensorboard. Nothing logged by this program')
# Debug
parser.add_argument('--debug', action='store_true',
                    help=('yield incorrect results! to verify things are'
                          'glued correctly (dataset, model, eval)'))
args.nms_threshold = NMS_THRESHOLD
args = parser.parse_args()


def main(args):
    "Put all the pieces together"
    if args.dump:
        args.disable_tqdm = True
        if len(args.logfile.name) == 0:
            basename = args.snapshot[0].with_suffix('')
            args.logfile = basename.parent.joinpath(
                args.output_prefix, basename.stem + '_corpus-2nd-eval')
            if not args.logfile.parent.exists():
                args.logfile.parent.mkdir()
        if args.logfile.exists():
            raise ValueError(
                f'{args.logfile} already exists. Please provide a logfile or'
                'backup existing results.')
    setup_logging(args)

    logging.info('Corpus Retrieval Evaluation for 2nd Stage')
    load_hyperparameters(args)
    logging.info(args)

    if args.arch == 'MCN':
        args.dataset = 'UntrimmedMCN'
    elif args.arch == 'SMCN':
        args.dataset = 'UntrimmedSMCN'
    else:
        ValueError('Unknown/unsupported architecture')

    logging.info('Loading dataset')
    if args.h5_path.exists():
        dataset_novisual = False
        dataset_cues = {args.feat: {'file': args.h5_path}}
    else:
        raise NotImplementedError('WIP')
    proposals_interface = proposals.__dict__[args.proposal_interface](
        args.min_length, args.scales, args.stride)
    dataset_setup = dict(
        json_file=args.test_list, cues=dataset_cues, loc=args.loc,
        context=args.context, debug=args.debug, eval=True,
        no_visual=dataset_novisual,
        proposals_interface=proposals_interface
    )
    dataset = dataset_untrimmed.__dict__[args.dataset](**dataset_setup)
    logging.info('Setting up models')
    arch_setup = dict(
        visual_size=dataset.visual_size[args.feat],
        lang_size=dataset.language_size,
        max_length=dataset.max_words,
        embedding_size=args.embedding_size,
        visual_hidden=args.visual_hidden,
        lang_hidden=args.lang_hidden,
        visual_layers=args.visual_layers,
        bi_lstm=args.bi_lstm,
        lang_dropout=args.lang_dropout
    )

    net = model.__dict__[args.arch](**arch_setup)
    model_param = setup_snapshot(args.snapshot)
    net.load_state_dict(model_param['state_dict'])
    net.eval()

    logging.info('Setting up engine')
    engine = setup_engine(args, dataset, net)

    logging.info('Launch evaluation...')
    # log-scale up to the end of the database
    if len(args.topk) == 1 and args.topk[0] == 0:
        exp = int(np.floor(np.log10(engine.num_moments)))
        args.topk = [10**i for i in range(0, exp + 1)]
        args.topk.append(engine.num_moments)
    num_instances_retrieved = []
    judge = CorpusVideoMomentRetrievalEval(topk=args.topk)
    args.n_display = max(int(args.n_display * len(dataset.metadata)), 1)
    for it, query_metadata in tqdm(enumerate(dataset.metadata),
                                   disable=args.disable_tqdm):
        vid_indices, segments = engine.query(
            query_metadata['language_input'], description_ind=it)
        judge.add_single_predicted_moment_info(
            query_metadata, vid_indices, segments, max_rank=engine.num_moments)
        num_instances_retrieved.append(len(vid_indices))
        if args.disable_tqdm and (it + 1) % args.n_display == 0:
            logging.info(f'Processed queries [{it}/{len(dataset.metadata)}]')

    logging.info('Summarizing results')
    num_instances_retrieved = np.array(num_instances_retrieved)
    logging.info(f'Number of queries: {len(judge.map_query)}')
    logging.info(f'Number of proposals: {engine.num_moments}')
    retrieved_proposals_median = int(np.median(num_instances_retrieved))
    retrieved_proposals_min = int(num_instances_retrieved.min())
    if (num_instances_retrieved != engine.num_moments).any():
        logging.info('Triggered approximate search')
        logging.info('Median numbers of retrieved proposals: '
                     f'{retrieved_proposals_median:d}')
        logging.info('Min numbers of retrieved proposals: '
                     f'{retrieved_proposals_min:d}')
    result = judge.evaluate()
    _ = [logging.info(f'{k}: {v}') for k, v in result.items()]
    if args.dump:
        filename = args.logfile.with_suffix('.json')
        logging.info(f'Dumping results into: {filename}')
        with open(filename, 'x') as fid:
            for key, value in result.items():
                result[key] = float(value)
            result['snapshot'] = [str(i) for i in args.snapshot]
            result['corpus'] = str(args.test_list)
            result['h5_path'] = str(args.h5_path)
            result['h5_1ststage'] = str(args.h5_1ststage)
            result['snapshot_1ststage'] = str(args.snapshot_1ststage)
            result['topk'] = args.topk
            result['iou_threshold'] = judge.iou_thresholds
            result['k_first'] = args.k_first
            result['median_proposals_retrieved'] = retrieved_proposals_median
            result['min_proposals_retrieved'] = retrieved_proposals_min
            result['nms_threshold'] = args.nms_threshold
            result['corpus_setup'] = args.corpus_setup
            result['date'] = datetime.now().isoformat()
            result['git_hash'] = get_git_revision_hash()
            json.dump(result, fid, indent=1, sort_keys=True)


def load_hyperparameters(args):
    "Update args with model hyperparameters"
    logging.info('Parsing JSON files with hyper-parameters')
    with open(args.snapshot[0], 'r') as fid:
        hyper_prm = json.load(fid)
        # TODO: clean this hacks by updating all the JSON (after deadline)
        if 'bi_lstm' not in hyper_prm:
            hyper_prm['bi_lstm'] = False
        if 'lang_dropout' not in hyper_prm:
            hyper_prm['lang_dropout'] = 0.0
        if hyper_prm['arch'] == 'ModelD':
            hyper_prm['arch'] = 'CALChamfer'

    for key, value in hyper_prm.items():
        if not hasattr(args, key):
            setattr(args, key, value)
        else:
            logging.debug(f'Ignored hyperparam: {key}')

    # TODO: is there a clean way to do this?
    if len(args.snapshot) == 2:
        args.arch = 'LateFusion'


def setup_engine(args, dataset, net):
    "Setup engine and deal with yet another dataset setup if any"
    engine_prm = {}
    if args.snapshot_1ststage is not None:
        with open(args.snapshot_1ststage, 'r') as fid:
            hyper_prm_1ststage = json.load(fid)
        # We use the same visual representation among stages to not complicate
        # things, but it's not a requirement
        dataset_cues = {
            hyper_prm_1ststage['feat']: {'file': args.h5_1ststage}
        }
        dataset_setup = dict(
            loc=hyper_prm_1ststage['loc'],
            context=hyper_prm_1ststage['context'],
            cues=dataset_cues,
            debug=args.debug,
            json_file=args.test_list,
            proposals_interface=dataset.proposals_interface,
            eval=True
        )
        # Hard-code UntrimmedSMCN and SMCN as they are the only clip-based
        # representation so far :)
        dataset_1ststage = dataset_untrimmed.UntrimmedSMCN(**dataset_setup)
        dataset_1ststage.set_padding(False)

        model_setup = dict(
            visual_size=dataset_1ststage.visual_size[args.feat],
            lang_size=dataset_1ststage.language_size,
            max_length=dataset_1ststage.max_words,
            embedding_size=hyper_prm_1ststage['embedding_size'],
            visual_hidden=hyper_prm_1ststage['visual_hidden'],
            lang_hidden=hyper_prm_1ststage['lang_hidden'],
            visual_layers=hyper_prm_1ststage['visual_layers'],
        )
        model_1ststage = model.SMCN(**model_setup)
        snapshot = torch.load(args.snapshot_1ststage.with_suffix('.pth.tar'),
                              map_location=lambda storage, loc: storage)
        model_1ststage.load_state_dict(snapshot['state_dict'])
        model_1ststage.eval()
        engine_prm['dataset_1stage'] = dataset_1ststage
        engine_prm['model_1ststage'] = [model_1ststage]
    else:
        engine_prm['h5_1ststage'] = args.h5_1ststage

    if args.corpus_setup == 'LoopOverKVideos':
        engine_prm['nms_threshold'] = args.nms_threshold
    engine_prm['topk'] = args.k_first

    engine = corpus.__dict__[args.corpus_setup](dataset, net, **engine_prm)
    return engine


def setup_snapshot(filenames):
    # Trick to setups snapshot when multiple are provided
    assert len(filenames) <= 2
    filename_1 = filenames[0].with_suffix('.pth.tar')
    snapshot_1 = torch.load(filename_1,
                          map_location=lambda storage, loc: storage)
    if len(filenames) == 1:
        return snapshot_1
    assert args.arch == 'LateFusion'
    snapshots = {'state_dict': [snapshot_1]}
    filename_2 = filenames[-1].with_suffix('.pth.tar')
    snapshot_2 = torch.load(filename_2,
                            map_location=lambda storage, loc: storage)
    snapshots['state_dict'].append(snapshot_2)

    return snapshots


if __name__ == '__main__':
    main(args)
