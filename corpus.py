import hashlib
import itertools
import json
import math

import h5py
import numpy as np
import torch

from np_segments_ops import non_maxima_suppresion
from utils import unique2d_perserve_order


class LoopOverKBase():
    "TODO: description"

    def __init__(self, dataset, model, h5_1ststage, topk=100,
                 nms_threshold=1.0):
        self.dataset = dataset
        self.model = model
        self.h5_file = h5_1ststage
        self.topk = topk
        self.nms_threshold = nms_threshold
        self.proposals = None  # torch 2D-tensor
        self.query2videos_ind = None  # numpy  2D-array
        self.query2videos_ind_per_proposal = None  # torch 2D-tensor
        self.query2proposals_ind = None  # torch 2D-tensor
        self._setup()

    @property
    def num_moments(self):
        return self.proposals.shape[0]

    def preprocess_description(self, description):
        "Return tensors representing description as 1) vectors and 2) length"
        # TODO (refactor): duplicate snippet from
        # CorpusVideoMomentRetrievalBase. Factor it out as function or apply
        # inheritance.

        # TODO (release): allow to tokenize description
        assert isinstance(description, list)
        lang_feature_, len_query_ = self.dataset._compute_language_feature(
            description)
        # torchify
        lang_feature = torch.from_numpy(lang_feature_)
        lang_feature.unsqueeze_(0)
        len_query = torch.tensor([len_query_])
        return lang_feature, len_query

    def query(self, description, description_ind):
        raise NotImplementedError('Subclass and implement')

    def _setup(self):
        "Misc stuff like load results from 1st retrieval stage"
        with h5py.File(self.h5_file, 'r') as fid:
            query2videos_ind = fid['vid_indices'][:]
            # Force us to examine a way to deal with approximate retrieval
            # approaches
            assert query2videos_ind.shape[1] >= self.dataset.num_videos
            assert (query2videos_ind >= 0).all()
            # Trigger post-processing in case we are dealing with retrieval
            # results from a moment-based approach
            if query2videos_ind.shape[1] > self.dataset.num_videos:
                self.query2videos_ind_per_proposal = torch.from_numpy(
                    query2videos_ind)
                query2videos_ind = unique2d_perserve_order(query2videos_ind)
            self.query2videos_ind = query2videos_ind

            # Note: self.proposals may be redudant and we could create a table
            # to save storage in practice
            if 'proposals' in fid:
                self.proposals = torch.from_numpy(fid['proposals'][:])
            else:
                proposals = []
                for video_ind in range(self.dataset.num_videos):
                    _, proposals_i = self.dataset.video_item(video_ind)
                    proposals.append(proposals_i)
                self.proposals = torch.from_numpy(
                    np.concatenate(proposals, axis=0))

            if 'proposals_ind' in fid:
                self.query2proposals_ind = fid['proposals_ind'][:]


class LoopOverKMoments(LoopOverKBase):
    """Re-rank topk moments

    For text-to-video retrieval algorithms, we evaluate enough videos such
    that the number of retrieved moments is bounded.

    TODO: description
    """

    def __init__(self, *args, **kwargs):
        self.moment_based_reranking = False
        super(LoopOverKMoments, self).__init__(*args, **kwargs)

    def query(self, description, description_ind):
        "Return videos and moments aligned with a text description"
        # TODO (tier-2): remove 2nd-stage results from 1st-stage to make them
        # exhaustive
        torch.set_grad_enabled(False)
        lang_feature, len_query = self.preprocess_description(description)

        video_ind_1ststage = self.query2videos_ind[description_ind, :]
        # Sorry for this dirty trick
        video_indices, proposals, scores = [], [], []
        if self.moment_based_reranking:
            proposals_ind = self.query2proposals_ind[
                description_ind, :self.topk]
            video_indices = self.query2videos_ind_per_proposal[
                description_ind, :self.topk]
            proposals = self.proposals[proposals_ind, :]

        proposals_counter = 0
        for i in range(self.topk):
            # branch according to 1st-stage
            if self.moment_based_reranking:
                video_id = self.dataset.videos[video_indices[i]]
                # There is only a single candidate in this case
                candidates_i_feat = self.dataset._compute_visual_feature(
                    video_id, proposals[i, :].numpy())
                for k, v in candidates_i_feat.items():
                    if isinstance(v, np.ndarray):
                        candidates_i_feat[k] = v[None, :]
                proposals_i = proposals[i, :].unsqueeze_(dim=0)
                proposals_counter += 1
            else:
                video_ind = int(video_ind_1ststage[i])
                candidates_i_feat, proposals_i = self.dataset.video_item(
                    video_ind)
                video_ind_i = video_ind * torch.ones(
                    len(proposals_i), dtype=torch.int32)
                proposals_counter += len(proposals_i)

            # torchify
            candidates_i_feat = {k: torch.from_numpy(v)
                                 for k, v in candidates_i_feat.items()}
            if isinstance(proposals_i, np.ndarray):
                proposals_i = torch.from_numpy(proposals_i)

            scores_i, descending_i = self.model.predict(
                lang_feature, len_query, candidates_i_feat)

            # TODO: add post-processing such as NMS
            if self.nms_threshold < 1:
                idx = non_maxima_suppresion(
                        proposals_i.numpy(), -scores_i.numpy(),
                        self.nms_threshold)
                proposals_i = proposals_i[idx, :]
                scores_i = scores_i[idx]

            scores.append(scores_i)
            if isinstance(proposals, list):
                proposals.append(proposals_i)
                video_indices.append(video_ind_i)

            if proposals_counter >= self.topk:
                break

        # Part of the dirty trick
        if isinstance(proposals, list):
            proposals = torch.cat(proposals, dim=0)
            video_indices = torch.cat(video_indices)
        scores = torch.cat(scores)
        scores, ind = scores.sort(descending=descending_i)
        return video_indices[ind], proposals[ind, :]

    def _setup(self):
        super(LoopOverKMoments, self)._setup()
        if self.query2videos_ind_per_proposal is not None:
            self.moment_based_reranking = True
