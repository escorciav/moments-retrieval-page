"""Operations for [N, 2] numpy arrays or torch tensors representing segments.
Example segment operations that are supported:
  * length: compute bounding box areas
  * IOU: pairwise intersection-over-union scores
  * intersection: pairwise intersection-over-union scores

TODO (refactor): rename module to segments_ops
"""
import numpy as np
import torch


def intersection(segments1, segments2):
    """Compute pairwise intersection length between segments.

    Args:
        segments1 (numpy array): shape [N, 2] holding N segments
        segments2 (numpy array): shape [M, 2] holding M segments
    Returns:
        a numpy array with shape [N, M] representing pairwise intersection length
    """
    [t_min1, t_max1] = np.split(segments1, 2, axis=1)
    [t_min2, t_max2] = np.split(segments2, 2, axis=1)

    all_pairs_min_tmax = np.minimum(t_max1, np.transpose(t_max2))
    all_pairs_max_tmin = np.maximum(t_min1, np.transpose(t_min2))
    intersect_length = np.maximum(np.zeros(all_pairs_max_tmin.shape),
                                  all_pairs_min_tmax - all_pairs_max_tmin)
    return intersect_length


def length(segments):
    """Computes length of segments.

    Args:
        segments (numpy array or torch tensor): shape [N, 2] holding N
            segments.
    Returns:
        a numpy array (or torch tensor) with shape [N] representing segment
        length.
    Note:
        it works with time, it would be off if using frames.
    """
    return segments[:, 1] - segments[:, 0]


def iou(segments1, segments2):
    """Computes pairwise intersection-over-union between segment collections.

    Args:
        segments1 (numpy array): shape [N, 2] holding N segments
        segments2 (numpy array): shape [M, 4] holding N boxes.
    Returns:
        a numpy array with shape [N, M] representing pairwise iou scores.
    """
    intersect = intersection(segments1, segments2)
    length1 = length(segments1)
    length2 = length(segments2)
    union = np.expand_dims(length1, axis=1) + np.expand_dims(
        length2, axis=0) - intersect
    return intersect / union


def non_maxima_suppresion(segments, scores, nms_threshold):
    """non-maxima suppresion over segments

    Args:
        segments (numpy array): shape [N, 2] holding N segments
        scores (numpy array): shape [N] holding score of each segment.
    Returns:
        a numpy array with shape [M] representing indexes to pick after nms.
    """
    t1, t2 = np.split(segments, 2, axis=1)
    area = t2 - t1
    idx = np.argsort(scores)
    ind_pick = []
    for i in range(len(idx)):
        if len(idx) == 0:
            break
        p = idx[len(idx) - 1]
        ind_pick.append(p)

        tt1 = np.maximum(t1[p], t1[idx])
        tt2 = np.minimum(t2[p], t2[idx])
        wh = np.maximum(0, tt2 - tt1)
        o = wh / (area[p] + area[idx] - wh)

        ind_rm_i = np.where(o >= nms_threshold)[0]
        idx = np.delete(idx, ind_rm_i)
    ind_pick = np.array(ind_pick)
    return ind_pick


def torch_intersection(segments1, segments2):
    """Compute pairwise intersection length between segments.

    Args:
        segments1 (torch tensor): shape [N, 2] holding N segments
        segments2 (torch tensor): shape [M, 2] holding M segments
    Returns:
        a torch tensor with shape [N, M] representing pairwise intersection length
    """
    [t_min1, t_max1] = torch.chunk(segments1, 2, dim=1)
    [t_min2, t_max2] = torch.chunk(segments2, 2, dim=1)
    # t_max1 and t_max2 are not contigous, does it matter?

    # it seems that t_*_[1-2] share memory with segments[1-2], thus I don't
    # transpose in-place
    all_pairs_min_tmax = torch.min(t_max1, t_max2.transpose(0, 1))
    all_pairs_max_tmin = torch.max(t_min1, t_min2.transpose(0, 1))
    intersect_length = torch.max(torch.zeros_like(all_pairs_max_tmin),
                                 all_pairs_min_tmax - all_pairs_max_tmin)
    return intersect_length


def torch_iou(segments1, segments2):
    """Computes pairwise intersection-over-union between segment collections.

    Args:
        segments1 (torch tensor): shape [N, 2] holding N segments
        segments2 (torch tensor): shape [M, 2] holding N boxes.
    Returns:
        a torch tensor with shape [N, M] representing pairwise iou scores.
    """
    intersect = torch_intersection(segments1, segments2)
    length1 = length(segments1)
    length2 = length(segments2)
    # print(length1.shape, length2.shape, intersect)
    union = length1.unsqueeze_(1) + length2.unsqueeze_(0) - intersect
    return intersect / union


if __name__ == '__main__':
    # kinda unit-test
    def random_segments(n):
        x_ = np.random.rand(n, 2).astype(np.float32)
        x = np.empty_like(x_)
        x[:, 0] = np.min(x_, axis=1)
        x[:, 1] = np.max(x_, axis=1)
        return x

    N, M = 1113, 2367
    a = random_segments(N)
    b = random_segments(M)
    a_torch = torch.from_numpy(a)
    b_torch = torch.from_numpy(b)
    length(a)
    length(a_torch)
    MAYBE_NUMERIC_FNS = {torch_iou}
    for functions_i in [(intersection, torch_intersection), (iou, torch_iou)]:
        numpy_fn, torch_fn = functions_i
        gt = numpy_fn(a, b)
        # torch
        for cuda in [False, True]:
            if cuda:
                a_torch = a_torch.cuda()
                b_torch = b_torch.cuda()
            else:
                a_torch = a_torch.cpu()
                b_torch = b_torch.cpu()

            rst = torch_fn(a_torch, b_torch)
            if cuda:
                rst = rst.cpu()

            testing_fn = np.testing.assert_array_equal
            if torch_fn in MAYBE_NUMERIC_FNS:
                testing_fn = np.testing.assert_array_almost_equal
            testing_fn(rst.numpy(), gt)

    scores = np.random.rand(N)
    nms_threshold = 0.75
    non_maxima_suppresion(a, scores, nms_threshold)
