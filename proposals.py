"Group multiple methods to generate salient temporal windows in a video"
import itertools

import numpy as np

PROPOSAL_SCHEMES = ['DidemoICCV17SS', 'SlidingWindowMSRSS']


class TemporalProposalsBase():
    "Base class (signature) to generate temporal candidate in video"

    def __call__(self, video_id, metadata=None, feature_collection=None):
        raise NotImplementedError('Implement with the signature above')


class DidemoICCV17SS(TemporalProposalsBase):
    """Original search space of moments proposed in ICCV-2017

    Attributes:
        clip_length_min (float) : minimum length, in seconds, of a video clip.
        proposals (numpy array) : of shape [21, 2] representing all the
            possible temporal segments of valid annotations of DiDeMo dataset.
            It represents the search space of a temporal localization
            algorithm.

    Reference: Hendricks et al. Localizing Moments in Video with Natural
        Language. ICCV 2017.
    """
    clip_length_min = 5.0

    def __init__(self, *args, dtype=np.float32, **kwargs):
        clips_indices = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
        for i in itertools.combinations(range(len(clips_indices)), 2):
            clips_indices.append(i)
        self.proposals = np.array(clips_indices, dtype=dtype)
        self.proposals *= self.clip_length_min
        self.proposals[:, 1] += self.clip_length_min

    def __call__(self, *args, **kwargs):
        return self.proposals


class SlidingWindowMSFS(TemporalProposalsBase):
    """Multi-scale (linear) sliding window with fixed stride

    TODO:
        - We are considering to deprecated this abstraction. Indeed, it's
          disabled from training.
        - documentation.
    """

    def __init__(self, length, num_scales, stride, unique=False,
                 dtype=np.float32):
        self.length = length
        self.num_scales = num_scales
        self.stride = stride
        self.unique = unique
        self.dtype = dtype
        self.canonical_windows = np.zeros((num_scales, 2), dtype=self.dtype)
        self.canonical_windows[:, 1] += (
            length * np.arange(1, num_scales + 1))

    def sliding_windows(self, t_end, t_start=0):
        "sliding canonical windows over a given time interval"
        t_zero = np.arange(t_start, t_end, self.stride, dtype=self.dtype)
        windows = (np.tile(self.canonical_windows, (len(t_zero), 1)) +
                   np.repeat(t_zero, len(self.canonical_windows))[:, None])
        # hacky way to make windows fit inside video
        # this means the lengths of the windows at the end are not in the set
        # spanned by length and num_scales
        windows[windows[:, 1] > t_end, 1] = t_end
        if self.unique:
            return np.unique(windows, axis=0)
        return windows

    def __call__(self, video_id, metadata=None, feature_collection=None):
        duration = metadata.get('duration')
        assert duration is not None
        return self.sliding_windows(duration)


class SlidingWindowMSRSS(TemporalProposalsBase):
    """Multi-scale sliding window with relative stride within the same scale

    Attributes:
        length (float) : length of smallest window.
        scales (sequence of int) : duration of moments relative to
            `lenght`.
        stride (float) : relative stride between two windows with the same
            duration. We used different strides for each scale rounding it
            towards a multiple of `length`. Note that the minimum stride is
            `length` for any window will be the `length` itself.
        dtype (numpy.dtype) : TODO

    TODO: documentation
    """

    def __init__(self, length, scales, stride=0.5, dtype=np.float32):
        self.length = length
        self.scales = scales
        self.relative_stride = stride
        # pick strides per scale that are multiples of length
        self.strides = [max(round(i * stride), 1) * length for i in scales]
        self.dtype = dtype
        assert len(scales) > 0

    def sliding_windows(self, t_end, t_start=0):
        "sliding canonical windows over a given time interval"
        windows_ = []
        for i, stride in enumerate(self.strides):
            num_i = np.ceil((t_end - t_start)/ stride)
            windows_i = np.empty((int(num_i), 2), dtype=np.float32)
            windows_i[:, 0] = np.arange(t_start, t_end, stride)
            windows_i[:, 1] = windows_i[:, 0] + self.length * self.scales[i]
            windows_i[windows_i[:, 1] > t_end, 1] = t_end
            windows_.append(windows_i)
        windows = np.concatenate(windows_, axis=0)
        # Hacky way to make windows fit inside video
        # It implies windows at the end may not belong to the set spanned by
        # length and scales.
        return np.unique(windows, axis=0)

    def __call__(self, video_id, metadata=None, feature_collection=None):
        duration = metadata.get('duration')
        assert duration is not None
        return self.sliding_windows(duration)


if __name__ == '__main__':
    test_fns_args = [(SlidingWindowMSFS, (3, 5, 3)),
                     (DidemoICCV17SS, (),),
                     (SlidingWindowMSRSS, (1.5, [2, 4, 6, 12]))]
    for fn_i, args_i in test_fns_args:
        proposal_fn = fn_i(*args_i)
        x = proposal_fn('hola', {'duration': 15})
        if fn_i == DidemoICCV17SS:
            assert len(x) == 21
