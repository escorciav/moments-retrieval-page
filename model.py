"""Moment based retrieval models

These models correspond to the aggregation mechanism computing the distance
between a visual features from temporal proposals, and a tokenized text query.
The video encoder is not included in these models.

Note: Be careful with the inheritance pattern used here. A refactoring would
come handy to avoid non-trivial errors.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence

MOMENT_RETRIEVAL_MODELS = ['MCN', 'SMCN']


class MCN(nn.Module):
    """MCN model
    """

    def __init__(self, visual_size=4096, lang_size=300, embedding_size=100,
                 dropout=0.3, max_length=None, visual_hidden=500,
                 lang_hidden=1000, visual_layers=1, unit_vector=False,
                 bi_lstm=False, lang_dropout=0.0):
        super(MCN, self).__init__()
        self.embedding_size = embedding_size
        self.max_length = max_length
        self.unit_vector = unit_vector

        self.lang_hidden = lang_hidden
        self.lang_size = lang_size
        self.bi_lstm = bi_lstm
        self.lang_dropout = lang_dropout
        bi_norm = 1
        if self.bi_lstm:
            bi_norm = 2

        visual_encoder = [nn.Linear(visual_size, visual_hidden),
                          nn.ReLU(inplace=True)]
        # (optional) add depth to visual encoder (capacity)
        for i in  range(visual_layers - 1):
            visual_encoder += [nn.Linear(visual_hidden, visual_hidden),
                               nn.ReLU(inplace=True)]
        self.visual_encoder = nn.Sequential(
            *visual_encoder,
            nn.Linear(visual_hidden, embedding_size),
            nn.Dropout(dropout)
        )
        # self.visual_embedding = nn.Sequential(
        #     nn.Linear(visual_hidden, embedding_size),
        #     nn.Dropout(dropout)
        # )

        self.sentence_encoder = nn.LSTM(self.lang_size, self.lang_hidden,
                                batch_first=True, bidirectional=self.bi_lstm)
        self.lang_encoder = nn.Linear(
            bi_norm * self.lang_hidden, self.embedding_size)
        self.init_parameters()

    def forward(self, padded_query, query_length, visual_pos,
                visual_neg_intra=None, visual_neg_inter=None):
        # v_v_embedded_* are tensors of shape [B, D]
        (v_embedded_pos, v_embedded_neg_intra,
         v_embedded_neg_inter) = self.encode_visual(
             visual_pos, visual_neg_intra, visual_neg_inter)

        l_embedded = self.encode_query(padded_query, query_length)
        c_pos = self.compare_emdeddings(l_embedded, v_embedded_pos)
        c_neg_intra, c_neg_inter = None, None
        if v_embedded_neg_intra is not None:
            c_neg_intra = self.compare_emdeddings(
                l_embedded, v_embedded_neg_intra)
        if v_embedded_neg_inter is not None:
            c_neg_inter = self.compare_emdeddings(
                l_embedded, v_embedded_neg_inter)
        return c_pos, c_neg_intra, c_neg_inter

    def encode_visual(self, pos, neg_intra, neg_inter):
        pos, neg_intra, neg_inter = self._unpack_visual(
            pos, neg_intra, neg_inter)
        embedded_neg_intra, embedded_neg_inter = None, None

        embedded_pos = self.visual_encoder(pos)
        # pos_encoded = self.visual_encoder(pos)
        # embedded_pos = self.visual_embedding(pos_encoded)
        if self.unit_vector:
            embedded_pos = F.normalize(embedded_pos, dim=-1)
        if neg_intra is not None:
            embedded_neg_intra = self.visual_encoder(neg_intra)
            # neg_intra_encoded = self.visual_encoder(neg_intra)
            # embedded_neg_intra = self.visual_embedding(neg_intra_encoded)
            if self.unit_vector:
                embedded_neg_intra = F.normalize(embedded_neg_intra, dim=-1)
        if neg_inter is not None:
            embedded_neg_inter = self.visual_encoder(neg_inter)
            # neg_inter_encoded = self.visual_encoder(neg_inter)
            # embedded_neg_inter = self.visual_embedding(neg_inter_encoded)
            if self.unit_vector:
                embedded_neg_inter = F.normalize(embedded_neg_inter, dim=-1)

        return embedded_pos, embedded_neg_intra, embedded_neg_inter

    def encode_query(self, padded_query, query_length):
        B = len(padded_query)
        packed_query = pack_padded_sequence(
            padded_query, query_length, batch_first=True)
        packed_output, _ = self.sentence_encoder(packed_query)
        output, _ = pad_packed_sequence(packed_output, batch_first=True,
                                        total_length=self.max_length)
        # TODO: try max-pooling
        last_output = output[range(B), query_length - 1, :]
        embedded_lang = self.lang_encoder(last_output)
        if self.unit_vector:
            embedded_lang = F.normalize(embedded_lang, dim=-1)
        return embedded_lang

    def compare_emdeddings(self, anchor, x, dim=-1):
        # TODO: generalize to other similarities
        return (anchor - x).pow(2).sum(dim=dim)

    def init_parameters(self):
        "Initialize network parameters"
        # if filename is not None and os.path.exists(filename):
        #    raise NotImplementedError('WIP')
        for name, prm in self.named_parameters():
            if 'bias' in name:
                prm.data.fill_(0)
            else:
                prm.data.uniform_(-0.08, 0.08)

    def optimization_parameters(
            self, initial_lr=1e-2, caffe_setup=False, freeze_visual=False,
            freeze_lang=False):
        # freeze_visual_encoder=True):
        if caffe_setup:
            return self.optimization_parameters_original(
                initial_lr, freeze_visual, freeze_lang)
                # freeze_visual_encoder)
        prm_policy = [
            {'params': self.sentence_encoder.parameters(),
             'lr': initial_lr * 10},
            {'params': self.visual_encoder.parameters()},
            {'params': self.lang_encoder.parameters()},
        ]
        return prm_policy

    def optimization_parameters_original(
            self, initial_lr, freeze_visual, freeze_lang):
        # freeze_visual_encoder):
        prm_policy = []

        for name, prm in self.named_parameters():
            is_lang_tower = ('sentence_encoder' in name or
                             'lang_encoder' in name)
            is_visual_tower = 'visual_encoder' in name
            # is_visual_tower = ('visual_encoder' in name or
            #                    'visual_embedding' in name)
            if freeze_visual and is_visual_tower:
                continue
            elif freeze_lang and is_lang_tower:
                continue
            # elif freeze_visual_encoder and 'visual_encoder' in name:
            #     continue
            elif 'sentence_encoder' in name and 'bias_ih_l' in name:
                continue
            elif 'sentence_encoder' in name:
                prm_policy.append({'params': prm, 'lr': initial_lr * 10})
            elif 'bias' in name:
                prm_policy.append({'params': prm, 'lr': initial_lr * 2})
            else:
                prm_policy.append({'params': prm})
        return prm_policy

    def predict(self, *args):
        "Compute distance between visual and sentence"
        d_pos, *_ = self.forward(*args)
        return d_pos, False

    def search(self, query, table):
        "Exhaustive search of single query in table"
        return self.compare_emdeddings(query, table), False

    def _unpack_visual(self, *args):
        "Get visual feature inside a dict"
        argout = ()
        for i in args:
            if isinstance(i, dict):
                assert len(i) == 1
                j = next(iter(i))
                argout += (i[j],)
            else:
                argout += (i,)
        return argout


class SMCN(MCN):
    "SMCN model"

    def __init__(self, *args, **kwargs):
        super(SMCN, self).__init__(*args, **kwargs)

    def forward(self, padded_query, query_length, visual_pos,
                visual_neg_intra=None, visual_neg_inter=None):
        # v_v_embedded_* are tensors of shape [B, N, D]
        (v_embedded_pos, v_embedded_neg_intra,
         v_embedded_neg_inter) = self.encode_visual(
             visual_pos, visual_neg_intra, visual_neg_inter)

        l_embedded = self.encode_query(padded_query, query_length)
        # transform l_emdedded into a tensor of shape [B, 1, D]
        l_embedded = l_embedded.unsqueeze(1)

        # meta-comparison
        c_pos, c_neg_intra, c_neg_inter = self.compare_emdedded_snippets(
            v_embedded_pos, v_embedded_neg_intra, v_embedded_neg_inter,
            l_embedded, visual_pos, visual_neg_intra, visual_neg_inter)
        return c_pos, c_neg_intra, c_neg_inter

    def encode_visual(self, pos, neg_intra, neg_inter):
        pos, _, neg_intra, _, neg_inter, _ = self._unpack_visual(
            pos, neg_intra, neg_inter)
        embedded_neg_intra, embedded_neg_inter = None, None

        embedded_pos = self.fwd_visual_snippets(pos)
        if neg_intra is not None:
            embedded_neg_intra = self.fwd_visual_snippets(neg_intra)
        if neg_inter is not None:
            embedded_neg_inter = self.fwd_visual_snippets(neg_inter)
        return embedded_pos, embedded_neg_intra, embedded_neg_inter

    def fwd_visual_snippets(self, x):
        B, N, D = x.shape
        x_ = x.view(-1, D)
        x_ = self.visual_encoder(x_)
        # x_ = self.visual_embedding(x_)
        if self.unit_vector:
            x_ = F.normalize(x_)
        return x_.view((B, N, -1))

    def pool_compared_snippets(self, x, mask):
        masked_x = x * mask
        K = mask.detach().sum(dim=-1)
        return masked_x.sum(dim=-1) / K

    def compare_emdedded_snippets(self, embedded_p, embedded_n_intra,
                                  embedded_n_inter, embedded_a,
                                  pos, neg_intra, neg_inter):
        _, mask_p, _, mask_n_intra, _, mask_n_inter = self._unpack_visual(
            pos, neg_intra, neg_inter)
        c_neg_intra, c_neg_inter = None, None

        c_pos = self.pool_compared_snippets(
            self.compare_emdeddings(embedded_a, embedded_p), mask_p)
        if embedded_n_intra is not None:
            c_neg_intra = self.pool_compared_snippets(
                self.compare_emdeddings(embedded_a, embedded_n_intra),
                mask_n_intra)
        if embedded_n_inter is not None:
            c_neg_inter = self.pool_compared_snippets(
                self.compare_emdeddings(embedded_a, embedded_n_inter),
                mask_n_inter)
        return c_pos, c_neg_intra, c_neg_inter

    def search(self, query, table, clips_per_segment, clips_per_segment_list):
        """Exhaustive search of query in table

        TODO: batch to avoid out of memory?
        """
        clip_distance = self.compare_emdeddings(
            query, table).split(clips_per_segment_list)
        sorted_clips_per_segment, ind = clips_per_segment.sort(
            descending=True)
        # distance >= 0 thus we pad with zeros
        clip_distance_padded = pad_sequence(
            [clip_distance[i] for i in ind], batch_first=True)
        sorted_segment_distance = (clip_distance_padded.sum(dim=1) /
                                   sorted_clips_per_segment)
        _, original_ind = ind.sort(descending=False)
        segment_distance = sorted_segment_distance[original_ind]
        return segment_distance, False

    def _unpack_visual(self, *args):
        "Get visual feature inside a dict"
        argout = ()
        for i in args:
            if isinstance(i, dict):
                assert len(i) == 2
                # only works in cpython >= 3.6
                argout += tuple(i.values())
            elif i is None:
                argout += (None, None)
            else:
                argout += (i,)
        return argout


if __name__ == '__main__':
    import torch, random
    from torch.nn.utils.rnn import pad_sequence
    B, LD = 3, 5
    net = MCN(lang_size=LD)
    x = torch.rand(B, 4096, requires_grad=True)
    z = [random.randint(2, 6) for i in range(B)]
    z.sort(reverse=True)
    y = [torch.rand(i, LD, requires_grad=True) for i in z]
    y_padded = pad_sequence(y, True)
    z = torch.tensor(z)
    a, b, c = net(y_padded, z, x, x, x)
    b.backward(b.clone())
    a, b, *c = net(y_padded, z, x)
    # Unsuccesful attempt tp check backward
    # b.backward(10000*b.clone())
    # print(z)
    # print(y_padded)
    # print(f'y.shape = {y_padded.shape}')
    # print(y_padded.grad)
    # print([i.grad for i in y])
