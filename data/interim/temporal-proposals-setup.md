This file shows the parameters used to generate the temporal proposals used in our [preprint-v1](https://arxiv.org/abs/1907.12763).

The parameter below should be used to instantiate the classes [here](https://github.com/escorciav/moments-retrieval-page/blob/master/proposals.py). Using those and our [train/val/test data](https://github.com/escorciav/moments-retrieval-page/blob/master/data/processed) should regenerate our temporal proposals.

TODO:sample-code-to-dump-temporal-proposals per-video

# DiDeMo

```JSON
{
    "proposal_interface": "DidemoICCV17SS",
}
```

For CAL models, we enforce that a moment must have more than 2 clips by using pooled features with clip-length of 2.5 seconds.

For MCN models, pooled features with clip-length of 5s achieves slightly better performance. Thus, we used&reported those.

# Charades-STA

```JSON
{
    "length": 3,
    "scales": [
        2,
        3,
        4,
        5,
        6,
        7,
        8
    ],
    "stride": 0.3,
    "proposal_interface": "SlidingWindowMSRSS",
}
```

# ActivityNet-Captions

```JSON
{
    "length": 5,
    "scales": [
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        24,
        26
    ],
    "stride": 0.3,
    "proposal_interface": "SlidingWindowMSRSS",
}
```