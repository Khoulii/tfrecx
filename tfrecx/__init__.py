"""
TFReX: A memory-efficient TFRecord utility library.

Modules:
    - core: Main functions for reading, writing, appending, merging, splitting, and shuffling TFRecords.
    - utils: Helper functions for TensorFlow features, schemas, and default values.

Example:
    from tfrex import pd_to_tfrec, tfrec_to_pd, append_pd_to_tfrec
"""

from .core import (
    count_records,
    head,
    get_schema,
    pd_to_tfrec,
    tfrec_to_pd,
    append_pd_to_tfrec,
    append_imgs_to_tfrec,
    merge_tfrecs,
    split_tfrec,
    shuffle_tfrec,
)

from .utils import (
    get_default_values,
    get_feature_description,
    to_feature,
    get_default_value,
)

__all__ = [
    "count_records",
    "head",
    "get_schema",
    "pd_to_tfrec",
    "tfrec_to_pd",
    "append_pd_to_tfrec",
    "append_imgs_to_tfrec",
    "merge_tfrecs",
    "split_tfrec",
    "shuffle_tfrec",
    "get_default_values",
    "get_feature_description",
    "to_feature",
    "get_default_value",
]
