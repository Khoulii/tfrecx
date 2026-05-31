import os
import io
import random
import pytest
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

from tfrecx import (
    count_records,
    head,
    get_schema,
    pd_to_tfrec,
    tfrec_to_pd,
    append_pd_to_tfrec,
    append_imgs_to_tfrec,
    merge_tfrecs,
)


# ----------------------
# Helper functions
# ----------------------
def _create_dummy_tfrecord(path: str, num_records: int = 5):
    with tf.io.TFRecordWriter(path) as writer:
        for i in range(num_records):
            feature = {
                "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
                "name": tf.train.Feature(bytes_list=tf.train.BytesList(value=[f"name{i}".encode()])),
                "value": tf.train.Feature(float_list=tf.train.FloatList(value=[i * 1.1]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

def _generate_dummy_df(n_rows=10, n_cols=3):
    data = {}
    for i in range(n_cols):
        if i == 0:
            data[f"col{i}"] = list(range(n_rows))
        elif i == 1:
            data[f"col{i}"] = [x * 1.1 for x in range(n_rows)]
        else:
            data[f"col{i}"] = [f"name{x}" for x in range(n_rows)]
    return pd.DataFrame(data)

def _create_dummy_image(color=(255, 0, 0), size=(64, 64)):
    return Image.new("RGB", size, color=color)

def _read_images_from_tfrecord(path):
    dataset = tf.data.TFRecordDataset(path)
    images = []
    for raw in dataset:
        example = tf.train.Example()
        example.ParseFromString(raw.numpy())
        img_bytes = example.features.feature["image"].bytes_list.value[0]
        images.append(Image.open(io.BytesIO(img_bytes)))
    return images

# ----------------------
# Tests
# ----------------------
def test_count_records(tmp_path):
    file_path = os.path.join(tmp_path, "dummy.tfrecord")
    _create_dummy_tfrecord(file_path, 7)
    assert count_records(file_path) == 7

def test_head(tmp_path):
    file_path = os.path.join(tmp_path, "dummy.tfrecord")
    _create_dummy_tfrecord(file_path, 5)
    records = head(file_path, n=2)
    assert len(records) == 2
    assert records[0]["id"] == [0]
    assert records[1]["name"] == ["name1"]

def test_get_schema(tmp_path):
    file_path = os.path.join(tmp_path, "dummy.tfrecord")
    _create_dummy_tfrecord(file_path, 3)
    schema = get_schema(file_path)
    assert schema == {"id": "int64", "name": "bytes", "value": "float"}

@pytest.mark.parametrize("max_rows", [None, 5])
def test_pd_to_tfrec(tmp_path, max_rows):
    df = _generate_dummy_df(n_rows=10, n_cols=3)
    out_file = os.path.join(tmp_path, "test.tfrecord")
    pd_to_tfrec(df, out_file, max_rows=max_rows)

    # Determine expected TFRecord files
    if max_rows is None:
        expected_files = [out_file]
    else:
        total_rows = len(df)
        num_files = (total_rows + max_rows - 1) // max_rows
        base_name = out_file.rsplit('.', 1)[0]
        expected_files = [f"{base_name}_{i}.tfrecord" for i in range(num_files)]

    # Check that all files exist and row count matches
    for f in expected_files:
        assert os.path.exists(f)
        if max_rows is None:
            expected_count = len(df)
        else:
            idx = int(f.rsplit("_", 1)[1].split(".")[0])
            start = idx * max_rows
            end = min((idx + 1) * max_rows, len(df))
            expected_count = end - start
        assert count_records(f) == expected_count

@pytest.mark.parametrize("batch_size", [None, 100])
def test_tfrec_to_pd(tmp_path, batch_size):
    df = _generate_dummy_df(n_rows=1000, n_cols=5)
    out_file = os.path.join(tmp_path, "test.tfrecord")
    pd_to_tfrec(df, out_file)
    df_read = tfrec_to_pd(out_file, schema=None, batch_size=batch_size)
    assert df_read.shape == df.shape
    for col in df.columns:
        if df[col].dtype == object:
            assert all(df[col] == df_read[col])
        else:
            assert np.allclose(df[col].values, df_read[col].values)

def test_append_pd_to_tfrec_normal(tmp_path):
    df1 = _generate_dummy_df(3)
    out_file = os.path.join(tmp_path, "test.tfrecord")
    pd_to_tfrec(df1, out_file)

    df2 = _generate_dummy_df(2)
    append_pd_to_tfrec(df2, out_file)

    assert count_records(out_file) == 5
    dataset = tf.data.TFRecordDataset(out_file)
    for i, raw in enumerate(dataset):
        example = tf.train.Example()
        example.ParseFromString(raw.numpy())
        if i == 0:
            assert example.features.feature["col0"].int64_list.value[0] == 0
        if i == 4:
            assert example.features.feature["col2"].bytes_list.value[0].decode("utf-8") == "name1"

def test_append_imgs_to_tfrec_single(tmp_path):
    file_path = os.path.join(tmp_path, "images.tfrecord")
    img = _create_dummy_image()
    append_imgs_to_tfrec(img, file_path)
    images = _read_images_from_tfrecord(file_path)
    assert len(images) == 1
    assert images[0].size == (64, 64)

def test_append_imgs_to_tfrec_multiple(tmp_path):
    file_path = os.path.join(tmp_path, "multi_images.tfrecord")
    imgs = [_create_dummy_image() for _ in range(3)]
    append_imgs_to_tfrec(imgs, file_path)
    more_imgs = [_create_dummy_image() for _ in range(2)]
    append_imgs_to_tfrec(more_imgs, file_path)
    images = _read_images_from_tfrecord(file_path)
    assert len(images) == 5

def test_merge_tfrecs_normal(tmp_path):
    df1 = _generate_dummy_df(3)
    df2 = _generate_dummy_df(2)
    file1 = os.path.join(tmp_path, "part1.tfrecord")
    file2 = os.path.join(tmp_path, "part2.tfrecord")
    merged = os.path.join(tmp_path, "merged.tfrecord")
    pd_to_tfrec(df1, file1)
    pd_to_tfrec(df2, file2)
    merge_tfrecs([file1, file2], merged)
    df_merged = tfrec_to_pd(merged)
    assert len(df_merged) == len(df1) + len(df2)
