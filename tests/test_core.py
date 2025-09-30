import tensorflow as tf
import os
import pytest
from tfrecx import count_records

def _create_dummy_tfrecord(path: str, num_records: int = 5):
    with tf.io.TFRecordWriter(path) as writer:
        for i in range(num_records):
            feature = {
                "value": tf.train.Feature(int64_list=tf.train.Int64List(value=[i]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

def test_count_records(tmp_path):
    file_path = os.path.join(tmp_path, "dummy.tfrecord")
    _create_dummy_tfrecord(file_path, 7)
    assert count_records(file_path) == 7
