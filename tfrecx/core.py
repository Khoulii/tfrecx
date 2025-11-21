import io
import os
import csv
import random
import tempfile

import tensorflow as tf
import pandas as pd
import numpy as np

from PIL import Image

from .utils import *

def count_records(path: str) -> int:
    """
    Count the number of records in a TFRecord file.
    
    Args:
        path (str): Path to the TFRecord file.
    
    Returns:
        int: Number of records in the file.
    """
    count = 0
    for _ in tf.data.TFRecordDataset(path):
        count += 1
    return count

def head(path: str, n: int = 5) -> list:
    """
    Reads the first N records from a TFRecord file.

    Args:
        path: Path to the TFRecord file.
        n: Number of records to read. Defaults to 5.

    Returns:
        A list of dictionaries. Each dictionary maps feature names
        to lists of values (decoded into Python types).
    """
    records = []
    dataset = tf.data.TFRecordDataset(path)

    for raw_record in dataset.take(n):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        parsed = {}
        for key, feature in example.features.feature.items():
            kind = feature.WhichOneof("kind")
            if kind == "bytes_list":
                parsed[key] = [v.decode("utf-8") for v in feature.bytes_list.value]
            elif kind == "float_list":
                parsed[key] = list(feature.float_list.value)
            elif kind == "int64_list":
                parsed[key] = list(feature.int64_list.value)
        records.append(parsed)
    return records

def get_schema(path: str, n:int=1) -> dict:
    """
    Returns the schema of a TFRecord file.

    Args:
        path: Path to the TFRecord file.

    Returns:
        A dictionary mapping feature names to their types
        ('int64', 'float', or 'bytes').
    """
    dataset = tf.data.TFRecordDataset(path)
    schema_set = set()
    raw_record = next(iter(dataset.take(n)))
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    for key, feature in example.features.feature.items():
        schema = {}
        kind = feature.WhichOneof("kind")
        if kind == "bytes_list":
            schema[key] = "bytes"
        elif kind == "float_list":
            schema[key] = "float"
        elif kind == "int64_list":
            schema[key] = "int64"
        schema_set|=set([(k, v) for k, v in schema.items()])
    return dict(schema_set)

def pd_to_tfrec(df: pd.DataFrame, out_file_path: str, max_rows: int = None, shuffle: bool = False) -> None:
    """
    Convert a pandas DataFrame to a TFRecord file.

    Args:
        df (pd.DataFrame): Input DataFrame.
        out_file_path (str): Path to the output TFRecord file.
        max_rows (int, optional): Maximum rows per TFRecord file.
                                If None, all rows go to a single file.
        shuffle (bool, optional): True to shuffle the before inserting to the tfrecord.
                                Defult is False.
    """
    if shuffle:
        df = df.sample(frac=1, random_state=None).reset_index(drop=True)

    if max_rows is None:
        dfs = [df]
        file_paths = [out_file_path]
    else:
        total_rows = len(df)
        num_files = (total_rows + max_rows - 1) // max_rows
        dfs = [df.iloc[i*max_rows : (i+1)*max_rows] for i in range(num_files)]
        base_name = out_file_path.rsplit('.', 1)[0]
        file_paths = [f"{base_name}_{i}.tfrecord" for i in range(num_files)]

    for path, chunk in zip(file_paths, dfs):
        with tf.io.TFRecordWriter(path) as writer:
            for row in chunk.itertuples(index=False, name=None):
                features = {col: to_feature(val) for col, val in zip(chunk.columns, row)}
                example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(example.SerializeToString())

def tfrec_to_pd(paths: str, schema: dict = None, batch_size: int = 1000) -> pd.DataFrame:
    """
    Convert one or more TFRecord files into a Pandas DataFrame in a memory-safe manner.
    Records are streamed in batches and written to a temporary CSV file, avoiding
    in-memory accumulation.

    Args:
        paths (str or list[str]): Path or list of paths to TFRecord file(s).
        schema (dict, optional): Dictionary mapping feature names to types
                                 ('int64', 'float', 'bytes').
                                 If None, the schema is inferred from the first TFRecord file.
        batch_size (int, optional): Number of records to read per batch. Default is 1000.

    Returns:
        pd.DataFrame: A DataFrame containing all records from the TFRecord files.
    """
    if isinstance(paths, str):
        paths = [paths]

    if schema is None:
        schema = get_schema(paths[0])
        if not schema:
            raise ValueError(f"TFRecord file {paths[0]} is empty or missing schema.")

    feature_description = get_feature_description(schema)

    dataset = tf.data.TFRecordDataset(paths).map(
        lambda x: tf.io.parse_single_example(x, feature_description)
    )

    if batch_size is None:
        batch_size = 1000
    dataset = dataset.batch(batch_size)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        csv_path = tmp.name

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(schema.keys())

        for batch in dataset:
            batch_len = len(next(iter(batch.values())))
            for i in range(batch_len):
                row = []
                for key, dtype in schema.items():
                    value = batch[key][i].numpy()
                    if dtype == "bytes":
                        value = value.decode("utf-8")
                    else:
                        if hasattr(value, "item"):
                            value = value.item()
                    row.append(value)
                writer.writerow(row)

    df = pd.read_csv(csv_path)
    os.remove(csv_path)
    return df

def append_pd_to_tfrec(df: pd.DataFrame, out_file_path: str, out_schema: dict = None) -> None:
    """
    Append rows from a Pandas DataFrame to an existing TFRecord file,
    while preserving schema and existing records.

    Args:
        df (pd.DataFrame): The DataFrame containing new rows to append.
        out_file_path (str): Path to the TFRecord file that will be appended to.
        out_schema (dict, optional): A dictionary mapping feature names to types.
            If not provided, the function will infer the schema from the existing TFRecord.
    """

    if out_schema is None:
        if not os.path.exists(out_file_path) or os.path.getsize(out_file_path) == 0:
            raise ValueError(
                f"TFRecord {out_file_path} is empty or missing schema. Provide out_schema.")
        out_schema = get_schema(out_file_path)

    if not out_schema:
        raise ValueError(
            f"TFRecord {out_file_path} is empty or missing schema. Provide out_schema explicitly.")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name

    with tf.io.TFRecordWriter(tmp_path) as writer:
        # Copy existing records
        if os.path.exists(out_file_path) and os.path.getsize(out_file_path) > 0:
            try:
                for record in tf.data.TFRecordDataset(out_file_path):
                    writer.write(record.numpy())
            except tf.errors.DataLossError as e:
                raise ValueError(f"TFRecord file {out_file_path} is corrupted.") from e
        # Append new rows
        for row in df.itertuples(index=False, name=None):
            features = {}
            for col in out_schema.keys():
                if col in df.columns:
                    val = row[df.columns.get_loc(col)]
                else:
                    val = get_default_value(out_schema[col])

                features[col] = to_feature(val)

            example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())
    os.replace(tmp_path, out_file_path)

def append_imgs_to_tfrec(images, path: str) -> None:
    """
    Append one or more images to a TFRecord file in a memory-safe way.

    Args:
        images: A single image (PIL Image, numpy array, or file path) or list of such items.
        path: Path to the TFRecord file to append to.
    """

    if not isinstance(images, (list, tuple)):
        images = [images]

    def _image_to_serialized(img):
        if isinstance(img, str):
            with tf.io.gfile.GFile(img, 'rb') as f:
                img_bytes = f.read()
        else:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            buf = io.BytesIO()
            img.save(buf, format='JPEG')
            img_bytes = buf.getvalue()

        feature = {"image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes]))}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example.SerializeToString()

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name

    with tf.io.TFRecordWriter(tmp_path) as writer:
        # Copy existing records
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                for record in tf.data.TFRecordDataset(path):
                    writer.write(record.numpy())
            except tf.errors.DataLossError as e:
                raise ValueError(f"TFRecord file {path} is corrupted.") from e
        # Append new images
        for img in images:
            writer.write(_image_to_serialized(img))
    os.replace(tmp_path, path)


def merge_tfrecs(input_paths: list[str], output_path: str) -> None:
    """
    Merge multiple TFRecord files into a single TFRecord file.

    Args:
        input_paths (list[str]): List of paths to input TFRecord files.
        output_path (str): Path to the output merged TFRecord file.
    """
    if not input_paths:
        raise ValueError("No input TFRecord files provided.")

    for path in input_paths:
        if not os.path.exists(path):
            raise ValueError(f"Input TFRecord file not found: {path}")

    with tf.io.TFRecordWriter(output_path) as writer:
        for path in input_paths:
            for record in tf.data.TFRecordDataset(path):
                writer.write(record.numpy())

def split_tfrec(input_file: str, output_dir: str, max_rows: int) -> None:
    """
    Split a large TFRecord file into multiple smaller TFRecord files.

    Args:
        input_file (str): Path to the input TFRecord.
        output_dir (str): Directory to save the split files.
        max_rows (int): Maximum number of records per split file.
                        If None, no splitting is performed and the file is simply copied.

    """
    os.makedirs(output_dir, exist_ok=True)

    dataset = list(tf.data.TFRecordDataset(input_file))
    total = len(dataset)

    base_name = os.path.splitext(os.path.basename(input_file))[0]

    if max_rows is None:
        out_file = os.path.join(output_dir, f"{base_name}.tfrecord")
        with tf.io.TFRecordWriter(out_file) as writer:
            for record in dataset:
                writer.write(record.numpy())
        return

    if max_rows <= 0:
        raise ValueError("batch_size must be a positive integer or None.")

    num_files = (total + max_rows - 1) // max_rows
    for i in range(num_files):
        out_file = os.path.join(output_dir, f"{base_name}_{i}.tfrecord")
        with tf.io.TFRecordWriter(out_file) as writer:
            for record in dataset[i*max_rows : (i+1)*max_rows]:
                writer.write(record.numpy())

def shuffle_tfrec(input_file: str, output_file: str, batch_size: int = None) -> None:
    """
    Shuffle a TFRecord file and save the result.

    Args:
        input_file (str): Path to input TFRecord.
        output_file (str): Path to save shuffled TFRecord.
        buffer_size (int): Buffer size for shuffling (larger = more random).
    """
    dataset = list(tf.data.TFRecordDataset(input_file))

    if batch_size is None:
        records = list(dataset)
        random.shuffle(records)
        with tf.io.TFRecordWriter(output_file) as writer:
            for record in records:
                writer.write(record.numpy())
        return

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer or None.")

    buffer = []
    writer = tf.io.TFRecordWriter(output_file)

    for record in dataset:
        buffer.append(record)
        if len(buffer) >= batch_size:
            random.shuffle(buffer)
            for r in buffer:
                writer.write(r.numpy())
            buffer.clear()
    if buffer:
        random.shuffle(buffer)
        for r in buffer:
            writer.write(r.numpy())
    writer.close()
