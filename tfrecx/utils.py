from typing import Union

import tensorflow as tf

def get_default_values(dtype: str, length: int) -> list:
    """
    Return a list of default values for a given dtype.

    Args:
        dtype: One of 'bytes', 'float', or 'int64'.
        length: Number of elements in the list.

    Returns:
        List of default values of the specified type.
    """
    if dtype == "bytes":
        return [""] * length
    elif dtype == "float":
        return [0.0] * length
    elif dtype == "int64":
        return [0] * length
    else:
        return [None] * length
    
def get_feature_description(schema: dict) -> dict:
    """
    Build a TensorFlow feature description from a schema.

    Args:
        schema: Dictionary mapping feature names to dtypes 
                ('int64', 'float', 'bytes').

    Returns:
        Dict suitable for tf.io.parse_single_example.
    """
    feature_description = {}
    for key, dtype in schema.items():
        if dtype == "int64":
            feature_description[key] = tf.io.FixedLenFeature([], tf.int64, default_value=0)
        elif dtype == "float":
            feature_description[key] = tf.io.FixedLenFeature([], tf.float32, default_value=0.0)
        elif dtype == "bytes":
            feature_description[key] = tf.io.FixedLenFeature([], tf.string, default_value=b"")
        else:
            raise ValueError(f"Unsupported dtype '{dtype}' for feature '{key}'")
    return feature_description

def to_feature(value: Union[int, float, str, bytes]) -> tf.train.Feature:
    """
    Convert a Python value to a TensorFlow Feature.

    Args:
        value: int, float, str, or bytes to convert.

    Returns:
        A tf.train.Feature containing the value.
    """
    if isinstance(value, int):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    if isinstance(value, float):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    if isinstance(value, str):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode("utf-8")]))
    if isinstance(value, bytes):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    raise ValueError(f"Unsupported value type: {type(value)}")

def get_default_value(dtype: str):
    """
    Return the default value for a given dtype.

    Args:
        dtype: One of 'int64', 'float', or 'bytes'.

    Returns:
        The default value corresponding to the dtype.
    """
    if dtype == "int64":
        return 0
    elif dtype == "float":
        return 0.0
    elif dtype == "bytes":
        return ""
    else:
        return None