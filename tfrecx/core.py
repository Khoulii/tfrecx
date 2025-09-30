import tensorflow as tf

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
