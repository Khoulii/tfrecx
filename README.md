# TFReCX
Elegant, minimal, and production-ready toolkit for working with TFRecords.

<img src="banner.png" alt="banner" width="600" height="auto">

---
## Overview
TFReCX is designed to help developers efficiently generate, manipulate, and manage TFRecords from various sources with clean abstractions, type safety, and customization. Its goal is to reduce the complexity of working with TFRecords and provide a production-ready workflow for TensorFlow pipelines.

---
## Features
- **Easy TFRecord Creation:** Convert pandas.DataFrame, image directories, or raw data directly into TFRecord files with a single function.
- **Flexible Shuffling & Splitting:** Global shuffling and split large datasets into multiple TFRecord files automatically.
- **Merging TFRecords:** Combine multiple TFRecord files into one while preserving schema and type consistency.
- **Customizable Serialization:** Supports custom feature converters for different data types (int, float, bytes, lists, images, etc.).
- **Streaming & Chunking:** Handle large datasets efficiently with minimal memory usage.
- **Production-Ready:** Type-safe, well-tested, and ready for TensorFlow pipelines.
- **Seamless Integration with TensorFlow:** Works out-of-the-box with `tf.data.TFRecordDataset` for training pipelines.
- **Installable via PyPI:** You can install it using pip or editable install from GitHub for development.

---
## Installation
```bash
pip install tfrecx
```

Or from source:

```bash
git clone https://github.com/yourusername/tfrecx.git
cd tfrecx
pip install -r requirements.txt
```

---

## Example Usage

```python
from tfrecx import head

head(path=tfrecord_path, n=5)
```
This prints the first 5 samples in the TFRecord file.

## Use Cases
- Training pipelines for TensorFlow models.
- Converting tabular datasets and images datasets into efficient binary form.
- Preparing large‑scale ML datasets.
- Ensuring consistent, validated, fast TFRecord generation.

## Contributing

We welcome contributions, and doing so is simple. Please follow the steps shown below.


### Step 1: Open a GitHub Issue
Before starting any work, create an issue describing:

- The problem or feature you want to address
- Your proposed solution
- Why the change is necessary

### Step 2: Create a Branch
Create a new branch for your changes:

```bash
git checkout -b feature/my-new-feature
```

Add your changes and commit them with clear messages.

### Step 3: **Set Up a Development Environment**
It’s recommended to use a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate   # Windows
   ```
Then install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```
This is to ensure your changes don’t affect your global Python environment.

### Step 4: Add Tests
Whenever you add features or fix bugs, include or update tests using pytest:

```bash
pytest -v
```

Make sure the updated code passes all existing and added tests.

### Step 5: Open a Pull Request
When your work is ready:

1. Push your branch to GitHub:
```bash
git push origin feature/my-new-feature
```
2. Open a pull request referencing the issue (e.g., Fixes #12).
3. Include a summary of what was changed and any notes for reviewers.

Thank you for contributing! Your effort makes TFReCX better for everyone.
