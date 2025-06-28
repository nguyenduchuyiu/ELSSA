# E.L.S.S.A - Even Lowkey, Still Superior Assistant

## Introduction

E.L.S.S.A is a powerful AI assistant that is easy to set up and use. Follow the instructions below to get started.

## Installation Guide

### 1. Create a Python Virtual Environment

Use [uv](https://github.com/astral-sh/uv) to create a virtual environment with Python 3.10.12:

```bash
uv venv elssa --python=3.10.12
source elssa/bin/activate
```

### 2. Install Dependencies

Install all required Python packages from `requirements.txt`:

```bash
uv pip install -r requirements.txt
```

### 3. Install OpenVoice

Clone the OpenVoice repository and install it in editable mode:

```bash
git clone git@github.com:myshell-ai/OpenVoice.git openvoice
cd openvoice
uv pip install -e .
```

### 4. Download and Extract Checkpoints

Download the sample checkpoint for OpenVoice:

```bash
wget https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_1226.zip
unzip checkpoints_1226.zip
```

## Notes

- Make sure you have [uv](https://github.com/astral-sh/uv) and Python 3.10.12 installed on your system.
- If you encounter permission issues when cloning OpenVoice, check your SSH keys or use HTTPS instead of SSH.

---

Happy installing and enjoy using E.L.S.S.A!