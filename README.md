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

### 3. Download OpenVoice Checkpoints

Download the sample checkpoint for OpenVoice:

```bash
mkdir -p libs/openvoice

wget -O libs/openvoice/checkpoints_1226.zip \
  https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_1226.zip

unzip libs/openvoice/checkpoints_1226.zip -d libs/openvoice

rm libs/openvoice/checkpoints_1226.zip

```

### 4. Choose a LLM

#### 4.1. Phi-2

```bash
# Manually download the model and run with local path
huggingface-cli download TheBloke/phi-2-GGUF --local-dir models/phi-2
```

#### 4.2. Llama-3.2-3B-Instruct

```bash
wget https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-IQ3_M.gguf -P models/Llama-3.2-3B-Instruct/
```

## Notes

- Make sure you have [uv](https://github.com/astral-sh/uv) and Python 3.10.12 installed on your system.

- If you encounter permission issues when cloning OpenVoice, check your SSH keys or use HTTPS instead of SSH.

---

Happy installing and enjoy using E.L.S.S.A!
