# SHAPE Hugging Face Space

This directory contains the Hugging Face Space demo for SHAPE.

## Overview

The Hugging Face Space provides an interactive web interface for SHAPE, allowing users to:

- Upload cell images
- Ask natural language queries
- View analysis results
- Explore morphological features

## Structure

- `app.py`: Gradio web application
- `requirements.txt`: Python dependencies
- `README.md`: Space description (auto-generated)

## Running Locally

```bash
cd hf_space
pip install -r requirements.txt
python app.py
```

## Deployment

The Space is deployed on Hugging Face Spaces. To update:

1. Push changes to the repository
2. The Space will automatically rebuild

## Note

This is a **demo/showcase** of SHAPE's capabilities. For production use and research, refer to:

- Framework code: `shape/`
- Examples: `examples/`
- Documentation: `docs/`

The core framework is designed for programmatic use, not just web interfaces.

