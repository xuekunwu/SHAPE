# Model Upload Guide for Cell Segmentation Tools

This guide explains how to upload custom Cellpose models to Hugging Face Hub for use with `Cell_Segmenter_Tool` and `Organoid_Segmenter_Tool`.

## Prerequisites

1. **Hugging Face Account**: Create an account at [huggingface.co](https://huggingface.co)
2. **Hugging Face CLI or Python Package**: Install `huggingface_hub`
   ```bash
   pip install huggingface_hub
   ```
3. **Hugging Face Token**: Generate an access token from [Settings > Access Tokens](https://huggingface.co/settings/tokens)
   - Set the token as an environment variable:
     ```bash
     # On Windows (PowerShell)
     $env:HUGGINGFACE_TOKEN="your_token_here"
     
     # On Linux/Mac
     export HUGGINGFACE_TOKEN="your_token_here"
     ```

## Uploading Models

### Method 1: Using Python Script (Recommended)

#### Step 1: Create a Repository on Hugging Face Hub

1. Go to [huggingface.co/new](https://huggingface.co/new)
2. Create a new repository:
   - **Repository name**: e.g., `cell-segmenter-cpsam-model` or `organoid-segmenter-model`
   - **Visibility**: Choose Public or Private
   - **Type**: Select "Model"

#### Step 2: Prepare Your Model File

For **Cell_Segmenter_Tool** (CPSAM model):
- Save your trained CPSAM model file (typically a `.pth` or `.pt` file)
- Recommended filename: `cpsam_model.pth` or `cpsam_model.pt`

For **Organoid_Segmenter_Tool**:
- **REQUIRED**: A specialized organoid segmentation model is mandatory
- **IMPORTANT**: Standard Cellpose models (cyto/cyto2) are NOT suitable for organoids and will NOT work
- Save your custom organoid segmentation model file (must be trained specifically for organoids)
- Recommended filename: `organoid_model.pth` or `organoid_model.pt` (or as per your naming convention)

#### Step 3: Upload Using Python

Create a Python script `upload_model.py`:

```python
from huggingface_hub import HfApi, upload_file
import os

# Initialize the API
api = HfApi()

# Configuration
REPO_ID = "your-username/cell-segmenter-cpsam-model"  # Update with your repo ID
MODEL_FILE_PATH = "path/to/your/cpsam_model.pth"  # Update with your model file path
MODEL_FILENAME = "cpsam_model"  # The filename to use in the repo (without extension)

# Check if token is set
token = os.getenv("HUGGINGFACE_TOKEN")
if not token:
    raise ValueError("HUGGINGFACE_TOKEN environment variable not set!")

# Upload the model
try:
    upload_file(
        path_or_fileobj=MODEL_FILE_PATH,
        path_in_repo=MODEL_FILENAME,
        repo_id=REPO_ID,
        repo_type="model",
        token=token,
    )
    print(f"✅ Successfully uploaded {MODEL_FILE_PATH} to {REPO_ID}")
    print(f"📁 File saved as: {MODEL_FILENAME}")
except Exception as e:
    print(f"❌ Error uploading model: {e}")
```

Run the script:
```bash
python upload_model.py
```

### Method 2: Using Hugging Face CLI

1. **Login to Hugging Face**:
   ```bash
   huggingface-cli login
   ```
   Enter your token when prompted.

2. **Clone the repository**:
   ```bash
   git clone https://huggingface.co/your-username/cell-segmenter-cpsam-model
   cd cell-segmenter-cpsam-model
   ```

3. **Copy your model file**:
   ```bash
   cp /path/to/your/cpsam_model.pth ./cpsam_model
   ```

4. **Commit and push**:
   ```bash
   git add cpsam_model
   git commit -m "Add CPSAM model for cell segmentation"
   git push
   ```

## Updating Tool Code

After uploading your model, update the tool code to use the correct repository ID:

### For Cell_Segmenter_Tool

Edit `octotools/tools/cell_segmenter/tool.py`:

```python
# Line ~51: Update the repo_id
model_path = hf_hub_download(
    repo_id="your-username/cell-segmenter-cpsam-model",  # UPDATE THIS
    filename="cpsam_model",  # UPDATE THIS (filename without extension)
    token=os.getenv("HUGGINGFACE_TOKEN")
)
```

### For Organoid_Segmenter_Tool

**IMPORTANT**: Organoid_Segmenter_Tool REQUIRES a specialized organoid model. The tool will fail if no organoid model is provided. Standard Cellpose models (cyto/cyto2) are NOT supported.

Edit `octotools/tools/organoid_segmenter/tool.py`:

```python
# Line ~60-63: Update the repo_id and filename
model_path = hf_hub_download(
    repo_id="your-username/organoid-segmenter-model",  # UPDATE THIS
    filename="CO_4x_V2",  # UPDATE THIS (filename as stored in your repo)
    token=os.getenv("HUGGINGFACE_TOKEN")
)
```

## Model Requirements

### Cellpose Model Format

The models should be compatible with Cellpose's `CellposeModel` class. Typical requirements:

1. **File Format**: 
   - PyTorch models (`.pth`, `.pt`) are standard
   - Cellpose expects specific model architectures

2. **Model Architecture**:
   - For CPSAM models: Should match the Cellpose CPSAM architecture
   - For custom organoid models: Can be based on Cellpose architectures or custom architectures compatible with Cellpose

3. **Loading**:
   - Models should load using: `models.CellposeModel(pretrained_model=model_path)`

### Training Your Own Model

If you want to train a custom model:

1. **Using Cellpose**:
   ```python
   from cellpose import models, io
   
   # Train a new model
   model = models.CellposeModel(gpu=True)
   model.train(train_data, train_labels, channels=[0,0], n_epochs=100)
   
   # Save the model
   model.save_model("path/to/save/cpsam_model.pth")
   ```

2. **Using CPSAM**:
   - Follow the CPSAM training procedure
   - Save the trained model weights

3. **For Organoids**:
   - Train on organoid-specific datasets
   - Use appropriate augmentation strategies for organoid images

## Testing the Uploaded Model

After uploading, test that the model downloads correctly:

```python
from huggingface_hub import hf_hub_download
from cellpose import models
import os

# Download the model
model_path = hf_hub_download(
    repo_id="your-username/cell-segmenter-cpsam-model",
    filename="cpsam_model",
    token=os.getenv("HUGGINGFACE_TOKEN")
)

# Load the model
model = models.CellposeModel(pretrained_model=model_path)
print(f"✅ Model loaded successfully from: {model_path}")
```

## Troubleshooting

### Issue: "File not found" error

**Solution**: 
- Verify the `repo_id` and `filename` match exactly what's in your Hugging Face repository
- Check that the file exists in the repository
- Ensure your token has read access to the repository

### Issue: Model loading fails

**Solution**:
- Verify the model file format is correct (should be compatible with Cellpose)
- Check that the model architecture matches what Cellpose expects
- Try loading the model locally first before uploading

### Issue: Permission denied

**Solution**:
- Ensure your `HUGGINGFACE_TOKEN` has write permissions
- For private repositories, make sure the token has access
- Check repository visibility settings

### Issue: Model works locally but not after upload

**Solution**:
- Verify the file was uploaded completely (check file size)
- Ensure no corruption during upload (try downloading and comparing checksums)
- Check that file extensions are handled correctly

## Example: Complete Upload Workflow

```python
# 1. Train or obtain your model
# ... training code ...

# 2. Save the model
model.save_model("my_cpsam_model.pth")

# 3. Upload to Hugging Face
from huggingface_hub import upload_file
import os

upload_file(
    path_or_fileobj="my_cpsam_model.pth",
    path_in_repo="cpsam_model",  # No extension needed
    repo_id="your-username/cell-segmenter-cpsam-model",
    repo_type="model",
    token=os.getenv("HUGGINGFACE_TOKEN"),
)

# 4. Update tool.py with the new repo_id and filename

# 5. Test the tool
from octotools.tools.cell_segmenter.tool import Cell_Segmenter_Tool
tool = Cell_Segmenter_Tool()
# ... use the tool ...
```

## Additional Resources

- [Hugging Face Hub Documentation](https://huggingface.co/docs/hub)
- [Cellpose Documentation](https://cellpose.readthedocs.io/)
- [Hugging Face Python API](https://huggingface.co/docs/huggingface_hub/quick-start)

## Notes

- For **public repositories**: Anyone can download your models
- For **private repositories**: Only you and authorized users can access
- Model files can be large; ensure stable internet connection for uploads
- Consider adding a README.md to your repository explaining the model, training data, and usage
