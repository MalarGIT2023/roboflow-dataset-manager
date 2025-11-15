# Roboflow Dataset Integration Guide

**Mission Tomorrow Career Exploration Event hosted by Chamber RVA**  
*Presented to 11,000+ eighth graders in Richmond*  
*Volunteered for IEEE Region3 Richmond*

---

## Overview

This guide explains how to locate datasets for emotion detection from **Roboflow Universe** and configure the dataset download script. This is **Step 1** of the IEEE Mission Tomorrow emotion detection workflow.

Roboflow provides a platform for hosting, managing, and distributing computer vision datasets. The `dataset-download-roboflow.py` script uses the Roboflow API to automatically download emotion detection datasets in YOLOv11 format.

### 🎯 Why Datasets Matter in AI

Before machines can detect emotions, they need to **learn** from examples. That's where datasets come in:

- **Dataset** = Thousands of labeled images showing different emotions
- **Training** = Teaching an AI model to recognize patterns
- **Deployment** = Using the trained model in real applications

Think of it like teaching a friend: "This is what 'happy' looks like, this is 'sad', this is 'angry'..." After seeing hundreds of examples, your friend gets good at recognizing emotions!

## Step 1: Finding Datasets in Roboflow Universe

### What is Roboflow Universe?

Roboflow Universe is a public repository of datasets where users share their projects. You can access it at:
```
https://universe.roboflow.com/
```

### Finding Emotion Detection Datasets

1. **Visit Roboflow Universe**: https://universe.roboflow.com/
2. **Search for emotion datasets**:
   - Search terms: "emotion detection", "facial emotion", "expression recognition"
   - Filter by: Dataset → Computer Vision Task → Object Detection

3. **Example Public Datasets**:
   - Facial Emotion Dataset (various versions)
   - Face Expression Recognition Dataset
   - Emotion Classification Dataset

### Dataset URL Format

When you find a dataset, the URL will look like:
```
https://universe.roboflow.com/WORKSPACE_NAME/PROJECT_NAME/dataset/VERSION_NUMBER
```

**Components**:
- `WORKSPACE_NAME` — The owner's workspace identifier
- `PROJECT_NAME` — The dataset project name
- `VERSION_NUMBER` — The specific version of the dataset

**Example**:
```
https://universe.roboflow.com/sina-awwg4/face-emotion-detection-fanof/dataset/5
```

Breaking this down:
- Workspace: `sina-awwg4`
- Project: `face-emotion-detection-fanof`
- Version: `5`

## Step 2: Getting Your API Key

### Option A: Using a Public Dataset (No API Key Needed)

For **publicly available** datasets in Roboflow Universe, you can sometimes download without authentication. However, using an API key allows:
- Automatic downloading
- Version tracking
- Workspace access

### Option B: Creating Your Own Dataset

If you want to create a custom dataset:

1. **Create a Roboflow Account**:
   - Go to https://roboflow.com
   - Sign up (free tier available)

2. **Create a Workspace**:
   - Create a new workspace for your project
   - Upload your annotated images
   - Configure preprocessing and augmentation

3. **Get Your API Key**:
   - Click on your profile icon (top-right)
   - Select "Account"
   - Copy your **Private API Key**
   - ⚠️ **Keep this secret** — never commit to public repositories

## Step 3: Extracting Dataset Information

### Finding Workspace and Project Names

**Method 1: From the URL**
```
https://universe.roboflow.com/sina-awwg4/face-emotion-detection-fanof/dataset/5
                           ↓              ↓
                       WORKSPACE      PROJECT
```

**Method 2: From Roboflow Dashboard**
1. Log in to https://roboflow.com
2. Navigate to your project
3. Go to **Export/Download** section
4. The code snippet will show your workspace and project names

### Finding Version Number

- The version is shown in the URL or in the project's **Versions** tab
- Latest version is recommended
- Each version may have different preprocessing applied

## Step 4: Configuring the Script

Edit `dataset-download-roboflow.py`:

```python
from roboflow import Roboflow

# Replace with your API key (from https://roboflow.com/settings/account)
rf = Roboflow(api_key="YOUR_API_KEY_HERE")

# Replace with your workspace name
# Found in: https://universe.roboflow.com/YOUR_WORKSPACE_NAME/...
project = rf.workspace("your_workspace_name").project("your_project_name")

# Replace with desired version number
version = project.version(5)

# Download in YOLOv11 format
dataset = version.download("yolov11")
```

## Step 5: Example Configurations

### Example 1: Using the Public Emotion Dataset

```python
from roboflow import Roboflow

# API key (required for downloading)
rf = Roboflow(api_key="YOUR_API_KEY")

# Public dataset in Roboflow Universe
project = rf.workspace("sina-awwg4").project("face-emotion-detection-fanof")
version = project.version(5)
dataset = version.download("yolov11")
```

**Dataset URL**: https://universe.roboflow.com/sina-awwg4/face-emotion-detection-fanof/dataset/5

### Example 2: Your Custom Dataset

```python
from roboflow import Roboflow

# Your API key from account settings
rf = Roboflow(api_key="f2hJNi4Y8VSJSVHT8a66")

# Your workspace and project
project = rf.workspace("your-workspace").project("your-emotion-dataset")
version = project.version(3)
dataset = version.download("yolov11")
```

## Step 6: Running the Script

### Prerequisites

Install Roboflow Python SDK:
```bash
pip install -r roboflow-requirements.txt
# OR manually:
pip install roboflow
```

### Download Dataset

```bash
python dataset-download-roboflow.py
```

**Output**:
- Downloads to `./datasets/` (or current directory)
- Creates YOLOv11-compatible structure:
  ```
  datasets/
  ├── data.yaml           # Dataset configuration
  ├── train/
  │   ├── images/        # Training images
  │   └── labels/        # Training annotations
  ├── valid/
  │   ├── images/
  │   └── labels/
  └── test/
      ├── images/
      └── labels/
  ```

## Understanding Dataset Structure

### data.yaml

This file is used by YOLOv11 for training:

```yaml
path: /path/to/datasets
train: train/images
val: valid/images
test: test/images

nc: 10  # Number of classes (emotions)
names: ['Angry', 'Disgust', 'Excited', 'Fear', 'Happy', 
        'Sad', 'Serious', 'Thinking', 'Worried', 'neutral']
```

### Image Annotations

Each image has a corresponding `.txt` label file in the same directory structure:

```
image.jpg → image.txt

Format (YOLO):
<class_id> <x_center> <y_center> <width> <height>
<class_id> <x_center> <y_center> <width> <height>
...
```

## Troubleshooting

### Issue: "Authentication Failed"
- Verify API key is correct
- Check that API key is from the correct Roboflow account
- Ensure API key has permission to access the project

### Issue: "Project Not Found"
- Verify workspace and project names match the URL
- Check for typos (case-sensitive)
- Confirm project is public or you have access

### Issue: "No Such Format Available"
- Ensure `"yolov11"` is the format requested (not `"yolov8"`)
- Check Roboflow documentation for available formats

### Issue: Downloaded Dataset is Empty
- Verify the version number is correct
- Check that the project has data uploaded
- Confirm sufficient disk space

## Security Best Practices

⚠️ **Important Security Notes**:

1. **Never commit API keys** to version control:
   ```bash
   # Add to .gitignore
   echo "*api*key*" >> .gitignore
   echo "*secret*" >> .gitignore
   ```

2. **Use environment variables** instead:
   ```python
   import os
   api_key = os.getenv('ROBOFLOW_API_KEY')
   rf = Roboflow(api_key=api_key)
   ```

3. **Rotate API keys** regularly in Roboflow account settings

4. **Limit API key permissions** to only required projects

## Next Steps

After downloading the dataset:

1. **Prepare for training**:
   ```bash
   cd ../yolo-model-training
   cp ../roboflow/datasets/* .
   ```

2. **Train the model**:
   ```bash
   python model-training.py --model yolo11n.pt --data data.yaml
   ```

3. **Evaluate and export**:
   - Check training results in `runs/detect/train/`
   - Export best model: `runs/detect/train/weights/best.pt`

## Additional Resources

- **Roboflow Website**: https://roboflow.com
- **Roboflow Universe**: https://universe.roboflow.com
- **Roboflow Docs**: https://docs.roboflow.com
- **YOLOv11 Guide**: https://docs.ultralytics.com/models/yolov11/
- **API Reference**: https://docs.roboflow.com/python-sdk

## Example Workflows

### Workflow 1: Use Public Dataset
```
1. Find dataset in Roboflow Universe
2. Copy workspace/project names from URL
3. Set API key (if required)
4. Run dataset-download-roboflow.py
5. Dataset ready for training
```

### Workflow 2: Create Custom Dataset
```
1. Upload annotated images to Roboflow
2. Configure preprocessing (resize, augment)
3. Export YOLOv11 format
4. Update script with workspace/project/version
5. Run dataset-download-roboflow.py
6. Train custom model
```

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) file for full details.

**MIT License Summary**: You are free to use, modify, and distribute this software for any purpose, provided you include the original license and copyright notice.

## Credits & Acknowledgments

**Created for**: IEEE Mission Tomorrow Career Exploration Event  
**Event**: Hosted by Chamber RVA for 11,000+ eighth graders in Richmond  
**Presented by**: IEEE Region 3 Richmond

**External Dependencies**:
- **Roboflow** — Dataset hosting and distribution platform
- **Roboflow Python SDK** — Dataset downloading and management
- **YOLOv11 Format** — Object detection data format standard
- **OpenCV** — Computer vision library

**Data Sources**:
- Public datasets from Roboflow Universe

**Special Thanks**:
- IEEE Region 3 Richmond for volunteering
- Chamber RVA for organizing Mission Tomorrow
- All educators supporting STEM education

---

**Last Updated**: November 2025  
**Compatible With**: YOLOv11, Roboflow Python SDK 0.2.32+
