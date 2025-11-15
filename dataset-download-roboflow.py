"""
Roboflow Emotion Detection Dataset Downloader
==============================================

This script downloads emotion detection datasets from Roboflow Universe
in YOLOv11-compatible format for training and testing.

Prerequisites:
- Python 3.8+
- roboflow Python SDK: pip install -r roboflow-requirements.txt

Configuration:
- Update API_KEY with your Roboflow account API key
- Update WORKSPACE_NAME with the Roboflow workspace
- Update PROJECT_NAME with the Roboflow project name
- Update VERSION with the desired dataset version

For detailed setup instructions, see ROBOFLOW_GUIDE.md
"""

from roboflow import Roboflow

# ============================================================================
# CONFIGURATION - UPDATE THESE VALUES
# ============================================================================

# Your Roboflow API Key
# Get from: https://roboflow.com/settings/account
# WARNING: Keep this secret, do NOT commit to public repositories
# Consider using environment variables instead: os.getenv('ROBOFLOW_API_KEY')
API_KEY = "f2hJNi4Y8VSJSVHT8a66"

# Roboflow Workspace Name
# Found in URL: https://universe.roboflow.com/WORKSPACE_NAME/project-name/...
WORKSPACE_NAME = "sina-awwg4"

# Roboflow Project Name
# Found in URL: https://universe.roboflow.com/workspace/PROJECT_NAME/...
PROJECT_NAME = "face-emotion-detection-fanof"

# Dataset Version Number
# Found in the dataset URL or Roboflow dashboard Versions tab
# Example dataset URL:
# https://universe.roboflow.com/sina-awwg4/face-emotion-detection-fanof/dataset/5
DATASET_VERSION = 5

# Output Format (YOLOv11 compatible)
# Options: "yolov11", "yolov8", "coco", "pascal", "vott", etc.
FORMAT = "yolov11"

# ============================================================================
# DOWNLOAD DATASET
# ============================================================================

print("="*70)
print("Roboflow Emotion Detection Dataset Downloader")
print("="*70)

print(f"\nConfiguration:")
print(f"  Workspace: {WORKSPACE_NAME}")
print(f"  Project: {PROJECT_NAME}")
print(f"  Version: {DATASET_VERSION}")
print(f"  Format: {FORMAT}")
print()

try:
    # Initialize Roboflow client with API key
    # This authenticates your request to access Roboflow API
    print("Authenticating with Roboflow...")
    rf = Roboflow(api_key=API_KEY)
    
    # Access the workspace (organization/team container)
    print(f"Accessing workspace: {WORKSPACE_NAME}")
    
    # Access the specific project within the workspace
    print(f"Accessing project: {PROJECT_NAME}")
    project = rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME)
    
    # Retrieve the specific dataset version
    # Different versions may have different preprocessing/augmentation
    print(f"Retrieving dataset version: {DATASET_VERSION}")
    version = project.version(DATASET_VERSION)
    
    # Download dataset in YOLOv11 format
    # This creates a directory structure:
    # - data.yaml (dataset config)
    # - train/ (training images and labels)
    # - valid/ (validation images and labels)
    # - test/ (test images and labels)
    print(f"\nDownloading dataset in {FORMAT} format...")
    print("This may take several minutes depending on dataset size...")
    dataset = version.download(FORMAT)
    
    print("\n" + "="*70)
    print("✓ Dataset downloaded successfully!")
    print("="*70)
    print(f"\nDataset location: {dataset.location}")
    print(f"Data configuration: {dataset.location}/data.yaml")
    print("\nNext steps:")
    print("1. Review data.yaml to verify class names and paths")
    print("2. Copy dataset to model-training directory")
    print("3. Run model training: python model-training.py")
    print("="*70)

except Exception as e:
    print("\n" + "="*70)
    print("❌ Error downloading dataset")
    print("="*70)
    print(f"Error: {str(e)}")
    print("\nTroubleshooting:")
    print("- Verify API_KEY is correct (check Roboflow account settings)")
    print("- Verify WORKSPACE_NAME matches the URL")
    print("- Verify PROJECT_NAME matches the URL")
    print("- Verify DATASET_VERSION exists in the project")
    print("- Check internet connection")
    print("\nFor detailed instructions, see ROBOFLOW_GUIDE.md")
    print("="*70)
    exit(1)

# ============================================================================
# REFERENCE: EXAMPLE ROBOFLOW UNIVERSE DATASETS
# ============================================================================

# Example Public Datasets:
# 1. Facial Emotion Dataset (Public)
#    URL: https://universe.roboflow.com/sina-awwg4/face-emotion-detection-fanof/dataset/5
#    Workspace: sina-awwg4
#    Project: face-emotion-detection-fanof
#    Version: 5

# 2. Another Example Dataset:
#    URL: https://universe.roboflow.com/workenv-dayet/facial-emotion-dataset-7g1jd-hipbk/dataset/3
#    Workspace: workenv-dayet
#    Project: facial-emotion-dataset-7g1jd-hipbk
#    Version: 3

# To use a different dataset, update the three configuration variables above
# with the values extracted from the Roboflow Universe URL

