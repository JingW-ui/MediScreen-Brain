# MediScreen-Brain Project Documentation

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Core Features](#core-features)
- [Technical Architecture](#technical-architecture)
- [System Modules](#system-modules)
- [User Guide](#user-guide)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)
- [Application Scenarios](#application-scenarios)
- [Output Formats](#output-formats)
- [Important Notes](#important-notes)

---

## Project Overview

**MediScreen-Brain** is a brain tumor detection system based on YOLO deep learning models, designed specifically for medical image analysis. The system provides a graphical user interface (GUI) that supports real-time detection and batch processing of multiple medical imaging formats, automatically identifying tumor regions in brain MRI images and generating detailed diagnostic reports.

### Key Features

- 🎯 **High-Precision Detection**: Based on YOLO object detection algorithm for accurate brain tumor identification
- 🖥️ **User-Friendly Interface**: Modern gradient UI design with intuitive operation
- 📊 **Multi-Format Support**: Supports NIfTI (.nii/.nii.gz), DICOM, and common image formats
- 🔄 **Real-Time Monitoring**: Multi-camera real-time monitoring and recording
- 📈 **Batch Processing**: Folder-based batch detection to improve workflow efficiency
- 📝 **Report Export**: Automatic PDF/Excel diagnostic report generation
- 💾 **History Records**: Automatic save of detection and monitoring snapshots with playback support

---

## Core Features

### 1. Real-time Detection

#### Feature Description
Performs real-time brain tumor detection on single images, video files, or cameras, displaying detection results and statistics instantly.

#### Supported Source Types
- 📷 **Single Image**: JPG, JPEG, PNG, BMP, TIFF, WEBP
- 🎬 **Video File**: MP4, AVI, MOV, MKV, WMV, FLV
- 📹 **Camera**: Multi-camera switching and real-time detection
- 🧠 **NIfTI File**: Professional medical imaging format (.nii, .nii.gz)

#### Detection Workflow
1. Select detection model (supports .pt and .onnx formats)
2. Set confidence threshold (0.01-1.0)
3. Choose detection source (image/video/camera/NIfTI)
4. Click "Start Detection"
5. View real-time detection results and statistics

#### Result Display
- Original image vs. detection result comparison
- Number of detected objects and class distribution
- Confidence range and average
- Estimated tumor diameter (pixels)
- Estimated tumor area (pixels²)

---

### 2. NIfTI Medical Imaging Detection

#### Feature Description
Specialized three-dimensional spatial detection for brain MRI NIfTI format files, providing tri-planar (axial, sagittal, coronal) analysis and detailed measurement data.

#### Detection Steps
1. Load NIfTI file (.nii or .nii.gz)
2. System automatically locates optimal tumor slices
3. Extract key slices from three orthogonal planes
4. Perform tumor detection using specialized models
5. Calculate tumor dimensions and volume

#### Tri-Planar Display
- **Axial**: Horizontal cross-section view
- **Sagittal**: Lateral cross-section view
- **Coronal**: Frontal cross-section view

#### Measurement Metrics
- Maximum tumor diameter in each view (mm)
- Tumor cross-sectional area (mm²)
- Estimated tumor volume (mm³)
- Tumor center coordinates

#### Interactive Functions
- **Select Target**: Display detection results with annotation boxes
- **Remove Annotation**: Show original unannotated image
- **Export Report**: Generate PDF diagnostic report with tri-planar views

---

### 3. Batch Detection

#### Feature Description
Automated batch detection of all images in a folder, suitable for large-scale data processing and statistical analysis.

#### Operation Flow
1. Select "Batch Folder" mode
2. Choose folder containing images
3. System automatically scans all supported image formats
4. Process each image sequentially and record results
5. Generate batch detection report

#### Result Navigation
- ⬅️ Previous / ➡️ Next: Browse detection results
- Display current index (e.g., 5/100)
- Detailed detection information for each image

#### Batch Reports
The system generates reports in two formats:

**TXT Text Report** (`detection_report.txt`)
- Processing time and parameter settings
- Detection details for each image
- Class distribution statistics
- Diameter and area data

**Excel Spreadsheet** (`detection_report.xlsx`)
- **Detection Results Worksheet**:
  - Index, file name
  - Number of detected objects, inference time
  - Confidence, class distribution
  - Maximum diameter, total area
  
- **Summary Worksheet**:
  - Processing time, confidence threshold
  - Total number of images, total detected objects
  - Average inference time

---

### 4. Real-time Monitoring

#### Feature Description
Supports simultaneous multi-camera monitoring, suitable for clinical real-time observation and long-term monitoring scenarios.

#### Monitoring Features
- 📹 **Multi-Camera Support**: Monitor up to 4 cameras simultaneously
- 🖥️ **Dynamic Layout**: Automatic grid layout adjustment based on camera count (1×1, 1×2, 2×2)
- ⏸️ **Pause/Resume**: Individual or batch camera control
- 📊 **Real-Time Statistics**: Display detection count and system time

#### Control Panel
- **Start All**: Launch all selected cameras simultaneously
- **Pause All**: Pause/resume all cameras
- **Stop All**: Stop all camera threads
- **Clear Display**: Clear all display content

#### Camera Management
- Automatic scanning of available cameras
- Display camera resolution and frame rate
- Support hot-plugging and automatic reconnection

---

### 5. Monitoring Snapshot

#### Feature Description
Automatically records detection videos during monitoring, saves as MP4 format with JSON metadata, supporting subsequent playback and analysis.

#### Recording Mechanism
- **Trigger Condition**: Recording only when targets are detected
- **File Format**: MP4 video + JSON metadata
- **Storage Location**: `monitor_history/` and `detection_history/` directories
- **Memory Management**: Automatic cleanup of old records exceeding limits (default 500MB)

#### Snapshot Information
JSON file contains:
```json
{
  "camera_id": 0,
  "source_name": "Camera0",
  "start_time": 1234567890,
  "end_time": 1234567920,
  "fps": 20,
  "total_detections": 150,
  "detection_stats": {
    "tumor": 120,
    "lesion": 30
  },
  "frame_count": 600,
  "mp4_filename": "Camera0_1234567890.mp4",
  "json_filename": "Camera0_1234567890.json"
}
```

#### Playback Function
- 📋 **Snapshot List**: Display all snapshots in reverse chronological order
- ▶️ **Playback Control**: Play, pause, stop, progress bar dragging
- 📊 **Information Display**: Camera name, time, duration, detection statistics
- 🗑️ **Delete Management**: Remove unwanted snapshots
- 📤 **Export Function**: Export snapshots to specified directory

---

### 6. NIfTI Format Conversion

#### Feature Description
Converts three-dimensional NIfTI medical images into two-dimensional PNG image sequences for subsequent processing and visualization.

#### Conversion Settings
- **Slice Direction**: Sagittal, Coronal, Axial
- **Slice Range**: Custom start and end slice indices
- **Output Directory**: Auto-generated (input directory + `_swift_normal`)

#### Preview Function
- Display 5 representative slice previews
- Right-click menu for zoomed detail view
- Scroll wheel for slice browsing

#### Conversion Process
1. Load NIfTI file
2. Select slice direction and range
3. Preview confirmation
4. Execute conversion (with progress bar)
5. Save as PNG sequence

#### File Information Display
- File size, modification date
- Image dimensions (e.g., 256×256×176)
- Data type (e.g., float32)
- Spatial resolution (voxel size)

---

### 7. Diagnostic Report Export

#### PDF Report Function
The system can generate standardized MRI examination reports containing:

**Report Structure**
1. **Title**: MRI Examination Report
2. **Basic Information Table**:
   - Name, Gender, Age
   - Date, ID, Department
   - Examination Site, Sequence, Equipment

3. **Detection Result Images**:
   - Detection image with annotation boxes
   - Tri-planar comparison (NIfTI mode)

4. **Scan Findings**:
   - Lesion type
   - Probability (confidence)
   - Maximum diameter
   - Estimated area/volume

5. **Diagnostic Conclusion**:
   - Diagnostic recommendations based on detection results
   - Lesion nature assessment

6. **Diagnostic Recommendations**:
   - Enhanced scan suggestions
   - Clinical comprehensive judgment
   - Regular follow-up plan

7. **Notes**:
   - Clinical symptom correlation
   - Prompt medical attention reminder
   - Regular monitoring suggestions

8. **Signature Area**:
   - Reporting physician, reviewing physician
   - Report time
   - Hospital name

9. **Disclaimer**:
   - Prototype version declaration
   - Non-diagnostic purpose statement
   - For medical research only

#### Font Support
- Automatic registration of Chinese fonts (Microsoft YaHei)
- Fallback font: simhei
- Support for mixed Chinese-English typesetting

#### Open Function
Option to directly open PDF file after successful export (cross-platform support for Windows/macOS/Linux)

---

## Technical Architecture

### Core Technology Stack

| Technology | Purpose | Version Requirement |
|------------|---------|---------------------|
| **Python** | Programming Language | 3.8+ |
| **PySide6** | GUI Framework | 6.x |
| **Ultralytics YOLO** | Object Detection Model | 8.x |
| **OpenCV** | Image Processing | 4.x |
| **NiBabel** | NIfTI File I/O | 5.x |
| **NumPy** | Numerical Computing | 1.21+ |
| **ReportLab** | PDF Generation | 4.x |
| **OpenPyXL** | Excel File Operations | 3.x |
| **Matplotlib** | Image Visualization | 3.x |
| **Pillow** | Image Processing | 9.x |

### Architecture Design

```
┌─────────────────────────────────────────┐
│         EnhancedDetectionUI             │
│       (Main Window - QMainWindow)        │
├─────────────────────────────────────────┤
│  Tab Widget (Tab Manager)                │
│  ┌──────────┬──────────┬──────────┐    │
│  │ Realtime │  Batch   │ Monitor  │    │
│  │Detection │ Results  │          │    │
│  ├──────────┼──────────┼──────────┤    │
│  │  NIfTI   │ Snapshot │  NIfTI   │    │
│  │Detection │          │Conversion│    │
│  └──────────┴──────────┴──────────┘    │
├─────────────────────────────────────────┤
│         Business Logic Layer            │
│  ┌────────────┐  ┌──────────────┐      │
│  │ModelManager│  │CameraManager │      │
│  └────────────┘  └──────────────┘      │
│  ┌────────────┐  ┌──────────────┐      │
│  │ StyleMgr   │  │ TumorFinder  │      │
│  └────────────┘  └──────────────┘      │
├─────────────────────────────────────────┤
│         Thread Management Layer         │
│  ┌──────────────┐  ┌─────────────┐     │
│  │DetectionThread│ │BatchThread  │     │
│  └──────────────┘  └─────────────┘     │
│  ┌──────────────┐  ┌─────────────┐     │
│  │MonitorThread │ │CameraThread │     │
│  └──────────────┘  └─────────────┘     │
├─────────────────────────────────────────┤
│         Data Persistence Layer          │
│  ┌────────────┐  ┌──────────────┐      │
│  │VideoRecorder│ │SnapshotWidget│      │
│  └────────────┘  └──────────────┘      │
└─────────────────────────────────────────┘
```

### Key Classes Description

#### 1. EnhancedDetectionUI (Main Window)
- **Responsibility**: Integrate all functional modules, manage user interaction
- **Main Methods**:
  - `init_ui()`: Initialize interface
  - `start_detection()`: Start detection
  - `export_report()`: Export PDF report
  - `save_batch_results()`: Save batch results

#### 2. ModelManager (Model Manager)
- **Responsibility**: Scan, load, and manage detection models
- **Supported Formats**: .pt (PyTorch), .onnx (ONNX Runtime)
- **Main Methods**:
  - `scan_models()`: Scan model files
  - `load_model()`: Load specified model

#### 3. CameraManager (Camera Manager)
- **Responsibility**: Detect and manage local camera devices
- **Main Methods**:
  - `scan_cameras()`: Scan available cameras
  - `get_available_cameras()`: Get available camera list

#### 4. DetectionThread (Detection Thread)
- **Responsibility**: Execute detection tasks in background thread to avoid UI freezing
- **Signals**:
  - `result_ready`: Detection result ready
  - `progress_updated`: Progress update
  - `status_changed`: Status change
  - `error_occurred`: Error occurred

#### 5. BatchDetectionThread (Batch Detection Thread)
- **Responsibility**: Handle batch detection of all images in folder
- **Features**: Support pause, stop, progress tracking

#### 6. MultiCameraMonitorThread (Multi-Camera Monitoring Thread)
- **Responsibility**: Simultaneously manage multiple cameras for real-time monitoring
- **Features**:
  - Independent frame rate control
  - Automatic reconnection mechanism
  - Pause/resume support

#### 7. TumorSliceFinder (Tumor Slice Finder)
- **Responsibility**: Locate optimal tumor slices in NIfTI 3D data
- **Algorithm**: Hierarchical search strategy for rapid tumor region localization

#### 8. DetectionVideoRecorder (Detection Recorder)
- **Responsibility**: Record video snapshots of detection process
- **Output**: MP4 video + JSON metadata

#### 9. SnapshotWidget (Snapshot Component)
- **Responsibility**: Manage and playback saved monitoring snapshots
- **Functions**: List display, video playback, information management

---

## System Modules

### Module 1: User Interface Module

#### Style Manager (StyleManager)
Provides unified modern UI styling:
- Gradient backgrounds
- Rounded buttons and controls
- Hover effects and animations
- Responsive layout

#### Main Interface Layout
```
┌──────────────────────────────────────────────────┐
│  Title Bar: MediScreen-Brain Tumor Detection Sys │
├──────────────┬───────────────────────────────────┤
│  Left Control│      Right Display Area            │
│  Panel       │      (Adaptive)                    │
│  (450px)     │                                   │
│              │  ┌─────────────────────────┐    │
│ ┌──────────┐ │  │                         │    │
│ │Model     │ │  │   Tab Display Area      │    │
│ │Config    │ │  │                         │    │
│ └──────────┘ │  │ • Real-time Detection   │    │
│ ┌──────────┐ │  │ • Batch Results         │    │
│ │Source    │ │  │ • NIfTI Detection       │    │
│ │Config    │ │  │ • Real-time Monitoring  │    │
│ └──────────┘ │  │ • Monitoring Snapshot   │    │
│ ┌──────────┐ │  │ • NIfTI Conversion      │    │
│ │Detection │ │  └─────────────────────────┘    │
│ │Control   │ │                                   │
│ └──────────┘ │                                   │
│ ┌──────────┐ │                                   │
│ │Run Log   │ │                                   │
│ └──────────┘ │                                   │
└──────────────┴───────────────────────────────────┘
```

---

### Module 2: Model Management Module

#### Model Scanning
- Automatically scans the following directories:
  - `pt_models/`
  - `onnx_models/`
  - `../models/`
  - `weights/`
- Supports custom model paths

#### Model Loading
```python
# PT Model
model = YOLO("pt_models/best.pt")

# ONNX Model
model = YOLO("onnx_models/segmentation.onnx", task='detect')
```

#### Advanced Model Selection Dialog
- Table display of model information (name, size, modification time, path)
- Support for custom path browsing
- Double-click for quick selection

---

### Module 3: Image Processing Module

#### Image Format Conversion
- BGR ↔ RGB color space conversion
- NumPy array ↔ QImage conversion
- Image scaling and anti-aliasing

#### NIfTI Data Processing
```python
# Load NIfTI file
img = nib.load("brain_scan.nii.gz")
data = img.get_fdata()

# Normalize to uint8
data = np.clip(data, 0, np.percentile(data, 99))
data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)

# Extract slices
axial_slice = data[:, :, slice_index]
sagittal_slice = data[slice_index, :, :]
coronal_slice = data[:, slice_index, :]
```

#### Slice Preview
- Rendered using Matplotlib
- Right-click menu for zoom
- Scroll wheel for slice switching

---

### Module 4: Detection Algorithm Module

#### YOLO Detection Workflow
```python
# Single image detection
results = model(image_path, conf=0.25, verbose=False)

# Get detection boxes
boxes = results[0].boxes
confidences = boxes.conf.cpu().numpy()
classes = boxes.cls.cpu().numpy().astype(int)
xyxy = boxes.xyxy.cpu().numpy()

# Draw results
result_image = results[0].plot()
```

#### Tumor Measurement Calculation
```python
# Diameter calculation (assuming circular)
width = x2 - x1
height = y2 - y1
diameter = (width + height) / 2

# Area calculation (ellipse approximation)
area = 0.785 * width * height  # π/4 ≈ 0.785

# Volume estimation (ellipsoid)
volume = (4/3) * π * (a/2) * (b/2) * (c/2)
```

#### Confidence Level Display
- **High Confidence (>0.8)**: Green background
- **Medium Confidence (0.5-0.8)**: Yellow background
- **Low Confidence (<0.5)**: Red background

---

### Module 5: Video Recording Module

#### CameraVideoRecorder (Monitoring Recording)
```python
recorder = CameraVideoRecorder(
    camera_id=0,
    camera_name="Camera0",
    output_dir=Path("monitor_history"),
    fps=20
)

recorder.start_recording()
recorder.add_frame(frame, detection_info)
recorder.stop_recording()
```

#### DetectionVideoRecorder (Detection Recording)
```python
recorder = DetectionVideoRecorder(
    source_name="Camera0",
    output_dir=Path("detection_history"),
    fps=20
)
```

#### Video Encoding
- Codec: MP4V (H.264)
- Frame rate: Configurable (default 20fps)
- Color space: RGB → BGR conversion

---

### Module 6: Report Generation Module

#### PDF Report Generation
```python
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Table

doc = SimpleDocTemplate("report.pdf", pagesize=A4)
story = []

# Add title, tables, images, etc.
story.append(title)
story.append(info_table)
story.append(detection_image)
story.append(findings_paragraph)

doc.build(story)
```

#### Excel Report Generation
```python
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill

wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Detection Results"

# Set header style
header_font = Font(bold=True, size=12, color="FFFFFF")
header_fill = PatternFill(start_color="3498DB", end_color="3498DB", fill_type="solid")

# Write data
for row_idx, result in enumerate(batch_results, 2):
    ws.cell(row=row_idx, column=1, value=result['file_name'])
    # ... other columns

wb.save("detection_report.xlsx")
```

---

## User Guide

### Quick Start

#### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Or use conda
conda create -n mediscreen python=3.9
conda activate mediscreen
pip install -r requirements.txt
```

#### 2. Model Preparation
Place trained model files in one of the following directories:
```
pt_models/
├── best.pt
└── Brain_Tumor.pt

onnx_models/
├── segmentation.onnx
└── Brain_Tumor.onnx
```

#### 3. Launch Program
```bash
python Brain_Tumor_detection_ui.py
```

---

### Detailed Feature Usage

#### Scenario 1: Single Image Detection

1. **Select Model**
   - Choose pre-loaded model in "Model Configuration" area
   - Or click "🔧" button to select custom model

2. **Set Parameters**
   - Adjust confidence threshold slider (recommended 0.25-0.65)

3. **Select Detection Source**
   - Choose "📷 Single Image" from dropdown menu
   - Click "📁 Select File/Folder" to select image

4. **Start Detection**
   - Click "▶️ Start" button
   - Wait for detection completion

5. **View Results**
   - Left side displays original image
   - Right side displays annotated result
   - Bottom table shows detailed detection data

6. **Export Report**
   - Click "📤 Export Report" button
   - Choose save location
   - PDF report generated automatically

---

#### Scenario 2: NIfTI Medical Image Analysis

1. **Switch to NIfTI Detection Tab**
   - Click top tab "🧠 NIfTI Detection"

2. **Load NIfTI File**
   - In real-time detection tab, select "🧠 NIfTI File"
   - Choose .nii or .nii.gz file
   - Click "▶️ Start" to begin analysis

3. **View Tri-Planar Views**
   - Axial: Horizontal cross-section
   - Sagittal: Lateral cross-section
   - Coronal: Frontal cross-section

4. **Analyze Detection Results**
   - Review tumor information in detection table
   - Focus on diameter, area, confidence

5. **Interactive Operations**
   - Click "📤 Select Target" to show annotation boxes
   - Click "❌ Remove Target" to remove annotations
   - Click "📤 Export Report" to export diagnostic report

6. **View Detailed Slices**
   - Load same file in NIfTI Conversion tab
   - Use scroll wheel to browse different slices
   - Right-click preview image to zoom

---

#### Scenario 3: Batch Folder Processing

1. **Select Batch Mode**
   - Choose "📂 Batch Folder" from dropdown menu

2. **Select Folder**
   - Click "📁 Select File/Folder"
   - Choose folder containing images

3. **Start Batch Detection**
   - Click "▶️ Start"
   - System automatically processes all images

4. **Browse Results**
   - Switch to "📊 Batch Results" tab
   - Use "⬅️ Previous" and "Next ➡️" to browse
   - View detection details for each image

5. **Save Results**
   - Click "💾 Save Results"
   - Choose save directory
   - Automatically generates:
     - Annotated result images
     - TXT text report
     - Excel spreadsheet

---

#### Scenario 4: Multi-Camera Real-time Monitoring

1. **Switch to Monitoring Tab**
   - Click "🖥️ Real-time Monitoring"

2. **Select Cameras**
   - Check cameras to monitor in camera list
   - Click "🔄 Refresh" to refresh device list

3. **Select Model**
   - Ensure detection model is loaded

4. **Start Monitoring**
   - Click "🚀 Start monitoring"
   - Camera feeds display in real-time

5. **Control Monitoring**
   - "⏸️ Pause": Pause/resume monitoring
   - "🗑️ Clear Monitor": Stop and clear

6. **Enable Auto-Recording**
   - Click "🎬 Auto-save Monitoring Snapshots"
   - Set frame rate (recommended 20fps)
   - Set memory limit (default 500MB)
   - Automatic recording when targets detected

7. **View Recorded Snapshots**
   - Switch to "🎬 Monitoring Snapshot" tab
   - Select snapshot and click "▶️ Play" for playback
   - Can export or delete snapshots

---

#### Scenario 5: NIfTI Format Conversion

1. **Switch to Conversion Tab**
   - Click "NIfTI Conversion"

2. **Load File**
   - Click "Browse" to select NIfTI file
   - File information displayed automatically

3. **Set Conversion Parameters**
   - Select slice direction (Sagittal/Coronal/Axial)
   - Set slice range (start-end)
   - Review preview for confirmation

4. **Execute Conversion**
   - Click "🔄 Convert"
   - Wait for conversion completion (with progress bar)

5. **View Results**
   - Output directory auto-generates PNG sequence
   - File naming: slice_0060.png, slice_0061.png...

---

### Keyboard Shortcuts

| Shortcut | Function |
|----------|----------|
| **F2** or **Ctrl+R** | Edit window title |
| **Scroll Wheel** | Switch slices in slice preview |
| **Right-Click** | Show slice zoom menu |

---

## File Structure

```
MediScreen-Brain/
│
├── Brain_Tumor_detection_ui.py          # Main program entry (6355 lines)
├── batch_compare_ui.py                  # Batch comparison tool
├── requirements.txt                     # Python dependency list
├── README.md                            # Project introduction
├── PROJECT_DOCUMENTATION.md             # This document
│
├── pt_models/                           # PyTorch model directory
│   ├── best.pt
│   └── Brain_Tumor.pt
│
├── onnx_models/                         # ONNX model directory
│   ├── segmentation.onnx
│   └── Brain_Tumor.onnx
│
├── utils/                               # Utility functions directory
│   ├── TumorSliceFinder.py             # Tumor slice finder
│   ├── YOLOONNX.py                     # ONNX inference wrapper
│   ├── data_utils/                     # Data processing utilities
│   ├── methods_ui/                     # UI methods
│   └── utils_for_data/                 # Data utilities
│
├── data_test/                           # Test data
│   ├── 545140_swust.nii.gz
│   └── MRBrainTumor2.nii.gz
│
├── monitor_history/                     # Monitoring snapshot storage
│   ├── Camera0_1234567890.mp4
│   └── Camera0_1234567890.json
│
├── detection_history/                   # Detection snapshot storage
│   ├── Camera0_1234567890.mp4
│   └── Camera0_1234567890.json
│
├── best_slices_results_batch_data2_94/  # Batch detection results example
│   ├── [PatientID]_[ScanID]/
│   │   ├── comparison_reports/
│   │   ├── full_search/
│   │   └── hierarchical_search/
│   └── batch_summary/
│
├── compares_resluts/                    # Comparison results
├── detection_history/                   # Detection history
├── img/                                 # Interface screenshots
│   ├── 1.png
│   ├── 2.png
│   └── ui1.png
│
├── build/                               # Build directory
├── dist/                                # Distribution directory
├── .idea/                               # PyCharm configuration
├── .vscode/                             # VSCode configuration
└── .gitignore                           # Git ignore rules
```

---

## Dependencies

### Python Version
- **Recommended**: Python 3.9
- **Minimum**: Python 3.8
- **Compatible**: Python 3.10, 3.11

### Core Dependencies

```txt
# Deep Learning
ultralytics>=8.0.0          # YOLO framework
torch>=1.12.0               # PyTorch
torchvision>=0.13.0         # TorchVision
onnxruntime>=1.12.0         # ONNX inference engine

# GUI Framework
PySide6>=6.4.0              # Qt6 Python bindings

# Image Processing
opencv-python>=4.6.0        # OpenCV
Pillow>=9.0.0               # PIL image processing
matplotlib>=3.5.0           # Plotting library
numpy>=1.21.0               # Numerical computing

# Medical Imaging
nibabel>=5.0.0              # NIfTI file processing

# Report Generation
reportlab>=4.0.0            # PDF generation
openpyxl>=3.0.0             # Excel file operations

# Other Utilities
pathlib                     # Path handling (built-in)
json                        # JSON processing (built-in)
threading                   # Thread management (built-in)
```

### Installation Commands

```bash
# Method 1: Using requirements.txt
pip install -r requirements.txt

# Method 2: Manual installation of core packages
pip install ultralytics PySide6 opencv-python nibabel reportlab openpyxl matplotlib numpy Pillow

# Method 3: Using conda
conda create -n mediscreen python=3.9
conda activate mediscreen
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install ultralytics PySide6 opencv-python nibabel reportlab openpyxl matplotlib numpy Pillow
```

### System Requirements

| Item | Minimum Configuration | Recommended Configuration |
|------|----------------------|---------------------------|
| **Operating System** | Windows 10 / macOS 10.15 / Ubuntu 18.04 | Windows 11 / macOS 12 / Ubuntu 22.04 |
| **CPU** | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| **RAM** | 8 GB RAM | 16 GB RAM |
| **GPU** | Integrated Graphics | NVIDIA GTX 1060+ (4GB VRAM) |
| **Storage** | 2 GB free space | 10 GB SSD |
| **Camera** | Optional (720p) | Recommended (1080p) |

### GPU Acceleration (Optional)

For GPU-accelerated detection:

```bash
# Uninstall CPU version
pip uninstall torch torchvision

# Install CUDA version (example: CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Application Scenarios

### 1. Clinical Auxiliary Diagnosis
- **Purpose**: Help radiologists quickly locate brain tumors
- **Advantage**: Reduce missed diagnoses, improve diagnostic efficiency
- **Workflow**:
  1. Import patient MRI scans (NIfTI format)
  2. System automatically detects tumor locations
  3. Generate tri-planar analysis report
  4. Physician reviews and signs off

### 2. Research Data Analysis
- **Purpose**: Large-scale medical image dataset annotation and statistics
- **Advantage**: Batch processing, automatic Excel statistical reports
- **Workflow**:
  1. Prepare dataset folder
  2. Start batch detection
  3. Wait for automatic processing completion
  4. Export structured data for statistical analysis

### 3. Teaching Demonstration
- **Purpose**: Medical image recognition teaching and student training
- **Advantage**: Visual interface, real-time feedback
- **Workflow**:
  1. Prepare typical case images
  2. Real-time demonstration in class
  3. Explain detection results and confidence
  4. Student hands-on practice

### 4. Remote Medical Monitoring
- **Purpose**: Multi-location real-time video monitoring and anomaly detection
- **Advantage**: Multi-camera support, automatic recording of abnormal segments
- **Workflow**:
  1. Deploy multiple cameras
  2. Start real-time monitoring mode
  3. System automatically detects and records
  4. Post-event playback and analysis

### 5. Preoperative Planning
- **Purpose**: Pre-surgical tumor location and size assessment
- **Advantage**: Precise measurement, three-dimensional spatial localization
- **Workflow**:
  1. Load patient's latest MRI data
  2. Analyze tumor position in tri-planar views
  3. Measure tumor diameter and volume
  4. Export report for surgical team reference

---

## Output Formats

### 1. Detection Results Table

| Column | Description | Example |
|--------|-------------|---------|
| Class | Detection category | tumor, lesion |
| Confidence | Confidence level | 0.923 |
| Coordinates (x,y) | Top-left corner coordinates | (120,85) |
| Size (w×h) | Bounding box dimensions | 45×38 |
| Estimated Diameter | Estimated diameter | 41.50 pixel |
| Pixel Area | Pixel area | 1343.70 pixel² |

### 2. PDF Report Structure

```
MRI Examination Report
═══════════════════════════

[Basic Information Table]
Name: ______  Gender: ______  Age: ______
Date: 2024-01-15  ID: ______  Department: ______
Site: Brain  Sequence: T1/T2  Equipment: MRI Scanner

[Detection Result Image]
┌─────────────────┐
│ Annotated Image │
└─────────────────┘

Scan Findings:
tumor Possibility: 0.923, Maximum diameter: 41.50pixel, 
Estimated pixel area: 1343.70pixel²

Diagnostic Conclusion:
Based on imaging findings, brain tumor lesion is considered. 
Further examination is recommended for confirmation.

Diagnostic Recommendations:
1. Enhanced scan is recommended to further clarify the nature of the lesion
2. Comprehensive judgment based on clinical symptoms and laboratory test results
3. Regular follow-up to observe changes in the lesion

Notes:
1. Please conduct comprehensive analysis in combination with clinical symptoms
2. If you feel unwell, please seek medical attention promptly
3. Regular follow-up is recommended to monitor disease changes

[Signature Area]
Reporting Physician: _______________
Reviewing Physician: _______________
Report Time: 2024-01-15 14:30:25
Hospital Name: XXX Hospital Imaging Department

Disclaimer:
This software is a prototype version and is not designed or intended for use 
in diagnosis or classification of any medical problems or other medical purposes.
```

### 3. Excel Report Format

**Detection Results Worksheet**

| Index | File Name | Detected Objects | Inference Time (s) | Confidence | Class Distribution | Max Diameter (pixel) | Total Area (pixel²) |
|-------|-----------|------------------|-------------------|------------|-------------------|---------------------|-------------------|
| 1 | patient_001.jpg | 2 | 0.045 | 0.856 - 0.923 | tumor:2 | 45.50 | 1580.25 |
| 2 | patient_002.jpg | 1 | 0.038 | 0.789 | lesion:1 | 32.00 | 804.25 |
| ... | ... | ... | ... | ... | ... | ... | ... |

**Summary Worksheet**

| Metric | Value |
|--------|-------|
| Processing Time | 2024-01-15 14:30:25 |
| Confidence Threshold | 0.25 |
| Number of Images | 100 |
| Total Detected Objects | 87 |
| Average Inference Time (s) | 0.042 |

### 4. JSON Metadata Format

```json
{
  "camera_id": 0,
  "source_name": "Camera0",
  "start_time": 1705312225,
  "end_time": 1705312255,
  "fps": 20,
  "total_detections": 150,
  "detection_stats": {
    "tumor": 120,
    "lesion": 30
  },
  "frame_count": 600,
  "mp4_filename": "Camera0_1705312225.mp4",
  "json_filename": "Camera0_1705312225.json"
}
```

### 5. Log Format

```
[14:30:25] 🚀 MediScreen-Brain Tumor Detection System started
[14:30:26] ✅ Model loaded successfully: best.pt
[14:30:30] 📁 Selected: H:\data\patient_001.jpg
[14:30:31] 🚀 Starting image detection...
[14:30:31] 🎯 Detected 2 objects: tumor:2 (Time: 0.045s)
[14:30:35] 📤 Exporting report...
[14:30:36] ✅ Report exported successfully: MRI_Diagnostic_Report_20240115_143036.pdf
```

---

## Important Notes

### ⚠️ Important Declarations

1. **Not a Medical Device**
   - This software is a **prototype version** for medical research and teaching demonstration only
   - **NOT suitable** for clinical diagnosis, disease classification, or other medical purposes
   - Users acknowledge and warrant not to use this software for the above purposes

2. **Disclaimer**
   - Software provided "as is" without warranty of any kind
   - Developers not liable for any losses resulting from software use
   - All detection results for reference only, must be confirmed by professional physicians

3. **Data Security**
   - Local processing: All data processed locally, not uploaded to cloud
   - Privacy protection: Patient data should be anonymized
   - Compliance requirements: Follow HIPAA/GDPR and other data protection regulations

### 🔧 Usage Recommendations

1. **Model Selection**
   - Use thoroughly validated trained models
   - Regularly update models to improve accuracy
   - Adjust confidence thresholds for different equipment

2. **Image Quality**
   - Ensure input images are clear and artifact-free
   - NIfTI files should be standard format
   - Recommend high-resolution scans (≥1mm isotropic)

3. **Performance Optimization**
   - Close other large applications during batch processing
   - Use SSD storage for faster read/write speeds
   - GPU acceleration significantly improves detection speed

4. **Storage Space**
   - Regularly clean `monitor_history/` and `detection_history/` directories
   - Adjust memory limit parameters to avoid disk full
   - Backup important data promptly

5. **Camera Usage**
   - Ensure camera drivers are correctly installed
   - Windows systems recommend CAP_DSHOW backend
   - Adequate lighting improves detection accuracy

### 🐛 Common Issues

#### Q1: Model Loading Failure
**Solution**:
- Check if model file path is correct
- Confirm file format (.pt or .onnx)
- Verify PyTorch/ONNX Runtime version compatibility

#### Q2: Slow Detection Speed
**Solution**:
- Reduce input image resolution
- Use GPU acceleration
- Narrow confidence threshold range

#### Q3: NIfTI File Cannot Open
**Solution**:
- Confirm file is not corrupted
- Check nibabel library version
- Try verifying file with other software (e.g., ITK-SNAP)

#### Q4: Camera Not Recognized
**Solution**:
- Check camera connection
- Update camera drivers
- Try different USB port
- Restart application

#### Q5: Chinese Characters Garbled in PDF Report
**Solution**:
- Confirm Microsoft YaHei font exists in system
- Check `C:\Windows\Fonts\msyh.ttc` path
- Modify font path in code to actual location

#### Q6: Batch Processing Interrupted
**Solution**:
- Check folder permissions
- Confirm sufficient disk space
- View logs to locate specific error files
- Process large datasets in batches

### 📞 Technical Support

If you encounter issues:
1. Check run logs (log area at bottom of interface)
2. Examine console output error messages
3. Confirm dependency library versions meet requirements
4. Consult this project documentation and related library official documentation

---

## Version History

### v2.0 (Current Version)
- ✨ Brand new gradient UI design
- 📹 Multi-camera real-time monitoring
- 🎬 Automatic monitoring snapshot recording
- 📊 Batch detection Excel reports
- 🧠 NIfTI tri-planar analysis
- 📤 PDF diagnostic report export
- 🔄 NIfTI format conversion tool
- 📋 Snapshot playback and management

### v1.x (Early Versions)
- Basic single image detection
- Simple GUI interface
- Basic logging functionality

---

## License

This project is for academic research and teaching use only. For commercial use, please contact the author for authorization.

---

## Contact Information

- **Project Name**: MediScreen-Brain
- **Development Language**: Python 3.9+
- **GUI Framework**: PySide6
- **Detection Model**: YOLO (Ultralytics)

### Team & Developer

- **Research Team**: [Neuroengineering Lab, UESTC](https://www.neuro.uestc.edu.cn/)
- **Developer Portfolio**: [JingW-ui](https://jingw-ui.github.io/resume/)、[MYiu](https://github.com/MYiuCode)

---

**Last Updated**: 2026-05-13  
**Document Version**: v2.0  
**Applicable Software Version**: MediScreen-Brain v2.0+

---

*This document was generated with AI assistance, based on project code analysis. If there are errors or omissions, please refer to the actual code.*
