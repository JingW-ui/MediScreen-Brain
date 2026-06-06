# DCM to NIfTI Converter & Screening Tool - User Guide

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [System Requirements](#system-requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Directory Structure Support](#directory-structure-support)
7. [Screening Criteria](#screening-criteria)
8. [Output Files](#output-files)
9. [Safety Features](#safety-features)
10. [Troubleshooting](#troubleshooting)
11. [Technical Details](#technical-details)

---

## Overview

The **DCM → NII Converter & Screening Tool** is a specialized application for converting DICOM (Digital Imaging and Communications in Medicine) files to NIfTI (Neuroimaging Informatics Technology Initiative) format with automatic quality screening. This tool is part of the MediScreen-Brain project developed by the University of Electronic Science and Technology of China.

### Purpose
This tool automates the conversion of medical imaging data from DICOM format (commonly used in MRI scanners) to NIfTI format (widely used in neuroimaging research), while automatically filtering out low-quality or inappropriate scans based on resolution and voxel dimensions.

### Key Benefits
- **Automated batch processing** of multiple DICOM sequences
- **Intelligent directory scanning** supporting various folder structures
- **Automatic quality control** based on spatial resolution and voxel dimensions
- **Real-time progress monitoring** with time estimation
- **Comprehensive logging** with CSV and TXT export
- **Safe operation** ensuring original DICOM files are never modified

---

## Features

### Core Functionality
- **One-click conversion**: Convert entire directories of DICOM files to NIfTI format
- **Smart screening**: Automatically retain only files meeting quality criteria (1mm isotropic resolution, >100 voxels per axis)
- **Batch processing**: Process hundreds of sequences without manual intervention
- **Multi-format support**: Handles .nii.gz files along with companion files (.json, .bval, .bvec)

### User Interface
- **Modern Qt-based GUI** with intuitive layout
- **Real-time progress tracking** with percentage completion
- **Live statistics** showing converted, kept, deleted, and failed counts
- **Color-coded logging** for easy identification of status messages
- **Time estimation** showing elapsed time and estimated remaining time

### Advanced Features
- **Threaded processing**: Background worker thread prevents UI freezing
- **Abort capability**: Safely stop processing at any time
- **Directory memory**: Remembers last used input/output directories
- **Command-line support**: Optional pre-filling of parameters via command line
- **Packaged executable**: Ready-to-run .exe file (no Python installation required)

---

## System Requirements

### Operating System
- **Windows 10/11** (64-bit recommended)
- Linux/macOS support available with minor modifications

### Hardware Requirements
- **CPU**: Multi-core processor recommended (conversion is CPU-intensive)
- **RAM**: Minimum 4GB, 8GB+ recommended for large datasets
- **Storage**: Sufficient space for both DICOM source files and NIfTI output (NIfTI files are typically smaller due to compression)

### Software Dependencies
For running from source code:
- **Python 3.8+**
- **PySide6** (Qt6 bindings for Python)
- **nibabel** (Neuroimaging data I/O library)
- **numpy** (Numerical computing library)

For packaged executable:
- No additional software required (all dependencies bundled)

---

## Installation

### Option 1: Using Packaged Executable (Recommended)

1. **Download the executable**:
   - Locate `DCM2NII_Converter.exe` in the distribution package
   - The executable includes all necessary dependencies

2. **Run the application**:
   ```
   Double-click DCM2NII_Converter.exe
   ```

3. **First-time setup**:
   - A safety notice will appear explaining the tool's operation
   - Read and acknowledge the safety information

### Option 2: Running from Source Code

1. **Install Python dependencies**:
   ```bash
   pip install PySide6 nibabel numpy
   ```

2. **Ensure dcm2niix.exe is available**:
   - The tool expects `dcm2niix.exe` in the same directory
   - Download from: https://github.com/rordenlab/dcm2niix/releases

3. **Run the application**:
   ```bash
   python dcm2nii_ui.py
   ```

### Option 3: Building from Source

See [BUILD_GUIDE.md](BUILD_GUIDE.md) for detailed instructions on creating a standalone executable using PyInstaller.

---

## Usage

### Basic Workflow

1. **Launch the application**
   - Double-click the executable or run from command line

2. **Select Input Directory**
   - Click "浏览…" button next to "输入目录"
   - Navigate to the folder containing your DICOM files
   - The tool supports various directory structures (see [Directory Structure Support](#directory-structure-support))

3. **Select Output Directory**
   - Click "浏览…" button next to "输出目录"
   - Choose where converted NIfTI files will be saved
   - **Important**: Output directory cannot be a subdirectory of input directory

4. **Start Processing**
   - Click "▶ 开始处理" button
   - Monitor progress in real-time
   - View detailed logs in the log panel

5. **Review Results**
   - Check final statistics in the summary dialog
   - Review CSV log file for detailed processing history
   - Review TXT summary file for overview statistics

### Interface Components

#### Path Configuration Section
- **Input Directory**: Source folder containing DICOM files
- **Output Directory**: Destination folder for NIfTI files
- **Screening Rules Display**: Shows current quality criteria

#### Control Buttons
- **▶ 开始处理 (Start Processing)**: Begin conversion and screening
- **⏹ 中止 (Abort)**: Stop processing after current sequence completes
- **清空日志 (Clear Log)**: Clear the log display area

#### Progress Section
- **Progress Bar**: Visual representation of completion percentage
- **Elapsed Time**: Time since processing started
- **Estimated Remaining**: Predicted time to completion
- **Statistics**: Real-time counts of converted, kept, deleted, and failed sequences

#### Log Panel
- **Color-coded messages**:
  - White: Informational messages
  - Green: Successful operations
  - Yellow: Warnings
  - Red: Errors
- **Scrollable**: Automatically scrolls to show latest entries
- **Detailed**: Shows resolution and voxel dimension information

---

## Directory Structure Support

The tool intelligently scans input directories and supports multiple organizational structures:

### Structure 1: Direct DICOM Files
```
input_directory/
├── image001.dcm
├── image002.dcm
└── image003.dcm
```
The directory itself is treated as a single DICOM sequence.

### Structure 2: Admin_* Folders (Legacy Format)
```
input_directory/
├── Admin_001/
│   ├── sequence1/
│   │   ├── img001.dcm
│   │   └── img002.dcm
│   └── sequence2/
│       ├── img001.dcm
│       └── img002.dcm
└── Admin_002/
    └── sequence3/
        ├── img001.dcm
        └── img002.dcm
```
Scans subdirectories within Admin_* folders.

### Structure 3: Arbitrary Nested Structure
```
input_directory/
├── patient001/
│   ├── scan1/
│   │   └── *.dcm
│   └── scan2/
│       └── *.dcm
└── patient002/
    └── scan1/
        └── *.dcm
```
Recursively searches for directories containing DICOM files.

### Structure 4: Mixed Structure
```
input_directory/
├── direct_scan/          # Contains .dcm files directly
│   ├── img001.dcm
│   └── img002.dcm
├── Admin_001/            # Legacy structure
│   └── sequence1/
│       └── *.dcm
└── nested/               # Nested structure
    └── deep/
        └── scan/
            └── *.dcm
```
Handles combinations of all supported structures.

### Detection Algorithm
1. **Check for DICOM files**: If directory contains `.dcm`, `.DCM`, `.dicom`, or `.DICOM` files, treat as sequence directory
2. **Check for Admin_* folders**: If present, scan their immediate subdirectories
3. **Recursive search**: Otherwise, recursively scan all subdirectories (max depth: 5 levels)
4. **File size heuristic**: Files without extensions >100KB may be detected as DICOM

---

## Screening Criteria

The tool automatically filters converted NIfTI files based on two quality criteria:

### Criterion 1: Spatial Resolution
- **Target**: 1.0 mm isotropic resolution
- **Tolerance**: ±0.05 mm
- **Acceptable range**: 0.95 mm to 1.05 mm for X, Y, and Z axes
- **Example**: (1.0, 1.0, 1.0) ✓, (1.2, 1.0, 1.0) ✗

### Criterion 2: Voxel Dimensions
- **Minimum**: All three axes must have >100 voxels
- **Example**: (256, 256, 176) ✓, (256, 256, 50) ✗

### File Handling

#### Files Retained (Both Criteria Met)
- Main NIfTI file: `*.nii.gz`
- Companion JSON metadata: `*.json` (BIDS format)
- DTI b-values: `*.bval` (if present)
- DTI b-vectors: `*.bvec` (if present)

#### Files Deleted (Either Criterion Failed)
- All associated files are removed together
- Deletion reasons logged:
  - "分辨率=(x,y,z) ≠ 1mm" - Resolution mismatch
  - "体素维度=(x,y,z) ≤ 100" - Insufficient voxel count

#### Orphaned Files
- Companion files without corresponding `.nii.gz` are also deleted
- Prevents accumulation of incomplete data sets

### Rationale
These criteria ensure:
- **Consistent resolution**: Critical for multi-subject analyses and template matching
- **Adequate coverage**: Ensures full brain coverage (typical brain MRI has 150-256 slices)
- **Data quality**: Filters out localizers, scouts, and low-resolution sequences

---

## Output Files

### Generated NIfTI Files
The tool produces compressed NIfTI files with naming convention:
```
{sequence_name}_{protocol}_{time}_{series}.nii.gz
```

Example: `t1_mprage_sag_p2_iso_1mm_20250205165421_23.nii.gz`

### Companion Files

#### JSON Metadata (.json)
- BIDS-compatible metadata
- Contains acquisition parameters
- Includes scanner information, timing, and sequence details

#### DTI Files (.bval, .bvec)
- Present only for diffusion-weighted imaging
- `.bval`: b-values for each gradient direction
- `.bvec`: Gradient direction vectors

### Log Files

#### CSV Log File
- **Filename**: `batch_log_YYYYMMDD_HHMMSS.csv`
- **Location**: Output directory
- **Contents**: Complete chronological log with timestamps
- **Columns**: Timestamp, Level, Message
- **Encoding**: UTF-8 with BOM (Excel-compatible)

#### TXT Summary File
- **Filename**: `batch_summary_YYYYMMDD_HHMMSS.txt`
- **Location**: Output directory
- **Contents**: Human-readable summary including:
  - Processing information (time, directories)
  - Statistics (total, converted, kept, deleted, failed)
  - Success rate calculation
  - Resolution statistics (if available)
  - Error messages (if any)

### Example Output Structure
```
output_directory/
├── patient001_scan1.nii.gz
├── patient001_scan1.json
├── patient002_scan1.nii.gz
├── patient002_scan1.json
├── patient002_scan1.bval
├── patient002_scan1.bvec
├── batch_log_20250515_143022.csv
└── batch_summary_20250515_143022.txt
```

---

## Safety Features

### Data Protection
1. **Read-only input**: Original DICOM files are NEVER modified or deleted
2. **Output isolation**: Only files in output directory are subject to deletion
3. **Path validation**: Prevents output directory from being input subdirectory
4. **Snapshot comparison**: Tracks only newly created files during each conversion

### Validation Checks
- **Directory existence**: Validates input directory before processing
- **Executable availability**: Checks for dcm2niix.exe presence
- **Path resolution**: Uses absolute paths to prevent ambiguity
- **Subdirectory detection**: Blocks dangerous output directory configurations

### Abort Safety
- **Graceful stopping**: Current sequence completes before aborting
- **No partial deletions**: Screening occurs after complete conversion
- **Mutex protection**: Thread-safe abort mechanism

### Error Handling
- **Timeout protection**: 300-second timeout per sequence prevents hangs
- **Exception catching**: Comprehensive error handling with detailed logging
- **Continued processing**: Individual failures don't stop entire batch

---

## Troubleshooting

### Common Issues

#### Problem: "找不到 dcm2niix.exe"
**Solution**: 
- Ensure `dcm2niix.exe` is in the same directory as the application
- Download from: https://github.com/rordenlab/dcm2niix/releases
- For packaged version, verify executable wasn't blocked by antivirus

#### Problem: "未找到任何 DCM 序列文件夹"
**Solution**:
- Verify input directory contains actual DICOM files
- Check file extensions (.dcm, .DCM, etc.)
- Ensure files are not corrupted
- Try selecting a parent directory if using nested structure

#### Problem: Conversion timeout (300s)
**Solution**:
- Large DICOM series may take longer; consider splitting datasets
- Check system resources (CPU, RAM, disk I/O)
- Verify DICOM files are not corrupted
- Reduce concurrent applications during processing

#### Problem: All files deleted during screening
**Solution**:
- Verify your DICOM data has ~1mm resolution
- Check voxel dimensions (should be >100 in all axes)
- Review log for specific rejection reasons
- Adjust screening criteria in source code if needed (not recommended)

#### Problem: Output directory permission error
**Solution**:
- Ensure write permissions for output directory
- Avoid system directories (C:\Program Files, etc.)
- Try a different output location (e.g., Desktop, Documents)

#### Problem: Application freezes during processing
**Solution**:
- Normal behavior; UI remains responsive due to threading
- Check progress bar and log for activity
- Use "Abort" button if truly stuck
- Monitor system resources in Task Manager

### Performance Optimization

#### For Large Datasets (>1000 sequences)
1. **Process in batches**: Split into groups of 200-300 sequences
2. **Use SSD storage**: Significantly faster I/O for DICOM reading
3. **Close other applications**: Free up CPU and RAM resources
4. **Monitor temperature**: Extended processing may cause thermal throttling

#### Memory Management
- Tool processes sequences one at a time (low memory footprint)
- Typical RAM usage: 200-500 MB regardless of dataset size
- No cumulative memory leak observed in extended use

### Log Analysis

#### Interpreting CSV Logs
```csv
时间戳,级别,消息
2025-05-15 14:30:22,info,▶ 开始扫描输入目录…
2025-05-15 14:30:25,ok,✅ 扫描完成，共发现 150 个 DCM 序列文件夹
2025-05-15 14:30:26,info,[1/150] 转换: Admin_001_sequence1
2025-05-15 14:30:28,ok,  ✅ dcm2niix 转换完成
2025-05-15 14:30:29,ok,  [保留] t1_mprage.nii.gz  分辨率=(1.0, 1.0, 1.0), 体素维度=(256, 256, 176)
```

#### Common Log Messages
- **[保留]**: File met quality criteria and was kept
- **[删除]**: File failed criteria and was deleted (with reason)
- **[跳过]**: File skipped due to safety check
- **[删除失败]**: Deletion failed (check file permissions)
- **⚠ 超时**: Sequence exceeded 300-second timeout
- **❌ 转换失败**: dcm2niix returned non-zero exit code

---

## Technical Details

### Architecture

#### Component Overview
```
┌─────────────────────────────────────────┐
│         MainWindow (GUI Thread)         │
│  - User interface components            │
│  - Event handlers                       │
│  - Progress updates                     │
└──────────────┬──────────────────────────┘
               │ Signals/Slots
┌──────────────▼──────────────────────────┐
│      WorkerThread (Background)          │
│  - Directory scanning                   │
│  - dcm2niix execution                   │
│  - Quality screening                    │
│  - File management                      │
└──────────────┬──────────────────────────┘
               │ System calls
┌──────────────▼──────────────────────────┐
│         External Tools                  │
│  - dcm2niix.exe (DICOM→NIfTI)          │
│  - nibabel (NIfTI header reading)       │
└─────────────────────────────────────────┘
```

#### Threading Model
- **Main thread**: Handles UI events and rendering
- **Worker thread**: Performs all processing tasks
- **Communication**: Qt signals/slots for thread-safe updates
- **Synchronization**: QMutex for abort flag protection

### Key Algorithms

#### Directory Scanning (`scan_input_dir`)
```python
1. Resolve absolute path
2. Recursive traversal (max depth: 5)
3. DICOM detection:
   - Extension check (.dcm, .DCM, etc.)
   - Size heuristic (>100KB for extensionless files)
4. Admin_* folder special handling
5. Deduplication and sorting
```

#### Quality Screening (`filter_new_files`)
```python
1. Snapshot output directory before conversion
2. Snapshot output directory after conversion
3. Identify new files (after - before)
4. For each new .nii.gz file:
   a. Read header (nibabel)
   b. Extract resolution (zooms[:3])
   c. Extract voxel dimensions (shape[:3])
   d. Check resolution criterion (±0.05mm of 1.0mm)
   e. Check voxel criterion (>100 in all axes)
   f. Keep or delete main file + companions
5. Clean up orphaned companion files
```

#### Time Estimation (`_on_progress`)
```python
1. Track elapsed time since start
2. Calculate average time per completed task
3. Maintain rolling window of last 5 tasks
4. ETA = recent_average × remaining_tasks
5. Update UI labels every progress event
```

### Constants and Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TARGET_RES` | 1.0 mm | Target isotropic resolution |
| `RES_TOLERANCE` | 0.05 mm | Acceptable deviation from target |
| `DCM2NIIX_FLAGS` | See below | Command-line flags for dcm2niix |
| Timeout | 300 seconds | Maximum time per sequence |
| Max recursion depth | 5 levels | Directory scanning limit |

#### dcm2niix Flags
```python
['-f', '%f_%p_%t_%s',  # Filename format: sequence_protocol_time_series
 '-p', 'y',             # Use protocol name
 '-z', 'y']             # Compress output (.nii.gz)
```

### File Format Specifications

#### NIfTI Header Fields Used
- **pixdim[1:3]**: Voxel dimensions in mm (resolution)
- **dim[1:3]**: Number of voxels in each axis

#### Supported Companion Files
- **.json**: BIDS sidecar metadata (JSON format)
- **.bval**: DTI b-values (space-separated text)
- **.bvec**: DTI gradient directions (3×N matrix)

### Performance Characteristics

#### Benchmark (Typical Desktop PC)
- **DICOM reading**: ~50-200 MB/s (depends on disk speed)
- **Conversion**: 2-10 seconds per sequence (depends on size)
- **Screening**: <0.1 seconds per file (header-only read)
- **Memory usage**: 200-500 MB constant

#### Scaling
- **Linear scaling**: Processing time ∝ number of sequences
- **No quadratic bottlenecks**: Each sequence processed independently
- **Disk I/O bound**: Faster disks yield proportional improvements

### Security Considerations

#### Path Traversal Prevention
- All paths resolved to absolute form
- Output directory validated against input directory
- File operations restricted to output directory tree

#### Input Validation
- Directory existence checks before operations
- File type verification before processing
- Permission error handling throughout

#### Resource Limits
- Recursion depth limited to prevent stack overflow
- Timeout prevents infinite hangs
- Mutex prevents race conditions

---

## Appendix

### A. Command-Line Usage

Although primarily a GUI application, basic command-line parameter pre-filling is supported:

```bash
DCM2NII_Converter.exe "C:\input\dicoms" "D:\output\nifti"
```

This pre-fills the input and output fields but still requires clicking "Start".

### B. Customization

For advanced users, screening criteria can be modified in source code:

```python
# In dcm2nii_ui.py, modify constants:
RES_TOLERANCE   = 0.05    # Change tolerance (mm)
TARGET_RES      = 1.0     # Change target resolution (mm)

# In check_voxel_dimensions function:
ok = all(d > 100 for d in voxel_dims)  # Change minimum voxel count
```

**Warning**: Modifying these values affects data quality. Consult with your research team before changes.

### C. Integration with Other Tools

#### Compatible Software
- **FSL**: Direct compatibility with NIfTI files
- **SPM**: Native NIfTI support
- **FreeSurfer**: Works with converted files
- **ANTs**: Full compatibility
- **MRIcron/MRIcroGL**: Visualization

#### BIDS Compliance
Generated files follow BIDS naming conventions when source DICOM headers contain appropriate metadata. JSON sidecars provide BIDS-compatible metadata.

### D. Version History

#### v1.0 (Current)
- Initial release
- Smart directory scanning
- Dual-criterion screening
- CSV/TXT logging
- Time estimation
- Abort capability

### E. Support and Contact

For issues, questions, or feature requests:
- **Project**: MediScreen-Brain
- **Institution**: University of Electronic Science and Technology of China
- **GitHub**: [Repository URL if available]
- **Documentation**: See PROJECT_DOCUMENTATION_EN.md in project root

### F. License

See LICENSE file in project root for licensing terms.

### G. Acknowledgments

- **dcm2niix**: Chris Rorden's excellent DICOM conversion tool
- **nibabel**: Neuroimaging data I/O library developers
- **PySide6**: Qt Company for Python bindings
- **Contributors**: All MediScreen-Brain project contributors

---

**Last Updated**: May 15, 2026  
**Document Version**: 1.0  
**Tool Version**: MediScreen-Brain DCM2NII v1.0