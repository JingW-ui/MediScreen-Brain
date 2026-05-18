# MediScreen-Brain 项目完整说明文档

## 📋 目录

- [项目概述](#项目概述)
- [核心功能](#核心功能)
- [技术架构](#技术架构)
- [系统模块](#系统模块)
- [使用指南](#使用指南)
- [文件结构](#文件结构)
- [依赖环境](#依赖环境)
- [应用场景](#应用场景)
- [输出格式](#输出格式)
- [注意事项](#注意事项)

---

## 项目概述

**MediScreen-Brain** 是一款基于 YOLO 深度学习模型的脑肿瘤检测系统，专为医疗影像分析设计。该系统提供图形化用户界面（GUI），支持多种医学影像格式的实时检测和批量处理，能够自动识别脑部 MRI 图像中的肿瘤区域，并生成详细的诊断报告。

### 主要特点

- 🎯 **高精度检测**：基于 YOLO 目标检测算法，准确识别脑肿瘤位置
- 🖥️ **友好界面**：现代化渐变 UI 设计，操作简便直观
- 📊 **多格式支持**：支持 NIfTI (.nii/.nii.gz)、DICOM、常见图片格式
- 🔄 **实时监控**：支持多摄像头实时监测和录像
- 📈 **批量处理**：支持文件夹批量检测，提高工作效率
- 📝 **报告导出**：自动生成 PDF/Excel 格式的诊断报告
- 💾 **历史记录**：自动保存检测和监控快照，支持回放

---

## 核心功能

### 1. 实时检测 (Real-time Detection)

#### 功能描述
对单张图片、视频文件或摄像头进行实时脑肿瘤检测，即时显示检测结果和统计信息。

#### 支持源类型
- 📷 **单张图片**：JPG, JPEG, PNG, BMP, TIFF, WEBP
- 🎬 **视频文件**：MP4, AVI, MOV, MKV, WMV, FLV
- 📹 **摄像头**：支持多摄像头切换和实时检测
- 🧠 **NIfTI 文件**：专业的医学影像格式 (.nii, .nii.gz)

#### 检测流程
1. 选择检测模型（支持 .pt 和 .onnx 格式）
2. 设置置信度阈值（0.01-1.0）
3. 选择检测源（图片/视频/摄像头/NIfTI）
4. 点击"开始检测"
5. 查看实时检测结果和统计数据

#### 结果显示
- 原始图像与检测结果对比显示
- 检测目标数量、类别分布
- 置信度范围及平均值
- 肿瘤直径估算（像素单位）
- 肿瘤面积估算（像素²）

---

### 2. NIfTI 医学影像检测

#### 功能描述
专门针对脑部 MRI 的 NIfTI 格式文件进行三维空间检测，提供三视图（轴位、矢状位、冠状位）分析和详细测量数据。

#### 检测步骤
1. 加载 NIfTI 文件 (.nii 或 .nii.gz)
2. 系统自动定位最佳肿瘤切片
3. 提取三个正交平面的关键切片
4. 使用专用模型进行肿瘤检测
5. 计算肿瘤尺寸和体积

#### 三视图显示
- **Axial (轴位)**：水平切面视图
- **Sagittal (矢状位)**：侧面切面视图
- **Coronal (冠状位)**：正面切面视图

#### 测量指标
- 每个视图中肿瘤的最大直径（mm）
- 肿瘤截面积（mm²）
- 估算肿瘤体积（mm³）
- 肿瘤中心坐标

#### 交互功能
- **选择目标**：显示带标注框的检测结果
- **移除标注**：显示原始无标注图像
- **导出报告**：生成包含三视图的 PDF 诊断报告

---

### 3. 批量检测 (Batch Detection)

#### 功能描述
对整个文件夹中的图片进行批量自动化检测，适合大规模数据处理和统计分析。

#### 操作流程
1. 选择"批量文件夹"模式
2. 选择包含图片的文件夹
3. 系统自动扫描所有支持的图片格式
4. 逐张处理并记录检测结果
5. 生成批量检测报告

#### 结果导航
- ⬅️ 上一张 / ➡️ 下一张：浏览检测结果
- 显示当前索引（如 5/100）
- 每张图片的详细检测信息

#### 批量报告
系统生成两种格式的报告：

**TXT 文本报告** (`detection_report.txt`)
- 处理时间和参数设置
- 每张图片的检测详情
- 类别分布统计
- 直径和面积数据

**Excel 电子表格** (`detection_report.xlsx`)
- **Detection Results 工作表**：
  - 序号、文件名
  - 检测目标数、推理时间
  - 置信度、类别分布
  - 最大直径、总面积
  
- **Summary 工作表**：
  - 处理时间、置信度阈值
  - 图片总数、总检测目标数
  - 平均推理时间

---

### 4. 实时监控 (Real-time Monitoring)

#### 功能描述
支持多摄像头同时监控，适用于临床实时观察和长期监测场景。

#### 监控特性
- 📹 **多摄像头支持**：最多同时监控 4 个摄像头
- 🖥️ **动态布局**：根据摄像头数量自动调整网格布局（1×1, 1×2, 2×2）
- ⏸️ **暂停/继续**：可单独或批量控制摄像头状态
- 📊 **实时统计**：显示检测数量和系统时间

#### 控制面板
- **启动全部**：同时启动所有选中摄像头
- **暂停全部**：暂停/继续所有摄像头
- **停止全部**：停止所有摄像头线程
- **清空画面**：清除所有显示内容

#### 摄像头管理
- 自动扫描可用摄像头
- 显示摄像头分辨率和帧率
- 支持热插拔和自动重连

---

### 5. 监控快照 (Monitoring Snapshot)

#### 功能描述
自动录制监控过程中的检测视频，保存为 MP4 格式并附带 JSON 元数据，支持后续回放和分析。

#### 录制机制
- **触发条件**：仅在检测到目标时录制
- **文件格式**：MP4 视频 + JSON 元数据
- **存储位置**：`monitor_history/` 和 `detection_history/` 目录
- **内存管理**：自动清理超出限制的旧记录（默认 500MB）

#### 快照信息
JSON 文件包含：
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

#### 回放功能
- 📋 **快照列表**：按时间倒序显示所有快照
- ▶️ **播放控制**：播放、暂停、停止、进度条拖动
- 📊 **信息显示**：摄像头名称、时间、时长、检测统计
- 🗑️ **删除管理**：删除不需要的快照
- 📤 **导出功能**：导出快照到指定目录

---

### 6. NIfTI 格式转换

#### 功能描述
将 NIfTI 格式的三维医学影像转换为二维 PNG 图片序列，便于后续处理和可视化。

#### 转换设置
- **切片方向**：Sagittal（矢状位）、Coronal（冠状位）、Axial（轴位）
- **切片范围**：自定义起始和结束切片索引
- **输出目录**：自动生成（输入目录 + `_swift_normal`）

#### 预览功能
- 显示 5 个代表性切片预览
- 右键菜单支持放大查看详情
- 滚轮切换切片浏览

#### 转换过程
1. 加载 NIfTI 文件
2. 选择切片方向和范围
3. 预览确认
4. 执行转换（带进度条）
5. 保存为 PNG 序列

#### 文件信息展示
- 文件大小、修改日期
- 图像维度（如 256×256×176）
- 数据类型（如 float32）
- 空间分辨率（体素大小）

---

### 7. 诊断报告导出

#### PDF 报告功能
系统可生成标准化的 MRI 检查报告，包含以下内容：

**报告结构**
1. **标题**：MRI Examination Report
2. **基本信息表**：
   - 姓名、性别、年龄
   - 日期、ID、科室
   - 检查部位、序列、设备

3. **检测结果图像**：
   - 带标注框的检测图
   - 三视图对比（NIfTI 模式）

4. **扫描发现 (Scan Findings)**：
   - 病变类型
   - 可能性（置信度）
   - 最大直径
   - 估算面积/体积

5. **诊断结论 (Diagnostic Conclusion)**：
   - 基于检测结果的诊断建议
   - 病变性质评估

6. **诊断建议 (Diagnostic Recommendations)**：
   - 增强扫描建议
   - 临床综合判断
   - 定期随访计划

7. **注意事项 (Notes)**：
   - 临床症状结合
   - 及时就医提醒
   - 定期监测建议

8. **签名区**：
   - 报告医师、审核医师
   - 报告时间
   - 医院名称

9. **免责声明**：
   - 原型版本声明
   - 非诊断用途说明
   - 仅供医学研究

#### 字体支持
- 自动注册中文字体（微软雅黑）
- 备用字体：simhei
- 支持中英文混合排版

#### 打开功能
导出成功后可选择直接打开 PDF 文件（跨平台支持 Windows/macOS/Linux）

---

## 技术架构

### 核心技术栈

| 技术 | 用途 | 版本要求 |
|------|------|----------|
| **Python** | 编程语言 | 3.8+ |
| **PySide6** | GUI 框架 | 6.x |
| **Ultralytics YOLO** | 目标检测模型 | 8.x |
| **OpenCV** | 图像处理 | 4.x |
| **NiBabel** | NIfTI 文件读写 | 5.x |
| **NumPy** | 数值计算 | 1.21+ |
| **ReportLab** | PDF 生成 | 4.x |
| **OpenPyXL** | Excel 文件操作 | 3.x |
| **Matplotlib** | 图像可视化 | 3.x |
| **Pillow** | 图像处理 | 9.x |

### 架构设计

```
┌─────────────────────────────────────────┐
│         EnhancedDetectionUI             │
│       (主窗口 - QMainWindow)             │
├─────────────────────────────────────────┤
│  Tab Widget (标签页管理器)               │
│  ┌──────────┬──────────┬──────────┐    │
│  │ Realtime │  Batch   │ Monitor  │    │
│  │Detection │ Results  │          │    │
│  ├──────────┼──────────┼──────────┤    │
│  │  NIfTI   │ Snapshot │  NIfTI   │    │
│  │Detection │          │Conversion│    │
│  └──────────┴──────────┴──────────┘    │
├─────────────────────────────────────────┤
│         业务逻辑层                       │
│  ┌────────────┐  ┌──────────────┐      │
│  │ModelManager│  │CameraManager │      │
│  └────────────┘  └──────────────┘      │
│  ┌────────────┐  ┌──────────────┐      │
│  │ StyleMgr   │  │ TumorFinder  │      │
│  └────────────┘  └──────────────┘      │
├─────────────────────────────────────────┤
│         线程管理层                       │
│  ┌──────────────┐  ┌─────────────┐     │
│  │DetectionThread│ │BatchThread  │     │
│  └──────────────┘  └─────────────┘     │
│  ┌──────────────┐  ┌─────────────┐     │
│  │MonitorThread │ │CameraThread │     │
│  └──────────────┘  └─────────────┘     │
├─────────────────────────────────────────┤
│         数据持久层                       │
│  ┌────────────┐  ┌──────────────┐      │
│  │VideoRecorder│ │SnapshotWidget│      │
│  └────────────┘  └──────────────┘      │
└─────────────────────────────────────────┘
```

### 关键类说明

#### 1. EnhancedDetectionUI (主窗口)
- **职责**：整合所有功能模块，管理用户交互
- **主要方法**：
  - `init_ui()`：初始化界面
  - `start_detection()`：启动检测
  - `export_report()`：导出 PDF 报告
  - `save_batch_results()`：保存批量结果

#### 2. ModelManager (模型管理器)
- **职责**：扫描、加载和管理检测模型
- **支持格式**：.pt (PyTorch), .onnx (ONNX Runtime)
- **主要方法**：
  - `scan_models()`：扫描模型文件
  - `load_model()`：加载指定模型

#### 3. CameraManager (摄像头管理器)
- **职责**：检测和管理局部摄像头设备
- **主要方法**：
  - `scan_cameras()`：扫描可用摄像头
  - `get_available_cameras()`：获取可用摄像头列表

#### 4. DetectionThread (检测线程)
- **职责**：在后台线程执行检测任务，避免界面卡顿
- **信号**：
  - `result_ready`：检测结果就绪
  - `progress_updated`：进度更新
  - `status_changed`：状态变化
  - `error_occurred`：错误发生

#### 5. BatchDetectionThread (批量检测线程)
- **职责**：处理文件夹中所有图片的批量检测
- **特性**：支持暂停、停止、进度追踪

#### 6. MultiCameraMonitorThread (多摄像头监控线程)
- **职责**：同时管理多个摄像头的实时监控
- **特性**：
  - 独立帧率控制
  - 自动重连机制
  - 暂停/恢复支持

#### 7. TumorSliceFinder (肿瘤切片查找器)
- **职责**：在 NIfTI 三维数据中定位最佳肿瘤切片
- **算法**：分层搜索策略，快速定位肿瘤区域

#### 8. DetectionVideoRecorder (检测录像器)
- **职责**：录制检测过程的视频快照
- **输出**：MP4 视频 + JSON 元数据

#### 9. SnapshotWidget (快照组件)
- **职责**：管理和回放已保存的监控快照
- **功能**：列表显示、视频播放、信息管理

---

## 系统模块

### 模块一：用户界面模块

#### 样式管理器 (StyleManager)
提供统一的现代化 UI 样式：
- 渐变色背景
- 圆角按钮和控件
- 悬停效果和动画
- 响应式布局

#### 主界面布局
```
┌──────────────────────────────────────────────────┐
│  标题栏: MediScreen-Brain Tumor Detection System │
├──────────────┬───────────────────────────────────┤
│  左侧控制面板 │      右侧显示区域                  │
│  (450px)     │      (自适应)                      │
│              │                                   │
│ ┌──────────┐ │  ┌─────────────────────────┐    │
│ │模型配置   │ │  │                         │    │
│ └──────────┘ │  │   标签页显示区域         │    │
│ ┌──────────┐ │  │                         │    │
│ │检测源配置 │ │  │ • 实时检测              │    │
│ └──────────┘ │  │ • 批量结果              │    │
│ ┌──────────┐ │  │ • NIfTI检测             │    │
│ │检测控制   │ │  │ • 实时监控              │    │
│ └──────────┘ │  │ • 监控快照              │    │
│ ┌──────────┐ │  │ • NIfTI转换             │    │
│ │运行日志   │ │  └─────────────────────────┘    │
│ └──────────┘ │                                   │
└──────────────┴───────────────────────────────────┘
```

---

### 模块二：模型管理模块

#### 模型扫描
- 自动扫描以下目录：
  - `pt_models/`
  - `onnx_models/`
  - `../models/`
  - `weights/`
- 支持自定义模型路径

#### 模型加载
```python
# PT 模型
model = YOLO("pt_models/best.pt")

# ONNX 模型
model = YOLO("onnx_models/segmentation.onnx", task='detect')
```

#### 高级模型选择对话框
- 表格显示模型信息（名称、大小、修改时间、路径）
- 支持自定义路径浏览
- 双击快速选择

---

### 模块三：图像处理模块

#### 图像格式转换
- BGR ↔ RGB 颜色空间转换
- NumPy 数组 ↔ QImage 转换
- 图像缩放和抗锯齿处理

#### NIfTI 数据处理
```python
# 加载 NIfTI 文件
img = nib.load("brain_scan.nii.gz")
data = img.get_fdata()

# 归一化到 uint8
data = np.clip(data, 0, np.percentile(data, 99))
data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)

# 提取切片
axial_slice = data[:, :, slice_index]
sagittal_slice = data[slice_index, :, :]
coronal_slice = data[:, slice_index, :]
```

#### 切片预览
- 使用 Matplotlib 渲染
- 支持右键菜单放大
- 滚轮切换切片

---

### 模块四：检测算法模块

#### YOLO 检测流程
```python
# 单张图片检测
results = model(image_path, conf=0.25, verbose=False)

# 获取检测框
boxes = results[0].boxes
confidences = boxes.conf.cpu().numpy()
classes = boxes.cls.cpu().numpy().astype(int)
xyxy = boxes.xyxy.cpu().numpy()

# 绘制结果
result_image = results[0].plot()
```

#### 肿瘤测量计算
```python
# 直径计算（假设圆形）
width = x2 - x1
height = y2 - y1
diameter = (width + height) / 2

# 面积计算（椭圆近似）
area = 0.785 * width * height  # π/4 ≈ 0.785

# 体积估算（椭球体）
volume = (4/3) * π * (a/2) * (b/2) * (c/2)
```

#### 置信度分级显示
- **高置信度 (>0.8)**：绿色背景
- **中置信度 (0.5-0.8)**：黄色背景
- **低置信度 (<0.5)**：红色背景

---

### 模块五：视频录制模块

#### CameraVideoRecorder (监控录像)
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

#### DetectionVideoRecorder (检测录像)
```python
recorder = DetectionVideoRecorder(
    source_name="Camera0",
    output_dir=Path("detection_history"),
    fps=20
)
```

#### 视频编码
- 编解码器：MP4V (H.264)
- 帧率：可配置（默认 20fps）
- 颜色空间：RGB → BGR 转换

---

### 模块六：报告生成模块

#### PDF 报告生成
```python
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Table

doc = SimpleDocTemplate("report.pdf", pagesize=A4)
story = []

# 添加标题、表格、图像等元素
story.append(title)
story.append(info_table)
story.append(detection_image)
story.append(findings_paragraph)

doc.build(story)
```

#### Excel 报告生成
```python
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill

wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Detection Results"

# 设置表头样式
header_font = Font(bold=True, size=12, color="FFFFFF")
header_fill = PatternFill(start_color="3498DB", end_color="3498DB", fill_type="solid")

# 写入数据
for row_idx, result in enumerate(batch_results, 2):
    ws.cell(row=row_idx, column=1, value=result['file_name'])
    # ... 其他列

wb.save("detection_report.xlsx")
```

---

## 使用指南

### 快速开始

#### 1. 环境准备
```bash
# 安装依赖
pip install -r requirements.txt

# 或使用 conda
conda create -n mediscreen python=3.9
conda activate mediscreen
pip install -r requirements.txt
```

#### 2. 模型准备
将训练好的模型文件放入以下目录之一：
```
pt_models/
├── best.pt
└── Brain_Tumor.pt

onnx_models/
├── segmentation.onnx
└── Brain_Tumor.onnx
```

#### 3. 启动程序
```bash
python Brain_Tumor_detection_ui.py
```

---

### 功能使用详解

#### 场景一：单张图片检测

1. **选择模型**
   - 在"模型配置"区域选择预加载模型
   - 或点击"🔧"按钮选择自定义模型

2. **设置参数**
   - 调整置信度阈值滑块（推荐 0.25-0.65）

3. **选择检测源**
   - 下拉菜单选择"📷 Single Image"
   - 点击"📁 Select File/Folder"选择图片

4. **开始检测**
   - 点击"▶️ Start"按钮
   - 等待检测完成

5. **查看结果**
   - 左侧显示原始图片
   - 右侧显示标注结果
   - 下方表格显示详细检测数据

6. **导出报告**
   - 点击"📤 Export Report"按钮
   - 选择保存路径
   - 自动生成 PDF 报告

---

#### 场景二：NIfTI 医学影像分析

1. **切换到 NIfTI 检测标签页**
   - 点击顶部标签"🧠 NIfTI Detection"

2. **加载 NIfTI 文件**
   - 在实时检测标签页选择"🧠 NIfTI File"
   - 选择 .nii 或 .nii.gz 文件
   - 点击"▶️ Start"开始分析

3. **查看三视图**
   - Axial（轴位）：水平切面
   - Sagittal（矢状位）：侧面切面
   - Coronal（冠状位）：正面切面

4. **分析检测结果**
   - 查看检测表格中的肿瘤信息
   - 关注直径、面积、置信度

5. **交互操作**
   - 点击"📤 Select Target"显示标注框
   - 点击"❌ Remove Target"移除标注
   - 点击"📤 Export Report"导出诊断报告

6. **查看详细切片**
   - 在 NIfTI 转换标签页加载同一文件
   - 使用滚轮浏览不同切片
   - 右键点击预览图放大查看

---

#### 场景三：批量文件夹处理

1. **选择批量模式**
   - 下拉菜单选择"📂 Batch Folder"

2. **选择文件夹**
   - 点击"📁 Select File/Folder"
   - 选择包含图片的文件夹

3. **开始批量检测**
   - 点击"▶️ Start"
   - 系统自动处理所有图片

4. **浏览结果**
   - 切换到"📊 Batch Results"标签页
   - 使用"⬅️ Previous"和"Next ➡️"浏览
   - 查看每张图片的检测详情

5. **保存结果**
   - 点击"💾 Save Results"
   - 选择保存目录
   - 自动生成：
     - 带标注的结果图片
     - TXT 文本报告
     - Excel 电子表格

---

#### 场景四：多摄像头实时监控

1. **切换到监控标签页**
   - 点击"🖥️ Real-time Monitoring"

2. **选择摄像头**
   - 在摄像头列表中勾选要监控的摄像头
   - 点击"🔄 Refresh"刷新设备列表

3. **选择模型**
   - 确保已加载检测模型

4. **启动监控**
   - 点击"🚀 Start monitoring"
   - 摄像头画面实时显示

5. **控制监控**
   - "⏸️ Pause"：暂停/继续监控
   - "🗑️ Clear Monitor"：停止并清空

6. **启用自动录制**
   - 点击"🎬 Auto-save Monitoring Snapshots"
   - 设置帧率（推荐 20fps）
   - 设置内存限制（默认 500MB）
   - 检测到目标时自动录制

7. **查看录制快照**
   - 切换到"🎬 Monitoring Snapshot"标签页
   - 选择快照点击"▶️ Play"回放
   - 可导出或删除快照

---

#### 场景五：NIfTI 格式转换

1. **切换到转换标签页**
   - 点击"NIfTI Conversion"

2. **加载文件**
   - 点击"Browse"选择 NIfTI 文件
   - 自动显示文件信息

3. **设置转换参数**
   - 选择切片方向（Sagittal/Coronal/Axial）
   - 设置切片范围（起始-结束）
   - 查看预览确认

4. **执行转换**
   - 点击"🔄 Convert"
   - 等待转换完成（带进度条）

5. **查看结果**
   - 输出目录自动生成 PNG 序列
   - 文件命名：slice_0060.png, slice_0061.png...

---

### 快捷键

| 快捷键 | 功能 |
|--------|------|
| **F2** 或 **Ctrl+R** | 编辑窗口标题 |
| **滚轮** | 在切片预览中切换切片 |
| **右键点击** | 显示切片放大菜单 |

---

## 文件结构

```
MediScreen-Brain/
│
├── Brain_Tumor_detection_ui.py          # 主程序入口（6355行）
├── batch_compare_ui.py                  # 批量对比工具
├── requirements.txt                     # Python 依赖清单
├── README.md                            # 项目简介
├── PROJECT_DOCUMENTATION.md             # 本文档
│
├── pt_models/                           # PyTorch 模型目录
│   ├── best.pt
│   └── Brain_Tumor.pt
│
├── onnx_models/                         # ONNX 模型目录
│   ├── segmentation.onnx
│   └── Brain_Tumor.onnx
│
├── utils/                               # 工具函数目录
│   ├── TumorSliceFinder.py             # 肿瘤切片查找器
│   ├── YOLOONNX.py                     # ONNX 推理封装
│   ├── data_utils/                     # 数据处理工具
│   ├── methods_ui/                     # UI 方法
│   └── utils_for_data/                 # 数据工具
│
├── data_test/                           # 测试数据
│   ├── 545140_swust.nii.gz
│   └── MRBrainTumor2.nii.gz
│
├── monitor_history/                     # 监控快照存储
│   ├── Camera0_1234567890.mp4
│   └── Camera0_1234567890.json
│
├── detection_history/                   # 检测快照存储
│   ├── Camera0_1234567890.mp4
│   └── Camera0_1234567890.json
│
├── best_slices_results_batch_data2_94/  # 批量检测结果示例
│   ├── [患者ID]_[扫描ID]/
│   │   ├── comparison_reports/
│   │   ├── full_search/
│   │   └── hierarchical_search/
│   └── batch_summary/
│
├── compares_resluts/                    # 对比结果
├── detection_history/                   # 检测历史
├── img/                                 # 界面截图
│   ├── 1.png
│   ├── 2.png
│   └── ui1.png
│
├── build/                               # 打包构建目录
├── dist/                                # 分发目录
├── .idea/                               # PyCharm 配置
├── .vscode/                             # VSCode 配置
└── .gitignore                           # Git 忽略规则
```

---

## 依赖环境

### Python 版本
- **推荐**：Python 3.9
- **最低**：Python 3.8
- **兼容**：Python 3.10, 3.11

### 核心依赖

```txt
# 深度学习
ultralytics>=8.0.0          # YOLO 框架
torch>=1.12.0               # PyTorch
torchvision>=0.13.0         # TorchVision
onnxruntime>=1.12.0         # ONNX 推理引擎

# GUI 框架
PySide6>=6.4.0              # Qt6 Python 绑定

# 图像处理
opencv-python>=4.6.0        # OpenCV
Pillow>=9.0.0               # PIL 图像处理
matplotlib>=3.5.0           # 绘图库
numpy>=1.21.0               # 数值计算

# 医学影像
nibabel>=5.0.0              # NIfTI 文件处理

# 报告生成
reportlab>=4.0.0            # PDF 生成
openpyxl>=3.0.0             # Excel 文件操作

# 其他工具
pathlib                     # 路径处理（内置）
json                        # JSON 处理（内置）
threading                   # 线程管理（内置）
```

### 安装命令

```bash
# 方式一：使用 requirements.txt
pip install -r requirements.txt

# 方式二：手动安装核心包
pip install ultralytics PySide6 opencv-python nibabel reportlab openpyxl matplotlib numpy Pillow

# 方式三：使用 conda
conda create -n mediscreen python=3.9
conda activate mediscreen
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install ultralytics PySide6 opencv-python nibabel reportlab openpyxl matplotlib numpy Pillow
```

### 系统要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| **操作系统** | Windows 10 / macOS 10.15 / Ubuntu 18.04 | Windows 11 / macOS 12 / Ubuntu 22.04 |
| **CPU** | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| **内存** | 8 GB RAM | 16 GB RAM |
| **GPU** | 集成显卡 | NVIDIA GTX 1060+ (4GB VRAM) |
| **存储** | 2 GB 可用空间 | 10 GB SSD |
| **摄像头** | 可选（720p） | 推荐（1080p） |

### GPU 加速（可选）

如需使用 GPU 加速检测：

```bash
# 卸载 CPU 版本
pip uninstall torch torchvision

# 安装 CUDA 版本（以 CUDA 11.8 为例）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 验证 GPU 可用性
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 应用场景

### 1. 临床辅助诊断
- **用途**：帮助放射科医生快速定位脑部肿瘤
- **优势**：减少漏诊，提高诊断效率
- **流程**：
  1. 导入患者 MRI 扫描（NIfTI 格式）
  2. 系统自动检测肿瘤位置
  3. 生成三视图分析报告
  4. 医生审核并签字确认

### 2. 科研数据分析
- **用途**：大规模医学影像数据集标注和统计
- **优势**：批量处理，自动生成 Excel 统计报表
- **流程**：
  1. 准备数据集文件夹
  2. 启动批量检测
  3. 等待自动处理完成
  4. 导出结构化数据用于统计分析

### 3. 教学演示
- **用途**：医学影像识别教学和学生培训
- **优势**：可视化界面，实时反馈
- **流程**：
  1. 准备典型病例图片
  2. 课堂实时演示检测过程
  3. 讲解检测结果和置信度
  4. 学生动手实践操作

### 4. 远程医疗监控
- **用途**：多地点实时视频监控和异常检测
- **优势**：支持多摄像头，自动录制异常片段
- **流程**：
  1. 部署多个摄像头
  2. 启动实时监控模式
  3. 系统自动检测并录制
  4. 事后回放和分析

### 5. 术前规划
- **用途**：手术前肿瘤位置和尺寸评估
- **优势**：精确测量，三维空间定位
- **流程**：
  1. 加载患者最新 MRI 数据
  2. 分析三视图肿瘤位置
  3. 测量肿瘤直径和体积
  4. 导出报告供手术团队参考

---

## 输出格式

### 1. 检测结果表格

| 列名 | 说明 | 示例 |
|------|------|------|
| Class | 检测类别 | tumor, lesion |
| Confidence | 置信度 | 0.923 |
| Coordinates (x,y) | 左上角坐标 | (120,85) |
| Size (w×h) | 边界框尺寸 | 45×38 |
| Estimated Diameter | 估算直径 | 41.50 pixel |
| Pixel Area | 像素面积 | 1343.70 pixel² |

### 2. PDF 报告结构

```
MRI Examination Report
═══════════════════════════

[基本信息表]
姓名: ______  性别: ______  年龄: ______
日期: 2024-01-15  ID: ______  科室: ______
部位: Brain  序列: T1/T2  设备: MRI Scanner

[检测结果图像]
┌─────────────────┐
│  带标注的检测图  │
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

[签名区]
Reporting Physician: _______________
Reviewing Physician: _______________
Report Time: 2024-01-15 14:30:25
Hospital Name: XXX Hospital Imaging Department

Disclaimer:
This software is a prototype version and is not designed or intended for use 
in diagnosis or classification of any medical problems or other medical purposes.
```

### 3. Excel 报告格式

**Detection Results 工作表**

| Index | File Name | Detected Objects | Inference Time (s) | Confidence | Class Distribution | Max Diameter (pixel) | Total Area (pixel²) |
|-------|-----------|------------------|-------------------|------------|-------------------|---------------------|-------------------|
| 1 | patient_001.jpg | 2 | 0.045 | 0.856 - 0.923 | tumor:2 | 45.50 | 1580.25 |
| 2 | patient_002.jpg | 1 | 0.038 | 0.789 | lesion:1 | 32.00 | 804.25 |
| ... | ... | ... | ... | ... | ... | ... | ... |

**Summary 工作表**

| Metric | Value |
|--------|-------|
| Processing Time | 2024-01-15 14:30:25 |
| Confidence Threshold | 0.25 |
| Number of Images | 100 |
| Total Detected Objects | 87 |
| Average Inference Time (s) | 0.042 |

### 4. JSON 元数据格式

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

### 5. 日志格式

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

## 注意事项

### ⚠️ 重要声明

1. **非医疗设备**
   - 本软件为**原型版本**，仅用于医学研究和教学演示
   - **不适用于**临床诊断、疾病分类或其他医疗目的
   - 用户承认并保证不会将本软件用于上述用途

2. **免责声明**
   - 软件按"现状"提供，不提供任何形式的担保
   - 开发者不对因使用本软件导致的任何损失负责
   - 所有检测结果仅供参考，需由专业医生确认

3. **数据安全**
   - 本地处理：所有数据在本地计算机处理，不上传云端
   - 隐私保护：患者数据应匿名化处理
   - 合规要求：遵循 HIPAA/GDPR 等数据保护法规

### 🔧 使用建议

1. **模型选择**
   - 使用经过充分验证的训练模型
   - 定期更新模型以提高准确性
   - 针对不同设备调整置信度阈值

2. **图像质量**
   - 确保输入图像清晰、无伪影
   - NIfTI 文件应为标准格式
   - 建议使用高分辨率扫描（≥1mm 各向同性）

3. **性能优化**
   - 批量处理时关闭其他大型应用
   - 使用 SSD 存储提高读写速度
   - GPU 加速可显著提升检测速度

4. **存储空间**
   - 定期清理 `monitor_history/` 和 `detection_history/` 目录
   - 调整内存限制参数避免磁盘占满
   - 重要数据及时备份

5. **摄像头使用**
   - 确保摄像头驱动已正确安装
   - Windows 系统建议使用 CAP_DSHOW 后端
   - 光线充足可提高检测准确率

### 🐛 常见问题

#### Q1: 模型加载失败
**解决**：
- 检查模型文件路径是否正确
- 确认文件格式（.pt 或 .onnx）
- 验证 PyTorch/ONNX Runtime 版本兼容性

#### Q2: 检测速度慢
**解决**：
- 降低输入图像分辨率
- 使用 GPU 加速
- 减少置信度阈值范围

#### Q3: NIfTI 文件无法打开
**解决**：
- 确认文件未损坏
- 检查 nibabel 库版本
- 尝试使用其他软件（如 ITK-SNAP）验证文件

#### Q4: 摄像头无法识别
**解决**：
- 检查摄像头连接
- 更新摄像头驱动
- 尝试其他 USB 端口
- 重启应用程序

#### Q5: PDF 报告中文字符乱码
**解决**：
- 确认系统中存在微软雅黑字体
- 检查 `C:\Windows\Fonts\msyh.ttc` 路径
- 修改代码中的字体路径为实际位置

#### Q6: 批量处理中断
**解决**：
- 检查文件夹权限
- 确认磁盘空间充足
- 查看日志定位具体错误文件
- 分批处理大量数据

### 📞 技术支持

如遇问题，请：
1. 查看运行日志（界面底部日志区域）
2. 检查控制台输出错误信息
3. 确认依赖库版本符合要求
4. 查阅本项目文档和相关库官方文档

---

## 版本历史

### v2.0 (当前版本)
- ✨ 全新渐变 UI 设计
- 📹 多摄像头实时监控
- 🎬 自动录制监控快照
- 📊 批量检测 Excel 报告
- 🧠 NIfTI 三视图分析
- 📤 PDF 诊断报告导出
- 🔄 NIfTI 格式转换工具
- 📋 快照回放和管理

### v1.x (早期版本)
- 基础单图片检测
- 简单 GUI 界面
- 基础日志功能

---

## 许可证

本项目仅供学术研究和教学使用。商业用途请联系作者获取授权。

---

## 联系方式

- **项目名称**：MediScreen-Brain
- **开发语言**：Python 3.9+
- **GUI 框架**：PySide6
- **检测模型**：YOLO (Ultralytics)

### 团队与个人

- **研发团队**：[电子科技大学神经工程实验室](https://www.neuro.uestc.edu.cn/)
- **开发者主页**：[JingW-ui](https://jingw-ui.github.io/resume/)、[MYiu](https://github.com/MYiuCode)

---

**最后更新**：2026-05-13  
**文档版本**：v2.0  
**适用软件版本**：MediScreen-Brain v2.0+

---

*本文档由 AI 辅助生成，内容基于项目代码分析。如有错误或遗漏，请以实际代码为准。*
