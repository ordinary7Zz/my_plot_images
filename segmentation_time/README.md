# 人工 vs AI辅助 分割时间效率对比实验套件

## 目录结构

```
datasets/
├── images/          # 原始超声图像
│   ├── TN3K_test_0000.jpg
│   └── ...
├── masks/           # 预分割 mask（由模型生成）
│   ├── TN3K_test_0000.jpg
│   └── ...
└── gt/              # GT mask（人工标注金标准, 右侧参考）
    ├── TN3K_test_0000.jpg
    └── ...

segmentation_time/
├── annotator.py                        # 核心标注工具 (多边形轮廓 + 画笔)
├── generate_masks.py                   # 批量推理生成预分割 mask
├── run_experiment.py                   # 实验运行脚本 (交叉设计、断点续传)
├── analyze_results.py                  # 统计分析 (检验 + 图表 + LaTeX 表格)
├── plot_case_time_saving_standalone.py # 时间效率可视化 (三子图 b/c/d + 组合图)
└── README.md                           # 本文档
```

---

## 实验流程

### 单人实验 (1 位标注者)

同一人对同一批图像做两轮: Round 1 全手动 → 洗脱期 ≥ 1 周 → Round 2 全辅助。

```bash
# 生成配置 (单人 + GT 参考)
python run_experiment.py generate-config \
    --image-dir ./datasets/images \
    --mask-dir ./datasets/masks \
    --gt-dir ./datasets/gt \
    --annotators alice \
    --output-dir ./experiment_single \
    --no-cross-over

# Windows 示例
python run_experiment.py generate-config ^
    --image-dir ./datasets/images ^
    --mask-dir ./datasets/masks ^
    --gt-dir ./datasets/gt ^
    --annotators alice ^
    --output-dir ./experiment_single ^
    --no-cross-over

# Round 1: 运行 (手动标注)
python run_experiment.py run --config experiment_single/experiment_config.json

# ===== 等待 ≥ 1 周 (洗脱期) =====

# Round 2: 继续运行 (AI辅助标注, 断点续传自动接上)
python run_experiment.py run --config experiment_single/experiment_config.json

# 统计分析 (按图像配对检验)
python analyze_results.py --log experiment_single/experiment_log.csv --output-dir ./analysis
```

> **单人实验要点**: Round 2 图像顺序与 Round 1 不同（已自动打乱），洗脱期 ≥ 1 周避免记忆效应。分析时使用配对 Wilcoxon signed-rank 检验。

---

## 一、标注工具 (annotator.py)

### 功能

**多边形轮廓模式 (默认, 医学标注标准方式):**
- 左键点击 → 添加顶点
- 左键拖拽顶点 → 移动顶点
- 右键点击顶点 → 删除顶点
- Shift+点击边 → 插入顶点
- Enter → 闭合多边形并填充区域

**AI辅助模式:**
- 模型预分割 mask 自动提取为多边形轮廓
- 标注者直接拖拽顶点调整边界即可（无需从零描点）

**GT 参考模式 (`--gt` 或 `--gt-dir`):**
- 窗口左右并排: 左侧 YOURS (可编辑) | 右侧 GT (蓝色只读)
- 鼠标仅在左侧工作区操作，右侧仅供对比参考

**画笔模式 (Tab 切换):**
- 左键涂抹 / 右键擦除

**全局:**
- **撤销/重做**: Ctrl+Z / Ctrl+Y
- **暂停计时**: 空格键
- **自动计时**: 精确到 0.1 秒
- **笔刷调节**: 数字键 1-9, `+/-` 微调
- **透明度**: `[` `]`
- **R**: 重置到初始状态

### 操作流程

```
纯人工: 打开 → 沿结节边界点一圈(15-25个点) → Enter闭合 → S保存
 AI辅助: 打开 → 轮廓已显示 → 拖拽偏离顶点调整 → S保存
```

### 使用

```bash
# AI 辅助 + GT 参考
python annotator.py --image datasets/images/patient_001.png \
    --mask datasets/masks/patient_001.jpg \
    --gt datasets/gt/patient_001.jpg \
    --annotator alice --output-log experiment_log.csv

# 纯人工 + GT 参考
python annotator.py --image datasets/images/patient_001.png \
    --gt datasets/gt/patient_001.jpg \
    --annotator alice --output-log experiment_log.csv

# 无 GT（单面板）
python annotator.py --image datasets/images/patient_001.png \
    --annotator alice --output-log experiment_log.csv
```

### 日志 CSV 字段

| 字段 | 含义 |
|------|------|
| `time_seconds` | 有效标注耗时 (秒) |
| `vertex_adds` | 添加顶点次数 |
| `vertex_deletes` | 删除顶点次数 |
| `vertex_drags` | 拖拽顶点次数 |
| `brush_strokes` | 涂抹笔画数 |
| `eraser_strokes` | 擦除笔画数 |
| `undo_count` | 撤销次数 |
| `mask_pixel_count` | 最终 mask 像素数 |

---

## 二、批量推理 (generate_masks.py)

**使用前需修改 `model_inference()` 函数**, 替换为你自己的模型推理逻辑。

```python
def model_inference(image_path: str) -> np.ndarray:
    # TODO: 替换为你自己的推理代码
    return binary_mask  # shape=(H,W), dtype=uint8, 值 0/255
```

```bash
python generate_masks.py \
    --image-dir ./datasets/images \
    --output-dir ./datasets/masks
```

---

## 三、实验运行 (run_experiment.py)

### 设计模式

| 模式 | 说明 | 适用 |
|------|------|------|
| 交叉设计 (默认) | 每位标注者一半手动一半辅助，同一张图不同人用不同模式 | 2+ 人 |
| 全量设计 (`--no-cross-over`) | 同一人对同一批图做两轮，先全手动后全辅助 | 1 人 |

### 子命令

```bash
# 生成配置 (含 GT 参考)
python run_experiment.py generate-config \
    --image-dir ./datasets/images \
    --mask-dir ./datasets/masks \
    --gt-dir ./datasets/gt \
    --annotators alice bob \
    --output-dir ./experiment_001 \
    --num-images 30 --seed 42

# 单人模式 (含 GT)
python run_experiment.py generate-config \
    --image-dir ./datasets/images \
    --mask-dir ./datasets/masks \
    --gt-dir ./datasets/gt \
    --annotators alice --output-dir ./experiment_single --no-cross-over

# 无 GT (单面板, 向后兼容)
python run_experiment.py generate-config \
    --image-dir ./datasets/images --mask-dir ./datasets/masks \
    --annotators alice --output-dir ./experiment_single --no-cross-over

# 运行实验 (支持断点续传)
python run_experiment.py run --config experiment_001/experiment_config.json

# 只运行某位标注者的任务
python run_experiment.py run --config experiment_001/experiment_config.json --annotator alice

# 查看进度
python run_experiment.py status --config experiment_001/experiment_config.json
```

---

## 四、统计分析 (analyze_results.py)

### 生成内容

| 输出 | 说明 |
|------|------|
| `analysis_report.md` | 完整 Markdown 报告 |
| `statistics.json` | JSON 统计数据 |
| `time_comparison_boxplot.png` | 时间箱线图 (全局 + 每位标注者) |
| `metrics_bar_comparison.png` | 多指标柱状图 |
| `scatter_paired_comparison.png` | 按图像配对的散点图 |

### 使用

```bash
python analyze_results.py \
    --log experiment_001/experiment_log.csv \
    --output-dir ./analysis_output

# 只输出 LaTeX 表格
python analyze_results.py \
    --log experiment_001/experiment_log.csv \
    --latex-only
```

---

## 五、时间效率可视化 (plot_case_time_saving_standalone.py)

### 功能

生成 **三个子图 + 一个组合图**，完整展示人工 vs AI辅助分割的时间效率对比:

| 子图 | 标题 | 内容 |
|------|------|------|
| b | Segmentation time | 小提琴图 + 箱线图 + 散点，对比 Manual vs AI 分割耗时分布 (秒)，标注缩短百分比和 P 值 |
| c | Within-case time saving | 病例级柱状图，绿色 = AI更快，红色 = AI更慢，橙色虚线 = 均值 (秒) |
| d | Annotator-stratified time saving | 按标注者分层展示分割时间减少百分比，含 Bootstrap 95% 置信区间 |
| bcd | 三子图组合 | b/c/d 水平并排 |

### 数据来源

读取 `annotator.py` 输出的日志 CSV，按 `image_name` 透视 `mode` 列，
将"每行一次标注"转换为"每行一个病例"的配对格式 (`manual_time_sec` vs `ai_time_sec`)，
并从 `annotator` 列提取 `ai_physician` 用于子图 d 的分层分析。

### 使用

```bash
# 默认读取 segmentation_time/experiment_log.csv，输出到 segmentation_time/output/
python plot_case_time_saving_standalone.py

# 指定输入文件和输出目录
python plot_case_time_saving_standalone.py \
    --input experiment_single/experiment_log.csv \
    --output-dir ./figures

# 自定义输出文件名前缀
python plot_case_time_saving_standalone.py \
    --input experiment_single/experiment_log.csv \
    --stem my_experiment
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | `experiment_log.csv` | annotator.py 输出的日志 CSV |
| `--output-dir` | `output/` | 输出目录，自动创建 |
| `--stem` | `segmentation_efficiency` | 输出文件名前缀 |

### 输出 (4 图表 × 3 格式 = 12 文件)

| 文件（以默认 stem 为例） | 说明 |
|------|------|
| `segmentation_efficiency_b_segmentation_time.{png,pdf,svg}` | 子图 b: 分割耗时分布 |
| `segmentation_efficiency_c_case_time_saving.{png,pdf,svg}` | 子图 c: 病例级时间节省 |
| `segmentation_efficiency_d_annotator_time_saving.{png,pdf,svg}` | 子图 d: 标注者分层 |
| `segmentation_efficiency_bcd_combined.{png,pdf,svg}` | 三子图组合 |

---

## 六、实验设计要点

### 参与者
- 多人: 2-3 人 (推荐交叉设计)
- 单人: 1 人 + 洗脱期 ≥ 1 周

### 图像数量
- 最少 10 张 (配对检验需要足够样本)
- 推荐 20-40 张

### 洗脱期
- 两轮之间至少间隔 1 周

### 额外记录建议
- 标注前的练习期 (不计入统计)
- 标注后主观难度评分 (1-5)

---

## 依赖安装

```bash
pip install opencv-python numpy scipy matplotlib pandas
```
