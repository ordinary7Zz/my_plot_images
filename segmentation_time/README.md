# 人工 vs AI辅助 分割时间效率对比实验套件

## 目录结构

```
segmentation_time/
├── annotator.py          # 核心标注工具 (支持撤销、操作计数)
├── generate_masks.py     # 批量推理生成预分割 mask
├── run_experiment.py     # 实验运行脚本 (交叉设计、断点续传)
├── analyze_results.py    # 统计分析 (检验 + 图表 + LaTeX 表格)
└── README.md             # 本文档
```

## 实验流程概览

```
Step 1: 用模型生成预分割 mask
  python generate_masks.py --image-dir ./images --output-dir ./masks

Step 2: 生成实验配置 (随机分配图像、交叉设计)
  python run_experiment.py generate-config \
      --image-dir ./images --mask-dir ./masks \
      --annotators alice bob \
      --output-dir ./experiment_001

Step 3: 运行实验 (交互式, 逐个标注)
  python run_experiment.py run --config experiment_001/experiment_config.json

  # 查看进度
  python run_experiment.py status --config experiment_001/experiment_config.json

Step 4: 统计分析
  python analyze_results.py --log experiment_001/experiment_log.csv --output-dir ./analysis
```

---

## 一、标注工具 (annotator.py)

### 功能
- **两种模式**: 纯人工 (空白mask) / AI辅助 (导入预分割mask修改)
- **画笔 (左键)**: 涂抹添加分割区域
- **橡皮擦 (右键)**: 擦除错误区域
- **撤销/重做**: Ctrl+Z / Ctrl+Y
- **暂停计时**: 空格键
- **操作统计**: 涂抹次数、擦除次数、撤销次数、像素变化量
- **自动计时**: 精确到 0.1 秒
- **笔刷调节**: 数字键 1-9 快速切换, `+/-` 微调
- **透明度调节**: `[` `]` 键
- **中键拖拽**: 平移画面

### 使用

```bash
# AI 辅助模式 (有预分割 mask)
python annotator.py --image patient_001.png --mask patient_001_mask.png \
    --annotator alice --output-log experiment_log.csv

# 纯人工模式 (无 mask)
python annotator.py --image patient_001.png \
    --annotator alice --output-log experiment_log.csv
```

### 输出的日志 CSV 字段

| 字段 | 含义 |
|------|------|
| `time_seconds` | 有效标注耗时 (秒) |
| `brush_strokes` | 涂抹笔画数 |
| `eraser_strokes` | 擦除笔画数 |
| `total_strokes` | 总笔画数 |
| `undo_count` | 撤销次数 |
| `brush_pixels` | 涂抹总像素数 |
| `eraser_pixels` | 擦除总像素数 |
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
    --image-dir /path/to/test_images \
    --output-dir /path/to/output_masks \
    --trim-suffixes _img _image   # 可选: 去掉文件名后缀
```

---

## 三、实验运行 (run_experiment.py)

### 交叉设计

默认启用交叉设计 (cross-over):
- 每位标注者分配到一半的手动任务和一半的辅助任务
- 不同标注者之间, 同一张图的模式可以不同
- 避免了同一标注者对同一张图做两次 (消除记忆效应)

### 子命令

```bash
# 1. 生成配置
python run_experiment.py generate-config \
    --image-dir ./images \
    --mask-dir ./masks \
    --annotators alice bob charlie \
    --output-dir ./experiment_001 \
    --num-images 30 \
    --seed 42

# 2. 运行实验 (交互式, 支持断点续传)
python run_experiment.py run --config experiment_001/experiment_config.json

# 只运行某位标注者的任务
python run_experiment.py run --config experiment_001/experiment_config.json --annotator alice

# 3. 查看进度
python run_experiment.py status --config experiment_001/experiment_config.json
```

### 交互说明

运行时会逐个提示任务:
```
Task 1/30
Annotator: alice | Round: 1 | Mode: manual
Press ENTER to start, or type 'skip'/'q' to quit
```

- 按 Enter → 启动标注工具
- 输入 `skip` → 跳过当前
- 输入 `q` → 退出 (可断点续传)
- 标注完成后按 S 保存, 自动进入下一个任务

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

### 统计检验

- **Mann-Whitney U 检验**: 全局手动 vs 辅助比较
- **Wilcoxon signed-rank 检验**: 按图像配对比较
- 每位标注者单独分析
- 自动计算时间节省百分比

### 使用

```bash
# 完整分析
python analyze_results.py \
    --log experiment_001/experiment_log.csv \
    --output-dir ./analysis_output

# 只输出 LaTeX 表格
python analyze_results.py \
    --log experiment_001/experiment_log.csv \
    --latex-only
```

---

## 五、实验设计要点

### 参与者数量
- 最少 2 人 (1 资深 + 1 初级)
- 推荐 3-5 人

### 图像数量
- 最少 10 张 (配对检验需要足够样本)
- 推荐 20-40 张

### 洗脱期 (Washout)
- 两轮之间至少间隔 1 周
- 由实验组织者手动控制 (两轮之间暂停实验)

### 数据收集建议
除了计时和操作次数, 建议额外记录:
- 标注开始前的练习期 (不计入统计)
- 标注后的主观难度评分 (1-5)
- 修正前后的 Dice (对比标注质量)

---

## 六、快速测试

```bash
# 在任意几张图像上快速测试工具链
mkdir -p test_data/images test_data/masks test_experiment

# 1. 放入几张测试图像到 test_data/images/

# 2. 生成 mask (先用脚本里的简单阈值)
python generate_masks.py \
    --image-dir test_data/images \
    --output-dir test_data/masks

# 3. 生成实验配置
python run_experiment.py generate-config \
    --image-dir test_data/images \
    --mask-dir test_data/masks \
    --annotators test_user \
    --output-dir test_experiment

# 4. 运行实验
python run_experiment.py run --config test_experiment/experiment_config.json

# 5. 分析
python analyze_results.py \
    --log test_experiment/experiment_log.csv \
    --output-dir test_analysis
```

## 依赖安装

```bash
pip install opencv-python numpy scipy matplotlib
```
