# nature_shap_draft.py 使用说明

## 作用
这个脚本会生成一张 Nature 风格的 SHAP 整图草案：

- 顶排嵌入 3 张已有的 beeswarm 图
- 中、底排使用 `D:\WorkFiles\A DataSets\ThyroidAgent\test_image` 中随机抽取的真实超声样例
- 每个样例左侧为真实超声图像，右侧为局部 SHAP 条形图
- 最终输出 PNG 和 PDF

## 运行环境
在 `D:\workspace\ThyroidAgent\plot_images` 目录下运行。

脚本依赖：

- Python 3
- `matplotlib`
- `numpy`

如果缺依赖，先安装：

```bash
pip install matplotlib numpy
```

## 运行方式
在项目根目录执行：

```bash
python nature_shap_draft.py
```

## 输出文件
脚本会自动把结果写到 `out/` 目录：

- `out/nature_shap_draft.png`
- `out/nature_shap_draft.pdf`

## 输入文件要求
脚本默认读取这 3 张 beeswarm 图：

- `all_images/LightGBM_BAG_L1_beeswarm_LymphUs.png`
- `all_images/LightGBMXT_BAG_L1_beeswarm_BM.png`
- `all_images/LightGBMXT_BAG_L1_beeswarm_FTCPTC_FangDai.png`

如果文件名变了，需要同步修改脚本里的 `BEE_SWARMS`。

## 需要改图时改哪里
如果你想调整草案结构，直接改 `nature_shap_draft.py` 里的这些位置：

- `BEE_SWARMS`：控制顶排 3 张 beeswarm 图
- `TASK_FEATURES`：控制每个任务右侧局部 SHAP 图的特征名
- `pick_test_images()`：控制从 `test_image` 里抽取哪几张真实超声图
- `load_ultrasound_image()`：控制真实超声图的灰度归一化方式
- `sample_panels`：控制样例顺序、正负模式和随机种子

## 建议的使用流程
1. 先运行脚本生成草案图
2. 检查排版是否合适
3. 再把随机样例图微调成更符合论文叙述的真实案例
4. 最后根据论文版面微调字体、间距和图注
