# ROI_level 目录结构实验运行说明

这里的目标是让 `QCL_ROI_level` 的脚本具备与 patch-level 相同的“入口约定”：
`src/` 入口脚本 + `configs/` YAML 配置 +（可选）`local_overrides.yaml` / 环境变量优先级覆盖。

## 1) 运行方式（通用）

每个脚本都支持：
- 直接运行（使用脚本源代码里的默认绝对路径）
- 使用 `--config` 指定 YAML（YAML 里的 `paths.*` 会覆盖脚本常量）

示例（以分类/回归等 ROI 任务脚本为例）：
```bash
cd src
python roi_rf_regression.py --config ../configs/roi_rf_regression_pimo_roi_level.example.yaml
```

## 2) 路径解析优先级（与 patch_level 对齐的精神）

由于 YAML 模板目前使用“相对路径”（相对于 `QCL_DATA_PATH` 解析），因此当你想用 YAML 覆盖默认路径时：

1. 设置环境变量 `QCL_DATA_PATH`（建议指向 `E:\temp project code\PIMO`）
2. 再运行脚本并传入 `--config`

Windows PowerShell 示例：
```powershell
$env:QCL_DATA_PATH="E:\temp project code\PIMO"
python roi_rf_regression.py --config ../configs/roi_rf_regression_pimo_roi_level.example.yaml
```

或者创建（本机专用，不提交）：
- `configs/local_overrides.yaml`（从 `local_overrides.example.yaml` 复制并按需修改）

`experiment_config.py` 的相对路径解析顺序为：
- `QCL_DATA_PATH` 环境变量
- `configs/local_overrides.yaml` 里的 `data_path`
- 如果两者都缺失，则不会用相对路径去覆盖脚本默认值（保留脚本内默认绝对路径）

## 3) 各脚本命令（推荐）

### `roi_rf_regression.py`（ROI 级 RF 回归）
```bash
cd src
python roi_rf_regression.py --config ../configs/roi_rf_regression_pimo_roi_level.example.yaml
```

### `build_roi_raw_data.py`（从 core + ROI_CLICKS 生成 padded raw ROI）
```bash
cd src
python build_roi_raw_data.py --config ../configs/build_roi_raw_data_pimo_roi_level.example.yaml
```

### `roi_distribution_plots.py`（ROI 分布可视化与统计检验）
```bash
cd src
python roi_distribution_plots.py --config ../configs/roi_distribution_plots_pimo_roi_level.example.yaml
```

### `roi_hypoxia_rf_classifier_full_vis.py`（ROI 像素级特征抽样 + 反复 hold-out 的 RF 缺氧二分类全可视化）
```bash
cd src
python roi_hypoxia_rf_classifier_full_vis.py --config ../configs/roi_hypoxia_rf_classifier_full_vis_roi_level.example.yaml
```

## 4) 最小验证清单（不做大规模训练）

1. 先确保 Python 能成功导入配置模块，并在控制台看到脚本开头的提示，例如：
   - `[config] roi_rf_regression: config loaded / overrides applied.`
2. 确认输出目录能创建（脚本里会 `os.makedirs(..., exist_ok=True)`）。
3. 若数据路径较大导致耗时，请先用一小段数据/子集运行（如果你能调整 manifest 为小文件）。

