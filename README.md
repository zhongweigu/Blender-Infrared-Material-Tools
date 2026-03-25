

# BLIR (Blender InfraRed Material Tools)

一个 Blender 插件/脚本，用于为 3D 模型生成红外热成像材质。  
支持考虑太阳照射、气动热量和发动机等热源，适合航空器或其他工程模型的热分布可视化。

## 功能

- 为 Blender 网格对象生成红外材质
- 支持彩色（蓝-黄-红）和黑白模式
- 可考虑环境温度、太阳照射和气动热量
- 可扩展其他热源计算方法
- 输出顶点辐射数据，可用于材质驱动 Shader

## 安装

1. 将 `bl_IR` 文件夹放入 Blender 脚本路径或者项目目录
2. 将项目目录添加到sys目录
3. 在 Blender 脚本编辑器中运行 `main.py`

## 批量处理

使用 `aircraft/batch_apply_ir.py` 批量为指定目录中的 .obj 模型应用红外材质。

### 配置

在脚本开头修改路径配置：

```python
SHAPENET_ROOT = r"你的模型路径"
OUTPUT_DIR = r"输出目录"
RENDER_IMAGE = False  # True 时同时渲染预览 PNG
```

### 运行

命令行运行（需要 Blender 4.2）：

```bash
blender -b --python ./aircraft/batch_apply_ir.py
```

或在 Blender GUI 中打开脚本后 Alt+P 运行。

### 输出

- `.blend` 文件：每个模型一个，以模型 ID 命名
- `_render.png`（可选）：渲染的 IR 图像

## 使用示例

```python
import bpy
script_dir = os.path.dirname(bpy.data.filepath)
if script_dir not in sys.path:
    sys.path.append(script_dir)
from bl_IR import material, config, radiation, location

# 获取对象
obj = bpy.data.objects["Aircraft"]
mesh = obj.data

# 计算辐射值
radiation_values = [radiation.calculate(temp_K) for temp_K in temps_list]

# 应用材质
material.assign(obj, mesh, radiation_values, mode=0)  # 彩色
material.assign(obj, mesh, radiation_values, mode=1)  # 黑白


```

## 说明

| **代码部分**                                                 | **对应公式**                                                 | **物理方法/依据**                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------------------------- |
| `E_self = radiation.calculate(T_inf_K)`                      | $E_{\text{self}} = \varepsilon \sigma T^4$                   | 斯特藩–玻尔兹曼定律，物体自身热辐射。                      |
| `E_sun = config.emissivity * I_sun * cos_theta`              | $E_{\text{sun}} = \alpha_{sun} I_{sun} \cos \theta$          | 太阳辐射吸收，余弦定律，近似大气透过率 0.7；α_sun≈ε。      |
| `T_recover = radiation.recovery_temperature(T_inf_K)`  `E_aero = radiation.calculate(T_recover) - radiation.calculate(T_inf_K)` | $T_r = T_\infty \big(1 + r \frac{\gamma - 1}{2} M^2 \big)$  $E_{\text{aero}} = \varepsilon \sigma T_r^4 - \varepsilon \sigma T_\infty^4$ | 气动加热采用恢复温度模型（航空热力学），再转化为辐射差值。 |
| `dT_jet = max(dTL, dTR)`  `E_jet = radiation.calculate(T_inf_K + dT_jet) - radiation.calculate(T_inf_K)` | $E_{\text{jet}} = \varepsilon \sigma (T_\infty+\Delta T_{jet})^4 - \varepsilon \sigma T_\infty^4$ | 发动机喷流加热，经验衰减模型 → 转换为辐射差分。            |
| `E_boost = radiation.calculate(T_inf_K + engine_heat_delta) - radiation.calculate(T_inf_K)` | 同上（固定加热量ΔT_engine）                                  | 发动机本体附加温升（设计参数），转为辐射差分。             |
| `E_total = E_self + E_sun + E_aero + E_jet + E_boost`        | $E_{total} = E_{\text{self}} + E_{\text{sun}} + E_{\text{aero}} + E_{\text{jet}} + E_{\text{boost}}$ | 多热源辐射通量线性叠加（符合辐射传输理论）。               |
| `rad = config.TAU * E_total + (1.0 - config.TAU) * E_bg`     | $E_{\text{sensor}} = \tau E_{total} + (1-\tau)E_{bg}$        | 大气透过率修正，Beer–Lambert 定律。                        |
| `temp_K = radiation.cfd_analysis(...)`                       | （修正项，不固定公式）                                       | CFD 数值修正，补充局部流场/传热效应。                      |
