# ============================================================
# 新管线 —— 统一参数配置
# 基于 pipeline.md 稳态热传导 + 红外辐射模型
# 所有温度单位: K，长度单位: m，功率单位: W，辐亮度单位: W/(m²·sr)
# ============================================================

# ---------- 项目路径 ----------
PROJECT_ROOT = r"D:\codes\MTIR-Blender-InfraRed-Material-Tools"  # 项目根目录，Blender文本编辑器运行时必须配置

# ---------- 模型缩放 ----------
MODEL_SCALE = 50.0           # 模型→真实尺寸的缩放因子 (1.0=不缩放)
MERGE_VERTEX_DIST = 0.0001    # 顶点合并距离（模型尺度），合并前焊接未对齐的接缝顶点
                              # 注意: 此值×MODEL_SCALE=实际合并距离。过大会导致薄壁结构(机翼)塌陷
SYMMETRIZE_MESH = False       # 计算前将网格沿X=0强制对称化 (消除左右几何不对称)

# ---------- 初始温度 ----------
T_AIRCRAFT_INIT = 280.0     # 蒙皮远场/初始温度 (非热源面片均初始化为此值)

# ---------- 热源参数 ----------
T_EXHAUST = 900.0           # 发动机尾焰热源温度 (T_o)
EXHAUST_RADIUS_MULT = 0.5   # 尾焰半径 = 发动机半径 × 此值
ENGINE_PROXIMITY_RADIUS = 0.08  # 蒙皮贴近发动机的距离阈值 (m)，面片中心距发动机表面小于此值即为热源

# ---------- 材料属性 ----------
EMISSIVITY = 0.85           # 蒙皮红外发射率 ε
K_STRUCTURE = 205.0         # 连接结构件导热系数 (W/(m·K))，铝 ≈ 205
A_STRUCTURE = 3.14e-4       # 等效传导截面积 (m²)，参考 Φ20 mm 铝撑杆
K_SKIN = 205.0              # 蒙皮导热系数 (W/(m·K))，铝 ≈ 205
SKIN_THICKNESS = 0.002      # 蒙皮厚度 (m)，典型 2 mm

# ---------- 环境参数 ----------
T_AMB = 280.0               # 辐射 sink 环境温度 (K)
Q_I = 0.0                   # 入射辐射项 (W)，基线设为 0

# ---------- 物理常数 ----------
SIGMA = 5.670374419e-8      # 斯特藩-玻尔兹曼常数 (W/(m²·K⁴))
C1 = 3.7418e-16             # 第一辐射常数 2πhc² (W·m²)
C2 = 1.4388e-2              # 第二辐射常数 hc/k (m·K)

# ---------- 气动加热 ----------
MACH_NUMBER = 0.8            # 飞行马赫数，摩擦升温 ΔT = T₀·0.16·M²

# ---------- 热源功率 ----------
Q_O = 9.564                  # 发动机热功率 per 连接面片 (W)，None=需先运行 calibrate_qo.py 校准

# ---------- 求解器参数 ----------
HEAT_SOURCE_TOL = 1e-3      # 热源求解器容差 (W)，对应 F(T) 残差
DIFFUSION_TOL = 0.1         # 扩散迭代收敛容差 (K)
MAX_ITERATIONS = 20000      # 扩散最大迭代次数
DIFFUSION_DECAY = 0.0          # 扩散衰减系数 α ∈ [0,1)，每步向 T_AMB 衰减: T_new = (1-α)·ΣwT + α·T_amb
                              # 0=纯扩散，0.01=轻微衰减，越大热量随距离消散越快
DECIMATE_RATIO = 1.0       # 面片缩减比例，1.0=不减, 0.15=保留15%面数
USE_EXTERNAL_COMPUTE = True # 是否使用外部 Python 进程加速 (需要 .venv 已安装 numpy/numba/scipy)

# ---------- 辐射计算 ----------
LAMBDA_1 = 8.0e-6           # 探测波段下限 8 μm → 真实 SI: 8×10⁻⁶ m
LAMBDA_2 = 12.0e-6          # 探测波段上限 12 μm → 真实 SI: 12×10⁻⁶ m

# ---------- 大气衰减 ----------
MU_ATM = 2.0e-5             # 8-12 μm 波段大气平均衰减系数 (m⁻¹)
# 典型值: 2e-5=良好条件(10km τ≈0.82), 5e-5=中等, 1e-4=高湿/低能见度
DETECTOR_POS = (0.0, 3000.0, -10000.0)  # 探测器世界坐标 (x, y, z) 单位 m
# 默认: 飞机前方3km, 下方10km (地面探测站视角)
DETECTOR_LOS = None              # 探测器视线方向 (x, y, z) 单位向量, None=自动指向目标中心

# ---------- 环境辐射 ----------
ENV_RADIATION_ENABLED = False  # 是否启用环境反射辐射 (太阳+天空+地面)，默认关闭
SUN_CONSTANT = 1353.0        # 太阳常数 I₀ (W/m²)，大气层外垂直入射辐照度
ATM_TRANSPARENCY = 0.75      # 大气透明度 P，范围 (0,1]，1=完全透明
SUN_ELEVATION = 0.785        # 太阳高度角 h (rad)，π/4=45°
SUN_AZIMUTH = 0.0            # 太阳方位角 (rad)，0=正南，π/2=正西 (gLTF/Blender坐标系 Y轴为北)
DAY_NUMBER = 182             # 日期序号 n，1=1月1日，182=7月1日
WATER_VAPOR_PRESSURE = 15.0  # 近地面水汽压 e (hPa)，典型值 5-30
AIR_TEMPERATURE = 280.0      # 飞行高度大气温度 T_air (K)
ALPHA_1 = 0.4                # 地面辐射比例常数 α₁ (原文未给出具体值，取合理估计)
EARTH_ANGLE_COEFF = 0.7      # 地球辐射角系数 f_{f_i} ∈ [0,1]，与面片朝向有关

# ---------- 传感器能量衰减 ----------
TAU0 = 0.85                   # 光学系统透射系数 τ₀，范围 (0,1]
K_E = 2.0                     # 有效光圈数 K_e (F数)，典型 1.0–4.0
BETA_RATIO = 0.0              # 线放大率比 β'/β_p，远距离目标≈0

# ---------- 灰度转换 ----------
DN_MIN = 0                     # 灰度输出下限 (DN)
DN_MAX = 255                   # 灰度输出上限 (DN)

# ---------- 输出保存 ----------
SAVE_PROCESSED_BLEND = True                   # 处理完后是否保存 .blend 文件
PROCESSED_BLEND_SUFFIX = "_IR"                # 处理后文件的后缀 (例如 aircraft_IR.blend)

# ---------- 过程图片输出 ----------
PROCESS_IMAGES_ENABLED = False              # 是否输出管线各阶段的过程图片
PROCESS_IMAGES_DIR = "//process_images/"     # 过程图片输出目录 (// = .blend 所在目录)

# ---------- 渲染输出 ----------
RENDER_ENABLED = False                       # 是否输出多视角辐射渲染图 (暂关闭)
RENDER_OUTPUT_DIR = "//thermal_renders/"     # 多视角输出目录 (// = .blend 所在目录)
RENDER_COLOR_MODE = "bw"                # 'thermal' (蓝-红) 或 'bw' (黑白)
RENDER_VMAX_PERCENTILE = 97.0               # vmax 取数据的此百分位 (0-100)
RENDER_RESOLUTION = (1920, 1080)

# ---------- 材质着色 ----------
RENDER_GAMMA = 0.7                       # Gamma Power 指数, <1 提亮中间调, 1.0=无变化
RENDER_EMISSION_STRENGTH = 1.2           # Emission 着色器强度
RENDER_BW_THRESHOLD = 0.30              # bw模式灰度-白色分界点 (0-1)，低于此值的都显示灰色，高于此值渐变到白
RENDER_BW_BASE_GRAY = 0.0               # bw模式低温区灰度 (0-1)，0=纯黑, 1=纯白

# ---------- 物体名称 ----------
OBJ_NAMES = ["Aircraft", "Engin_L", "Engin_R"]
