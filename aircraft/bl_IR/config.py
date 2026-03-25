import bpy
import numpy as np

# 基本环境参数
ambient_temp_C = -50.0 # 飞行环境的自由来流温度（通常指高空外界空气温度）
solar_delta = 25.0
aero_delta = 10.0
emissivity = 0.85
sigma = 5.670374419e-8
# 普朗克常量
h = 6.62607015e-34
c = 2.99792458e8
kB = 1.380649e-23
wavelength = 10e-6  # 10 μm

emission_strength = 245.0 # 整体辐射显示强度

# 大气传输修正（可选）
USE_ATMOS_CORR = True
TAU = 0.85                # 透过率
T_BACKGROUND_C = 15.0     # 大气辐射背景温度，即通过大气层衰减之后，传到传感器的背景辐射等效温度

# 计算方法:stefan_boltzmann, plank_law
METHOD="plank_law"

# 是否考虑太阳辐射
CONSIDER_SUN = True
sun_dir = np.array([0, -1, -1])
sun_dir = sun_dir / np.linalg.norm(sun_dir)
# 是否考虑气流
CONSIDER_AERO = True
forward_dir = np.array([0, 1, 0])

# 输出模型，0是RGB材质， 1是仿红外黑白材质
OUTPUT_MODE = 1

# 是否考虑温度场
CONSIDER_CFD = True
# ---------- 解析近似 CFD 场参数 ----------
JET_CENTERLINE_DT0 = 100.0      # 发动机喷流中心线最大升温（K）
JET_Y_DECAY = 15.0              # 沿 y 的指数衰减长度（m）
JET_RADIAL_SIGMA = 1.5          # 径向高斯标准差（m）
JET_MAX_Y_LENGTH = 20.0         # 喷流影响最大长度（m）
JET_DIRS = [(0.0, 1.0, 0.0), (0.0, 1.0, 0.0)]  # 发动机喷流方向

# 边界层/气动加热（恢复温度）
GAMMA = 1.4
PR = 0.71
MACH = 0.8
RECOVERY_FACTOR = PR ** (1.0/3.0)  # ~0.88

# 传感器噪声
CONSIDER_NOISE = True
NOISE_LEVEL = 0.05   # 噪声水平 (5% 标准差)

# 传感器位置
CAMERA_POS = (1.50, -3.00, 2.00)
KAPPA = 0.01                      # 大气消光系数 (1/m)，典型值 0.005~0.05

# 物体属性名及温度配置
obj_names = {
    "Aircraft": 0.0,
    "Engin_L": 140.0,
    "Engin_R": 140.0
}
