## 2.3 Atmospheric transmittance model
## 2.3 大气透射模型

For ground-based infrared detection systems, the atmospheric environment absorbs and scatters infrared radiation within different wavelength bands. Radiation emitted from the target surface is inevitably attenuated by the atmosphere before reaching the detector. Atmospheric transmittance along the detection path under cloud and rain conditions is calculated using the MODTRAN software [20].

对于地面红外探测系统，大气环境会吸收并散射不同波长波段的红外辐射。从目标表面发射的辐射在到达探测器之前必然会被大气衰减。在云和雨条件下，沿探测路径的大气透射率通过 MODTRAN 软件[20]计算。

The relationship between the atmospheric spectral transmittance $ \tau_a(\lambda) $ and the attenuation coefficient $ \mu(\lambda) $ in atmospheric transmission can be described by Bouguer-Lambert's law [21], expressed as Equation 8:

大气光谱透射率 $ \tau_a(\lambda) $ 与大气透射衰减系数 $ \mu(\lambda) $ 之间的关系可以用布格-朗伯特定律[21]描述，表达为方程 8：

$$
\tau_a(\lambda) = \frac{\varphi_e(\lambda,R)}{\varphi_e(\lambda,0)} = e^{-[\mu(\lambda)R]} \quad \{8\}
$$

where $ R $ is the distance between the infrared detection system and the target, $ \varphi_e(\lambda, R) $ represents the spectral density of the target or background radiation flux at a distance $ R $, and $ \varphi_e(\lambda, R) $ denotes the spectral density of the radiation flux at $ R = 0 $, $ \lambda $ represents the wavelength. Atmospheric attenuation of IR radiation consists mainly of absorption by CO₂ and H₂O and scattering by some suspended particles in the atmosphere. The total transmittance is calculated using Equation 9:

其中 $ R $ 是红外探测系统与目标之间的距离，$ \varphi_e(\lambda, R) $ 表示目标或背景辐射通量在距离 $ R $ 处的光谱密度，$ \varphi_e(\lambda, R) $ 表示 $ = 0 $ 处 $ R $ 的辐射通量光谱密度，$ \lambda $ 表示波长。大气中红外辐射的衰减主要包括一氧化碳 2 和氢 2 气吸收，以及大气中悬浮颗粒的散射。总透射率通过方程 9 计算：

$$
\tau_a(\lambda) = \tau_{H_2O}(\lambda) \cdot \tau_{CO_2}(\lambda) \cdot \tau_s \quad \{9\}
$$

where $ \tau_{H_2O}(\lambda) $ represents the transmittance of H₂O, $ \tau_{CO_2}(\lambda) $ represents the transmittance of CO₂, and $ \tau_s $ denotes the scattering transmittance.

其中 $ \tau_{H_2O}(\lambda) $ 表示 H₂O 的透射率，$ \tau_{CO_2}(\lambda) $ 表示 CO₂ 的透射率，$ \tau_s $ 表示散射透射率。