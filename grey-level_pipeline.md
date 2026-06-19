2. Calibration-based NUC2. 基于校准的 NUC 分析

Infrared systems, in many applications, are operated in a range of irradiance within which detectors exhibit linear input-output characteristics. The output gray value (DN, digital number) of a single detector is given by the approximate linear relation [21], [22]:

红外系统在许多应用中工作于一个辐照度范围内，探测器在该范围内表现出线性输入-输出特性。单个探测器的输出灰值（DN，数字数）由近似线性关系[21]，[22]给出：

$$
Y_{i,j}^{(k)} = G_{i,j}^{(k)} \times X^{(k)} + O_{i,j}^{(k)} + \sigma_{i,j}^{(k)},
\tag{1}
$$

where $k$ denotes the frame number. $G_{i,j}$ and $O_{i,j}$ are respectively the gain and offset of the $(i,j)$th detector, and $X$ denotes the true infrared radiance of the imaging target. $\sigma_{i,j}$ refers to the temporal noise which can be effectively reduced by averaging 20 or more frames. In general, gains and offsets of pixels are different, and NUC intends to draw them to uniform. After NUC, the calibration formula, Eq. (1), should be theoretically corrected to

其中 $k$ 表示帧数。$G_{i,j}$ 和 $O_{i,j}$ 分别是 $(i,j)$ 个探测器的增益和偏移，$X$ 表示成像目标的真实红外辐射。$\sigma_{i,j}$ 指的是通过平均 20 帧或更多帧可以有效减少的时间噪声。一般来说，像素的增益和偏移不同，NUC 意图将其绘制为均匀。经过 NUC 后，校准公式（方程 1）应理论上修正为

$$
\overline{Y} = \overline{G} \times X + \overline{O}
\tag{2}
$$

where $\overline{Y}$ denotes the average of pixel outputs, $\overline{G}$ is the corrected gain, and $\overline{O}$ is the corrected offset. Actually, the desired response of each detector to the uniform radiation is the average of pixel outputs, namely $\overline{Y}$. The purpose of NUC is just to correct the response of each detector to the uniform radiation to be uniform. However, it does not mean that the gain and offset are corrected to the mean ones, that is to say Eq. (2) does not holds.

其中 $\overline{Y}$ 表示像素输出的平均值，$\overline{G}$ 是修正增益，$\overline{O}$ 是修正后的偏移量。实际上，每个探测器对均匀辐射的期望响应是像素输出的平均值，即 $\overline{Y}$。NUC 的目的只是校正每个探测器对均匀辐射的响应，使其均匀。然而，这并不意味着增益和偏移被修正为均值，也就是说，方程（2）不成立。