Fig. 3. Physical effects simulation process for an infrared sensor.

## 4.1. Sensor modeling for physical effects of spatial domain

Fig. 2 shows some typical sensor spatial effects. Here we take the vignetting, motion blur and geometric distortion effects as an example to illustrate the pixel based simulation method.

Energy degradation effect is caused by optical components which partially absorb or reflect the incident radiation energy of optical system. The expression of energy degradation is shown as formula (4):

$$
E(\lambda) = \frac{\tau\pi\tau_0(\lambda)M(\lambda,T)}{4K_e^2(1-\beta'/\beta_p)^2} \tag{4}
$$

where $M(\lambda,T)$ is radiant exitance, $\tau$ is atmospheric spectral transmission coefficient and $\tau_0(\lambda)$ represents attenuation coefficient. $K_e^2$ stands for effective raster number of lens. $\beta'$ and $\beta_p$ is line magnification of optical system imaging and lens pupil respectively [20]. Due to the existence of $\tau_0(\lambda)$, so that incident infrared radiation energy of optical system is degraded.

For the vignetting effect, suppose an image with pixels of $N \times M$ and the center coordinate of the image is $(N/2, M/2)$. Define a vignetting coefficient shown as formula (5):

$$
l = \frac{\sqrt{(N/2-i)^2 + (M/2-j)^2}}{\sqrt{(N/2)^2 + (M/2)^2}} d \tan w \tag{5}
$$

where $w$ is half field of view, $d$ is distance between entrance pupil and entrance window [21]. The gray level of each pixel of the original image is multiplied by its corresponding vignetting coefficient to simulate the vignetting effect.

For the geometric distortion effect, a kind of geometry transformation relation between an original pixel's position and its distorted pixel's position is built and shown in formula (6):

$$
\begin{cases}
x_i = x_i' + \delta_x(x_i', y_i') \\
y_i = y_i' + \delta_y(x_i', y_i')
\end{cases} \tag{6}
$$

where $(x_i', y_i')$ is the original pixel coordinates and $\delta(x_i', y_i')$ is the corresponding offset [6]. Although this pixel based simulation method can be used to adjust a result image flexibly, we need to further discovery the correspondence between processed result images and actual physical parameters of infrared sensors. In addition, this method cannot simulate the physical effects in frequency domain.

If there is a large relative velocity between the infrared sensor and the detected entity, it will lead to an abnormal color accumulation of the output infrared image which is called as motion blur. To simulating this effect, a current frame is mixed with its last frame, rendering to texture and outputting to the screen.

In well-designed infrared imaging system, detector noise is the main noise source. Even if there is no signal input, it will produce some erratic and unpredictable output inevitably. Detector noise can be classified as Poisson noise and Gaussian noise from the perspective of probability theory. Poisson noise probability density distribution is:

$$
P(x) = \frac{\lambda^x e^{-\lambda}}{x!} (x = 0,1,2,\cdots \lambda > 0) \tag{7}
$$

where $x$ stands for the gray level of noise pixel. Many types of detector noise can be summarized as Poisson noise, such as photoelectric conversion noise, signal circuit noise and transition noise.

Gaussian noise probability density distribution is:

$$
P(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-x^2/2\sigma^2} \tag{8}
$$

where $x$ stands for the gray level of noise pixel and $\sigma$ is the RMS (root mean square) value of noise. According to the statistical analysis, it can be found that the superimposition of many kinds of noises such as thermal noise and temperature noise can be regarded as white Gaussian noise [22].