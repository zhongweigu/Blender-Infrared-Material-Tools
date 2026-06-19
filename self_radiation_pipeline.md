## 4.3. Zero-distance IR modeling  

The radiation released from target is made up of two main parts: self-radiation and reflect radiation. After getting the self-radiation and the reflect radiation, radiation of detector's direction should also be calculated.  


### 4.3.1. Self-radiation  

Planck’s law reveals the distribution law of relationship between black-body radiation energy and wavelength \(\boldsymbol{\lambda}\) and temperature \(\boldsymbol{T}\). The equation is:  
\[ M_\lambda = C_1 \lambda^{-5} \cdot \frac{1}{e^{C_2/(\lambda T)} - 1} \tag{4} \]  
where \( C_1 \) and \( C_2 \) are radiation constant. Since most objects in reality are not black-body, the concept of emissivity is introduced to illustrate the level of correlation between real object radiation and black-body radiation. For lambertian emitter, target radiance is not relevant to direction. The spectral radiance can be expressed as:  
\[ L_\lambda = \frac{\varepsilon_0 \cdot M_\lambda}{\pi} \tag{5} \]  
\(\varepsilon_0\) is the emissivity. By computing the integration over needed bands, the target radiance can be calculated with the following expression. Normally there are two detection bands of infrared detector: \( 3\!-\!5\,\mu\text{m} \) and \( 8\!-\!12\,\mu\text{m} \).  
\[ L_{\text{self}} = \int_{\lambda_1}^{\lambda_2} \frac{\varepsilon_0}{\pi} \cdot C_1 \lambda^{-5} \cdot \frac{1}{e^{C_2/(\lambda T)} - 1} d\lambda \tag{6} \]  

Common real-time calculation methods are to use empirical-formula instead of solving numerical integration directly. This paper adopts the formula as follows [17]:  
\[ 
\begin{aligned}
L_{\text{self}} &= \frac{\varepsilon_0}{\pi} \cdot \frac{C_1}{C_2/T} \cdot e^{-C_2/(x T)} \\
&\quad \times \bigg \{ 
x^3 + \frac{3}{C_2/T \cdot \left[ x^2 + \frac{2}{C_2/T \cdot \left( x + \frac{1}{C_2/T} \right)} \right]} 
\bigg \}  \bigg|_{x=\frac{1}\lambda_1}^{x=\frac{1}\lambda_2} \tag{7}
\end{aligned}
\] 
Simulation results show that error of this formula is less than \( 1\% \) when \( T < 3000\,\text{K} \). For example, assuming the temperature of a point is \( 498\,\text{mK} \), the accurate value of radiance is \( 511.5\,\text{W/(m}^2\cdot\text{sr)} \) between \( 3 \) and \( 5\,\mu\text{m} \). While the approximation value is \( 510.9\,\text{W/(m}^2\cdot\text{sr)} \) calculated with this formula, and the relative error is \( 0.12\% \).