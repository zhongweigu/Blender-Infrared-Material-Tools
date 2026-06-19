# An Infrared Image Synthesis Model Based on Infrared Physics and Heat Transfer  
**Authors:** Weijie Yu, Qunsheng Peng, Hongming Tu, Zhangye Wang  
**Affiliation:**  
1 State Key Laboratory of CAD and CG, Zhejiang University, Yugu road 20, Hangzhou 310027, P.R. China  
2 Wenzhou Teachers College, Wenzhou 325003, P.R. China  
**Email:** zucad@public.hz.zj.cn  
**Received:** August 25, 1998  
**Journal:** *International Journal of Infrared and Millimeter Waves*, Vol. 19, No. 12, 1998  

---

## Abstract
Most of current image synthesis models are based on optics of visible-spectrum. While simulating the illumination effect of visible light, they cannot show the infrared signature of the objects. This paper presents a synthesis model for generating realistic infrared images. We first establish a heat equilibrium equation of the object surface. Then according to it and the heat transfer inside the object and on the boundary surface of the object, we compute the temperature and radiometries of each surface patch of the object. Finally on the basis of the radiometries, each patch is drawn by Gouraud Shading. Experimental examples of the generated infrared images are presented, which illustrate the potential of our method.

**Keywords:** infrared image; infrared physics; heat transfer

---

## 1. Introduction
Infrared (IR) images of scenes have found great applications in the military area, aviation and spaceflight, for system performance evaluation, algorithm development, mission planning, training, remote sensing[1] and recently in multisensor fusion for scene interpretation[2]. For all these applications, understanding and modeling the thermal behavior of objects in the scene is of great interest. However most of the current shading models for realistic image synthesis are based on optics of the visible-band, while simulating the illumination effect of visible light vividly these models cannot properly reproduce the IR images of objects caused by the surface temperature and intrinsic thermal properties.

It has been an exciting challenge for researchers to synthesize the IR image. In 1987, J. Hinderer presented an empirical approach. It divides the object surface into facets, which are assigned certain temperatures based on pre-compiled thermal data and heuristics[3]. In the same year, W.R. Owens presented an approach for extrapolating a thermal image to meet environmental condition[4]. Here again, the object surface is divided into facets, which are further subdivided into several thermal classes. A major drawback of these approaches is that they rely on heuristics and require a large thermal database, and the heat flow between the adjacent facets and between the interior volumetric elements is not accounted for.

In order to overcome the shortcoming in previous works and improve the fidelity of the synthetic infrared image, in 1987, G. Gerhart developed a model-based approach by which the radiance of each isothermal facet is computed by program[5]. In 1989, Chanhee Oh modeled accurate IR images of the object by using octree[6]. However, these works take no account of non-homogeneous structure material and the internal heat source of the object.

The method proposed by N. Nandhakumar in 1994 is the first method capable of simulating the thermal effects of the internal heat source[7]. The method constructs a hierarchical model of the object called V s-tree and evaluates the surface temperature by computing each element (node) of the object in sequence. However, as the shape of the object goes more complex, the method has to compute a large number of nodes necessitating a considerably large amount of computations load. In addition, this method is difficult to represent accurately the complex shapes of various objects.

In 1996, Hyum-Ki Hong developed an integrated object model that represents the heat transfer process within the non-homogeneous object as the equivalent thermal circuit[8]. It represents the behavior of heat conduction between the internal heat source and the surface of the object as thermal resistance. The proposed model effectively shows the influence of the internal heat source, but it determines the surface radiance of each facet by merely considering convection and the conduction from the internal heat source, hence ignoring the effects caused by heat flow between the adjacent facets.

After a comprehensive investigation of previous works, we recognize that a model for IR image synthesis should account for all forms of heat transfer within the object as well as the dynamic interaction between the object's surface and the environment. Also as the heat may flow from one part of the object to the other part, it is important to incorporate the 3D-object structure in the model. It is based on this idea that we proposed an infrared image synthesis model based on infrared physics and heat transfer.

The remainder of this paper is organized as follows. In section 2 we briefly review some basic concepts and equations in heat transfer and infrared physics. The heat transfer within an object is carefully studied and a new infrared image synthesis model is proposed in section 3. Section 4 demonstrates several examples rendered by our model and the difference between our method and the related works is discussed in section 5.

---

## 2. The fundamentals of heat transfer and infrared physics

### 2.1 Heat transfer
Whenever a temperature gradient exists within a system, or when two systems at different temperatures are brought into contact, energy is transferred. The process by which the energy transport takes place is known as heat transfer. Heat transfer has three distinct modes: **conduction**, **convection** and **radiation**.

- **Conduction** may be viewed as the heat transfer from the higher-temperature to the lower-temperature region of a solid medium. The heat transfer by conduction is described by Fourier’s Law. For the one-dimensional heat transfer by conduction illustrated in Fig.1, the rate equation is expressed as

$$q_{c d}=-k A\frac{\Delta t}{\delta}$$

where  
$q_{c d}$ — rate of heat transfer by conduction,  
$A$ — passed area,  
$k$ — thermal conductivity,  
$\Delta t$ — temperature difference,  
$\delta$ — length.  
The minus sign reflects the fact that heat is transferred in the direction of decreasing temperature. The quantity $\delta/ k A$ can be regarded as a thermal resistance $R_t$ of the in-between solid medium:

$$q_{c d}=-\frac{\Delta t}{R_t}$$

For the two-dimensional, steady-state conduction, the heat flow equation is:

$$\frac{\partial^2 t}{\partial x^2}+\frac{\partial^2 t}{\partial y^2}=0$$

- **Convection** may occur between a fluid and a bounding surface when the two are at different temperature. In this case, the rate of heat transfer by convection is calculated from the relation

$$q_{c v}=\bar{h} A\Delta t \qquad (5)$$

where  
$q_{c v}$ — rate of heat transfer by convection,  
$A$ — heat transfer area,  
$\bar{h}$ — convection coefficient.

(The third mode, **radiation**, is described in the next section.)

### 2.2 Infrared radiation
The process by which heat is transferred from a body by virtue of its temperature, without the aid of any intervening medium, is called thermal radiation. The radiation in the infrared region is called infrared radiation.

The spectral radiant energy emission per unit time and per unit area from a blackbody at wavelength $\lambda$ in the wavelength range $d\lambda$ is described by **Planck’s Law**:

$$E_{b\lambda}(T)=\frac{C_1\lambda^{-5}}{e^{C_2/(\lambda T)}-1}$$

where  
$E_{b\lambda}$ is the spectral radiant emittance of a blackbody at absolute temperature $T$,  
$\lambda$ is the wavelength,  
$C_{1}=3.742\times 10^{-16}\ \mathrm{W\cdot m^{2}}$,  
$C_{2}=1.4388\times 10^{-2}\ \mathrm{m\cdot K}$.

The wavelength at which the spectral radiant emittance attains its maximum, $\lambda_{\max}$, decreases with increasing temperature. The relationship

$$\lambda_m T=2.9\times 10^{-3}\ \mathrm{m\cdot K} \qquad (7)$$

is called **Wien’s displacement law**.

The total emission of radiation per unit surface area per unit time from a blackbody is related to the fourth power of the absolute temperature according to the **Stefan–Boltzmann law**:

$$E_{b}(T)=\sigma_{0} T^{4} \quad (8)$$

with the Stefan–Boltzmann constant $\sigma_{0}=5.67\times 10^{-8}\ \mathrm{W/(m^{2}\cdot K^{4})}$.

---

## 3. A synthesis model for infrared image generation

### 3.1 Analysis of infrared image generation
The factors affecting the thermal image of an object include object surface temperature, emissivity, atmospheric propagation, sensor characteristics and its geometry. Among them, the surface temperature of the object plays a leading role.

The distribution of surface temperature depends on:
- internal heat conditions (power and locations of internal heat sources),
- object structure,
- intrinsic thermal properties (material density, specific heat, conductivity, volume, surface geometry),
- dynamic interaction between the object’s surface and the environment.

Energy conservation across the body surface is described by a **heat equilibrium equation**:

$$q_1+q_{c d o}=q_{a b s}+q_{r a d}+q_{c d i}+q_{c v} \quad (9)$$

where  

- \(q_1 = q_{\text{sun}}+q_{\text{sky}}+q_{\text{ground}}\)
  - $q_{\text{sun}}$: solar irradiation  
  - $q_{\text{sky}}$: atmospheric scattered energy  
  - $q_{\text{ground}}$: ground emitted energy  
- $q_{c d o}$: conducted heat caused by internal heat sources  
- $q_{a b s}$: absorbed heat  
- $q_{rad}$: radiated heat into the environment  

$$q_{rad}=\varepsilon\sigma\big(T_{s}^{4}-T_{amb}^{4}\big),\quad \varepsilon\ \text{is emissivity}$$

- $q_{cdi}$: conducted heat into the object (conductive loss into interior / through structure)
- $q_{c v}$: convected heat  

$$q_{c v}=\bar{h}\left(T_{s}-T_{amb}\right)$$

- $T_s$: surface temperature  
- $T_{amb}$: ambient temperature  

According to Eq. (9), we can compute the surface temperature and generate the infrared image of the object. For a real object, its internal heat sources determine its principal infrared signature; therefore we study the component caused by the internal heat sources in more detail in the next section.

### 3.2 Heat transfer within the object
It is known that an internal heat source is connected with the outer shell of the object by some metal structural parts. As metal conducts heat very well, the conductive heat from the internal heat source plays a chief role in determining the surface temperature.

To calculate the conductive heat, we first subdivide the object surface into patches. For those patches connected with the heat source through the metal structural parts, we construct the equivalent thermal circuit:

$$R_N=\frac{L_N}{k\cdot A_N}
=\frac{1}{k}\int_0^{L_N}\frac{1}{A(l)}\,dl$$

$$A_N=\frac{L_N}{\displaystyle\int_0^{L_N} 1/A(l)\,dl}$$

where  
$A_N$ — effective cross-section area of the circuit,  
$A(l)$ — cross-section of the metal structural part at position $l$,  
$L_N$ — distance from the heat source to the center of each surface patch,  
$k$ — thermal conductivity of the metal structural parts.

According to Fourier’s Law, the heat conduction equation is expressed as

$$q_o-q_{c d o}=\frac{T_s-T_o}{R_N}$$

where  
$q_o$ is energy from the internal heat source;  
$T_o$ is the temperature of the heat source.

Using Stefan–Boltzmann’s law and the heat equilibrium Eq. (9), we obtain:

$$q_o+q_i-\varepsilon\sigma_{0}\left(T_s^4-T_{amb}^4\right)A_N=\frac{T_s-T_o}{R_N}$$

(原文编号此处给出 (13)，可用其确定与内热源通过金属结构相连的那些面片温度。)

### 3.3 Surface thermal diffusion
On the object surface, heat is transferred from the high temperature area (adjacent to the internal heat source) to the low temperature area. This is regarded as **thermal diffusion**, essentially conduction from hotter surface patches to surrounding adjacent patches.

Assuming the object is in a state of balance (steady-state conduction on the surface) and adopting an implicit finite-difference treatment of the 2D conduction operator:

$$\frac{\partial^2 t}{\partial x^2}\approx
\frac{t_{m+1,n}+t_{m-1,n}-2t_{m,n}}{\Delta x^2},\quad
\frac{\partial^2 t}{\partial y^2}\approx
\frac{t_{m,n+1}+t_{m,n-1}-2t_{m,n}}{\Delta y^2}$$

Let $\Delta x=\Delta y$; substituting into the 2D steady Laplace-type form gives the discrete update:

$$t_{m,n}=\frac{t_{m+1,n}+t_{m-1,n}+t_{m,n+1}+t_{m,n-1}}{4}$$

With this relation, by adopting the **Gauss–Seidel iterative method**, we can progressively calculate the temperature of all other unknown surface patches.

### 3.4 Generating thermal image based on surface temperature variance
From atmospheric physics we know that infrared radiation at different wavelengths experiences different propagation characteristics. In most spectral ranges the infrared radiation is seriously attenuated; only in certain **atmospheric windows** is the source radiation observable.

Therefore, when simulating the infrared images, we calculate the radiance of the object surface using Planck’s law integrated over an atmospheric window $[\lambda_1,\lambda_2]$. Noting that \(e^{C_2/(\lambda T)}\gg 1\) holds well in the infrared region, an approximation of Planck’s law is adopted:

$$E\approx\int_{\lambda_1}^{\lambda_2}\frac{C_1}{\lambda^3}\,
e^{-C_2/(\lambda T)}\,d\lambda$$

Using a partial-integral method, this integral can be approximated as

$$E\approx\frac{C_1}{C_2/T}\,
e^{-C_2 X/T}\Bigl\{
X^3+\frac{3}{C_2/T}\Bigl[
X^2+\frac{2}{C_2/T}\Bigl(X+\frac{1}{C_2/T}\Bigr)\Bigr]\Bigr\}
\Big|_{X=1/\lambda_1}^{X=1/\lambda_2}$$

According to Wien’s Displacement Law, if $\lambda T<3000\ \mu\mathrm{m\cdot K}$, the introduced relative error by the above approximation is less than about one percent.

After determining the radiance of each surface patch by the above integration, we calculate the radiance at each vertex by interpolation, and finally each surface patch is drawn by **Gouraud Shading**.

---

## 4. Results
Infrared images of an airplane were generated on an SGI IRIS 4D/35 workstation at State Key Laboratory of CAD&CG, Zhejiang University.

- Figures 6–8: infrared images in the **3–5 μm** band, with engine temperatures assumed as **900 K, 1200 K, 1300 K** (regular / first thrust augmentation / second thrust augmentation). The surface area near the engine appears brighter; brightness increases with engine temperature.
- Figure 9: infrared image in the **8–12 μm** band, engine at **1300 K**.  
Comparison: high-temperature region (~900–1200 K) looks brighter in mid-IR (3–5 μm) than in long-wave IR; low-temperature region (~273–350 K) looks brighter in long-wave IR (8–12 μm) than in mid-IR, consistent with blackbody spectral behavior.

(Real IR images would additionally be affected by atmospheric attenuation and sensor characteristics, reducing apparent brightness and contrast, and potentially deforming shape cues.)

---

## 5. Discussions and conclusions
Compared to the works of Nandhakumar and Hyum-Ki Hong, the proposed method emphasizes:

1. The model is set up based on infrared physics and heat transfer, simulating heat transfer **inside the object**, **on the boundary surface**, and **between object and environment**, with lower computation load than the Nandhakumar model.  
2. Only patches directly connected with the internal heat source via metal structural parts compute conductive heat from that source—more consistent with the real situation than treating the whole interior uniformly.  
3. Heat conduction among adjacent surface patches is handled as **surface thermal diffusion**, improving fidelity.  
4. The model can provide an initial surface-temperature value from internal heat-transfer alone, allowing internal vs. environmental effects to be separated/composited.

Future work: thermal contrast; synthesis for high-speed objects; atmospheric/sensor-effect modeling.

---

## References (as given in your text)
1. A.B. Kahle. “A simple thermal model of the earth's surface for geological mapping by remote sensing”, *Journal of Geophysical Research*, 82(11), 1979, pp.1673–1680  
2. N. Nandhakumar and J.K. Aggarwal. “Integrated modeling of thermal and visual image for scene interpretation”, *IEEE Trans. PAMI*, 10(4), 1988, pp.469–480  
3. J. Hinderer, “Model for Generating Synthetic Three-dimensional(3D) Images of Small Vehicles”, *Proc. SPIE Conf. On Infrared Sensor Fusion*, Vol.782, 1987, pp.9–12  
4. W.R. Owens, “Data-based Methodology for Infrared Signature Projection”, *Proc. SPIE Conf. On Infrared Sensors and Sensor Fusion*, Vol.782, 1987, pp.96–99  
5. G. Gerhart et al., “Thermal Image Modeling”, *SPIE Vol.782 Infrared Sensors and Sensor Fusion*, 1987, pp.3–9  
6. Chanhee Oh et al., “Integrated Modeling of Thermal and Visual Image Generation”, IEEE, 1989, pp.356–362  
7. N. Nandhakumar et al., “United Modeling of Non-homogeneous 3D Objects for Thermal and Visual Images Synthesis”, *Pattern Recognition*, Vol.27, No.10, 1994, pp.1303–1316  
8. Hyun-Ki Hong et al., “Simulation of Reticle Seekers Using the Generated Thermal Images”, IEEE, 1996, pp.183–186