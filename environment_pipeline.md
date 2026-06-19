### 4.3.2. Reflection radiation

When the environment radiation reaches the aircraft surface, part of the radiation is absorbed and transformed into the heat of the aircraft. Another part is reflected by the aircraft surface and superimposes together with the aircraft's own radiation. Without considering the situation of multiple reflections, in such a natural environment including the sun, sky, ground and sea, the reflection radiation of the aircraft can be calculated by the following equation [18]:

$$ E_{i}=I_{d}\cos\theta_{i}+I_{s c}+I_{sky}+I_{ground\mid sea}\qquad(8) $$

where $l_{d}$ and $l_{sc}$ represent solar direct and scattering radiation; $\theta_{i}$ is solar incident angles; $l_{scy}$ is atmospheric scattering radiation; $l_{v}$ is the Earth radiation. The accurate simulation of the solar and atmospheric radiation is a very complex problem. Taking into account the real-time requirements, this paper uses empirical formulas for simulation.

The direct solar radiation is obtained by Eq. (9), assuming that the radiation intensity is uniform because it is far from the sun.

$$ l_{d}=\xi\cdot l_{0}\cdot P^{m}\qquad(9) $$

where $l_{0}$ is the solar constant and equals $1353\text{W/m}^{2}$; $P$ is the atmosphere transparent rate at a certain region and a certain time; $h$ is solar elevation angle; $m$ is air quality and is expressed as:

$$ m=\frac{1}{\sin h}\qquad(10) $$

$\xi$ is the Sun–Earth distance correction coefficient:

$$ \xi=1+0.034\cos\left(\frac{2\pi}{365}n\right)\qquad(11) $$

$n$ is serial number of the measurement date in a year.

Berlage formula is used to calculate $l_{sc}$.

$$ l_{sc}=\frac{1}{2}l_{0}\sin\frac{1-p^{m}}{1-1.4\ln^{2}C}\frac{\theta}{2}\qquad(12) $$

where $\theta$ is the angle of the plane and the horizontal plane.

The sky scattering radiation is shown as the following equation:

$$ l_{sky}=(a+b\sqrt{e})\sigma T^{4}=\varepsilon\sigma T^{4}\qquad(13) $$

where $a$ and $b$ are empirical parameters; and they value as 0.58 and 0.061 respectively by tests. $e$ is water vapor pressure.

The Earth's infrared radiation comes from the energy of solar radiation absorbed by the Earth's surface. Assuming the Earth is a uniform radiant body, the thermal radiation intensity is the same throughout the Earth.

If $E_{0}$ is the solar infrared radiation, and it is expressed as:

$$ E_{0}=\frac{(1-\rho_{E})S_{0}}{4}\qquad(14) $$

where $\rho_{E}$ is reflection coefficient, with the range of [0, 1]. It is 0.7 in this paper. Then the Earth's thermal radiation intensity received by the aircraft can be shown as:

$$ l_{e}=\alpha_{1}l_{0}l_{f_{i}}\qquad(15) $$

where $f_{f_{i}}$ is the Earth's radiation angle coefficient, with the range of [0, 1]. It is 0.7 in this paper.