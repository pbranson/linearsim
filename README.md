# linearsim - Time series generation of random seas

This library contains code to model ocean surface gravity waves using the JONSWAP spectra 
and generate timeseries of water elevation and time-domain statistics of gaussian random seas.

It implements both the 'deterministic amplitude, random phase' (DAS) and 'random amplitude' (RAS) schemes
for the generation of 1D timeseries of water surface elevation. See references below for additional details. 

Notebook [Step3_Compare_RAS_DAS.ipynb](Step3_Compare_RAS_DAS.ipynb) compares the coefficient of variation 
in several time-domain statistics and demonstrates that DAS does not reproduce the statistics of the underlying 
random gaussian process. 

Utilising the RAS technique we generate large ensembles of random realisation using a Monte-Carlo technique and
model the parametric probability distributions of several important sea-state parameters ($H_s$, $T_z$, $H_{max}$, 
$H_{max}/H_s$ and $r$ - crest to trough correlation) as a function of increasing number of waves (N) and JONSWAP gamma factor. 
These can be used to as first order estimates of parameter uncertainty or variability due to the stochastic 
nature of ocean surface waves.

Finally, we generate a large (2 million) member ensemble and extract random seeds at specific quantiles of the univariate
and marginal multivariate distributions of the parameters. This subset of seeds is published here as a netCDF file and allows 
for timeseries with specific characteristics to be regenerated. These subsets of the stochastic space can then be used as 
timeseries boundary conditions for time-domain simulations where the non-linearity
of ocean waves or structure response is accounted for. Examples may include for modelling Wave Energy Converters (WECs), wave runup on beaches or 
forces on marine structures. 

# Citation

This work was undertaken by Dr. Paul Branson during a Fulbright post-doctoral scholarship to Oregon State University hosted by Dr. Bryson Robertson, director of the Pacific Marine Energy Centre. If you find this useful please get in contact via the issues list. A publication is forth-coming.

(In preparation) Branson, P. and Robertson B. (2023) Short-time stochasticity of irregular ocean waves. TBD

# References

Tucker, M. J., Challenor, P. G., & Carter, D. J. T. (1984). Numerical 
simulation of a random sea: a common error and its effect upon wave 
group statistics. Applied Ocean Research, 6(2), 118–122. 
https://doi.org/10.1016/0141-1187(84)90050-6

Merigaud, A., & Ringwood, J. V. (2018). Free-Surface Time-Series 
Generation for Wave Energy Applications. IEEE Journal of Oceanic 
Engineering, 43(1), 19–35. 
https://doi.org/10.1109/JOE.2017.2691199


