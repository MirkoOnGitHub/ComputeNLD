# ComputeNLD

This is a script for calculating nuclear level densities based on a given spectrum.
Requires Python 3.8 or higher.

## Arguments
There are five mandatory arguments (refer to the parseArguments function below):
The spectrum is given in form of a file in which the first column contains the energy values and a second column contains the incidence or crosssection valus respectively.
In case of a crosssection spectrum there must be a third column giving the uncertainties of the crosssection values.
The parameter corrWidth determines the interval length of the autocorrelation function and must be given in the same units as the energy entries of the spectrum.
The third argument is the experimental resolution given in the same units as the energy entries of the spectrum.
The parameters sigmaLow and sigmaHigh determine the standard deviations of the Gaussians with which the spectrum is convolved. They must be lower/higher than the experimental resolution. 

The minimum input looks like this:
python computeNLD.py <spectrum> <correlation interval> <resolution> <lower sigma> <upper sigma>

Additionally, there are five optional arguments:
The first optional argument -en contains energy values for which to calculate level densities. It must be given as a string of values delimited by commas.
The second optional argument -pr determines which units are used for the input, either 'k' for kev or 'M' for Mev
The third optional argument -pl determines which plots are displayed: 'stat' for the stationary spectrum, 'ac' for the autocorrelation function, 'nld' for level densities. 
They can be combined by putting them in the same string.
The fourth optional argument -st gives the type of the input spectrum, either 'incidence' or 'crosssection'. Depending on this, uncertainties are clculated using a Poisson distribution or a Gaussian distribution
for the randomly generated values of the Monte Carlo simulation.
The fivth optional argument -it contains the number of iterations for the Monte Carlo simulation that is used to calulate the uncertainty of the level densitis. If no uncertainties are required,
it can be left at the default of one.

Using all ten arguments would look look like this:
python computeNLD.py <spectrum> <correlation interval> <resolution> <lower sigma> <upper sigma> -en <energies> -pr <prefix> -pl <plots> -st <spectrum type> -it <iterations>

## Results
Results are given in form of a plot (or several) and a text file called nld.txt that contains energy values corresponding level denisties as well as uncertainties for both methods of calculation. 

