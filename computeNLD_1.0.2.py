#!/usr/bin/env python
__author__ = "Mirko Strauss"
__version__ = 1.02
__date__ = "2022-05_02"

'''
This is a script for calculating nuclear level densities based on a given spectrum.
Requires Python 3.8 or higher.

------------Arguments------------
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

------------Results------------
Results are given in form of a plot (or several) and a text file called nld.txt that contains energy values corresponding level denisties as well as uncertainties for both methods of calculation. 

'''




import sys
import argparse
from numpy import array,linspace,arange,exp,zeros,empty,genfromtxt,convolve,where,random,transpose,log,sqrt,round
from scipy.optimize import curve_fit
from math import pi,floor,ceil
from statistics import multimode
from matplotlib.pyplot import *
from scipy.stats import norm, poisson



def parseArguments():
    '''Parses the input. There are five mandatory arguments, the rest are optional arguments with default values.'''
    parser = argparse.ArgumentParser()

    #mandatory arguments
    parser.add_argument("spectrum", help="input spectrum")
    parser.add_argument("corrWidth",help="width for which the autocorrelation is calculated",type=float)
    parser.add_argument("resolution",help="experimental resolution",type=float)
    parser.add_argument("sigmaLow",help="parameter for convolution, lower than resolution",type=float)
    parser.add_argument("sigmaHigh",help="parameter for convolution, higher than resolution",type=float)
    #optional arguments
    parser.add_argument('-en',"--energies",help="energies for which to calculate level densities, contained in a string and delimited by commas ", type=str, default=-1.)
    parser.add_argument('-pr',"--prefix",help="prefix of the energy units used in the input spectrum, 'k' or 'M'",type=str, default='M')
    parser.add_argument('-pl',"--plots",help="output plots, 'stat' for stationary spectrum, 'ac' for autocorrelation, 'nld' for level densities - can be combined",type=str, default='nld')
    parser.add_argument('-st',"--spectrumType",help="type of input spectrum, either 'incidence' or 'crosssection",type=str, default='crosssection')
    parser.add_argument('-it',"--iterations", help="number of iterations for Monte Carlo simulation used to calculate uncertainty",type=int, default=1)

    args = parser.parse_args()
    return args



def gauss(x, sigma, mu):
    '''A regular Gaussian function for convolution.'''
    return 1/(sqrt(2*pi)*sigma)*exp(-1/2*((x-mu)/sigma)**2)



def gaussConvolve(eRange, crossSection, sigmaLow, sigmaHigh):
    '''
    Convolves each bin of the given crosssection data with Gaussian functions 
    with the standard deviations sigmaLow and sigmaHigh respectively,
    eRange stores the energy values of the input data.
    '''
    gLow = zeros(len(eRange))
    gHigh = zeros(len(eRange))
    eWidth = (eRange[-1]-eRange[0])/(len(eRange)-1)
    
    for i in range(len(eRange)):
        if crossSection[i] != 0:
            singleBar = array(crossSection[i])
            gaussLow = gauss(eRange, sigmaLow, eRange[i])
            gaussHigh = gauss(eRange, sigmaHigh, eRange[i])
            #In order for the convoluted spectrum to have the same area (sum of bin height*bin width) 
            #as the original spectrum, the area of the gauss function must be considered, 
            #e.g. sum(gaussLow)*eWidth. eWidth is therefore introduced as a normalization factor.        
            gLow += convolve(singleBar, gaussLow, mode='same')*eWidth
            gHigh += convolve(singleBar, gaussHigh, mode='same')*eWidth
        
    return gLow, gHigh



def autoCorr(eRange, d, lowerBound, width):
    '''Computes the autocorrelation of the input spectrum d from lowerBound to lowerBound+width.'''
    start = where(eRange >= lowerBound)[0][0]
    end = where(eRange >= lowerBound + 2*width)[0][0]
    dPart = d[start:end]
    l = floor((end-start)/2)
    corr = zeros([l])
    err = 0

    for i in range(l):
        mean = dPart[:l].mean()
        dPartEps = dPart[i:l+i]
        meanEps = dPartEps.mean()
        if mean == 0 or meanEps == 0:
            err = 1
        else:
            corr[i] = dPart[:l]@dPartEps/l/mean/meanEps
    return corr, err    



def dPlot(eRange, d, lowerBound, width, ax=None, prefix = 'M'):
    '''Plots the stationary spectrum d.'''
    start = where(eRange >= lowerBound)[0][0]
    end = where(eRange >= lowerBound + width)[0][0]
    ax.plot(eRange[0:end-start]-eRange[0], d[start:end], linewidth = 1)
    ax.plot(eRange[0:end-start]-eRange[0], len(eRange[start:end])*[1], color = 'black', linewidth = 2)
    ax.set_xlim(0,eRange[end-start]-eRange[0])



def d2Plot(eRange, d0, d1, lowerBound, width, ax=None, prefix = 'M'):
    '''Plots the raw stationary spectrum d0 and the smoothed stationary spectrum d1 side by side.'''
    dPlot(eRange,d0,lowerBound,width,ax[0],prefix)
    dPlot(eRange,d1,lowerBound,width,ax[1],prefix)



def autoCorrPlot(epsRange, C, aC, param, ax=None, prefix = 'M'):
    '''Plots the experimental autocorrelation function aC as well as the theoretical autocorrelation function C.'''
    ax.plot(epsRange,(aC-1)*1e4,label='experiment',drawstyle='steps-post',linewidth=1)
    ax.plot(epsRange,len(epsRange)*[0],color='black',linewidth=1)
    if C != None:
        ax.plot(epsRange,(C(epsRange,param)-1)*1e4,label='theory')
    ax.set_xlim(0,epsRange[floor(len(aC)/2)])
    



def autoCorr2Plot(epsRange, C, aC0, params0, aC1, ax=None, prefix='M'):
    '''
    Plots the experimental and theoretical autocorrelation functions of the raw and smoothed spectrum side by side.
    '''
    autoCorrPlot(epsRange,C,aC0,params0,ax[0],prefix)
    autoCorrPlot(epsRange,None,aC1,None,ax[1],prefix)



def CSig(sigmaLow,sigmaHigh):
    '''
    Curried function that returns the theoretical autocorrelation function C for given values sigmaLow and sigmaHigh.
    This is necessary so that sigmaLow and sigmaHigh can be changed inside the NLD calculation function 
    based on the input spectrum, but are not interpreted as free parameters by curve_fit.
    '''
    y = sigmaHigh/sigmaLow
    def C(epsilon,D):
        cDef = 1+alpha*D/(2*sigmaLow*sqrt(pi))*(exp(-epsilon**2/(4*sigmaLow**2))
                                           +1/y*exp(-epsilon**2/(4*sigmaLow**2*y**2))
                              -sqrt(8/(1+y**2))*exp(-epsilon**2/(2*sigmaLow**2*(1+y**2))))
        return cDef
    return C



def fitAutoCorr(eRange,
                crossSection,
                boundaries,
                corrWidth,
                resolution,
                sigmaLow, sigmaHigh,
                prefix='M'):
    '''
    Computes level densities based on the given cross section spectrum for the raw and the smoothed spectrum.
    '''
    #alpha is the sum of intensity variance and spacing variance 
    #eRWidth is the bin width of the input spectrum
    #dSmoothed is the smoothed stationary spectrum, dRaw the non-smoothed
    y = sigmaHigh/sigmaLow
    alpha = 2.273  
    eWidth = (eRange[-1]-eRange[0])/(len(eRange)-1) 
    gLow, gHigh = gaussConvolve(eRange,crossSection,sigmaLow,sigmaHigh)  

    if not all(gHigh):
        print("Error: g<(E) contains zeros. Results may be affected.")

    dSmoothed = gLow/gHigh
    dRaw = crossSection/gHigh   
    #NLDSmoothed saves the the level densities based on the smoothed stationary spectrum,
    #NLDRaw the nld based on the non-smoothed stationary spectrum for the specified energies
    #Similarly, acRawCompl and acSmoothedCompl save the respective autocorrelation functions for both spectra
    #stdRaw collects the sum of standard deviations out of the autocorrelation fits
    NLDRaw = empty(len(boundaries))
    NLDSmoothed = empty(len(boundaries))
    acRawCompl = []
    acSmoothedCompl = []
    err = empty(len(boundaries))

    for i in range(len(boundaries)):
        acRaw, errRaw = autoCorr(eRange,dRaw,boundaries[i],corrWidth)
        acRawCompl.append(acRaw)

        acSmoothed, errSmoothed = autoCorr(eRange,dSmoothed,boundaries[i],corrWidth)       
        acSmoothedCompl.append(acSmoothed)
        
        err[i] = errRaw + errSmoothed

        #The NLD from the smoothed spectrum is calculated by taking the autocorrelation value at epsilon=0,
        #where epsilon is the energy offset from boundaries[i]
        NLDSmoothed[i] = alpha/2/(resolution/2/sqrt(2*log(2)))/sqrt(pi)*(1+1/y-sqrt(8/(1+y**2)))/(acSmoothed[0]-1)
        
        #epsRange defines a range of values for the offset epsilon
        #The function C with the current values for sigmLow and sigmaHigh is fitted to the non-smoothed autocorrelation
        #The level density is given by the inverse of the function parameter D
        epsRange = linspace(0,len(acRaw)*eWidth,len(acRaw))
        endFit = where(acRaw<=1)[0][0]
        if endFit < 5:
            endFit = 20
        paramsRaw, covRaw = curve_fit(CSig(resolution/2/sqrt(2*log(2)),sigmaHigh),epsRange[1:endFit],acRaw[1:endFit])
        
        NLDRaw[i] = 1/paramsRaw[0]

    #if the spectrum is given in units of keV, the NLD are converted from 1/keV to 1/MeV
    if prefix == 'k':
        NLDRaw, NLDSmoothed = array([NLDRaw, NLDSmoothed])*1e3

    return array([NLDRaw,NLDSmoothed,dRaw,dSmoothed,acRawCompl,acSmoothedCompl,eRange,boundaries,err], dtype='object')



def plotResults(results, corrWidth, resolution, sigmaLow, sigmaHigh, plots='statacnld', prefix='M', deltaRaw=0, deltaSmoothed=0):
    '''Plots the stationary spectra, the autocorrelation functions and/or the resulting nuclear level densities'''
    if plots.__contains__('stat') or plots.__contains__('ac'):
        NLDRaw,NLDSmoothed,dRaw,dSmoothed,acRaw,acSmoothed,eRange,boundaries,err = results
        paramsRaw = 1/NLDRaw
        if prefix == 'k':
            paramsRaw = 1e3*paramsRaw
        eRW = (eRange[-1]-eRange[0])/(len(eRange)-1)

    if plots.__contains__('stat'):
        f, axs = subplots(len(boundaries), 2, squeeze=False, sharex='col', tight_layout=True, gridspec_kw={'hspace':0})
        anc = ceil(len(boundaries)/2)-1
        axs[0][0].set_title('Raw spectrum')
        axs[0][1].set_title('Smoothed spectrum')
        axs[anc][0].set_ylabel('$d(E)$',fontsize=20)
        axs[anc][0].yaxis.set_label_coords(-0.1,anc+1-len(boundaries)/2)
        

        for i in range(len(boundaries)):
            d2Plot(eRange,dRaw,dSmoothed,boundaries[i],corrWidth,axs[i],prefix)

        axs[-1][0].set_xlabel('Energy offset $\epsilon$ ('+prefix+'eV)',fontsize=10)
        axs[-1][1].set_xlabel('Energy offset $\epsilon$ ('+prefix+'eV)',fontsize=10)
            
    if plots.__contains__('ac'):
        f, axs = subplots(len(boundaries), 2,squeeze=False, sharex='col', tight_layout=True, gridspec_kw={'hspace':0})
        center = floor(len(boundaries)/2)-1
        axs[0][0].set_title('Raw spectrum')
        axs[0][1].set_title('Smoothed spectrum')
        axs[center][0].set_ylabel('(C$(\epsilon)-1)\cdot 10^{-4}$',fontsize=20)
        axs[center][0].yaxis.set_label_coords(-0.05,-(len(boundaries)/2-center-1))

        for i in range(len(boundaries)):
            epsRange = linspace(0,len(acRaw[i])*eRW,len(acRaw[i]))
            autoCorr2Plot(epsRange,CSig(resolution/2/sqrt(2*log(2)),sigmaHigh),
                          acRaw[i],paramsRaw[i],
                          acSmoothed[i],
                          axs[i],prefix)

        axs[0][0].legend()
        axs[0][1].legend()
        axs[-1][0].set_xlabel('Energy offset $\epsilon$ ('+prefix+'eV)',fontsize=10)
        axs[-1][1].set_xlabel('Energy offset $\epsilon$ ('+prefix+'eV)',fontsize=10)

    if plots.__contains__('nld'):
        NLDRaw, NLDSmoothed = results[:2]
        boundaries = results[7]
        f = figure()
        energies = boundaries+0.5*corrWidth
        errorbar(energies, NLDRaw, deltaRaw, fmt='o', label='using raw spectrum (fit)', markersize=3)
        errorbar(energies, NLDSmoothed, deltaSmoothed, fmt='o', label='using smoothed spectrum', markersize=3)

        ylabel('Level density (MeV$^{-1})$')
        yscale('log')
        legend()
        if prefix == 'M':
            xlabel('Energy (MeV)')        
        else:
            xlabel('Energy (keV)')




def setBounds(e, rawData, corrWidth):
    '''Restricts data to what is needed for calculations and sets boundaries for the autocorrelation functions'''
    e = array(e)
    boundaries = e - corrWidth/2

    #Only data between e0 and e1 is used.
    e0 = boundaries[0] - corrWidth/2
    e1 = boundaries[-1] + 2*corrWidth
    i = 1

    while rawData[:,0][-1] < e1 and i < len(boundaries):
        e1 = boundaries[-1-i] + 2*corrWidth
        boundaries = boundaries[:-1]
        i += 1         

    start = where(rawData[:,0]>=e0)[0][0]
    end = where(rawData[:,0]>=e1)[0][0]+1
    data = rawData[start:end]
    
    return boundaries, data

def calcDelta(e, monteCarloNLD, confidence, method, spectrumType):
    '''Calculates uncertainties based on the given confidence intervall'''
    nld = zeros([len(e)])
    confInt = zeros([len(e),2])

    for i in range(len(e)):
        #The most common value is selected from the results of the Monte Carlo Simulation
        modes = multimode(round(monteCarloNLD[:,i],-1))
        nld[i] = modes[floor(len(modes)/2)]
        if spectrumType == 'crosssection':
            confInt[i] = norm.interval(alpha=confidence,loc=nld[i], scale=monteCarloNLD[:,i].std())
        if spectrumType == 'incidence':
            confInt[i] = poisson.interval(alpha=confidence,mu=nld[i])
        
    delta = array([abs(nld-confInt[:,0]),abs(confInt[:,1]-nld)])

    return nld, delta

def calcNLD(rawData,
            corrWidth,
            resolution,
            sigmaLow,
            sigmaHigh,
            e=-1,
            prefix='M',
            plots='nld',
            iterations=1,
            spectrumType='crosssection',
            confidence=.68):
    '''Calculates level densities based on input data.'''

    if not all(rawData[:,1]):
        print('Warning: Zeros in input spectrum can distort results.')
    boundaries, data = setBounds(e, rawData, corrWidth) 

    monteCarloData = zeros([len(data),iterations])
    results = fitAutoCorr(data[:,0],data[:,1],boundaries,corrWidth,resolution,sigmaLow,sigmaHigh,prefix)

    #If the mean of dPart is zero, an error message is printed.
    errMessage = zeros(len(boundaries))
    for i in range(len(boundaries)):
        if results[-1][i] > 0 and errMessage[i] == 0:
            print("Error: NLD for E="+str(boundaries[i]+corrWidth/2)+" "+str(prefix)+"eV could not properly be calculated due to the mean of dPart being zero.")
            errMessage[i] = 1

    if iterations <= 1:
        deltaRaw = zeros([2,len(boundaries)])
        deltaSmoothed = zeros([2,len(boundaries)])
    
    else:  
        #For each spectrum entry values are picked randomly from either a Gaussian or Poisson distribution.
        for i in range(len(data)):
            if data[i,1] > 0:
                #Depending on the type of input spectrum, a distribution for the randoml values is selected.
                if spectrumType == 'crosssection':
                    if len(data[0])<3:
                        print('Error: Missing third column. If the spectrumType is set to crosssection, the spectrum file must contain a third column that provides uncertainties.')
                        exit()
                    else:
                        monteCarloData[i] = random.normal(data[i,1],data[i,2],iterations)
                if spectrumType == 'incidence':
                    monteCarloData[i] = random.poisson(data[i,1],iterations)
            if data[i,1] < 0:
                data[i,1] = 0
        
        monteCarloData = transpose(monteCarloData)
        monteCarloNLDRaw = zeros([iterations+1,len(boundaries)])
        monteCarloNLDSmoothed = zeros([iterations+1,len(boundaries)])
        monteCarloNLDRaw[-1] = results[0]
        monteCarloNLDSmoothed[-1] = results[1]

        for i in range(iterations):
            rndResults = fitAutoCorr(data[:,0],
                                  monteCarloData[i],
                                  boundaries,
                                  corrWidth,
                                  resolution,
                                  sigmaLow,sigmaHigh,
                                  prefix)

            for k in range(len(rndResults[-1])):
                if rndResults[-1][k] > 0 and errMessage[k] == 0:
                    print("Error: NLD for E="+str(boundaries[k]+corrWidth/2)+" "+str(prefix)+"eV could not properly be calculated due to the mean of dPart being zero.")
                    errMessage[k] = 1

            monteCarloNLDRaw[i] = rndResults[0]
            monteCarloNLDSmoothed[i] = rndResults[1]

        #Uncertainties and most frequent value of the nld distribution are calculated.
        NLDRawMode, deltaRaw = calcDelta(boundaries, monteCarloNLDRaw, confidence, 'raw', spectrumType)
        NLDSmoothedMode, deltaSmoothed = calcDelta(boundaries, monteCarloNLDSmoothed, confidence, 'smoothed', spectrumType)
        #Original results are replaced with the mode of the Monte Carlo simulation.
        results[0] = NLDRawMode
        results[1] = NLDSmoothedMode
        

    plotResults(results, corrWidth, resolution, sigmaLow, sigmaHigh, plots=plots, prefix=prefix, deltaRaw=deltaRaw, deltaSmoothed=deltaSmoothed)
    show()

    return [[results[0][i],deltaRaw[0][i],deltaRaw[1][i],results[1][i],deltaSmoothed[0][i],deltaSmoothed[1][i]]for i in range(len(results[0]))] 


if __name__=='__main__':
    args = parseArguments()
    alpha = 2.273


    with open(args.spectrum) as file:
        spectrum = genfromtxt(file)
    
    #If the energies for which to calculate NLD are not specified, an array of energies in with distances equal to the correlation width is generated.
    if args.energies == -1:
        lowIndex = where(spectrum[:,1] > 0)[0][0]
        highIndex = where(spectrum[:,1] > 0)[0][-1]
        energies = arange(spectrum[:,0][lowIndex], spectrum[:,0][highIndex], args.corrWidth)
    else:
        energies = array([int(item) for item in args.energies.split(',')])

    #Conversion from FWHM to standard deviation
    sigmaLow = args.sigmaLow/(2*sqrt(2*log(2)))
    sigmaHigh = args.sigmaHigh/(2*sqrt(2*log(2)))

    #Calculation of level densities.
    out = calcNLD(spectrum,
            args.corrWidth,
            args.resolution,
            sigmaLow,
            sigmaHigh,
            e=energies,
            prefix=args.prefix,
            plots=args.plots,
            iterations=args.iterations,
            spectrumType=args.spectrumType,
            confidence=.68)

    #Level densities and uncertainties are put into a text file.
    file = open('nld.txt', 'w')
    file.write('Energy ('+str(args.prefix)+'eV) NLDRaw (1/MeV) Lower error raw (1/MeV) Upper error raw (1/MeV) NLDSmoothed (1/MeV) Lower error smoothed (1/MeV) Upper error smoothed (1/MeV) \n')
    for i in range(len(out)):
        file.write(str(energies[i])+' ')
        for j in range(len(out[0])):
            file.write(str(out[i][j])+' ')
        file.write('\n')
    file.close()
