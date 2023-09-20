#!/usr/bin/env python
# coding: utf-8

#===========================================================================================================================
#============= ============= ============= ============  FOLPSν  ============== ============== ============= =============== 
# This is a code for efficiently evaluating the redshift space power spectrum in the presence of massive neutrinos.

# We recommend to use NumPy versions ≥ 1.20.0. For older versions, one needs to rescale by a factor 1/N the FFT computation. (see: https://github.com/henoriega/FOLPS-nu).
#===========================================================================================================================

#standard libraries
import numpy as np
import scipy
from scipy import integrate
from scipy.interpolate import CubicSpline
from scipy import interpolate
from scipy.fft import dst, idst
from scipy.special import gamma
from scipy.special import spherical_jn
from scipy.special import eval_legendre
from scipy.integrate import quad
import sys



def interp(k, x, y):
    '''Cubic spline interpolation.
    
    Args:
        k: coordinates at which to evaluate the interpolated values.
        x: x-coordinates of the data points.
        y: y-coordinates of the data points.
    Returns:
        Cubic interpolation of ‘y’ evaluated at ‘k’.
    '''
    inter = CubicSpline(x, y)
    return inter(k)  
    


def Matrices(Nfftlog = None):    
    '''M matrices. They do not depend on the cosmology, so they are computed only one time.
    
    Args:
        if 'Nfftlog = None' (or not specified) the code use as default 'Nfftlog = 128'. 
        to use a different number of sample points, just specify it as 'Nfftlog =  number'.
        we recommend using the default mode, see Fig.~8 at arXiv:2208.02791. 
    Returns:
        All the M matrices.
    '''
    global M22matrices, M13vectors, bnu_b, N
    
    k_min = 10**(-7); k_max = 100.
    b_nu = -0.1;   #Not yet tested for other values
    
    if Nfftlog == None:
        N = 128
        
    else:
        N = Nfftlog
        
    
    #Eq.~ 4.19 at arXiv:2208.02791
    def Imatrix(nu1, nu2):
        return 1/(8 * np.pi**(3/2)) * ( gamma(3/2-nu1)*gamma(3/2-nu2)*gamma(nu1+nu2-3/2) )/( gamma(nu1)*gamma(nu2)*gamma(3-nu1-nu2) )

    
    #M22-type
    def M22(nu1, nu2):
        
        #Overdensity and velocity
        def M22_dd(nu1, nu2):
            return Imatrix(nu1,nu2)*(3/2-nu1-nu2)*(1/2-nu1-nu2)*( (nu1*nu2)*(98*(nu1+nu2)**2 - 14*(nu1+nu2) + 36) - 91*(nu1+nu2)**2+ 3*(nu1+nu2) + 58)/(196*nu1*(1+nu1)*(1/2-nu1)*nu2*(1+nu2)*(1/2-nu2))
        
        def M22_dt_fp(nu1, nu2):
            return Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-1+2*(nu1+nu2))*(-23-21*nu1+(-38+7*nu1*(-1+7*nu1))*nu2+7*(3+7*nu1)*nu2**2) )/(196*nu1*(1+nu1)*nu2*(1+nu2)*(-1+2*nu2))
        
        def M22_tt_fpfp(nu1, nu2):
            return Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-12*(1-2*nu2)**2 + 98*nu1**(3)*nu2 + 7*nu1**2*(1+2*nu2*(-8+7*nu2))- nu1*(53+2*nu2*(17+7*nu2))))/(98*nu1*(1+nu1)*nu2*(1+nu2)*(-1+2*nu2))
        
        def M22_tt_fkmpfp(nu1, nu2):
            return Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-37+7*nu1**(2)*(3+7*nu2) + nu2*(-10+21*nu2) + nu1*(-10+7*nu2*(-1+7*nu2))))/(98*nu1*(1+nu1)*nu2*(1+nu2))
        
        #A function
        def MtAfp_11(nu1, nu2):
            return Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-1+2*(nu1+nu2))*(-5+nu1*(-4+7*(nu1+nu2))))/(7*nu1*(1+nu1)*(-1+2*nu1)*nu2)
        
        def MtAfkmpfp_12(nu1, nu2):
            return -Imatrix(nu1,nu2)*(((-3+2*(nu1+nu2))*(-1+2*(nu1+nu2))*(6+7*(nu1+nu2)))/(56*nu1*(1+nu1)*nu2*(1+nu2)))
        
        def MtAfkmpfp_22(nu1, nu2):
            return Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-18+3*nu1*(1+4*(10-9*nu1)*nu1)+75*nu2+8*nu1*(41+2*nu1*(-28+nu1*(-4+7*nu1)))*nu2+48*nu1*(-9+nu1*(-3+7*nu1))*nu2**2+4*(-39+4*nu1*(-19+35*nu1))*nu2**3+336*nu1*nu2**4) )/(56*nu1*(1+nu1)*(-1+2*nu1)*nu2*(1+nu2)*(-1+2*nu2))
        
        def MtAfpfp_22(nu1, nu2):
            return Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-5+3*nu2+nu1*(-4+7*(nu1+nu2))))/(7*nu1*(1+nu1)*nu2)
        
        def MtAfkmpfpfp_23(nu1, nu2):
            return -Imatrix(nu1,nu2)*(((-1+7*nu1)*(-3+2*(nu1+nu2))*(-1+2*(nu1+nu2)))/(28*nu1*(1+nu1)*nu2*(1+nu2)))
        
        def MtAfkmpfpfp_33(nu1, nu2):
            return Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-1+2*(nu1+nu2))*(-13*(1+nu1)+2*(-11+nu1*(-1+14*nu1))*nu2 + 4*(3+7*nu1)*nu2**2))/(28*nu1*(1+nu1)*nu2*(1+nu2)*(-1+2*nu2))
        
        #D function
        def MB1_11(nu1, nu2):
            return Imatrix(nu1,nu2)*(3-2*(nu1+nu2))/(4*nu1*nu2)
        
        def MC1_11(nu1, nu2):
            return Imatrix(nu1,nu2)*((-3+2*nu1)*(-3+2*(nu1+nu2)))/(4*nu2*(1+nu2)*(-1+2*nu2))
        
        def MB2_11(nu1, nu2):
            return Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-1+2*(nu1+nu2)))/(4*nu1*nu2)
        
        def MC2_11(nu1, nu2):
            return Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-1+2*(nu1+nu2)))/(4*nu2*(1+nu2))
        
        def MD2_21(nu1, nu2):
            return Imatrix(nu1,nu2)*((-1+2*nu1-4*nu2)*(-3+2*(nu1+nu2))*(-1+2*(nu1+nu2)))/(4*nu1*nu2*(-1+nu2+2*nu2**2))
        
        def MD3_21(nu1, nu2):
            return Imatrix(nu1,nu2)*((3-2*(nu1+nu2))*(1-4*(nu1+nu2)**2))/(4*nu1*nu2*(1+nu2))
        
        def MD2_22(nu1, nu2):
            return Imatrix(nu1,nu2)*(3*(3-2*(nu1+nu2))*(1-2*(nu1+nu2)))/(32*nu1*(1+nu1)*nu2*(1+nu2))
        
        def MD3_22(nu1, nu2):
            return Imatrix(nu1,nu2)*((3-2*(nu1+nu2))*(1-4*(nu1+nu2)**2)*(1+2*(nu1**2-4*nu1*nu2+nu2**2)))/(16*nu1*(1+nu1)*(-1+2*nu1)*nu2*(1+nu2)*(-1+2*nu2))
        
        def MD4_22(nu1, nu2):
            return Imatrix(nu1,nu2)*((9-4*(nu1+nu2)**2)*(1-4*(nu1+nu2)**2))/(32*nu1*(1+nu1)*nu2*(1+nu2))
    
        
        return (M22_dd(nu1, nu2), M22_dt_fp(nu1, nu2), M22_tt_fpfp(nu1, nu2), M22_tt_fkmpfp(nu1, nu2),
                MtAfp_11(nu1, nu2), MtAfkmpfp_12(nu1, nu2), MtAfkmpfp_22(nu1, nu2), MtAfpfp_22(nu1, nu2), 
                MtAfkmpfpfp_23(nu1, nu2), MtAfkmpfpfp_33(nu1, nu2), MB1_11(nu1, nu2), MC1_11(nu1, nu2), 
                MB2_11(nu1, nu2), MC2_11(nu1, nu2), MD2_21(nu1, nu2), MD3_21(nu1, nu2), MD2_22(nu1, nu2), 
                MD3_22(nu1, nu2), MD4_22(nu1, nu2))
    
    
    #M22-type Biasing
    def M22bias(nu1, nu2):
        
        def MPb1b2(nu1, nu2):
            return Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-4+7*(nu1+nu2)))/(28*nu1*nu2)
        
        def MPb1bs2(nu1, nu2):
            return Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(2+14*nu1**2 *(-1+2*nu2)-nu2*(3+14*nu2)+nu1*(-3+4*nu2*(-11+7*nu2))))/(168*nu1*(1+nu1)*nu2*(1+nu2))
        
        def MPb22(nu1, nu2):
            return 1/2 * Imatrix(nu1, nu2)

        def MPb2bs2(nu1, nu2):
            return Imatrix(nu1,nu2)*((-3+2*nu1)*(-3+2*nu2))/(12*nu1*nu2)

        def MPb2s2(nu1, nu2):
            return Imatrix(nu1,nu2)*((63-60*nu2+4*(3*(-5+nu1)*nu1+(17-4*nu1)*nu1*nu2+(3+2*(-2+nu1)*nu1)*nu2**2)))/(36*nu1*(1+nu1)*nu2*(1+nu2))

        def MPb2t(nu1, nu2):
            return Imatrix(nu1,nu2)*((-4+7*nu1)*(-3+2*(nu1+nu2)))/(14*nu1*nu2)

        def MPbs2t(nu1, nu2):
            return  Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-19-10*nu2+nu1*(39-30*nu2+14*nu1*(-1+2*nu2))))/(84*nu1*(1+nu1)*nu2*(1+nu2))

        
        return (MPb1b2(nu1, nu2), MPb1bs2(nu1, nu2), MPb22(nu1, nu2), MPb2bs2(nu1, nu2), 
                MPb2s2(nu1, nu2), MPb2t(nu1, nu2), MPbs2t(nu1, nu2))
    
    
    #M13-type
    def M13(nu1):
        
        #Overdensity and velocity
        def M13_dd(nu1):
            return ((1+9*nu1)/4) * np.tan(nu1*np.pi)/( 28*np.pi*(nu1+1)*nu1*(nu1-1)*(nu1-2)*(nu1-3) )
        
        def M13_dt_fk(nu1):
            return ((-7+9*nu1)*np.tan(nu1*np.pi))/(112*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1*(1+nu1))
        
        def M13_tt_fk(nu1):
            return -(np.tan(nu1*np.pi)/(14*np.pi*(-3 + nu1)*(-2 + nu1)*(-1 + nu1)*nu1*(1 + nu1) ))
        
        # A function
        def Mafk_11(nu1):
            return ((15-7*nu1)*np.tan(nu1*np.pi))/(56*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1)
        
        def Mafp_11(nu1):
            return ((-6+7*nu1)*np.tan(nu1*np.pi))/(56*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1)
        
        def Mafkfp_12(nu1):
            return (3*(-13+7*nu1)*np.tan(nu1*np.pi))/(224*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1*(1+nu1))
        
        def Mafpfp_12(nu1):
            return (3*(1-7*nu1)*np.tan(nu1*np.pi))/(224*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1*(1+nu1))
        
        def Mafkfkfp_33(nu1):
            return ((21+(53-28*nu1)*nu1)*np.tan(nu1*np.pi))/(224*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1*(1+nu1))
        
        def Mafkfpfp_33(nu1):
            return ((-21+nu1*(-17+28*nu1))*np.tan(nu1*np.pi))/(224*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1*(1+nu1))

        
        return (M13_dd(nu1), M13_dt_fk(nu1), M13_tt_fk(nu1), Mafk_11(nu1),  Mafp_11(nu1), Mafkfp_12(nu1),
                Mafpfp_12(nu1), Mafkfkfp_33(nu1), Mafkfpfp_33(nu1))

    
    #M13-type Biasing
    def M13bias(nu1):
        
        def Msigma23(nu1):
            return (45*np.tan(nu1*np.pi))/(128*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1*(1+nu1))
        
        return (Msigma23(nu1))
    
    
    #Computation of M22-type matrices
    def M22type(k_min, k_max, N, b_nu, M22):
        
        #nuT = -etaT/2, etaT = bias_nu + i*eta_m        
        nuT = np.zeros(N+1, dtype = complex)
        
        for jj in range(N+1):
            nuT[jj] = -0.5 * (b_nu +  (2*np.pi*1j/np.log(k_max/k_min)) * (jj - N/2) *(N-1)/(N))
            
        #reduce time x10 compared to "for" iterations
        nuT_x, nuT_y = np.meshgrid(nuT, nuT) 
        M22matrix = M22(nuT_y, nuT_x)
        
        return np.array(M22matrix)
    
    
    #Computation of M13-type matrices
    def M13type(k_min, k_max, N, b_nu, M13):
           
        #nuT = -etaT/2, etaT = bias_nu + i*eta_m 
        nuT = np.zeros(N+1, dtype = complex)
        
        for ii in range(N+1):
            nuT[ii] = -0.5 * (b_nu +  (2*np.pi*1j/np.log(k_max/k_min)) * (ii - N/2) *(N-1)/(N))
        
        M13vector = M13(nuT)
            
        return np.array(M13vector)
    
    
    #FFTLog bias for the biasing spectra Pb1b2,...
    bnu_b = 15.1*b_nu
    

    M22T =  M22type(k_min, k_max, N, b_nu, M22)
    M22biasT = M22type(k_min, k_max, N, bnu_b, M22bias)
    M22matrices = np.concatenate((M22T, M22biasT))
    
    M13T = M13type(k_min, k_max, N, b_nu, M13)
    M13biasT = np.reshape(M13type(k_min, k_max, N, bnu_b, M13bias), (1, int(N+1)))
    M13vectors = np.concatenate((M13T, M13biasT))
    
    print('N = '+str(N)+' sampling points')
    print('M matrices have been computed')
  
    return (M22matrices, M13vectors)




def LinearRegression(inputxy): 
    '''Linear regression.
    
    Args:
        inputxy: data set with x- and y-coordinates.
    Returns:
        slope ‘m’ and the intercept ‘b’.
    '''
    xm = np.mean(inputxy[0])
    ym = np.mean(inputxy[1])
    Npts = len(inputxy[0])
    
    SS_xy = np.sum(inputxy[0]*inputxy[1]) - Npts*xm*ym
    SS_xx = np.sum(inputxy[0]**2) - Npts*xm**2
    m = SS_xy/SS_xx
    
    b = ym - m*xm
    return (m, b)




def Extrapolate(inputxy, outputx):
    '''Extrapolation.
    
    Args:
        inputxy: data set with x- and y-coordinates.
        outputx: x-coordinates of extrapolation.
    Returns:
        extrapolates the data set ‘inputxy’ to the range given by ‘outputx’.
    '''
    m, b = LinearRegression(inputxy)
    outxy = [(outputx[ii], m*outputx[ii]+b) for ii in range(len(outputx))]
    
    return np.array(np.transpose(outxy))




def ExtrapolateHighkLogLog(inputT, kcutmax, kmax):
    '''Extrapolation for high-k values.
    
    Args:
        inputT: k-coordinates and linear power spectrum.
        kcutmax: value of ‘k’ from which ‘inputT’ will be interpolated.
        kmax: value of ‘k’ up to which ‘inputT’ will be interpolated.
    Returns:
        extrapolation for high-k values (from ‘kcutmax’ to ‘kmax’) for a given linear power spectrum ‘ inputT’.
    '''
    cutrange = np.where(inputT[0]<= kcutmax)
    inputcutT = np.array([inputT[0][cutrange], inputT[1][cutrange]])
    listToExtT = inputcutT[0][-6:]
    tableToExtT = np.array([listToExtT, inputcutT[1][-6:]])
    delta = np.log10(listToExtT[2])-np.log10(listToExtT[1])
    lastk = np.log10(listToExtT[-1])
    
    logklist = [];
    while (lastk <= np.log10(kmax)):
        logklistT = lastk + delta;
        lastk = logklistT
        logklist.append(logklistT)
    logklist = np.array(logklist)
    
    sign = np.sign(tableToExtT[1][1])
    tableToExtT = np.log10(np.abs(tableToExtT))
    logextT = Extrapolate(tableToExtT, logklist)
    
    output = np.array([10**logextT[0], sign*10**logextT[1]])
    output = np.concatenate((inputcutT, output), axis=1)
        
    
    return output




def ExtrapolateLowkLogLog(inputT, kcutmin, kmin):
    '''Extrapolation for low-k values.
    
    Args:
        inputT: k-coordinates and linear power spectrum.
        kcutmin: value of ‘k’ from which ‘inputT’ will be interpolated.
        kmin: value of ‘k’ up to which ‘inputT’ will be interpolated.
    Returns:
        extrapolation for low-k values (from ‘kcutmin’ to ‘kmin’) for a given linear power spectrum ‘inputT’.
    '''
    cutrange = np.where(inputT[0] > kcutmin)
    inputcutT = np.array([inputT[0][cutrange], inputT[1][cutrange]])
    listToExtT = inputcutT[0][:5]
    tableToExtT = np.array([listToExtT, inputcutT[1][:5]])
    delta = np.log10(listToExtT[2])-np.log10(listToExtT[1])
    firstk = np.log10(listToExtT[0])
    
    logklist = [];
    while (firstk > np.log10(kmin)):
        logklistT = firstk - delta;
        firstk = logklistT
        logklist.append(logklistT)
    logklist = np.array(list(reversed(logklist)))
    
    sign = np.sign(tableToExtT[1][1])
    tableToExtT = np.log10(np.abs(tableToExtT))
    logextT = Extrapolate(tableToExtT, logklist)
    
    output = np.array([10**logextT[0], sign*10**logextT[1]])
    output = np.concatenate((output, inputcutT), axis=1)
        
    
    return output




def ExtrapolatekLogLog(inputT, kcutmin, kmin, kcutmax, kmax):
    '''Extrapolation at low-k and high-k.
    
    Args:
        inputT: k-coordinates and linear power spectrum.
        kcutmin, kcutmax: value of ‘k’ from which ‘inputT’ will be interpolated.
        kmin, kmax: value of ‘k’ up to which ‘inputT’ will be interpolated.
    Returns:
        combines extrapolation al low-k and high-k.
    '''
    output = ExtrapolateLowkLogLog(ExtrapolateHighkLogLog(inputT, kcutmax, kmax), kcutmin, kmin)
    
    return output




def Extrapolate_inputpkl(inputT):
    '''Extrapolation to the input linear power spectrum.
    
    Args:
        inputT: k-coordinates and linear power spectrum.
    Returns:
        extrapolates the input linear power spectrum ‘inputT’ to low-k or high-k if needed.
    '''
    kcutmin = min(inputT[0]); kmin = 10**(-5);
    kcutmax = max(inputT[0]); kmax = 200
    
    if ((kmin < kcutmin) or (kmax > kcutmax)):
        output = ExtrapolatekLogLog(inputT, kcutmin, kmin, kcutmax, kmax)
        
    else:
        output = inputT
        
    return output




def CosmoParam(h, ombh2, omch2, omnuh2):
    '''Gives some inputs for the function 'fOverf0EH'.
    
    Args:
        h = H0/100.
        ombh2: Omega_b h² (baryons)
        omch2: Omega_c h² (CDM)
        omnuh2: Omega_nu h² (massive neutrinos)
    Returns:
        h: H0/100.
        OmM0: Omega_b + Omega_c + Omega_nu (dimensionless matter density parameter)
        fnu: Omega_nu/OmM0
        Massnu: Total neutrino mass [eV]
    '''
                
    Omb = ombh2/h**2;
    Omc = omch2/h**2;
    Omnu = omnuh2/h**2;
        
    OmM0 = Omb + Omc + Omnu; 
    fnu = Omnu/OmM0;
    Massnu = Omnu*93.14*h**2;
        
    return(h, OmM0, fnu, Massnu)




def fOverf0EH(zev, k, OmM0, h, fnu):
    '''Rutine to get f(k)/f0 and f0.
    f(k)/f0 is obtained following H&E (1998), arXiv:astro-ph/9710216
    f0 is obtained by solving directly the differential equation for the linear growth at large scales.
    
    Args:
        zev: redshift
        k: wave-number
        OmM0: Omega_b + Omega_c + Omega_nu (dimensionless matter density parameter)
        h = H0/100
        fnu: Omega_nu/OmM0
    Returns:
        f(k)/f0 (when 'EdSkernels = True' f(k)/f0 = 1)
        f0
    '''
    if (fnu > 0):
        eta = np.log(1/(1+zev))   #log of scale factor
        Neff = 3.046              #effective number of neutrinos
        omrv = 2.469*10**(-5)/(h**2 * (1 + 7/8*(4/11)**(4/3)*Neff)) #rad: including neutrinos
        aeq = omrv/OmM0           #matter-radiation equality
            
        pcb = 5/4 - np.sqrt(1 + 24*(1 - fnu))/4     #neutrino supression
        c = 0.7         
        Nnu = 3                                     #number of neutrinos
        theta272 = (1.00)**2                        # T_{CMB} = 2.7*(theta272)
        pf = (k * theta272)/(OmM0 * h**2)  
        DEdS = np.exp(eta)/aeq                      #growth function: EdS cosmology
            
        yFS = 17.2*fnu*(1 + 0.488*fnu**(-7/6))*(pf*Nnu/fnu)**2  #yFreeStreaming 
        rf = DEdS/(1 + yFS)
        fFit = 1 - pcb/(1 + (rf)**c)                #f(k)/f0
            
    else:
        fFit = np.full(len(k), 1.0)
            
        
    #Getting f0
    def OmM(eta):
        return 1/(1 + ((1-OmM0)/OmM0)* np.exp(3*eta) )
        
    def f1(eta):
        return 2 - 3/2 * OmM(eta)
        
    def f2(eta):
        return 3/2 * OmM(eta)
        
    etaini = -6;  #initial eta, early enough to evolve as EdS (D + \propto a)
    zfin = -0.99;
        
    def etaofz(z):
        return np.log(1/(1 + z))
        
    etafin = etaofz(zfin); 
        
    from scipy.integrate import odeint
        
    #Differential eqs.
    def Deqs(Df, eta):
        Df, Dprime = Df
        return [Dprime, f2(eta)*Df - f1(eta)*Dprime]
        
    #eta range and initial conditions
    eta = np.linspace(etaini, etafin, 1001)   
    Df0 = np.exp(etaini)
    Df_p0 = np.exp(etaini)
        
    #solution
    Dplus, Dplusp = odeint(Deqs, [Df0,Df_p0], eta).T
    
    Dplusp_ = interp(etaofz(zev), eta, Dplusp)
    Dplus_ = interp(etaofz(zev), eta, Dplus)
    f0 = Dplusp_/Dplus_ 
        
    return (k, fFit, f0)




def pknwJ(k, PSLk, h):
    '''Routine (based on J. Hamann et. al. 2010, arXiv:1003.3999) to get the non-wiggle piece of the linear power spectrum.    
    
    Args:
        k: wave-number.
        PSLk: linear power spectrum.
        h: H0/100.
    Returns:
        non-wiggle piece of the linear power spectrum.
    '''
    #ksmin(max): k-range and Nks: points
    ksmin = 7*10**(-5)/h; ksmax = 7/h; Nks = 2**16

    #sample ln(kP_L(k)) in Nks points, k range (equidistant)
    ksT = [ksmin + ii*(ksmax-ksmin)/(Nks-1) for ii in range(Nks)]
    PSL = interp(ksT, k, PSLk)
    logkpkT = np.log(ksT*PSL)
        
    #Discrete sine transf., check documentation
    FSTtype = 1; m = int(len(ksT)/2)
    FSTlogkpkT = dst(logkpkT, type = FSTtype, norm = "ortho")
    FSTlogkpkOddT = FSTlogkpkT[::2]
    FSTlogkpkEvenT = FSTlogkpkT[1::2]
        
    #cut range (remove the harmonics around BAO peak)
    mcutmin = 120; mcutmax = 240;
        
    #Even
    xEvenTcutmin = np.linspace(1, mcutmin-2, mcutmin-2)
    xEvenTcutmax = np.linspace(mcutmax+2, len(FSTlogkpkEvenT), len(FSTlogkpkEvenT)-mcutmax-1)
    EvenTcutmin = FSTlogkpkEvenT[0:mcutmin-2] 
    EvenTcutmax = FSTlogkpkEvenT[mcutmax+1:len(FSTlogkpkEvenT)]
    xEvenTcuttedT = np.concatenate((xEvenTcutmin, xEvenTcutmax))
    nFSTlogkpkEvenTcuttedT = np.concatenate((EvenTcutmin, EvenTcutmax))


    #Odd
    xOddTcutmin = np.linspace(1, mcutmin-1, mcutmin-1)
    xOddTcutmax = np.linspace(mcutmax+1, len(FSTlogkpkEvenT), len(FSTlogkpkEvenT)-mcutmax)
    OddTcutmin = FSTlogkpkOddT[0:mcutmin-1]
    OddTcutmax = FSTlogkpkOddT[mcutmax:len(FSTlogkpkEvenT)]
    xOddTcuttedT = np.concatenate((xOddTcutmin, xOddTcutmax))
    nFSTlogkpkOddTcuttedT = np.concatenate((OddTcutmin, OddTcutmax))

    #Interpolate the FST harmonics in the BAO range
    preT, = map(np.zeros,(len(FSTlogkpkT),))
    PreEvenT = interp(np.linspace(2, mcutmax, mcutmax-1), xEvenTcuttedT, nFSTlogkpkEvenTcuttedT)
    PreOddT = interp(np.linspace(0, mcutmax-2, mcutmax-1), xOddTcuttedT, nFSTlogkpkOddTcuttedT)
    for ii in range(m):
        if (mcutmin < ii+1 < mcutmax):
            preT[2*ii+1] = PreEvenT[ii]
            preT[2*ii] = PreOddT[ii]
        if (mcutmin >= ii+1 or mcutmax <= ii+1):
            preT[2*ii+1] = FSTlogkpkT[2*ii+1]
            preT[2*ii] = FSTlogkpkT[2*ii]
                
        
    #Inverse Sine transf.
    FSTofFSTlogkpkNWT = idst(preT, type = FSTtype, norm = "ortho")
    PNWT = np.exp(FSTofFSTlogkpkNWT)/ksT

    PNWk = interp(k, ksT, PNWT)
    DeltaAppf = k*(PSL[7]-PNWT[7])/PNWT[7]/ksT[7]

    irange1 = np.where((k < 1e-3))
    PNWk1 = PSLk[irange1]/(DeltaAppf[irange1] + 1)

    irange2 = np.where((1e-3 <= k) & (k <= ksT[len(ksT)-1]))
    PNWk2 = PNWk[irange2]
        
    irange3 = np.where((k > ksT[len(ksT)-1]))
    PNWk3 = PSLk[irange3]
        
    PNWkTot = np.concatenate((PNWk1, PNWk2, PNWk3))
        
    return(k, PNWkTot)




def cmM(k_min, k_max, N, b_nu, inputpkT):
    '''coefficients c_m, see eq.~ 4.2 - 4.5 at arXiv:2208.02791
    
    Args:
        kmin, kmax: minimal and maximal range of the wave-number k.
        N: number of sampling points (we recommend using N=128).
        b_nu: FFTLog bias (use b_nu = -0.1. Not yet tested for other values).
        inputpkT: k-coordinates and linear power spectrum.
    Returns:
        coefficients c_m (cosmological dependent terms)
    '''
    #define de zero matrices
    M = int(N/2)
    kBins = np.zeros(N)
    c_m = np.zeros(N+1, dtype = complex)
        
    #"kBins" trought "delta" gives logspaced k's in [k_min,k_max] 
    for ii in range (N):
        delta = 1/(N-1) * np.log(k_max/k_min)
        kBins[ii] = k_min * np.exp((ii) * delta)
    f_kl = interp(kBins, inputpkT[0], inputpkT[1]) * (kBins/k_min)**(-b_nu)

    #F_m is the Discrete Fourier Transform (DFT) of f_kl
    #"forward" has the direct transforms scaled by 1/N (numpy version >= 1.20.0)
    F_m = np.fft.fft(f_kl, n = N, norm = "forward" )
        
    #etaT = bias_nu + i*eta_m
    #to get c_m: 1) reality condition, 2) W_m factor
    for ii in range(N+1):
        etaT = b_nu +  (2*np.pi*1j/np.log(k_max/k_min)) * (ii - N/2) *(N-1)/(N)
        if (ii - M < 0):
            c_m[ii] = k_min**(-(etaT))*np.conj(F_m[-ii+M])
        c_m[ii] = k_min**(-(etaT)) * F_m[ii-M]
    c_m[0] = c_m[0]/2
    c_m[int(N)] = c_m[int(N)]/2 
        
    return(c_m)




def NonLinear(inputpkl, CosmoParams, kminout=0.001, kmaxout=0.5, nk = 120, EdSkernels = False):
    '''1-loop corrections to the linear power spectrum.
    
    Args:
        If 'EdSkernels = True' (default: 'False', fk-kernels), EdS-kernels will be employed.        
        inputpkl: k-coordinates and linear power spectrum.
        CosmoParams: Set of cosmological parameters [z_pk, omega_b, omega_cdm, omega_ncdm, h] in that order.
                   z_pk: redshift.
                   omega_i = omega_i = Omega_i h², where i=baryons (b), CDM (cdm), massive neutrinos (ncdm).
                   h = H0/100. 
    Returns:
        list of 1-loop contributions for the wiggle and non-wiggle (also computed here) linear power spectra.
    '''
    global TableOut, TableOut_NW, f0, kTout, sigma2w, sigma2w_NW
    
    global z_pk, omega_b, omega_cdm, omega_ncdm, h

    #CosmoParams
    (z_pk, omega_b, omega_cdm, omega_ncdm, h) = CosmoParams
    
    k_min = 10**(-7); k_max = 100.
    b_nu = -0.1;   #Not yet tested for other values
    
          
    #Extrapolates the linear power spectrum if needed. 
    inputpkT = Extrapolate_inputpkl(inputpkl) 
    
    
    ################################ KTOUT ###########################################    
    
    #kminout = 0.001; kmaxout = 0.5;
    
    kTout = np.logspace(np.log10(kminout), np.log10(kmaxout), num=nk)

    
    ##################################################################################
    
    def P22type(kTout, inputpkT, inputpkTf, inputpkTff, M22matrices, k_min,
                k_max, N, b_nu):
        
        (M22_dd, M22_dt_fp, M22_tt_fpfp, M22_tt_fkmpfp, MtAfp_11, MtAfkmpfp_12, 
         MtAfkmpfp_22, MtAfpfp_22, MtAfkmpfpfp_23, MtAfkmpfpfp_33, MB1_11, MC1_11, 
         MB2_11, MC2_11, MD2_21, MD3_21, MD2_22, MD3_22, MD4_22, MPb1b2, MPb1bs2, 
         MPb22, MPb2bs2, MPb2s2, MPb2t, MPbs2t) = M22matrices
        
        #matter coefficients 
        cmT = cmM(k_min, k_max, N, b_nu, inputpkT)
        cmTf = cmM(k_min, k_max, N, b_nu, inputpkTf)
        cmTff = cmM(k_min, k_max, N, b_nu, inputpkTff)
        
        #biased tracers coefficients
        bnu_b = 15.1*b_nu
        cmT_b = cmM(k_min, k_max, N, bnu_b, inputpkT)
        cmTf_b = cmM(k_min, k_max, N, bnu_b, inputpkTf)
        
        #creating the zeros of P22
        #Ploop
        P22dd, P22dt, P22tt = map(np.zeros,3*(len(kTout),))
        #Bias
        Pb1b2, Pb1bs2, Pb22, Pb2bs2, Pb2s2, Pb2t, Pbs2t = map(np.zeros,7*(len(kTout),))
        #A-TNS
        I1udd_1b, I2uud_1b, I3uuu_3b, I2uud_2b, I3uuu_2b = map(np.zeros,5*(len(kTout),))  
        #D-RSD
        I2uudd_1D, I2uudd_2D, I3uuud_2D, I3uuud_3D, I4uuuu_2D, I4uuuu_3D, I4uuuu_4D = map(np.zeros,7*(len(kTout),))
        
        #etaT = bias_nu + i*eta_m
        etamT = np.zeros(N+1, dtype = complex)
        etamT_b = np.zeros(N+1, dtype = complex)
        for jj in range(N+1):
            ietam = (2*np.pi*1j/np.log(k_max/k_min)) * (jj - N/2) *(N-1)/(N)
            etamT[jj] = b_nu + ietam 
            etamT_b[jj] = bnu_b + ietam
            
        for ii in range(len(kTout)):
            K = kTout[ii]
            precvec = K**(etamT) 
            vec = cmT * precvec
            vecf = cmTf * precvec
            vecff = cmTff * precvec
            
            precvec_b = K**(etamT_b)
            vec_b = cmT_b * precvec_b
            vecf_b = cmTf_b * precvec_b
            
            P22dd[ii] = (K**3 * vec @ M22_dd @ vec).real 
            P22dt[ii] = (2*K**3 * vecf @ M22_dt_fp @ vec).real 
            P22tt[ii] = K**3 *(vecff @ M22_tt_fpfp @ vec + vecf @ M22_tt_fkmpfp @ vecf).real
            
            Pb1b2[ii] = (K**3 * vec_b @ MPb1b2 @ vec_b).real 
            Pb1bs2[ii] = (K**3 * vec_b @ MPb1bs2 @ vec_b).real
            Pb22[ii] = (K**3 * vec_b @ MPb22 @ vec_b).real  
            Pb2bs2[ii] = (K**3 * vec_b @ MPb2bs2 @ vec_b).real
            Pb2s2[ii] = (K**3 * vec_b @ MPb2s2 @ vec_b).real
            Pb2t[ii] = (K**3 * vecf_b @ MPb2t @ vec_b).real
            Pbs2t[ii] = (K**3 * vecf_b @ MPbs2t @ vec_b).real
            
            I1udd_1b[ii] = (K**3 * vecf @ MtAfp_11 @ vec).real 
            I2uud_1b[ii] = (K**3 * vecf @ MtAfkmpfp_12 @ vecf).real
            I3uuu_3b[ii] = (K**3 * vecff @ MtAfkmpfpfp_33 @ vecf).real
            I2uud_2b[ii] = K**3 * (vecf @ MtAfkmpfp_22 @ vecf + vecff @ MtAfpfp_22 @ vec).real 
            I3uuu_2b[ii] = (K**3 * vecff @ MtAfkmpfpfp_23 @ vecf).real
            
            I2uudd_1D[ii] = K**3 * (vecf @ MB1_11 @ vecf + vec @ MC1_11 @ vecff).real
            I2uudd_2D[ii] = K**3 * (vecf @ MB2_11 @ vecf + vec @ MC2_11 @ vecff).real
            I3uuud_2D[ii] = (K**3 * vecf @ MD2_21 @ vecff).real 
            I3uuud_3D[ii] = (K**3 * vecf @ MD3_21 @ vecff).real 
            I4uuuu_2D[ii] = (K**3 * vecff @ MD2_22 @ vecff).real 
            I4uuuu_3D[ii] = (K**3 * vecff @ MD3_22 @ vecff).real 
            I4uuuu_4D[ii] = (K**3 * vecff @ MD4_22 @ vecff).real 
        
        return (P22dd, P22dt, P22tt, Pb1b2, Pb1bs2, Pb22, Pb2bs2, Pb2s2, 
            Pb2t, Pbs2t, I1udd_1b, I2uud_1b, I3uuu_3b, I2uud_2b, 
            I3uuu_2b, I2uudd_1D, I2uudd_2D, I3uuud_2D, I3uuud_3D,
            I4uuuu_2D, I4uuuu_3D, I4uuuu_4D) 
    
    def P13type(kTout, inputpkT, inputpkTf, inputpkTff, inputfkT, M13vectors, 
                k_min, k_max, N, b_nu):
                
        (M13_dd, M13_dt_fk, M13_tt_fk, Mafk_11, Mafp_11, Mafkfp_12, Mafpfp_12, 
         Mafkfkfp_33, Mafkfpfp_33, Msigma23) = M13vectors
        
        
        #condition for EdS-kernels or fk-kernels (default: fk-kernels)
        if EdSkernels == False:
            Fkoverf0 = interp(kTout, inputfkT[0], inputfkT[1])
        else:
            Fkoverf0 = np.full(len(kTout), 1.0)
            
        
        #matter coefficients 
        cmT = cmM(k_min, k_max, N, b_nu, inputpkT)
        cmTf = cmM(k_min, k_max, N, b_nu, inputpkTf)
        cmTff = cmM(k_min, k_max, N, b_nu, inputpkTff)
        
        #biased tracers coefficients
        bnu_b = 15.1*b_nu
        cmT_b = cmM(k_min, k_max, N, bnu_b, inputpkT)
        cmTf_b = cmM(k_min, k_max, N, bnu_b, inputpkTf)
        
        #creating the zeros of P13 
        #Ploop
        P13dd, P13dt, P13tt = map(np.zeros,3*(len(kTout),))
        #Bias
        sigma23 = np.zeros(len(kTout))
        #A-TNS
        I1udd_1a, I2uud_1a, I3uuu_3a = map(np.zeros,3*(len(kTout),))
        
        #etaT = bias_nu + i*eta_m
        etamT = np.zeros(N+1, dtype = complex)
        etamT_b = np.zeros(N+1, dtype = complex)
        for jj in range(N+1):
            ietam = (2*np.pi*1j/np.log(k_max/k_min)) * (jj - N/2) *(N-1)/(N)
            etamT[jj] = b_nu + ietam 
            etamT_b[jj] = bnu_b + ietam
        
        sigma2psi = 1/(6 * np.pi**2) * scipy.integrate.simps(inputpkT[1], inputpkT[0])
        sigma2v = 1/(6 * np.pi**2) * scipy.integrate.simps(inputpkTf[1], inputpkTf[0]) 
        sigma2w = 1/(6 * np.pi**2) * scipy.integrate.simps(inputpkTff[1], inputpkTff[0])
        
        for ii in range(len(kTout)):
            K = kTout[ii]
            precvec = K**(etamT) 
            vec = cmT * precvec
            vecf = cmTf * precvec 
            vecff = cmTff * precvec
            vecfM13dt_fk = vecf @ M13_dt_fk
            
            precvec_b = K**(etamT_b)
            vec_b = cmT_b * precvec_b
            vecf_b = cmTf_b * precvec_b
            
            P13dd[ii] = (K**3 * vec @ M13_dd).real - 61/105 * K**2 * sigma2psi
            P13dt[ii] = 0.5 *(K**3 * (Fkoverf0[ii] * vec @ M13_dt_fk + vecfM13dt_fk)).real - (23/21*sigma2psi * Fkoverf0[ii] + 2/21*sigma2v)* K**2  
            P13tt[ii] = (K**3 * Fkoverf0[ii] * (Fkoverf0[ii] * vec @ M13_tt_fk + vecfM13dt_fk ) ).real - (169/105*sigma2psi * Fkoverf0[ii] + 4/21 * sigma2v)* Fkoverf0[ii]* K**2 
            
            sigma23[ii] = (K**3 * vec_b @ Msigma23).real 
            
            I1udd_1a[ii] = K**3 * (Fkoverf0[ii] * vec @ Mafk_11 + vecf @ Mafp_11).real + (92/35*sigma2psi * Fkoverf0[ii] - 18/7*sigma2v)*K**2 
            I2uud_1a[ii] = K**3 * (Fkoverf0[ii] * vecf @ Mafkfp_12 + vecff @ Mafpfp_12).real - (38/35*Fkoverf0[ii] *sigma2v + 2/7*sigma2w)*K**2 
            I3uuu_3a[ii] = K**3 * Fkoverf0[ii] * (Fkoverf0[ii] * vecf @ Mafkfkfp_33 + vecff @ Mafkfpfp_33).real - (16/35*Fkoverf0[ii]*sigma2v + 6/7*sigma2w)*Fkoverf0[ii]*K**2 
        
        return (P13dd, P13dt, P13tt, sigma23, I1udd_1a, I2uud_1a, I3uuu_3a)
       
    
    #Evaluation: f(k)/f0 and linear power spectrums
    h, OmM0, fnu, Massnu = CosmoParam(h, omega_b, omega_cdm, omega_ncdm)    
    inputfkT = fOverf0EH(z_pk, inputpkT[0], OmM0, h, fnu)
    f0 = inputfkT[2]
    
    
    
    #condition for EdS-kernels or fk-kernels (default: fk-kernels)
    if EdSkernels == False:
        Fkoverf0 = interp(kTout, inputfkT[0], inputfkT[1])
    else:
        Fkoverf0 = np.full(len(kTout), 1.0)
        
        
    
    #Non-wiggle linear power spectrum
    inputpkT_NW = pknwJ(inputpkT[0], inputpkT[1], h)
    
    
    #condition for EdS-kernels or fk-kernels (default: fk-kernels)
    if EdSkernels == False:
        
        inputpkTf = (inputpkT[0], inputpkT[1]*inputfkT[1])
        inputpkTff = (inputpkT[0], inputpkT[1]*(inputfkT[1])**2)
        
        inputpkTf_NW = (inputpkT_NW[0], inputpkT_NW[1]*inputfkT[1])
        inputpkTff_NW = (inputpkT_NW[0], inputpkT_NW[1]*(inputfkT[1])**2)
    
    else:
        inputpkTf = (inputpkT[0], inputpkT[1])
        inputpkTff = (inputpkT[0], inputpkT[1])
        
        inputpkTf_NW = (inputpkT_NW[0], inputpkT_NW[1])
        inputpkTff_NW = (inputpkT_NW[0], inputpkT_NW[1])
          
        
        
    #P22type contributions 
    P22 = P22type(kTout, inputpkT, inputpkTf, inputpkTff, M22matrices, k_min, k_max, N, b_nu)
    P22_NW = P22type(kTout, inputpkT_NW, inputpkTf_NW, inputpkTff_NW, M22matrices, k_min, k_max, N, b_nu)
    
    #P13type contributions
    P13overpkl = P13type(kTout, inputpkT, inputpkTf, inputpkTff, inputfkT, M13vectors, k_min, k_max, N, b_nu)
    P13overpkl_NW = P13type(kTout, inputpkT_NW, inputpkTf_NW, inputpkTff_NW, inputfkT, M13vectors, k_min, k_max, N, b_nu)
    
    #Computations for Table
    pk_l = np.interp(kTout, inputpkT[0], inputpkT[1])
    pk_l_NW = np.interp(kTout,inputpkT_NW[0], inputpkT_NW[1])
        
    
    sigma2w = 1/(6 * np.pi**2) * scipy.integrate.simps(inputpkTff[1], inputpkTff[0])
    sigma2w_NW = 1/(6 * np.pi**2) * scipy.integrate.simps(inputpkTff_NW[1], inputpkTff_NW[0])

    Ploop_dd = P22[0] + P13overpkl[0]*pk_l
    Ploop_dt = P22[1] + P13overpkl[1]*pk_l
    Ploop_tt = P22[2] + P13overpkl[2]*pk_l
    
    Pb1b2 = P22[3];        
    Pb1bs2 = P22[4];          
    Pb22 = P22[5]-interp(10**(-10), kTout, P22[5]);    
    Pb2bs2 = P22[6]-interp(10**(-10), kTout, P22[6]);
    Pb2s2 = P22[7]-interp(10**(-10), kTout, P22[7]); 
    sigma23pkl = P13overpkl[3]*pk_l
    Pb2t = P22[8]; 
    Pbs2t = P22[9];
    
    I1udd_1 = P13overpkl[4]*pk_l + P22[10];
    I2uud_1 = P13overpkl[5]*pk_l + P22[11];
    I2uud_2 = (P13overpkl[6]*pk_l)/Fkoverf0 + Fkoverf0*P13overpkl[4]*pk_l + P22[13];
    I3uuu_2 = Fkoverf0*P13overpkl[5]*pk_l + P22[14];
    I3uuu_3 = P13overpkl[6]*pk_l + P22[12];
    
    I2uudd_1D = P22[15];   I2uudd_2D = P22[16];   I3uuud_2D = P22[17];
    I3uuud_3D = P22[18];   I4uuuu_2D = P22[19];   I4uuuu_3D = P22[20];
    I4uuuu_4D = P22[21];
    
    TableOut = (kTout, pk_l, Fkoverf0, Ploop_dd, Ploop_dt, Ploop_tt, 
                Pb1b2, Pb1bs2, Pb22, Pb2bs2, Pb2s2, sigma23pkl, Pb2t, Pbs2t, 
                I1udd_1, I2uud_1, I2uud_2, I3uuu_2, I3uuu_3, I2uudd_1D, 
                I2uudd_2D, I3uuud_2D, I3uuud_3D, I4uuuu_2D, I4uuuu_3D, 
                I4uuuu_4D, f0, sigma2w)
    
    
    ######################## Non- Wiggle ########################################
    
    Ploop_dd_NW = P22_NW[0] + P13overpkl_NW[0]*pk_l_NW;
    Ploop_dt_NW = P22_NW[1] + P13overpkl_NW[1]*pk_l_NW;
    Ploop_tt_NW = P22_NW[2] + P13overpkl_NW[2]*pk_l_NW;
    
    Pb1b2_NW = P22_NW[3];        
    Pb1bs2_NW = P22_NW[4];          
    Pb22_NW = P22_NW[5]-interp(10**(-10), kTout, P22_NW[5]);
    Pb2bs2_NW = P22_NW[6]-interp(10**(-10), kTout, P22_NW[6]); 
    Pb2s2_NW = P22_NW[7]-interp(10**(-10), kTout, P22_NW[7]);
    sigma23pkl_NW = P13overpkl_NW[3]*pk_l_NW;
    Pb2t_NW = P22_NW[8];        
    Pbs2t_NW = P22_NW[9];         
    
    I1udd_1_NW = P13overpkl_NW[4]*pk_l_NW + P22_NW[10];
    I2uud_1_NW = P13overpkl_NW[5]*pk_l_NW + P22_NW[11];
    I2uud_2_NW = (P13overpkl_NW[6]*pk_l_NW)/Fkoverf0 + Fkoverf0*P13overpkl_NW[4]*pk_l_NW + P22_NW[13];
    I3uuu_2_NW = Fkoverf0*P13overpkl_NW[5]*pk_l_NW + P22_NW[14];
    I3uuu_3_NW = P13overpkl_NW[6]*pk_l_NW + P22_NW[12];
    
    I2uudd_1D_NW = P22_NW[15];   I2uudd_2D_NW = P22_NW[16];   I3uuud_2D_NW = P22_NW[17];
    I3uuud_3D_NW = P22_NW[18];   I4uuuu_2D_NW = P22_NW[19];   I4uuuu_3D_NW = P22_NW[20];
    I4uuuu_4D_NW = P22_NW[21];
    
    
    TableOut_NW = (kTout, pk_l_NW, Fkoverf0, Ploop_dd_NW, Ploop_dt_NW, Ploop_tt_NW, 
                   Pb1b2_NW, Pb1bs2_NW, Pb22_NW, Pb2bs2_NW, Pb2s2_NW, sigma23pkl_NW, 
                   Pb2t_NW, Pbs2t_NW, I1udd_1_NW, I2uud_1_NW, I2uud_2_NW, I3uuu_2_NW, 
                   I3uuu_3_NW, I2uudd_1D_NW, I2uudd_2D_NW, I3uuud_2D_NW, I3uuud_3D_NW, 
                   I4uuuu_2D_NW, I4uuuu_3D_NW, I4uuuu_4D_NW, f0, sigma2w_NW)
    
    return (TableOut, TableOut_NW)




def TableOut_interp(k):
    '''Interpolation of non-linear terms given by the wiggle power spectra.
    
    Args:
        k: wave-number.
    Returns:
        Interpolates the non-linear terms given by the wiggle power spectra.
    '''
    nobjects = 25
    Tableout = np.zeros((nobjects + 1, len(k)))
    for ii in range(nobjects):
        Tableout[ii][:] = interp(k, kTout, TableOut[1+ii])
        Tableout[25][:] = sigma2w
    return Tableout




def TableOut_NW_interp(k):
    '''Interpolation of non-linear terms given by the non-wiggle power spectra.
    
    Args:
        k: wave-number.
    Returns:
        Interpolates the non-linear terms given by the non-wiggle power spectra.
    '''
    nobjects = 25
    Tableout_NW = np.zeros((nobjects + 1, len(k)))
    for ii in range(nobjects):
        Tableout_NW[ii][:] = interp(k, kTout, TableOut_NW[1+ii])
        Tableout_NW[25][:] = sigma2w_NW
    return Tableout_NW




def PEFTs(kev, mu, NuisanParams, Table):
    '''EFT galaxy power spectrum, Eq. ~ 3.40 at arXiv: 2208.02791.
    
    Args: 
        kev: evaluation points (wave-number coordinates).
        mu: cosine angle between the wave-vector ‘\vec{k}’ and the line-of-sight direction ‘\hat{n}’.
        NuisamParams: set of nuisance parameters [b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, ctilde, 
                                                  alphashot0, alphashot2, PshotP] in that order.
                    b1, b2, bs2, b3nl: biasing parameters.
                    alpha0, alpha2, alpha4: EFT parameters.
                    ctilde: parameter for NL0 ∝ Kaiser power spectrum.
                    alphashot0, alphashot2, PshotP: stochastic noise parameters.
       Table: List of non-linear terms given by the wiggle or non-wiggle power spectra.
    Returns:
       EFT galaxy power spectrum in redshift space.
    '''
    
    #NuisanParams
    (b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, 
                ctilde, alphashot0, alphashot2, PshotP) = NuisanParams
    
    #Table
    (pkl, Fkoverf0, Ploop_dd, Ploop_dt, Ploop_tt, Pb1b2, Pb1bs2, Pb22, Pb2bs2, 
         Pb2s2, sigma23pkl, Pb2t, Pbs2t, I1udd_1, I2uud_1, I2uud_2, I3uuu_2, I3uuu_3, 
         I2uudd_1D, I2uudd_2D, I3uuud_2D, I3uuud_3D, I4uuuu_2D, I4uuuu_3D, I4uuuu_4D, sigma2w) = Table
    
    fk = Fkoverf0*f0
        
    #linear power spectrum
    Pdt_L = pkl*Fkoverf0; Ptt_L = pkl*Fkoverf0**2;
        
    #one-loop power spectrum 
    Pdd = pkl + Ploop_dd; Pdt = Pdt_L + Ploop_dt; Ptt = Ptt_L + Ploop_tt;
        
        
    #biasing
    def PddXloop(b1, b2, bs2, b3nl):
        return (b1**2 * Ploop_dd + 2*b1*b2*Pb1b2 + 2*b1*bs2*Pb1bs2 + b2**2 * Pb22
                   + 2*b2*bs2*Pb2bs2 + bs2**2 *Pb2s2 + 2*b1*b3nl*sigma23pkl)
        
    def PdtXloop(b1, b2, bs2, b3nl):
        return b1*Ploop_dt + b2*Pb2t + bs2*Pbs2t + b3nl*Fkoverf0*sigma23pkl
        
    def PttXloop(b1, b2, bs2, b3nl):
        return Ploop_tt
        
    #RSD functions       
    def Af(mu, f0):
        return (f0*mu**2 * I1udd_1 + f0**2 * (mu**2 * I2uud_1 + mu**4 * I2uud_2)
                    + f0**3 * (mu**4 * I3uuu_2 +  mu**6 * I3uuu_3)) 
        
    def Df(mu, f0):
        return (f0**2 * (mu**2 * I2uudd_1D + mu**4 * I2uudd_2D) 
                    + f0**3 * (mu**4 * I3uuud_2D + mu**6 * I3uuud_3D)
                    + f0**4 * (mu**4 * I4uuuu_2D + mu**6 * I4uuuu_3D + mu**8 * I4uuuu_4D))
        
        
    #Introducing bias in RSD functions, eq.~ A.32 & A.33 at arXiv: 2208.02791
    def ATNS(mu, b1):
        return b1**3 * Af(mu, f0/b1)
        
    def DRSD(mu, b1):
        return b1**4 * Df(mu, f0/b1)
        
    def GTNS(mu, b1):
        return -((kev*mu*f0)**2 *sigma2w*(b1**2 * pkl + 2*b1*f0*mu**2 * Pdt_L 
                                   + f0**2 * mu**4 * Ptt_L))
        
        
    #One-loop SPT power spectrum in redshift space
    def PloopSPTs(mu, b1, b2, bs2, b3nl):
        return (PddXloop(b1, b2, bs2, b3nl) + 2*f0*mu**2 * PdtXloop(b1, b2, bs2, b3nl)
                    + mu**4 * f0**2 * PttXloop(b1, b2, bs2, b3nl) + ATNS(mu, b1) + DRSD(mu, b1)
                    + GTNS(mu, b1))
        
        
    #Linear Kaiser power spectrum
    def PKaiserLs(mu, b1):
        return (b1 + mu**2 * fk)**2 * pkl
        
    def PctNLOs(mu, b1, ctilde):
        return ctilde*(mu*kev*f0)**4 * sigma2w**2 * PKaiserLs(mu, b1)
    
    # EFT counterterms
    def Pcts(mu, alpha0, alpha2, alpha4):
        return (alpha0 + alpha2 * mu**2 + alpha4 * mu**4)*kev**2 * pkl
    
    #Stochastics noise
    def Pshot(mu, alphashot0, alphashot2, PshotP):
        return PshotP*(alphashot0 + alphashot2 * (kev*mu)**2)
        
    return (PloopSPTs(mu, b1, b2, bs2, b3nl) + Pcts(mu, alpha0, alpha2, alpha4)
                + PctNLOs(mu, b1, ctilde) + Pshot(mu, alphashot0, alphashot2, PshotP))




def Sigma2Total(kev, mu, Table_NW):
    '''Sigma² tot for IR-resummations, see eq.~ 3.59 at arXiv:2208.02791
    
    Args:
        kev: evaluation points (wave-number coordinates). 
        mu: cosine angle between the wave-vector ‘\vec{k}’ and the line-of-sight direction ‘\hat{n}’.
        Table_NW: List of non-linear terms given by the non-wiggle power spectra.
    Returns:
        Sigma² tot for IR-resummations.
    '''
    kT = kev; pkl_NW = Table_NW[0];
        
    kinit = 10**(-6);  kS = 0.4;                                  #integration limits
    pT = np.logspace(np.log10(kinit),np.log10(kS), num = 10**2)   #integration range
        
    PSL_NW = interp(pT, kT, pkl_NW)
    k_BAO = 1/104                                                 #BAO scale
        
    Sigma2 = 1/(6 * np.pi**2)*scipy.integrate.simps(PSL_NW*(1 - spherical_jn(0, pT/k_BAO) 
                                                + 2*spherical_jn(2, pT/k_BAO)), pT)
        
    deltaSigma2 = 1/(2 * np.pi**2)*scipy.integrate.simps(PSL_NW*spherical_jn(2, pT/k_BAO), pT)
        
    def Sigma2T(mu):
        return (1 + f0*mu**2 *(2 + f0))*Sigma2 + (f0*mu)**2 * (mu**2 - 1)* deltaSigma2
        
    return Sigma2T(mu)




def k_AP(k_obs, mu_obs, qperp, qpar):
    '''True ‘k’ coordinates.
    
    Args: where ‘_obs’ denote quantities that are observed assuming the reference (fiducial) cosmology.
        k_obs: observed wave-number.
        mu_obs: observed cosine angle between the wave-vector ‘\vec{k}’ and the line-of-sight direction ‘\hat{n}’.
        qperp, qpar: AP parameters.
    Returns:
        True wave-number ‘k_AP’.
    '''
    F = qpar/qperp
    return (k_obs/qperp)*(1 + mu_obs**2 * (1./F**2 - 1))**(0.5)




def mu_AP(mu_obs, qperp, qpar):
    '''True ‘mu’ coordinates.
    
    Args: where ‘_obs’ denote quantities that are observed assuming the reference (fiducial) cosmology.
        mu_obs: observed cosine angle between the wave-vector ‘\vec{k}’ and the line-of-sight direction ‘\hat{n}’.
        qperp, qpar: AP parameters.
    Returns:
        True ‘mu_AP’.
    '''
    F = qpar/qperp
    return (mu_obs/F) * (1 + mu_obs**2 * (1/F**2 - 1))**(-0.5)




def Hubble(Om, z_ev):
    '''Hubble parameter.
    
    Args:
        Om: Omega_b + Omega_c + Omega_nu (dimensionless matter density parameter).
        z_ev: redshift of evaluation.
    Returns:
        Hubble parameter.
    '''
    return ((Om) * (1 + z_ev)**3. + (1 - Om))**0.5




def DA(Om, z_ev):
    '''Angular-diameter distance.
    
     Args:
        Om: Omega_b + Omega_c + Omega_nu (dimensionless matter density parameter).
        z_ev: redshift of evaluation.
    Returns:
        Angular diameter distance.
    '''
    r = quad(lambda x: 1. / Hubble(Om, x), 0, z_ev)[0]
    return r / (1 + z_ev)




def Table_interp(k, kev, Table):
    '''Cubic interpolator.
    
    Args:
        k: coordinates at which to evaluate the interpolated values.
        kev: x-coordinates of the data points.
        Table: list of 1-loop contributions for the wiggle and non-wiggle
    '''
    f = interpolate.interp1d(kev, Table, kind = 'cubic', fill_value = "extrapolate")
    Tableout = f(k) 

    return Tableout




def RSDmultipoles(kev, NuisanParams, Omfid = -1, AP = False):
    '''Redshift space power spectrum multipoles.
    
    Args:
        If 'AP=True' (default: 'False') the code perform the AP test.
        If 'AP=True'. Include the fiducial Omfid after ‘NuisanParams’.
        
        kev: wave-number coordinates of evaluation.
        NuisamParams: set of nuisance parameters [b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, ctilde, alphashot0, 
                                                  alphashot2, PshotP] in that order.
                   b1, b2, bs2, b3nl: biasing parameters.
                   alpha0, alpha2, alpha4: EFT parameters.
                   ctilde: parameter for NL0 ∝ Kaiser power spectrum.
                   alphashot0, alphashot2, PshotP: stochastic noise parameters.
    Returns:
       Redshift space power spectrum multipoles (monopole, quadrupole and hexadecapole) at 'kev'.
    '''
            
    #NuisanParams
    (b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, 
                ctilde, alphashot0, alphashot2, PshotP) = NuisanParams
    
    if AP == True and Omfid == -1:
        sys.exit("Introduce the fiducial value of the dimensionless matter density parameter as ‘Omfid = value’.")
     
    if AP == True and Omfid > 0:
                
        #Om computed for any cosmology
        OmM = CosmoParam(h, omega_b, omega_cdm, omega_ncdm)[1]
        
        #qperp, qpar: AP parameters.
        qperp = DA(OmM, z_pk)/DA(Omfid, z_pk) 
        qpar = Hubble(Omfid, z_pk)/Hubble(OmM, z_pk) 
        
        
    def PIRs(kev, mu, Table, Table_NW):
        
        if AP == True:
            
            k_true = k_AP(kev, mu, qperp, qpar)
            mu_true = mu_AP(mu, qperp, qpar)
            
            Table_true = Table_interp(k_true, kev, Table)
            Table_NW_true = Table_interp(k_true, kev, Table_NW)
            
            Sigma2T = Sigma2Total(k_true, mu_true, Table_NW_true)
            
            Fkoverf0 = Table_true[1]; fk = Fkoverf0*f0
            pkl = Table_true[0]; pkl_NW = Table_NW_true[0];
            
            
            return ((b1 + fk * mu_true**2)**2 * (pkl_NW + np.exp(-k_true**2 * Sigma2T)*(pkl-pkl_NW)*(1 + k_true**2 * Sigma2T) )
                + np.exp(-k_true**2 * Sigma2T)*PEFTs(k_true, mu_true, NuisanParams, Table_true) 
                + (1 - np.exp(-k_true**2 * Sigma2T))*PEFTs(k_true, mu_true, NuisanParams, Table_NW_true)) 
            
        else:
            
            k = kev; Fkoverf0 = Table[1]; fk = Fkoverf0*f0
            pkl = Table[0]; pkl_NW = Table_NW[0];
            Sigma2T = Sigma2Total(kev, mu, Table_NW)
            
            return ((b1 + fk * mu**2)**2 * (pkl_NW + np.exp(-k**2 * Sigma2T)*(pkl-pkl_NW)*(1 + k**2 * Sigma2T) )
                + np.exp(-k**2 * Sigma2T)*PEFTs(k, mu, NuisanParams, Table) 
                + (1 - np.exp(-k**2 * Sigma2T))*PEFTs(k, mu, NuisanParams, Table_NW))     
    
    
    if AP == True:
        
        Nx = 6                                         #Points
        xGL, wGL = scipy.special.roots_legendre(Nx)    #x=cosθ and weights
        
        def ModelPkl0(Table, Table_NW):
            monop = 0;
            for ii in range(Nx):
                monop = monop + 0.5/(qperp**2 * qpar)*wGL[ii]*PIRs(kev, xGL[ii], Table, Table_NW)
            return monop
        
        def ModelPkl2(Table, Table_NW):    
            quadrup = 0;
            for ii in range(Nx):
                quadrup = quadrup + 5/(2*qperp**2 * qpar)*wGL[ii]*PIRs(kev, xGL[ii], Table, Table_NW)*eval_legendre(2, xGL[ii])
            return quadrup
        
        def ModelPkl4(Table, Table_NW):
            hexadecap = 0;
            for ii in range(Nx):
                hexadecap = hexadecap + 9/(2*qperp**2 * qpar)*wGL[ii]*PIRs(kev, xGL[ii], Table, Table_NW)*eval_legendre(4, xGL[ii])
            return hexadecap
    
    else:
        
        Nx = 6                                         #Points
        xGL, wGL = scipy.special.roots_legendre(Nx)    #x=cosθ and weights
        
        def ModelPkl0(Table, Table_NW):
            monop = 0;
            for ii in range(Nx):
                monop = monop + 0.5*wGL[ii]*PIRs(kev, xGL[ii], Table, Table_NW)
            return monop
        
        def ModelPkl2(Table, Table_NW):    
            quadrup = 0;
            for ii in range(Nx):
                quadrup = quadrup + 5/2*wGL[ii]*PIRs(kev, xGL[ii], Table, Table_NW)*eval_legendre(2, xGL[ii])
            return quadrup
        
        def ModelPkl4(Table, Table_NW):
            hexadecap = 0;
            for ii in range(Nx):
                hexadecap = hexadecap + 9/2*wGL[ii]*PIRs(kev, xGL[ii], Table, Table_NW)*eval_legendre(4, xGL[ii])
            return hexadecap
        
    
    Pkl0 = ModelPkl0(TableOut_interp(kev), TableOut_NW_interp(kev));
    Pkl2 = ModelPkl2(TableOut_interp(kev), TableOut_NW_interp(kev));
    Pkl4 = ModelPkl4(TableOut_interp(kev), TableOut_NW_interp(kev));
    
    #print('Redshift space power spectrum multipoles have been computed')
    #print('')
    #print('All computations have been performed successfully ')
    
    return (kev, Pkl0, Pkl2, Pkl4)




def RSDmultipoles_marginalized_const(kev, NuisanParams, Omfid = -1, AP = False, Hexa = False):
    '''Redshift space power spectrum multipoles 'const': Pℓ,const 
      (α->0, marginalizing over the EFT and stochastic parameters).
    
    Args:
        If 'AP=True' (default: 'False') the code perform the AP test.
        If 'AP=True'. Include the fiducial Omfid after ‘NuisanParams’.
        If 'Hexa = True' (default: 'False') the code includes the hexadecapole.
        
        kev: wave-number coordinates of evaluation.
        NuisamParams: set of nuisance parameters [b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, ctilde, alphashot0, 
                                                  alphashot2, PshotP] in that order.
                   b1, b2, bs2, b3nl: biasing parameters.
                   alpha0, alpha2, alpha4: EFT parameters.
                   ctilde: parameter for NL0 ∝ Kaiser power spectrum.
                   alphashot0, alphashot2, PshotP: stochastic noise parameters.
    Returns:
       Redshift space power spectrum multipoles (monopole, quadrupole and hexadecapole) at 'kev'.
    '''
    
    #NuisanParams
    (b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, 
                ctilde, alphashot0, alphashot2, PshotP) = NuisanParams
    
    if AP == True and Omfid == -1:
        sys.exit("Introduce the fiducial value of the dimensionless matter density parameter as ‘Omfid = value’.")
     
    if AP == True and Omfid > 0:
                
        #Om computed for any cosmology
        OmM = CosmoParam(h, omega_b, omega_cdm, omega_ncdm)[1]
        
        #qperp, qpar: AP parameters.
        qperp = DA(OmM, z_pk)/DA(Omfid, z_pk) 
        qpar = Hubble(Omfid, z_pk)/Hubble(OmM, z_pk) 
        
        
    def PIRs_const(kev, mu, Table, Table_NW):
        
        #NuisanParams_const: α->0 (set to zero EFT and stochastic parameters)
        alpha0, alpha2, alpha4, alphashot0, alphashot2 = np.zeros(5)
        
        NuisanParams_const = (b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, 
                              ctilde, alphashot0, alphashot2, PshotP)
        
        
        if AP == True:
            
            k_true = k_AP(kev, mu, qperp, qpar)
            mu_true = mu_AP(mu, qperp, qpar)
            
            Table_true = Table_interp(k_true, kev, Table)
            Table_NW_true = Table_interp(k_true, kev, Table_NW)
            
            Sigma2T = Sigma2Total(k_true, mu_true, Table_NW_true)
            
            Fkoverf0 = Table_true[1]; fk = Fkoverf0*f0
            pkl = Table_true[0]; pkl_NW = Table_NW_true[0];
            
            
            return ((b1 + fk * mu_true**2)**2 * (pkl_NW + np.exp(-k_true**2 * Sigma2T)*(pkl-pkl_NW)*(1 + k_true**2 * Sigma2T) )
                + np.exp(-k_true**2 * Sigma2T)*PEFTs(k_true, mu_true, NuisanParams_const, Table_true) 
                + (1 - np.exp(-k_true**2 * Sigma2T))*PEFTs(k_true, mu_true, NuisanParams_const, Table_NW_true))
        
        
        else:
            
            k = kev; Fkoverf0 = Table[1]; fk = Fkoverf0*f0
            pkl = Table[0]; pkl_NW = Table_NW[0];
            Sigma2T = Sigma2Total(kev, mu, Table_NW)
            
            return ((b1 + fk * mu**2)**2 * (pkl_NW + np.exp(-k**2 * Sigma2T)*(pkl-pkl_NW)*(1 + k**2 * Sigma2T) )
                + np.exp(-k**2 * Sigma2T)*PEFTs(k, mu, NuisanParams_const, Table) 
                + (1 - np.exp(-k**2 * Sigma2T))*PEFTs(k, mu, NuisanParams_const, Table_NW))
        
        
    Nx = 6                                         #Points
    xGL, wGL = scipy.special.roots_legendre(Nx)    #x=cosθ and weights
    
    def ModelPkl0_const(Table, Table_NW):
        if AP == True:
            monop = 1/(qperp**2 * qpar) * sum(0.5*wGL[ii]*PIRs_const(kev, xGL[ii], Table, Table_NW) for ii in range(Nx))
            return monop
        else:
            monop = sum(0.5*wGL[ii]*PIRs_const(kev, xGL[ii], Table, Table_NW) for ii in range(Nx))
            return monop
        
        
    def ModelPkl2_const(Table, Table_NW):    
        if AP == True:
            quadrup = 1/(qperp**2 * qpar) * sum(5/2*wGL[ii]*PIRs_const(kev, xGL[ii], Table, Table_NW)*eval_legendre(2, xGL[ii]) for ii in range(Nx))
            return quadrup
        else:
            quadrup = sum(5/2*wGL[ii]*PIRs_const(kev, xGL[ii], Table, Table_NW)*eval_legendre(2, xGL[ii]) for ii in range(Nx))
            return quadrup
        
        
    def ModelPkl4_const(Table, Table_NW):
        if AP == True:
            hexadecap = 1/(qperp**2 * qpar) * sum(9/2*wGL[ii]*PIRs_const(kev, xGL[ii], Table, Table_NW)*eval_legendre(4, xGL[ii]) for ii in range(Nx))
            return hexadecap
        else:
            hexadecap = sum(9/2*wGL[ii]*PIRs_const(kev, xGL[ii], Table, Table_NW)*eval_legendre(4, xGL[ii]) for ii in range(Nx))
            return hexadecap
        
        
    if Hexa == False:
        Pkl0_const = ModelPkl0_const(TableOut_interp(kev), TableOut_NW_interp(kev));
        Pkl2_const = ModelPkl2_const(TableOut_interp(kev), TableOut_NW_interp(kev));
        return (Pkl0_const, Pkl2_const)
    
    else:
        Pkl0_const = ModelPkl0_const(TableOut_interp(kev), TableOut_NW_interp(kev));
        Pkl2_const = ModelPkl2_const(TableOut_interp(kev), TableOut_NW_interp(kev));
        Pkl4_const = ModelPkl4_const(TableOut_interp(kev), TableOut_NW_interp(kev));
        return (Pkl0_const, Pkl2_const, Pkl4_const)        
    
    
    

def PEFTs_derivatives(k, mu, pkl, PshotP):
    '''Derivatives of PEFTs with respect to the EFT and stochastic parameters.
    
    Args:
        k: wave-number coordinates of evaluation.
        mu: cosine angle between the wave-vector ‘\vec{k}’ and the line-of-sight direction ‘\hat{n}’.
        pkl: linear power spectrum.
        PshotP: stochastic nuisance parameter.
    Returns:
        ∂P_EFTs/∂α_i with: α_i = {alpha0, alpha2, alpha4, alphashot0, alphashot2}
    '''
    
    k2 = k**2
    k2mu2 = k2 * mu**2
    k2mu4 = k2mu2 * mu**2

    PEFTs_alpha0 = k2 * pkl
    PEFTs_alpha2 = k2mu2 * pkl 
    PEFTs_alpha4 = k2mu4 * pkl 
    PEFTs_alphashot0 = PshotP
    PEFTs_alphashot2 = k2mu2 * PshotP
    
    return (PEFTs_alpha0, PEFTs_alpha2, PEFTs_alpha4, PEFTs_alphashot0, PEFTs_alphashot2)  




def RSDmultipoles_marginalized_derivatives(kev, NuisanParams, Omfid = -1, AP = False, Hexa = False):
    '''Redshift space power spectrum multipoles 'derivatives': Pℓ,i=∂Pℓ/∂α_i 
      (derivatives with respect to the EFT and stochastic parameters).
    
    Args:
        If 'AP=True' (default: 'False') the code perform the AP test.
        If 'AP=True'. Include the fiducial Omfid after ‘NuisanParams’.
        If 'Hexa = True' (default: 'False') the code includes the hexadecapole.
        
        kev: wave-number coordinates of evaluation.
        NuisamParams: set of nuisance parameters [b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, ctilde, alphashot0, 
                                                  alphashot2, PshotP] in that order.
                   b1, b2, bs2, b3nl: biasing parameters.
                   alpha0, alpha2, alpha4: EFT parameters.
                   ctilde: parameter for NL0 ∝ Kaiser power spectrum.
                   alphashot0, alphashot2, PshotP: stochastic noise parameters.
    Returns:
       Redshift space power spectrum multipoles (monopole, quadrupole and hexadecapole) at 'kev'.
    '''
            
    #NuisanParams
    (b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, 
                ctilde, alphashot0, alphashot2, PshotP) = NuisanParams
    
    if AP == True and Omfid == -1:
        sys.exit("Introduce the fiducial value of the dimensionless matter density parameter as ‘Omfid = value’.")
     
    if AP == True and Omfid > 0:
                
        #Om computed for any cosmology
        OmM = CosmoParam(h, omega_b, omega_cdm, omega_ncdm)[1]
        
        #qperp, qpar: AP parameters.
        qperp = DA(OmM, z_pk)/DA(Omfid, z_pk) 
        qpar = Hubble(Omfid, z_pk)/Hubble(OmM, z_pk) 
        
    
    def PIRs_derivatives(kev, mu, Table, Table_NW):
        
        if AP == True:
            
            k_true = k_AP(kev, mu, qperp, qpar)
            mu_true = mu_AP(mu, qperp, qpar)
            
            Table_true = Table_interp(k_true, kev, Table)
            Table_NW_true = Table_interp(k_true, kev, Table_NW)
            
            Sigma2T = Sigma2Total(k_true, mu_true, Table_NW_true)
            
            Fkoverf0 = Table_true[1]; fk = Fkoverf0*f0
            pkl = Table_true[0]; pkl_NW = Table_NW_true[0];
            
            #computing PEFTs_derivatives: wiggle and non-wiggle terms
            PEFTs_alpha0, PEFTs_alpha2, PEFTs_alpha4, PEFTs_alphashot0, PEFTs_alphashot2 = PEFTs_derivatives(k_true, mu_true, pkl, PshotP)
            PEFTs_alpha0_NW, PEFTs_alpha2_NW, PEFTs_alpha4_NW, PEFTs_alphashot0_NW, PEFTs_alphashot2_NW = PEFTs_derivatives(k_true, mu_true, pkl_NW, PshotP)
            
            exp_term = np.exp(-k_true**2 * Sigma2T)
            exp_term_inv = 1 - exp_term
            
            #computing PIRs_derivatives for EFT and stochastic parameters
            PIRs_alpha0 = exp_term * PEFTs_alpha0 + exp_term_inv * PEFTs_alpha0_NW
            PIRs_alpha2 = exp_term * PEFTs_alpha2 + exp_term_inv * PEFTs_alpha2_NW
            PIRs_alpha4 = exp_term * PEFTs_alpha4 + exp_term_inv * PEFTs_alpha4_NW
            PIRs_alphashot0 = exp_term * PEFTs_alphashot0 + exp_term_inv * PEFTs_alphashot0_NW
            PIRs_alphashot2 = exp_term * PEFTs_alphashot2 + exp_term_inv * PEFTs_alphashot2_NW
            
            return (PIRs_alpha0, PIRs_alpha2, PIRs_alpha4, PIRs_alphashot0, PIRs_alphashot2)
        
        
        else:
            k = kev; Fkoverf0 = Table[1]; fk = Fkoverf0*f0
            pkl = Table[0]; pkl_NW = Table_NW[0];
            
            Sigma2T = Sigma2Total(kev, mu, Table_NW)
            
            #computing PEFTs_derivatives: wiggle and non-wiggle terms
            PEFTs_alpha0, PEFTs_alpha2, PEFTs_alpha4, PEFTs_alphashot0, PEFTs_alphashot2 = PEFTs_derivatives(k, mu, pkl, PshotP)
            PEFTs_alpha0_NW, PEFTs_alpha2_NW, PEFTs_alpha4_NW, PEFTs_alphashot0_NW, PEFTs_alphashot2_NW = PEFTs_derivatives(k, mu, pkl_NW, PshotP)
            
            exp_term = np.exp(-k**2 * Sigma2T)
            exp_term_inv = 1 - exp_term
            
            #computing PIRs_derivatives for EFT and stochastic parameters
            PIRs_alpha0 = exp_term * PEFTs_alpha0 + exp_term_inv * PEFTs_alpha0_NW
            PIRs_alpha2 = exp_term * PEFTs_alpha2 + exp_term_inv * PEFTs_alpha2_NW
            PIRs_alpha4 = exp_term * PEFTs_alpha4 + exp_term_inv * PEFTs_alpha4_NW
            PIRs_alphashot0 = exp_term * PEFTs_alphashot0 + exp_term_inv * PEFTs_alphashot0_NW
            PIRs_alphashot2 = exp_term * PEFTs_alphashot2 + exp_term_inv * PEFTs_alphashot2_NW
            
            return (PIRs_alpha0, PIRs_alpha2, PIRs_alpha4, PIRs_alphashot0, PIRs_alphashot2)
        
        
    Nx = 6    
    xGL, wGL = scipy.special.roots_legendre(Nx)    #x=cosθ and weights
    
    def ModelPkl0_derivatives(Table, Table_NW):
        if AP == True:
            monop = 1/(qperp**2 * qpar) * sum(0.5*wGL[ii]*np.array(PIRs_derivatives(kev, xGL[ii], Table, Table_NW)) for ii in range(Nx))
            return monop
        
        else:
            monop = sum(0.5*wGL[ii]*np.array(PIRs_derivatives(kev, xGL[ii], Table, Table_NW)) for ii in range(Nx))
            return monop
        
    
    def ModelPkl2_derivatives(Table, Table_NW):
        if AP == True:
            quadrup = 1/(qperp**2 * qpar) * sum(5/2*wGL[ii]*np.array(PIRs_derivatives(kev, xGL[ii], Table, Table_NW))*eval_legendre(2, xGL[ii]) for ii in range(Nx))
            return quadrup 
        
        else:
            quadrup = sum(5/2*wGL[ii]*np.array(PIRs_derivatives(kev, xGL[ii], Table, Table_NW))*eval_legendre(2, xGL[ii]) for ii in range(Nx))
            return quadrup
    
    
    def ModelPkl4_derivatives(Table, Table_NW):
        if AP == True:
            hexadecap = 1/(qperp**2 * qpar) * sum(9/2*wGL[ii]*np.array(PIRs_derivatives(kev, xGL[ii], Table, Table_NW))*eval_legendre(4, xGL[ii]) for ii in range(Nx))
            return hexadecap
        
        else:
            hexadecap = sum(9/2*wGL[ii]*np.array(PIRs_derivatives(kev, xGL[ii], Table, Table_NW))*eval_legendre(4, xGL[ii]) for ii in range(Nx))
            return hexadecap  
    
    if Hexa == False:   
        Pkl0_derivatives = ModelPkl0_derivatives(TableOut_interp(kev), TableOut_NW_interp(kev));
        Pkl2_derivatives = ModelPkl2_derivatives(TableOut_interp(kev), TableOut_NW_interp(kev));
        return (Pkl0_derivatives, Pkl2_derivatives)
    
    else:
        Pkl0_derivatives = ModelPkl0_derivatives(TableOut_interp(kev), TableOut_NW_interp(kev));
        Pkl2_derivatives = ModelPkl2_derivatives(TableOut_interp(kev), TableOut_NW_interp(kev));
        Pkl4_derivatives = ModelPkl4_derivatives(TableOut_interp(kev), TableOut_NW_interp(kev));
        return (Pkl0_derivatives, Pkl2_derivatives, Pkl4_derivatives)
    
    
    
    
#Marginalization matrices

def startProduct(A, B, invCov):
    '''Computes: A @ InvCov @ B^{T}, where 'T' means transpose.
    
    Args:
         A: first vector, array of the form 1 x n
         B: second vector, array of the form 1 x n
         invCov: inverse of covariance matrix, array of the form n x n
    
    Returns:
         The result of: A @ InvCov @ B^{T}
    '''
    
    return A @ invCov @ B.T 




def compute_L0(Pl_const, Pl_data, invCov):
    '''Computes the term L0 of the marginalized Likelihood.
    
    Args:
         Pl_const: model multipoles for the constant part (Pℓ,const = Pℓ(α->0)), array of the form 1 x n
         Pl_data: data multipoles, array of the form 1 x n 
         invCov: inverse of covariance matrix, array of the form n x n
         
    Return:
         Loglikelihood for the constant part of the model multipoles 
    '''
    
    D_const = Pl_const - Pl_data
    
    L0 = -0.5 * startProduct(D_const, D_const, invCov)             #eq. 2.4 notes on marginalization
    
    return L0




def compute_L1i(Pl_i, Pl_const, Pl_data, invCov):
    '''Computes the term L1i of the marginalized Likelihood.
    
    Args:
         Pl_i: array with the derivatives of the power spectrum multipoles with respect to 
               the EFT and stochastic parameters, i.e., Pℓ,i=∂Pℓ/∂α_i , i = 1,..., ndim
               array of the form ndim x n
         Pl_const: model multipoles for the constant part (Pℓ,const = Pℓ(α->0)), array of the form 1 x n
         Pl_data: data multipoles, array of the form 1 x n
         invCov: inverse of covariance matrix, array of the form n x n
    Return:
         array for L1i
    '''
    
    D_const = Pl_const - Pl_data  
    
    #ndim = len(Pl_i)
    
    #computing L1i
    #L1i = np.zeros(ndim)
    
    #for ii in range(ndim):
    #    term1 = startProduct(Pl_i[ii], D_const, invCov)
    #    term2 = startProduct(D_const, Pl_i[ii], invCov)
    #    L1i[ii] = -0.5 * (term1 + term2)
    
    L1i = - startProduct(Pl_i, D_const, invCov)
    
    return L1i




def compute_L2ij(Pl_i, invCov, sigma_prior = np.inf):
    '''Computes the term L2ij of the marginalized Likelihood.
    
    Args:
         Pl_i: array with the derivatives of the power spectrum multipoles with respect to 
               the EFT and stochastic parameters, i.e., Pℓ,i=∂Pℓ/∂α_i , i = 1,..., ndim
               array of the form ndim x n
         invCov: inverse of covariance matrix, array of the form n x n
    Return:
         array for L2ij
    '''
    
    #ndim = len(Pl_i)
    
    #Computing L2ij
    #L2ij = np.zeros((ndim, ndim))
    
    #for ii in range (ndim):
        #for jj in range (ndim):
            #L2ij[ii, jj] = startProduct(Pl_i[ii], Pl_i[jj], invCov)
    
    L2ij = startProduct(Pl_i, Pl_i, invCov)
            
    # Adding prior variances to L2ij
    if isinstance(sigma_prior, (int, float)):
        L2ij += 1 / (sigma_prior ** 2)
    else:
        L2ij += np.diag(1 / np.array(sigma_prior) ** 2)
            
    return L2ij 
