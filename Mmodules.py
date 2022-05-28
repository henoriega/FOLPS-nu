#!/usr/bin/env python
# coding: utf-8

# ###  Full one-loop power spectrum using FFTLog in fk-kernels
# 
# Here we compute all the contributions related to: the one-loop corrections, non-linear bias and redshift-space distortions (A-TNS and D-RSD).
# 
# **Requirement:** update your numpy to versions $\geq$ 1.20.0 or scale the DFT by $1/N$ in the $c_m$ computation.



import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy import integrate
from scipy.interpolate import CubicSpline
from scipy.fft import dst, idst
from scipy.special import gamma
from scipy.special import spherical_jn
from scipy.special import eval_legendre
from classy import Class
import time as tm






def Matrices(FFTLogParams):
    
    (k_min, k_max, N, b_nu) = FFTLogParams   
    
    global M22matrices, M13vectors, bnu_b

    def Imatrix(nu1, nu2):
        return 1/(8 * np.pi**(3/2)) * ( gamma(3/2-nu1)*gamma(3/2-nu2)*gamma(nu1+nu2-3/2) )/( gamma(nu1)*gamma(nu2)*gamma(3-nu1-nu2) )
    
    "TypeM22"
    
    def M22(nu1, nu2):
        
        "Ploop"
        def M22_dd(nu1, nu2):
            return Imatrix(nu1,nu2)*(3/2-nu1-nu2)*(1/2-nu1-nu2)*( (nu1*nu2)*(98*(nu1+nu2)**2 - 14*(nu1+nu2) + 36) - 91*(nu1+nu2)**2+ 3*(nu1+nu2) + 58)/(196*nu1*(1+nu1)*(1/2-nu1)*nu2*(1+nu2)*(1/2-nu2))
        
        def M22_dt_fp(nu1, nu2):
            return Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-1+2*(nu1+nu2))*(-23-21*nu1+(-38+7*nu1*(-1+7*nu1))*nu2+7*(3+7*nu1)*nu2**2) )/(196*nu1*(1+nu1)*nu2*(1+nu2)*(-1+2*nu2))
        
        def M22_tt_fp2(nu1, nu2):
            return Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-12*(1-2*nu2)**2 + 98*nu1**(3)*nu2 + 7*nu1**2*(1+2*nu2*(-8+7*nu2))- nu1*(53+2*nu2*(17+7*nu2))))/(196*nu1*(1+nu1)*nu2*(1+nu2)*(-1+2*nu2))
        
        def M22_tt_fkmpfp(nu1, nu2):
            return Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-37+7*nu1**(2)*(3+7*nu2) + nu2*(-10+21*nu2) + nu1*(-10+7*nu2*(-1+7*nu2))))/(98*nu1*(1+nu1)*nu2*(1+nu2))
        
        "A-TNS"
        def MtAfp_11(nu1, nu2):
            return Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-1+2*(nu1+nu2))*(-5+nu1*(-4+7*(nu1+nu2))))/(7*nu1*(1+nu1)*(-1+2*nu1)*nu2)
        
        def MtAfkmpfp_12(nu1, nu2):
            return -Imatrix(nu1,nu2)*(((-3+2*(nu1+nu2))*(-1+2*(nu1+nu2))*(6+7*(nu1+nu2)))/(56*nu1*(1+nu1)*nu2*(1+nu2)))
        
        def MtAfkmpfp_22(nu1, nu2):
            return Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-18+3*nu1*(1+4*(10-9*nu1)*nu1)+75*nu2+8*nu1*(41+2*nu1*(-28+nu1*(-4+7*nu1)))*nu2+48*nu1*(-9+nu1*(-3+7*nu1))*nu2**2+4*(-39+4*nu1*(-19+35*nu1))*nu2**3+336*nu1*nu2**4) )/(56*nu1*(1+nu1)*(-1+2*nu1)*nu2*(1+nu2)*(-1+2*nu2))
        
        def MtAfp2_22(nu1, nu2):
            return Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-5+3*nu2+nu1*(-4+7*(nu1+nu2))))/(7*nu1*(1+nu1)*nu2)
        
        def MtAfkmpfp2_23(nu1, nu2):
            return -Imatrix(nu1,nu2)*(((-1+7*nu1)*(-3+2*(nu1+nu2))*(-1+2*(nu1+nu2)))/(28*nu1*(1+nu1)*nu2*(1+nu2)))
        
        def MtAfkmpfp2_33(nu1, nu2):
            return Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-1+2*(nu1+nu2))*(-13*(1+nu1)+2*(-11+nu1*(-1+14*nu1))*nu2 + 4*(3+7*nu1)*nu2**2))/(28*nu1*(1+nu1)*nu2*(1+nu2)*(-1+2*nu2))
        
        "D-RSD"
        def MB1_11(nu1, nu2):
            return Imatrix(nu1,nu2)*(3-2*(nu1+nu2))/(4*nu1*nu2)
        
        def MK1_C2(nu1, nu2):
            return Imatrix(nu1,nu2)*((-3+2*nu1)*(-3+2*(nu1+nu2)))/(4*nu2*(1+nu2)*(-1+2*nu2))
        
        def MB2_11(nu1, nu2):
            return Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-1+2*(nu1+nu2)))/(4*nu1*nu2)
        
        def MK2_C2(nu1, nu2):
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
    
        
        return (M22_dd(nu1, nu2), M22_dt_fp(nu1, nu2), M22_tt_fp2(nu1, nu2), M22_tt_fkmpfp(nu1, nu2),
                MtAfp_11(nu1, nu2), MtAfkmpfp_12(nu1, nu2), MtAfkmpfp_22(nu1, nu2), MtAfp2_22(nu1, nu2), 
                MtAfkmpfp2_23(nu1, nu2), MtAfkmpfp2_33(nu1, nu2), MB1_11(nu1, nu2), MK1_C2(nu1, nu2), 
                MB2_11(nu1, nu2), MK2_C2(nu1, nu2), MD2_21(nu1, nu2), MD3_21(nu1, nu2), MD2_22(nu1, nu2), 
                MD3_22(nu1, nu2), MD4_22(nu1, nu2))
    
    def M22bias(nu1, nu2):
        
        "Bias"
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
    
    "TypeM13"

    def M13(nu1):
        
        "Ploop"
        def M13_dd(nu1):
            return ((1+9*nu1)/4) * np.tan(nu1*np.pi)/( 28*np.pi*(nu1+1)*nu1*(nu1-1)*(nu1-2)*(nu1-3) )
        
        def M13_dt_fk(nu1):
            return ((-7+9*nu1)*np.tan(nu1*np.pi))/(112*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1*(1+nu1))
        
        def M13_tt_fk(nu1):
            return -(np.tan(nu1*np.pi)/(14*np.pi*(-3 + nu1)*(-2 + nu1)*(-1 + nu1)*nu1*(1 + nu1) ))
        
        "A-TNS"
        def Mafk_11(nu1):
            return ((15-7*nu1)*np.tan(nu1*np.pi))/(56*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1)
        
        def Mafp_11(nu1):
            return ((-6+7*nu1)*np.tan(nu1*np.pi))/(56*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1)
        
        def Mafkfp_12(nu1):
            return (3*(-13+7*nu1)*np.tan(nu1*np.pi))/(224*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1*(1+nu1))
        
        def Mafp2_12(nu1):
            return (3*(1-7*nu1)*np.tan(nu1*np.pi))/(224*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1*(1+nu1))
        
        def Mafk2fp_33(nu1):
            return ((21+(53-28*nu1)*nu1)*np.tan(nu1*np.pi))/(224*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1*(1+nu1))
        
        def Mafkfp2_33(nu1):
            return ((-21+nu1*(-17+28*nu1))*np.tan(nu1*np.pi))/(224*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1*(1+nu1))
        
        return (M13_dd(nu1), M13_dt_fk(nu1), M13_tt_fk(nu1), Mafk_11(nu1),  Mafp_11(nu1), Mafkfp_12(nu1),
                Mafp2_12(nu1), Mafk2fp_33(nu1), Mafkfp2_33(nu1))
    
    def M13bias(nu1):
        
        "Bias"
        def Msigma23(nu1):
            return (45*np.tan(nu1*np.pi))/(128*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1*(1+nu1))
        
        return (Msigma23(nu1))

    
    
    def M22type(k_min, k_max, N, b_nu, M22):
        
        #nuT = -etaT/2, etaT = bias_nu + i*eta_m        
        nuT = np.zeros(N+1, dtype = complex)
        
        for jj in range(N+1):
            nuT[jj] = -0.5 * (b_nu +  (2*np.pi*1j/np.log(k_max/k_min)) * (jj - N/2) *(N-1)/(N))
            
        #reduce time x10 compared to "for" iterations
        nuT_x, nuT_y = np.meshgrid(nuT, nuT) 
        M22matrix = M22(nuT_y, nuT_x)
        
        return np.array(M22matrix)
    
    def M13type(k_min, k_max, N, b_nu, M13):
           
        #nuT = -etaT/2, etaT = bias_nu + i*eta_m 
        nuT = np.zeros(N+1, dtype = complex)
        
        for ii in range(N+1):
            nuT[ii] = -0.5 * (b_nu +  (2*np.pi*1j/np.log(k_max/k_min)) * (ii - N/2) *(N-1)/(N))
        
        M13vector = M13(nuT)
            
        return np.array(M13vector)
    
    bnu_b = 15.1*b_nu
    
    
    start_M = tm.time() #time start

    M22T =  M22type(k_min, k_max, N, b_nu, M22)
    M22biasT = M22type(k_min, k_max, N, bnu_b, M22bias)
    M22matrices = np.concatenate((M22T, M22biasT))
    
    M13T = M13type(k_min, k_max, N, b_nu, M13)
    M13biasT = np.reshape(M13type(k_min, k_max, N, bnu_b, M13bias), (1, int(N+1)))
    M13vectors = np.concatenate((M13T, M13biasT))
    
    end_M = tm.time(); time_M = end_M - start_M #time stop
    
    print('Cosmology independent terms (computed only once!):')
    print('Computation time for all matrices/vectors (N=256) =',time_M, 's')
  
    return (M22matrices, M13vectors)







#use: NeuEffect = y to consider kernels-fk. Otherwise use use: NeuEffect = n, and EdS kernels will b eemploy 
    
#NeuEffect = 'y'

def LoopCorr(inputpkT, CosmoParams, FFTLogParams, NeuEffect):
    
    "CosmoParams"
    (z_pk, omega_b, omega_cdm, omega_ncdm, h) = CosmoParams
    

    "FFTLogParams"
    (k_min, k_max, N, b_nu) = FFTLogParams
    
    global TableOut, TableOut_NW, f0
    
    
    #Returns some inputs for fOverf0EH
    def CosmoParam(h, ombh2, omch2, omnuh2):
                
        Omb = ombh2/h**2;
        Omc = omch2/h**2;
        Omnu = omnuh2/h**2;
        
        OmM0 = Omb + Omc + Omnu; 
        fnu = Omnu/OmM0;
        Massnu = Omnu*93.14*h**2;
        
        return(h, OmM0, fnu, Massnu)

    
    #Returns $f(k)/f_0$ & f0 following E&H (1997)
    def fOverf0EH(zev, k, OmM0, h, fnu):
        
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
    
    
    def interp(k, x, y):
        inter = CubicSpline(x, y)
        return inter(k)  
    
    
    #Gives the non-wiggle piece of the pkl following Julien 2010
    def pknwJ(k, PSLk, h):
        
        #ksmin(max): k range and Nks: points
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
        
        return(inputpkT[0], PNWkTot)
    
    #cmM: returns c_m
    def cmM(k_min, k_max, N, b_nu, inputpkT):

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
    
    ################################ KTOUT ###########################################    
    
    kminout = 0.001; kmaxout = 2;
    
    kTout = []
    for ii in range(len(inputpkT[0])):
        kv = inputpkT[0][ii]
        if (kv >= kminout and kv < kmaxout):
            x = kv
            kTout.append(x)
    kTout = np.array(kTout)
    
    #################################### KTOUT #########################################
    
    def P22type(kTout, inputpkT, inputpkTf, inputpkTff, M22matrices, k_min,
                k_max, N, b_nu):
        
        (M22_dd, M22_dt_fp, M22_tt_fp2, M22_tt_fkmpfp, MtAfp_11, MtAfkmpfp_12, 
         MtAfkmpfp_22, MtAfp2_22, MtAfkmpfp2_23, MtAfkmpfp2_33, MB1_11, MK1_C2, 
         MB2_11, MK2_C2, MD2_21, MD3_21, MD2_22, MD3_22, MD4_22, MPb1b2, MPb1bs2, 
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
            P22tt[ii] = K**3 *(2*vecff @ M22_tt_fp2 @ vec + vecf @ M22_tt_fkmpfp @ vecf).real
            
            Pb1b2[ii] = (K**3 * vec_b @ MPb1b2 @ vec_b).real 
            Pb1bs2[ii] = (K**3 * vec_b @ MPb1bs2 @ vec_b).real
            Pb22[ii] = (K**3 * vec_b @ MPb22 @ vec_b).real  
            Pb2bs2[ii] = (K**3 * vec_b @ MPb2bs2 @ vec_b).real
            Pb2s2[ii] = (K**3 * vec_b @ MPb2s2 @ vec_b).real
            Pb2t[ii] = (K**3 * vecf_b @ MPb2t @ vec_b).real
            Pbs2t[ii] = (K**3 * vecf_b @ MPbs2t @ vec_b).real
            
            I1udd_1b[ii] = (K**3 * vecf @ MtAfp_11 @ vec).real 
            I2uud_1b[ii] = (K**3 * vecf @ MtAfkmpfp_12 @ vecf).real
            I3uuu_3b[ii] = (K**3 * vecff @ MtAfkmpfp2_33 @ vecf).real
            I2uud_2b[ii] = K**3 * (vecf @ MtAfkmpfp_22 @ vecf + vecff @ MtAfp2_22 @ vec).real 
            I3uuu_2b[ii] = (K**3 * vecff @ MtAfkmpfp2_23 @ vecf).real
            
            I2uudd_1D[ii] = K**3 * (vecf @ MB1_11 @ vecf + vec @ MK1_C2 @ vecff).real
            I2uudd_2D[ii] = K**3 * (vecf @ MB2_11 @ vecf + vec @ MK2_C2 @ vecff).real
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
                
        (M13_dd, M13_dt_fk, M13_tt_fk, Mafk_11, Mafp_11, Mafkfp_12, Mafp2_12, 
         Mafk2fp_33, Mafkfp2_33, Msigma23) = M13vectors
        
        Fkoverf0 = interp(kTout, inputfkT[0], inputfkT[1])
        
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
        I1udd_1ac, I2uud_1ac, I3uuu_3ac = map(np.zeros,3*(len(kTout),))
        
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
            
            I1udd_1ac[ii] = K**3 * (Fkoverf0[ii] * vec @ Mafk_11 + vecf @ Mafp_11).real + (92/35*sigma2psi * Fkoverf0[ii] - 18/7*sigma2v)*K**2 
            I2uud_1ac[ii] = K**3 * (Fkoverf0[ii] * vecf @ Mafkfp_12 + vecff @ Mafp2_12).real - (38/35*Fkoverf0[ii] *sigma2v + 2/7*sigma2w)*K**2 
            I3uuu_3ac[ii] = K**3 * Fkoverf0[ii] * (Fkoverf0[ii] * vecf @ Mafk2fp_33 + vecff @ Mafkfp2_33).real - (16/35*Fkoverf0[ii]*sigma2v + 6/7*sigma2w)*Fkoverf0[ii]*K**2 
        
        return (P13dd, P13dt, P13tt, sigma23, I1udd_1ac, I2uud_1ac, I3uuu_3ac)  
    
    #Evaluation: f(k) and linear power spectrums
    h, OmM0, fnu, Massnu = CosmoParam(h, omega_b, omega_cdm, omega_ncdm)    
    inputfkT = fOverf0EH(z_pk, inputpkT[0], OmM0, h, fnu)
    f0 = inputfkT[2]
    
    inputpkT_NW = pknwJ(inputpkT[0], inputpkT[1], h)
    
    
    if NeuEffect in ['y', 'Y', 'yes', 'Yes', 'YES']:
        
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
    Fkoverf0 = interp(kTout, inputfkT[0], inputfkT[1])
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






def PklModel(inputpkT, CosmoParams, NuisanParams, FFTLogParams, NeuEffect):
    
    "NuisanParams"
    (b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, 
                ctilde, alphashot0, alphashot2, PshotP) = NuisanParams
    
    def PEFTs(mu, Table):
        
        (k, pkl, Fkoverf0, Ploop_dd, Ploop_dt, Ploop_tt, Pb1b2, Pb1bs2, Pb22, Pb2bs2, 
         Pb2s2, sigma23pkl, Pb2t, Pbs2t, I1udd_1, I2uud_1, I2uud_2, I3uuu_2, I3uuu_3, 
         I2uudd_1D, I2uudd_2D, I3uuud_2D, I3uuud_3D, I4uuuu_2D, I4uuuu_3D, I4uuuu_4D, 
         f0, sigma2w) = Table
        
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
        
        
        #Introducing bias in RSD functions
        def ATNS(mu, b1):
            return b1**3 * Af(mu, f0/b1)
        
        def DRSD(mu, b1):
            return b1**4 * Df(mu, f0/b1)
        
        def GTNS(mu, b1):
            return -((k*mu*f0)**2 *sigma2w*(b1**2 * pkl + 2*b1*f0*mu**2 * Pdt_L 
                                   + f0**2 * mu**4 * Ptt_L))
        
        
        #One-loop SPT power spectrum in redshift space
        def PloopSPTs(mu, b1, b2, bs2, b3nl):
            return (PddXloop(b1, b2, bs2, b3nl) + 2*f0*mu**2 * PdtXloop(b1, b2, bs2, b3nl)
                    + mu**4 * f0**2 * PttXloop(b1, b2, bs2, b3nl) + ATNS(mu, b1) + DRSD(mu, b1)
                    + GTNS(mu, b1))
        
        
        #Linear Kaiser power spectrum, counterterms and shot noise
        def PKaiserLs(mu, b1):
            return (b1 + mu**2 * fk)**2 * pkl
        
        def PctNLOs(mu, b1, ctilde):
            return ctilde*(mu*k*f0)**4 * sigma2w**2 * PKaiserLs(mu, b1)
        
        def Pcts(mu, alpha0, alpha2, alpha4):
            return (alpha0 + alpha2 * mu**2 + alpha4 * mu**4)*k**2 * pkl
        
        def Pshot(mu, alphashot0, alphashot2, PshotP):
            return PshotP*(alphashot0 + alphashot2 * (k*mu)**2)
        
        return (PloopSPTs(mu, b1, b2, bs2, b3nl) + Pcts(mu, alpha0, alpha2, alpha4)
                + PctNLOs(mu, b1, ctilde) + Pshot(mu, alphashot0, alphashot2, PshotP))
    
    def interp(k, x, y):
        inter = CubicSpline(x, y)
        return inter(k) 
    
    def Sigma2Total(mu, Table_NW):
        kT = Table_NW[0]; pkl_NW = Table_NW[1];
        
        kinit = 10**(-6);  kS = 0.4;                                  #integration limits
        pT = np.logspace(np.log10(kinit),np.log10(kS), num = 10**4)   #integration range
        
        PSL_NW = interp(pT, kT, pkl_NW)
        k_BAO = 1/104                                                 #BAO scale
        
        Sigma2 = 1/(6 * np.pi**2)*scipy.integrate.simps(PSL_NW*(1 - spherical_jn(0, pT/k_BAO) 
                                                + 2*spherical_jn(2, pT/k_BAO)), pT)
        
        deltaSigma2 = 1/(2 * np.pi**2)*scipy.integrate.simps(PSL_NW*spherical_jn(2, pT/k_BAO), pT)
        
        def Sigma2T(mu):
            return (1 + f0*mu**2 *(2 + f0))*Sigma2 + (f0*mu)**2 * (mu**2 - 1)* deltaSigma2
        
        return Sigma2T(mu)
    
    
    
    def PIRs(mu, Table, Table_NW):
        k = Table[0]; Fkoverf0 = Table[2]; fk = Fkoverf0*f0
        pkl = Table[1]; pkl_NW = Table_NW[1];
        
        Sigma2T = Sigma2Total(mu, Table_NW)
        
        return ((b1 + fk * mu**2)**2 * (pkl_NW + np.exp(-k**2 * Sigma2T)*(pkl-pkl_NW)*(1 + k**2 * Sigma2T) )
                + np.exp(-k**2 * Sigma2T)*PEFTs(mu, Table) 
                + (1 - np.exp(-k**2 * Sigma2T))*PEFTs(mu, Table_NW))  
    
    
    Nx = 6                                         #Points
    xGL, wGL = scipy.special.roots_legendre(Nx)    #x=cosθ and weights
    
    def ModelPkl0(Table, Table_NW):
        monop = 0;
        for ii in range(Nx):
            monop = monop + 0.5*wGL[ii]*PIRs(xGL[ii], Table, Table_NW)
        return monop
    
    def ModelPkl2(Table, Table_NW):    
        quadrup = 0;
        for ii in range(Nx):
            quadrup = quadrup + 5/2*wGL[ii]*PIRs(xGL[ii], Table, Table_NW)*eval_legendre(2, xGL[ii])
        return quadrup
    
    def ModelPkl4(Table, Table_NW):
        hexadecap = 0;
        for ii in range(Nx):
            hexadecap = hexadecap + 9/2*wGL[ii]*PIRs(xGL[ii], Table, Table_NW)*eval_legendre(4, xGL[ii])
        return hexadecap
    
    
    Pkl0 = ModelPkl0(TableOut, TableOut_NW);
    Pkl2 = ModelPkl2(TableOut, TableOut_NW);
    Pkl4 = ModelPkl4(TableOut, TableOut_NW);
    

    return (TableOut[0], Pkl0, Pkl2, Pkl4) 








def LinearRegression(inputxy): 
    
    xm = np.mean(inputxy[0])
    ym = np.mean(inputxy[1])
    Npts = len(inputxy[0])
    
    SS_xy = np.sum(inputxy[0]*inputxy[1]) - Npts*xm*ym
    SS_xx = np.sum(inputxy[0]**2) - Npts*xm**2
    m = SS_xy/SS_xx
    
    b = ym - m*xm
    return (m, b)



def Extrapolate(inputxy, outputx):
    
    m, b = LinearRegression(inputxy)
    outxy = [(outputx[ii], m*outputx[ii]+b) for ii in range(len(outputx))]
    
    return np.array(np.transpose(outxy))



def ExtrapolateHighkLogLog(inputT, kcutmax, kmax):
    
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
