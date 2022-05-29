# Fast One Loop Power Spectrum in the presence of massive neutrinos (FOLPSν)


**Emails**:henoriega@estudiantes.fisica.unam.mx, avilescervantes@gmail.com 


**Requirements:** 

- numpy (update your numpy to versions ≥ 1.20.0)
- scipy


FOLPSν (still in development) is a code for efficiently evaluating the redshift-space power spectrum in the presence of [massive neutrinos](https://arxiv.org/pdf/2106.13771.pdf). 
The code is based on the FFTLog formalism (hipervinculo FFTlog) and computes the one-loop power spectrum from Eulerian Perturbation Theory (hipervinculo Bernardeu) , incorporating into the model some standard ingredients such as non-linear bias, Infrared resummations, and Effective Field Theory counterterms.

FFTLog matrices and vectors do not depend on the cosmological parameters, so they only need to be computed once!
