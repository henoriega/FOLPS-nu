# Fast One Loop Power Spectrum in the presence of massive neutrinos (FOLPSν)
This code ......

#
**Author**: [Hernán E. Noriega](mailto:henoriega@estudiantes.fisica.unam.mx)

**Other people who contributed to this code**:
- Alejandro Aviles
- Sebastien Fromenteau
- Mariana Vargas Magaña


### Requirements:
- numpy (update your numpy to versions ≥ 1.20.0)
- scipy


FOLPSν (still in development) is a code for efficiently evaluating the redshift-space power spectrum in the presence of massive neutrinos.
The code is based on the FFTLog formalism (https://arxiv.org/abs/1603.04826, https://arxiv.org/abs/1603.04405) and computes the one-loop power spectrum from [Eulerian Perturbation Theory](https://arxiv.org/abs/astro-ph/0112551), incorporating into the model some standard ingredients such as non-linear bias, Infrared resummations, and Effective Field Theory counterterms.

**Remark:** FFTLog matrices and vectors do not depend on the cosmological parameters, so they only need to be computed once!



Attribution
-----------

Please cite <https://ARXIV_PONER> if you find this code useful in your research. 

    @article{LLENAR
    }
