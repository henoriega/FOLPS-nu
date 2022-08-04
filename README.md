# FOLPSν
FOLPSν computes the redshift space power spectrum multipoles in a fraction of second. The code combine analytical modeling and numerical methods based on the FFTLog formalism to speed up the loop calculations.


[![arXiv](https://img.shields.io/badge/arXiv-PONER_NUMERO-red)](https://ARXIV_PONER_LINK)


## Authors: 
- [Hernán E. Noriega](mailto:henoriega@estudiantes.fisica.unam.mx)

**Other people who contributed to this code**:
- Alejandro Aviles
- Sebastien Fromenteau
- Mariana Vargas-Magaña


## Run

**Dependences**

The code employs the standard libraries:
- NumPy 
- SciPy

We recommend to use NumPy versions ≥ 1.20.0. For older versions, one needs to rescale by a factor 1/N the [FFT computation](https://github.com/henoriega/FOLPS-nu/blob/main/Mmodules.py#L412). 

To run the code, first use git clone:

```
git clone https://github.com/henoriega/FOLPS-nu.git
```

or download it from https://github.com/henoriega/FOLPS-nu.

Once everything is ready, please check the [Jupyter Notebook](https://github.com/henoriega/FOLPS-nu/blob/main/notebooks/Example.ipynb) with some useful examples. 



Attribution
-----------

Please cite <https://ARXIV_PONER> if you find this code useful in your research. 

    @article{LLENAR
    }
