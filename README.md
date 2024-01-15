![logo](logo.eps)

# FOLPSν (aka Flops)
FOLPSν is a Python code that computes the galaxy redshift space power spectrum for cosmologies containing massive neutrinos. The code combines analytical modeling and numerical methods based on the FFTLog formalism. <!-- to speed up the calculations of loop integrals. -->


[![arXiv](https://img.shields.io/badge/arXiv-2208.02791-red)](https://arxiv.org/abs/2208.02791)


## Authors: 
- [Hernán E. Noriega](mailto:henoriega@estudiantes.fisica.unam.mx)
- [Alejandro Aviles](mailto:avilescervantes@gmail.com)


*Special thanks to other people who have tested this code*: Sebastien Fromenteau, Mariana Vargas-Magaña, Arnaud de Mattia, Gerardo Morales Navarrete 




## Run

**Dependences**

The code employs the standard libraries:
- NumPy 
- SciPy

We recommend to use NumPy versions ≥ 1.20.0. For older versions, one needs to rescale by a factor 1/N the [FFT computation](https://github.com/henoriega/FOLPS-nu/blob/main/FOLPSnu.py#L626). 

To run the code, first use git clone:

```
git clone https://github.com/henoriega/FOLPS-nu.git
```

or download it from https://github.com/henoriega/FOLPS-nu.

Once everything is ready, please check the [Jupyter Notebook](https://github.com/henoriega/FOLPS-nu/blob/main/notebooks/Example.ipynb) which contains some helpful examples. 



Attribution
-----------

Please cite <https://arxiv.org/abs/2208.02791> if you find this code useful in your research. 

    @article{Noriega:2022nhf,
    author = "Noriega, Hern\'an E. and Aviles, Alejandro and Fromenteau, Sebastien and Vargas-Maga\~na, Mariana",
    title = "{Fast computation of non-linear power spectrum in cosmologies with massive neutrinos}",
    eprint = "2208.02791",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    month = "8",
    year = "2022"
    }
