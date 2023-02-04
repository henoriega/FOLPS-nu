from setuptools import setup


package_basename = 'FOLPSnu'


setup(name=package_basename,
      version='1.0.0',
      author='Hernán E. Noriega & Alejandro Aviles',
      author_email='',
      description='Computation of the galaxy redshift space power spectrum for cosmologies containing massive neutrinos',
      license='',
      url='https://github.com/henoriega/FOLPS-nu',
      install_requires=['numpy', 'scipy'],
      py_modules=[package_basename])