# astro-dynamo

astro-dynamo applies machine learning methods to adapt an N-body model to your needs.
An example notebook showing how to use it can be found on google colab
[here](https://github.com/cwegg/astro-dynamo/blob/master/astro_dynamo_example.ipynb).

The primary usage envisioned is adapting an N-body model to fit data, or create
tailored initial conditions. 

The method is called the made-to-measure method in the astronomy literature 
([Syer and Tremaine, 1996](https://ui.adsabs.harvard.edu/abs/1996MNRAS.282..223S)). 
[de Lorenzi et al. (2007)](https://ui.adsabs.harvard.edu/abs/2007MNRAS.376...71D)
were the first to consider applying it to fitting data. 

The method adapts the masses of the particles to optimise a loss (normally data
likelihood) and so is analogous to training models in machine learning. Taking
advantage of this astro-dynamo uses the pytorch framework to do this fitting.
This gives access both to the machine learning features of pytorch 
(automatic differentiation, dropout, stochastic gradient descent, learning rate
scheduling etc) as well using it as an easy way to do all our work on the
GPU. The problem is naturally parallel across all the particles and so
using the GPU gives a big speed-up over the CPU.

### Installation

Installation is just the usual `python setup.py install` or `pip install
 .` from the repository top level.

Although not required, you may like to also install 
[nemo](https://github.com/teuben/nemo). This is needed
 for reading the input N-body files in the examples. Installation takes ~5
  minutes on a modern computer. To install to `[your_location]` run
 ```bash
wget -q https://teuben.github.io/nemo/install_nemo
chmod +x install_nemo
./install_nemo nemo=[your_location] 
```
and then before starting Python set the environment variable `NEMO_LOCATION
` to `[your_location]`.

