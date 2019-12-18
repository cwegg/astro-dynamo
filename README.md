# astro-dynamo

astro-dynamo applies machine learning codes and methods to adapt an N-body model to your needs.
An example notebook showing how to use it can be found on google colab
[here](https://colab.research.google.com/drive/1oFQdm0V3KfxtbmtoEwsG_KGeehdsjdDc).

The primary usage envisioned is adapting an N-body model to fit data. 

The method is called the made-to-measure method in the astronomy literature 
([Syer and Tremaine, 1996](https://ui.adsabs.harvard.edu/abs/1996MNRAS.282..223S)). 
[de Lorenzi et al. (2007)](https://ui.adsabs.harvard.edu/abs/2007MNRAS.376...71D)
were the first to consider applying it to fitting data. 

The method adapts the masses of the particles to optimise a loss (normally data
likihood) and so is analagous to training models in machine learning. Astro-dynamo
therefore uses the pytorch framework to do this fitting. In this way we take
advantage both of the machine learning features of pytorch 
(automatic differentiaion, dropout, learning rate scheduling etc) as well using
it as an easy way to do all our work on the GPU. The problem is naturally parallel
across all the particles this gives a big speed-up over the CPU.

