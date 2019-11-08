import torch
import math
import numpy as np
from abc import ABC, abstractmethod

class Target(ABC):
    @abstractmethod
    def observe(self,snap):
        pass

    @property
    @abstractmethod
    def target(self):
        pass

    @property
    def error(self):
        return self.target

    def loss(self,snap):
        return (((self.observe(snap) - self.target)/self.error)**2).sum()/len(self.target   )

class RadialProfile(Target):
    def __init__(self,surface_density=None,rrange=(0,10),rbins=50, device=None):
        self.device = torch.zeros((1,),device=device).device
        self.dr = (rrange[1]-rrange[0])/rbins
        self.rrange = rrange
        self.rbins = rbins
        self.area = math.pi*(self.redge[1:]**2 - self.redge[:-1]**2)

        if surface_density is not None:
            #We take in the surface_denisty function and sample it 10 times in each bin
            oversample = 100
            nrsamples = rbins * oversample
            dr = (rrange[1] - rrange[0]) / nrsamples
            rsamples = torch.arange(rrange[0] + dr / 2, rrange[1], dr, device=device)
            fvalues = surface_density(rsamples)
            values = 2*math.pi*rsamples*fvalues*dr
            self._target = values.view(rbins,oversample).sum(dim=-1)/self.area
        else:
            self._target=None

    @property
    def target(self):
        return self._target

    @property
    def redge(self):
        return torch.linspace(self.rrange[0],self.rrange[1],self.rbins+1,device=self.device)

    @property
    def rmid(self):
        return torch.arange(self.rrange[0]+self.dr/2,self.rrange[1],self.dr,device=self.device)

    def observe(self,snap):
        assert snap.positions.device == self.device
        i = ((snap.rcyl - self.rrange[0])/self.dr).floor().type(torch.long)
        gd=(i>=0) & (i<self.rbins)
        mass_in_bin = torch.sparse.FloatTensor(i[gd].unsqueeze(0), snap.masses[gd], size=(self.rbins,)).to_dense()
        surface_density = mass_in_bin/self.area
        return surface_density

    def interpolate_surface_density(self,snap,r):
        my_r = self.rmid.cpu().numpy()
        my_surface_density = self.observe(snap).cpu().numpy()
        interpolated_surface_density = torch.as_tensor(np.interp(r, my_r, my_surface_density),device=self.device)
        return interpolated_surface_density
