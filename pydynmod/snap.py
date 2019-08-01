import torch
import math
import numpy as np
from enum import IntFlag, unique


float_dtype=torch.float32

class ParticleType(IntFlag):
    """Enum for storing particle type: Gas, Star, DarkMatter"""
    #If the values are changed then SnapShot.__organise will need updating
    DarkMatter = 1
    Gas = 2
    Star = 4
    Baryonic = 6

def ensuretensor(x,dtype=float_dtype):
    if isinstance(x,torch.Tensor):
        return x
    else:
        return torch.tensor(x,dtype=dtype)

class SnapShot:
    def __init__(self,file=None,positions=None,velocities=None,
                 masses=None,particletype=None,time=0.,omega=0.):
        super(SnapShot, self).__init__()
        if file is None and (positions is None or velocities is None or masses is None):
            raise TypeError('Need either a snapshot to load, or positions, velocities and masses.')
        if positions is not None:
            self.positions = ensuretensor(positions)
            self.velocities = ensuretensor(velocities)
            self.masses = ensuretensor(masses)
        if file is not None:
            snap = torch.tensor(np.loadtxt(file),dtype=float_dtype)
            self.positions = snap[:,0:3]
            self.velocities = snap[:,3:6]
            self.masses = snap[:,6]
            if snap.shape[1]>=8:
                particletype = snap[:,7].type(torch.uint8)
        if particletype is None:
            particletype = torch.full(self.masses.shape,ParticleType.Star,dtype=torch.uint8)
        self.__particletype = particletype
        self.time = ensuretensor(time)
        self.omega = ensuretensor(omega)
        self.n = len(self.masses)
        self.dt = None
        self.__organise()

    def to(self,device):
        """Moves the snapshot to the specified device. If already on the device returns self, otherwise returns a new
        SnapShot on the device."""
        if self.positions.device == device:
            return self
        else:
            newsnap = SnapShot(positions = self.positions.to(device),
                   velocities = self.velocities.to(device),
                   masses = self.masses.to(device),
                   particletype = self.particletype.to(device),
                   time = self.time.to(device), omega=self.omega.to(device))
            if self.dt is not None:
                newsnap = self.dt.to(device)
            return newsnap
        
    @property
    def particletype(self):
        return self.__particletype
    
    @particletype.setter
    def particletype(self,particletype):
        self.__particletype=particletype
        self.__organise()
        
    def __organise(self):
        ind = torch.argsort(self.__particletype,descending=True)
        self.positions = self.positions[ind,:]
        self.velocities = self.velocities[ind,:]
        self.masses = self.masses[ind]
        self.__particletype = self.__particletype[ind]
        counts=torch.bincount(self.__particletype,minlength=int(max(ParticleType)))
        self.starrange=(0,counts[ParticleType.Star])
        self.gasrange=(self.starrange[1],self.starrange[1]+counts[ParticleType.Gas])
        self.dmrange=(self.gasrange[1],self.gasrange[1]+counts[ParticleType.DarkMatter])

    @property
    def dm(self):
        return self[self.dmrange[0]:self.dmrange[1]]
    @property
    def gas(self):
        return self[self.gasrange[0]:self.gasrange[1]]
    @property
    def stars(self):
        return self[self.starrange[0]:self.starrange[1]]

    @property
    def x(self):
        return self.positions[:,0]
    @property
    def y(self):
        return self.positions[:,1]
    @property
    def z(self):
        return self.positions[:,2]
    @property
    def vx(self):
        return self.velocities[:,0]
    @property
    def vy(self):
        return self.velocities[:,1]
    @property
    def vz(self):
        return self.velocities[:,2]
    @property
    def r(self):
        return self.positions.norm(dim=-1)
    @property
    def vr(self):
        return torch.tensordot(self.positions,self.velocities)/self.r


    @property
    def rcyl(self):
        return (self.positions[:,0]**2 + self.positions[:,1]**2).sqrt()

    
    def __getitem__(self,i):
        """Returns a new snapshot but should avoid unnessesary copies: if index
        is a slice that doesnt need a copy we will get views. Othewise we get a
        copy."""
        newsnap = SnapShot(positions = self.positions[i,:],
                           velocities = self.velocities[i,:],
                           masses = self.masses[i],
                           particletype = self.particletype[i],
                           time = self.time, omega=self.omega)
        if self.dt is not None:
            newsnap.dt = self.dt[i]
        return newsnap
    
    def integrate(self,time,potential,mindt=1e-6,stepsperorbit=800,verbose=False):
        """"We integrate in the inertial frame, but store the particles back to the corotating frame at the """
        if time==self.time:
            return
        if self.dt is None:
            self.dt = torch.full_like(self.masses,time)
        #Given each particles dt then compute the optimal number of steps rounded up to the nearest 2**N
        steps = 2**(((time-self.time)/self.dt).abs().log2().ceil()).clamp(min=0).type(torch.int32)
        maxsteps = 2**(((time-self.time)/mindt).abs().log2().ceil()).clamp(min=0).type(torch.int32)
        steps = steps.clamp(1,maxsteps) #dont exceed mindt stepsize, and make at least 1 step
        
        thissteps = steps.min()
        while thissteps <= steps.max():
            thisdt=(time-self.time)/thissteps
            i = (thissteps==steps)
            if verbose:
                print(thissteps,maxsteps,thisdt,i.sum())
            positions = self.positions[i,:] 
            velocities = self.velocities[i,:]
            dt = self.dt[i]
            positions += velocities*thisdt*0.5
            timenow = self.time + thisdt*0.5
            for step in range(thissteps):
                accelerations = potential.get_accelerations(self.CoRotatingFrame(self.omega,timenow-self.time,positions)) #Get accelerations in from corotating frame
                self.CoRotatingFrame(-self.omega,timenow-self.time,accelerations,inplace=True) #Move accelerations to inertial frame
                velocities -= accelerations*thisdt
                tau = 2*math.pi*((positions.norm(dim=-1)+1e-3)/accelerations.norm(dim=-1)).sqrt()/stepsperorbit
                dt = torch.min(dt,tau)
                positions += velocities*thisdt
                timenow += thisdt
            positions -= velocities*thisdt*0.5
            if thissteps<maxsteps:
                steps[i.nonzero()[:,0][dt<thisdt]]*=2
                gd = i.nonzero()[:,0][dt>=thisdt]
                if len(gd)>0:
                    self.positions[gd,:] = positions[dt>=thisdt]
                    self.velocities[gd,:] = velocities[dt>=thisdt]
            else:
                self.positions[i,:] = positions
                self.velocities[i,:] = velocities

            self.dt[i] = dt
            thissteps*=2
        self.CoRotatingFrame(self.omega,time-self.time,self.positions,self.velocities,inplace=True)
            
        self.time = time
        

    @classmethod
    def CoRotatingFrame(cls,omega,time,positions,velocities=None,inplace=False):
        """Returns positions, velocities in the corotating frame.
        Uses the time to rotate positions (and optionally velocities) from the rotating to the corotating frame."""
        if inplace:
            corotpositions=positions
        else:
            corotpositions = torch.zeros_like(positions)
            corotpositions[...,2] = positions[...,2]

        phase = ensuretensor(omega*time)
        R = torch.tensor(((torch.cos(phase),-torch.sin(phase)),
                          (torch.sin(phase),torch.cos(phase))),
                           dtype=positions.dtype,
                           device=positions.device)

        corotpositions[...,0:2] = (R @ positions[...,0:2,None])[...,0]
        if velocities is None:
            return corotpositions
        else:
            if inplace:
                corotvelocities=velocities
            else:
                corotvelocities = torch.zeros_like(velocities)
                corotvelocities[...,2] = velocities[...,2]
            corotvelocities[...,0:2] = (R @ velocities[...,0:2,None])[...,0]
            return corotpositions, corotvelocities
