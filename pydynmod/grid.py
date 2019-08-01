import torch
from functools import reduce

class Grid:
    """Class for gridding data. Assumes grids are symmertic about zero."""
    def __init__(self,gridedges=torch.Tensor((10.,10.,10.)),n=(256,256,256),data=None):
        self.n=n
        if gridedges.dim() == 1:
            self.min=-gridedges
            self.max=gridedges
        else:
            self.min=gridedges[...,0]
            self.max=gridedges[...,1]
        self.dx=(self.max-self.min)/(self.max.new_tensor(n)-1)
        self.ndim=len(self.n)
        self.data=data

    def to(self,device):
        if self.min.device == device:
            return self
        else:
            gridedges = torch.stack((self.min,self.max),dim=1).to(device)
            return Grid(gridedges=gridedges,n=self.n,data=self.data.to(device))
    
    @property
    def x(self):
        """Returns grid points in first dimension"""
        return torch.linspace(self.min[0],self.max[0],self.n[0])
    
    @property
    def y(self):
        """Returns grid points in second dimension"""
        return torch.linspace(self.min[1],self.max[1],self.n[1])
    
    @property
    def z(self):
        """Returns grid points in third dimension"""
        return torch.linspace(self.min[2],self.max[2],self.n[2])

        
    def ingrid(self,positions,h=None):
        """Test if positions are in the grid"""
        if h is None:
            return ((positions > self.min[None,:]) & 
                    (positions < self.max[None,:])).all(dim=1)
        else:
            return ((positions > self.min[None,:]+h*self.dx[None,:]) &
                    (positions < self.max[None,:]-h*self.dx[None,:])).all(dim=1)
    
    def fidx(self,positions):
        return (positions-self.min[None,:])/self.dx
    
    def __nd_to_1d(self,i):
        i1d=i[:,0]
        if self.ndim >= 2:
            i1d=i1d*self.n[1] + i[:,1]
        if self.ndim >=3:
            i1d=i1d*self.n[2] + i[:,2]
        return i1d
    
    def griddata(self,positions,weights=None,method='nearest'):
        """Places data from positions onto grid using method='nearest'|'cic'
        where cic=cloud in cell. Returns gridded data and stores it as class attribute
        data"""
        del self.data
        fi=self.fidx(positions)
        nelements=reduce(lambda x,y:x*y,self.n)
        if method=='nearest':
            gd=self.ingrid(positions) 
        else:
            gd=self.ingrid(positions,h=0.5)
        if gd.sum()==0:
            self.data = positions.new_zeros(nelements)
            return self.data

        if method=='nearest':
            i=(fi+0.5).type(torch.int64)
            if weights is not None:
                weights=weights[gd]
            self.data = torch.bincount(self.__nd_to_1d(i[gd]),minlength=nelements,weights=weights).type(dtype=positions.dtype)
        
        if method=='cic':
            i=fi[gd,...].floor()
            offset=fi[gd,...]-i
            i=i.type(torch.int64)
            if weights is None:
                weights=positions.new_ones(gd.sum())
            else:
                weights=weights[gd]
            self.data = weights.new_zeros(nelements)
            if len(self.n)==1:
                self.data += torch.bincount(self.__nd_to_1d(i), minlength=nelements, 
                                  weights=weights*(1-offset))
                self.data += torch.bincount(self.__nd_to_1d(i+1), minlength=nelements, 
                                      weights=weights*offset)
            else:
                twidle=torch.tensor([0, 1])
                indexes = torch.cartesian_prod(*torch.split(twidle.repeat(self.ndim),2))
                for offsetdims in indexes:
                    thisweights=torch.ones_like(weights)
                    for dimi,offsetdim in enumerate(offsetdims):
                        if offsetdim==0:
                            thisweights*=(1-offset[...,dimi])
                        if offsetdim==1:
                            thisweights*=offset[...,dimi]
                    self.data += torch.bincount(self.__nd_to_1d(i+offsetdims), minlength=nelements, 
                                          weights=thisweights*weights)
        self.data = self.data.reshape(self.n)                
        return self.data


class ForceGrid(Grid):
    def __init__(self,gridedges=torch.Tensor((10.,10.,10.)),n=(256,256,256),
                 data=None,smoothing=1.0):
        self.greenfft=None
        self.pot = None
        self.acc = None
        self.epsilon=smoothing
        super().__init__(gridedges,n,data)

    def to(self,device):
        if self.min.device == device:
            return self
        else:
            gridedges = torch.stack((self.min,self.max),dim=1).to(device)
            grid = ForceGrid(gridedges=gridedges,n=self.n,data=self.data.to(device))
            if self.acc is not None:
                grid.acc = self.acc.to(device)
            return grid


    @staticmethod
    def smoothing_pot(r,epsilon=1.):
        """Taken from the manual of the Galaxy code by Jerry Sellwood."""
        x=r/epsilon
        pot=-1/r
        i=(x<1)
        pot[i] = (-1.4 + x[i]**2*(2./3 - 0.3*x[i]**2 + 0.1*x[i]**3))/epsilon
        i=(x>=1) & (x<2)
        pot[i] = (-1.6 + 1/(15*x[i]) + x[i]**2*(4./3 - x[i] + 0.3*x[i]**2 - x[i]**3/30))/epsilon
        return pot
    
    @staticmethod 
    def complex_mul(a,b):
        """Function to do complex multiplicatinos: Pytorch has no complex dtype but 
        stores complex numbers as an additional two long dimension"""
        return torch.stack([a[...,0]*b[...,0] - a[...,1]*b[...,1],
                            a[...,0]*b[...,1] + a[...,1]*b[...,0]], dim = -1)
    
    def get_accelerations(self,positions):
        """Linear intepolate the gridded forces to the specified positions. This should be preceeded
        by a cool to grid_acc to (re)compute the accelerations on the grid."""
        fi=self.fidx(positions)
        gd=self.ingrid(positions) 
        i=fi[gd,...].floor()
        offset=fi[gd,...]-i
        i=i.type(torch.int64)
        ret = positions.new_zeros(positions.shape[0:-1]+(self.ndim,))
        if self.ndim == 1:
            ret[gd] = self.acc[i,:]*(1-offset[...,0]) + self.acc[i+1,:]*offset[...,0]
        if self.ndim == 2:
            ret[gd] = (self.acc[i[...,0],i[...,1],:]*(1-offset[...,0,None])*(1-offset[...,1,None]) +
                       self.acc[i[...,0],i[...,1]+1,:]*(1-offset[...,0,None])*offset[...,1,None] +
                       self.acc[i[...,0]+1,i[...,1],:]*offset[...,0,None]*(1-offset[...,1,None]) +
                       self.acc[i[...,0]+1,i[...,1]+1,:]*offset[...,0,None]*offset[...,1,None] )
        if self.ndim == 3:
            samples = (positions/self.max).unsqueeze(0).unsqueeze(2).unsqueeze(3)
            image = self.acc.permute(3,2,1,0)             # change to:      C x W x H x D
            image = image.unsqueeze(0)                  # change to:  1 x C x W x H x D
            return torch.nn.functional.grid_sample(image,samples,mode='bilinear').squeeze().t()
        return ret 
      
    def grid_accelerations(self,positions=None,weights=None,method='nearest'):
        """Takes the brute force approach of removing periodic images by grid doubling in every dimension
        returns (pot, acc, greenfft) greenfft can be repassed and reused on the next call"""
        if positions is not None:
            self.griddata(positions,weights,method)
        rho=self.data
        padrho=rho.new_zeros([n*2 for n in self.n])
        
        #Treat each case of different dimensions seperately since theres at most 3 in the real world
        if self.ndim == 1:
            nx=rho.shape[0]
            padrho[0:nx]=rho
            rhofft = torch.rfft(padrho,1,onesided=True)
            if self.greenfft is None or rhofft.shape != self.greenfft.shape:
                x=torch.arange(-nx,nx,dtype=rhofft.dtype,device=rhofft.device)*self.dx[0]
                padgreen = ForceGrid.smoothing_pot(x.abs(),epsilon=self.epsilon)
                self.greenfft=torch.rfft(padgreen,1,onesided=True)
            self.pot=torch.irfft(ForceGrid.complex_mul(rhofft,self.greenfft),1,onesided=True,signal_sizes=padrho.shape)
            self.pot=self.pot[nx:]

        if self.ndim == 2:
            nx,ny = rho.shape[0],rho.shape[1]
            padrho[0:nx,0:ny] = rho
            rhofft = torch.rfft(padrho,2,onesided=False)
            if self.greenfft is None or rhofft.shape != self.greenfft.shape:
                x=torch.arange(-nx,nx,dtype=rhofft.dtype,device=rhofft.device)*self.dx[0]
                y=torch.arange(-ny,ny,dtype=rhofft.dtype,device=rhofft.device)*self.dx[1]
                r=torch.sqrt(x[:,None]**2 + y[None,:]**2)
                padgreen = ForceGrid.smoothing_pot(r,epsilon=self.epsilon)
                self.greenfft=torch.rfft(padgreen,2,onesided=False)
            self.pot=torch.irfft(ForceGrid.complex_mul(rhofft,self.greenfft),2,onesided=False)
            self.pot=self.pot[nx:,ny:]
       
        if self.ndim == 3:
            nx,ny,nz=rho.shape[0],rho.shape[1],rho.shape[2]
            padrho[0:nx,0:ny,0:nz] = rho
            if self.greenfft is None or padrho.shape != self.greenfft.shape[:-1]:
                x=torch.arange(-nx,nx,dtype=rho.dtype,device=rho.device)*self.dx[0]
                y=torch.arange(-ny,ny,dtype=rho.dtype,device=rho.device)*self.dx[1]
                z=torch.arange(-nz,nz,dtype=rho.dtype,device=rho.device)*self.dx[2]
                self.greenfft=torch.rfft(ForceGrid.smoothing_pot(
                    x[:,None,None]**2 + y[None,:,None]**2 + z[None,None,:]**2,epsilon=self.epsilon),
                                         3,onesided=False)
            rhofft=torch.rfft(padrho,3,onesided=False)
            del padrho
            rhofft = ForceGrid.complex_mul(rhofft,self.greenfft)
            self.pot=torch.irfft(rhofft,3,onesided=False)
            del rhofft
            self.pot=self.pot[nx:,ny:,nz:]
            
        self.acc=self.pot.new_zeros(self.pot.shape+(self.ndim,))
        for dim in range(self.ndim):
            self.acc[...,dim] = self.__diff(self.pot,dim,d=self.dx[dim])
            
    @staticmethod
    def __diff(vector,dim,d=1.0):
        """Helper function for differentiaitnf the potential to get the accelerations"""
        ret = (vector.roll(-1,dim) - vector.roll(1,dim))/(2*d)
        if dim==0:
            ret[0,...] = (vector[1,...] - vector[0,...])/d
            ret[-1,...] = (vector[-1,...] - vector[-2,...])/d
        if dim==1:
            ret[:,0,...] = (vector[:,1,...] - vector[:,0,...])/d
            ret[:,-1,...] = (vector[:,-1,...] - vector[:,-2,...])/d
        if dim==2:
            ret[:,:,0,...] = (vector[:,:,1,...] - vector[:,:,0,...])/d
            ret[:,:,-1,...] = (vector[:,:,-1,...] - vector[:,:,-2,...])/d
        return ret
