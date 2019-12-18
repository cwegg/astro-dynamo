"""
Routines for calling nemo from python.

To use you need to have installed nemo from https://teuben.github.io/nemo/

Calling Nemo can be a bit tempramental - the executables are dyanmically linked, and the libraries
and executables are all in their own envirnoment. We try to handle this, but if it fails then first
add verbose=True to the failing command, and see if there's a useful message being printed.
"""
import errno
import os
import subprocess
import tempfile

import numpy as np
from dotenv import load_dotenv, find_dotenv

_dotenv_path = find_dotenv(usecwd=True)
load_dotenv(_dotenv_path, override=True)

_NEMO_LOCATION = os.environ.get("NEMO_LOCATION")
_NEMO_BIN_LOCATION = os.path.join(_NEMO_LOCATION, 'bin')


def _make_nemo_env():
    """
    Nemo doesnt make static executables for GyrFalcon/getgravity. normally nemo_start.csh would add the paths to the
    libraries. here we call nemo_start.csh and use the environment variables it sets to create our environment
    """
    command = ['csh', '-c', 'source ' + os.path.join(_NEMO_LOCATION, 'nemo_start.csh') + '  && env']
    proc = subprocess.Popen(command, stdout=subprocess.PIPE)

    for line in proc.stdout:
        (key, equals, value) = line.decode('utf-8').rstrip().partition('=')
        if equals == '=':
            os.environ[key] = value

    proc.communicate()

    env = dict(os.environ)
    return env


_NEMO_ENV = _make_nemo_env()


def call_nemo_executable(executable, arguments):
    """Call nemo executable (should be a string) with arguments (should be a list).
    Returns Popen object with .stdin, .stdout and .stderr attributes."""
    p = subprocess.Popen([os.path.join(_NEMO_BIN_LOCATION, executable)] + arguments,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=_NEMO_ENV)
    return p


def writesnap(particles, filename, time=0.0, verbose=False, positions_only=False, filter_mass=False):
    """
    Write a nemo snapshot of particles to filename, uses snapshot time=0 by deafult
    This writes a snapshot with postitions, velocities and masses. Use positions_only to just write only postitions.
    filter_mass=True removes zero (or negative) mass particles that can cause problems for various nemo routines
    """

    x = particles[:, 0:3]
    npart = len(particles[:, 0])
    if not positions_only:
        m = particles[:, 6]
        v = particles[:, 3:6]

    if positions_only:
        p = call_nemo_executable('atos', ["in=-", "out=" + filename, "options=pos"])
        p.stdin.write('{} 3 {} 0'.format(npart, time).encode('ascii'))
        np.savetxt(p.stdin, x, fmt='%e', delimiter=' ')
    else:
        if filter_mass:
            good = (m > 0)
            npart = np.sum(good)
            x, m, v = x[good], m[good], v[good]

        p = call_nemo_executable('atos', ["in=-", "out=" + filename, "options=mass,pos,vel"])
        p.stdin.write('{} 3 {} 0'.format(npart, time).encode('ascii'))
        np.savetxt(p.stdin, m, fmt='%e', delimiter=' ')
        np.savetxt(p.stdin, x, fmt='%e', delimiter=' ')
        np.savetxt(p.stdin, v, fmt='%e', delimiter=' ')
    p.stdin.flush()
    (stdoutdata, stderrdata) = p.communicate()
    if verbose:
        print(stdoutdata.decode('utf-8'), stderrdata.decode('utf-8'))


def readsnap(filename, times='all', verbose=False, give='t,x,y,z,vx,vy,vz,m'):
    """
    Read a nemo snapshot of particles
    This reads a snapshot with postitions, velocities and masses.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

    p = call_nemo_executable("snapprint", ["in=" + filename, "options="+give, f"times={times}"])

    snaps = np.genfromtxt(p.stdout)
    if verbose:
        print(p.stderr.read().decode('utf-8'))

    columns = give.count(',')+1
    tlist, ti = np.unique(snaps[:, 0], return_inverse=True)
    snaps = (np.reshape(snaps, (len(tlist), np.size(snaps) // (columns * len(tlist)), columns)))[:, :, 1:]
    return tlist, snaps



def rotationcurve(particles, nr=100, ntheta=100, rrange=(0, 10), verbose=False):
    """
    Computes the rotation curve of particles as a function of r in the X,Y plane using gyrfalcON

    Inputs:
        particles : n-body model of which to compute rotation curve
        nr : number of radial locations to compute curve (default:100)
        ntheta : number of azimuthal locations to average over in computation (default:100)
        rrange : range of radii to compute rotation curve, nr points are spaced linearly over this range
        (default:[0,10])
    Outputs:
        (r,vcirc2) : Where vcirc2 is the square of circular velocity and is computed at each point in r
                    return in internal units i.e. assuming G=1.
    """
    theta = np.linspace(0, 2 * np.pi, num=ntheta, endpoint=False)
    r = np.linspace(rrange[1], rrange[0], num=nr, endpoint=False)[::-1]
    thetamat, rmat = np.meshgrid(theta, r)
    x = rmat * np.sin(thetamat)
    y = rmat * np.cos(thetamat)
    positions = np.zeros((ntheta * nr, 3))
    positions[:, 0] = x.flatten()
    positions[:, 1] = y.flatten()
    ax, ay, _, _ = getgravity(particles, positions, verbose=verbose).T
    ax = ax.reshape(x.shape)
    ay = ay.reshape(y.shape)
    ar = - (x * ax + y * ay) / np.sqrt(x * x + y * y)
    vcirc2 = np.mean(ar, 1) * r
    return r, vcirc2


def getpartpot(particles, verbose=False):
    """
    Computes the potential at each particle using gyrfalcON.
    Inputs:
        particles : n-body model of which to compute rotation curve. Having zero or negative mass particles
                    would be a problem.
    Outputs:
        potential at each particle
        return in internal units i.e. assuming G=1.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        filein = tmpdirname + 'partpot.in'
        writesnap(particles, filename=filein, time=0.0)
        fileout = tmpdirname + 'partpot.falc'
        p = call_nemo_executable("gyrfalcON", ["in=" + filein, "out=" + fileout, "tstop=0", "theta=0.5",
                                               "eps=0.05", "kmax=6", "give=phi", "Ncrit=20"])
        (stdout, stderr) = p.communicate()
        if verbose:
            print(stdout.decode('utf-8'), stderr.decode('utf-8'))
        p = call_nemo_executable("snapprint", ["in=" + fileout, "options=phi"])
        fromfalc = np.genfromtxt(p.stdout)
        if verbose:
            print(p.stderr.read().decode('utf-8'))

    return fromfalc


def getgravity(particles, positions, verbose=False):
    """
    Computes the forces and potential at each position due to paticles using gyrfalcON
    Inputs:
        particles : n-body model of which to compute rotation curve
        postions : positions in z,y,z to compute gravity. Format (N,3) array
    Outputs:
        (Fx,Fy,Fz,potential) at each postion. return in internal units i.e. assuming G=1.
    """

    with tempfile.TemporaryDirectory() as tmpdirname:

        filein = os.path.join(tmpdirname, 'part.in')
        writesnap(particles, filename=filein, time=0.0, filter_mass=True, verbose=verbose)

        posfilein = os.path.join(tmpdirname, 'pos.in')
        writesnap(positions, filename=posfilein, time=0.0, positions_only=True, verbose=verbose)

        fileout = os.path.join(tmpdirname, 'gravity.out')
        p = call_nemo_executable("getgravity", ["srce=" + filein, "sink=" + posfilein, "out=" + fileout, "Ncrit=20"])
        stdout, stderr = p.communicate()
        if verbose:
            print(stdout.decode('utf-8'), stderr.decode('utf-8'))
        p = call_nemo_executable("snapprint", ["in=" + fileout, "options=ax,ay,az,phi"])
        fromfalc = np.genfromtxt(p.stdout)

        if verbose:
            print(p.stderr.read().decode('utf-8'))

        return fromfalc


def integrate(particles, t=1., step=None, verbose=False):
    """
    Integartes paticles using gyrfalcON
    Inputs:
        particles : n-body model of which to compute rotation curve
        t : time to integrate for
        step : return a snapshot each step units of time
    Outputs:
        a tuple (t,snaps) with snaps containing a snapshot at each t
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        filein = os.path.join(tmpdirname, 'integrate.in')
        writesnap(particles, filename=filein, time=0.0)
        fileout = os.path.join(tmpdirname, 'integrate.falc')
        p = call_nemo_executable("gyrfalcON", ["in=" + filein, "out=" + fileout,
                                               "tstop=" + str(t), "step=" + str(step), "theta=0.5", "eps=0.05",
                                               "kmax=6", "give=mxvp"])
        (stdout, stderr) = p.communicate()
        if verbose:
            print(stdout.decode('utf-8'), stderr.decode('utf-8'))
        p = call_nemo_executable("snapprint", ["in=" + fileout, "options=t,x,y,z,vx,vy,vz,phi"])

        snaps = np.genfromtxt(p.stdout)
        if verbose:
            print(p.stderr.read().decode('utf-8'))

        tlist, ti = np.unique(snaps[:, 0], return_inverse=True)
        snaps = (np.reshape(snaps, (len(tlist), np.size(snaps) // (9 * len(tlist)), 9)))[:, :, 1:]
    return tlist, snaps


def gravity_cartesian_grid(particles, x, y, z, polar_forces=False, verbose=False):
    """
    Takes vectors of r theta and phi and then on the 3d sphical polar grid and
    computes the gravity using gyrfalcON.

    Returns a dict with entries for (r,theta,phi,F,pot).
    F is [Fx,Fy,Fz] in cartesian, [Fr,Ftheta,Fphi] in polar.
    return in internal units i.e. assuming G=1.
    """

    xmat, ymat, zmat = np.meshgrid(x, y, z, indexing='ij')
    pos = np.zeros((len(xmat.flatten()), 3))
    pos[:, 0] = xmat.flatten()
    pos[:, 1] = ymat.flatten()
    pos[:, 2] = zmat.flatten()
    # gyrfalcon crashes if there are 2 particles in the same location, remove duplicates
    # (eg along z axis), but keep track of indexes to add them back at the end
    unique_pos, inds = np.unique(pos, axis=0, return_inverse=True)
    grav = getgravity(particles, unique_pos, verbose)
    ret = {}
    ret['x'] = x
    ret['y'] = y
    ret['z'] = z
    ret['pot'] = np.reshape(grav[inds, 3], xmat.shape)
    ret['F'] = np.reshape(grav[inds, 0:3], xmat.shape + (3,))
    if polar_forces:
        Fx = ret['F'][..., 0]
        Fy = ret['F'][..., 1]
        Fz = ret['F'][..., 2]
        rmat = np.sqrt(xmat ** 2 + ymat ** 2 + zmat ** 2)
        with np.errstate(divide='ignore', invalid='ignore'):
            Fr = (Fx * xmat + Fy * ymat + Fz * zmat) / rmat
            Rcyl = np.sqrt(xmat ** 2 + ymat ** 2)
            Ftheta = ((xmat * Fx + ymat * Fy) * zmat - Rcyl ** 2 * Fz) / rmat / Rcyl
            Fphi = (-ymat * Fx + xmat * Fy) / Rcyl
        Fr[rmat == 0] = 0
        Ftheta[Rcyl == 0] = 0
        Fphi[Rcyl == 0] = 0
        ret['F'] = np.stack((Fr, Ftheta, Fphi), axis=-1)
    ret['pot'] = ret['pot'].squeeze()
    ret['F'] = ret['F'].squeeze()
    return ret


def gravity_spherical_grid(particles, r, theta, phi, polar_forces=False, verbose=False):
    """
    Takes 1d vectors of x, y and z and then on the resultant 3d cartesian grid
    computes the gravity using gyrfalcON.

    Returns a dict with entries for (x,y,z,F,pot).
    F is [Fx,Fy,Fz] if polar_forces=False (default) and [Fr,Ftheta,Fphi] if polar_forces=True.
    return in internal units i.e. assuming G=1.
    """
    phi_v, theta_v, r_v  = np.meshgrid(phi, theta, r, indexing='ij')
    x = r_v * np.sin(theta_v) * np.cos(phi_v)
    y = r_v * np.sin(theta_v) * np.sin(phi_v)
    z = r_v * np.cos(theta_v)
    pos = np.zeros((len(x.flatten()), 3))
    pos[:, 0] = x.flatten()
    pos[:, 1] = y.flatten()
    pos[:, 2] = z.flatten()
    # gyrfalcon crashes if there are 2 particles in the same location, remove duplicates
    # (eg along z axis), but keep track of indexes to add them back at the end
    unique_pos, inds = np.unique(pos, axis=0, return_inverse=True)
    grav = getgravity(particles, unique_pos, verbose)
    ret = {}
    ret['r'] = r
    ret['theta'] = theta
    ret['phi'] = phi
    ret['pot'] = np.reshape(grav[inds, 3], x.shape)
    ret['F'] = np.reshape(grav[inds, 0:3], x.shape + (3,))
    if polar_forces:
        Fx = ret['F'][..., 0]
        Fy = ret['F'][..., 1]
        Fz = ret['F'][..., 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            Fr = (Fx * x + Fy * y + Fz * z) / r_v
            Rcyl = np.sqrt(x ** 2 + y ** 2)
            Ftheta = ((x * Fx + y * Fy) * z - Rcyl ** 2 * Fz) / r_v / Rcyl
            Fphi = (-y * Fx + x * Fy) / Rcyl
        Fr[r_v == 0] = 0
        Ftheta[Rcyl == 0] = 0
        Fphi[Rcyl == 0] = 0
        ret['F'] = np.stack((Fr, Ftheta, Fphi), axis=-1)
    ret['pot'] = ret['pot'].squeeze()
    ret['F'] = ret['F'].squeeze()
    return ret
