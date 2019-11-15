import torch
import numpy as np
import math


def patternspeed(snap, rrange=(1, 4), n=range(2, 12, 2), combine=True, plot=None):
    """Compute the pattern speed of a snapshot. Uses continuity equation in cylindrical coordinates. Integrating over
    all z and taking fourier transform in angular direction gives i m R F(Sigma) = d/dR{F(Sigma vR)} + i m F(sigma vT)
    where F(y) is the angular fourier transform of y. This provides an equation at each radius and m for Sigma. By
    default combine m=2,4,6,8,10 over the radial range 1->4 into one measurement."""
    rcyl = torch.norm(snap.positions[:, 0:2], dim=-1)
    vr = torch.einsum('...i,...i->...', snap.positions[:, 0:2], snap.velocities[:, 0:2]) / rcyl
    vt = (snap.positions[:, 1] * snap.velocities[:, 0] -
          snap.positions[:, 0] * snap.velocities[:, 1]) / rcyl

    rbins = np.linspace(0., 10, 50)
    rmid, (surfdensft, surfdensrvrft, surfdensvtft) = bar_cyl_fft(snap, rbins=rbins, weights=(None, vr * rcyl, vt))
    dr = rbins[1] - rbins[0]
    dsurfdensrvr_drft = np.zeros_like(surfdensrvrft)
    dsurfdensrvr_drft[0, :] = (surfdensrvrft[1, :] - surfdensrvrft[0, :]) / dr
    dsurfdensrvr_drft[-1, :] = (surfdensrvrft[-1, :] - surfdensrvrft[-2, :]) / dr
    dsurfdensrvr_drft[1:-1, :] = (surfdensrvrft[2:, :] - surfdensrvrft[:-2]) / (2 * dr)

    omegas = []
    omegaerrs = []
    idx_good_r = (rmid > rrange[0]) & (rmid < rrange[1])
    for i in n:
        omega = surfdensvtft[:, i] / surfdensft[:, i] / rmid + \
                1j * dsurfdensrvr_drft[:, i] / surfdensft[:, i] / i / rmid
        omegas.append(np.abs(np.mean(omega[idx_good_r])))
        omegaerrs.append(np.std(np.abs(omega[idx_good_r])) / np.sqrt(np.sum(idx_good_r)))
    omegas, omegaerrs = np.array(omegas), np.array(omegaerrs)

    # combine by weighted mean
    omega = np.sum(omegas / omegaerrs ** 2) / np.sum(1 / omegaerrs ** 2)
    omegaerr = 1 / np.sqrt(np.sum(1 / omegaerrs ** 2))

    if plot is not None:
        omega, omegaerr = np.sum(omegas / omegaerrs ** 2) / np.sum(1 / omegaerrs ** 2), 1 / np.sqrt(
            np.sum(1 / omegaerrs ** 2))
        plot.errorbar(np.arange(2, 12, 2), y=omegas, yerr=omegaerrs, fmt='ko', markersize=3)
        plot.axhline(y=omega, color='r')
        plot.axhspan(ymin=omega - omegaerr, ymax=omega + omegaerr, color='r', alpha=0.2)
        plot.set_ylim([omega - 10 * omegaerr, omega + 10 * omegaerr])
        plot.set_ylabel('$\Omega$')
        plot.set_xlabel('Mode $m$')
    if not combine:
        return omegas, omegaerrs
    else:
        return omega, omegaerr


def bar_cyl_fft(snap, rbins=None, phibins=None, weights=(None,)):
    """Takes fft in the phi direction in each radial bin. Bins in the fi direction using phibins. Returns list of ffts
    each weighted by respective weights (None corresponds to by mass.)"""
    if rbins is None:
        rbins = np.linspace(0., 10, 50)
    if phibins is None:
        phibins = np.linspace(-math.pi, math.pi, 361)

    rcyl = torch.norm(snap.positions[:, 0:2], dim=-1)
    phi = torch.atan2(snap.positions[:, 1], snap.positions[:, 0])

    ft_out = []
    for weight in weights:
        if weight is None:
            totalweight = snap.masses
        else:
            totalweight = snap.masses * weight
        h, redges, phiedges = np.histogram2d(rcyl.cpu().numpy(), phi.cpu().numpy(), (rbins, phibins),
                                             weights=totalweight.cpu())
        area = 0.5 * (redges[1:, np.newaxis] ** 2 - redges[:-1, np.newaxis] ** 2) * (
                phiedges[np.newaxis, 1:] - phiedges[np.newaxis, :-1])
        surfdens = h / area
        ft_out += [np.fft.fft(surfdens, axis=1)]
    if len(ft_out) == 1:
        ft_out = ft_out[0]
    rmid = 0.5 * (redges[:-1] + redges[1:])

    return rmid, ft_out


def compute_bar_angle(snap, max_r=5, deg=True):
    """Computes the bar angle of a snapshot from angle of the m=2 mode at its maximum"""
    r, surfdensft = bar_cyl_fft(snap)
    gd_i = (r < max_r)
    m2 = np.abs(surfdensft[gd_i, 2]) / np.abs(surfdensft[gd_i, 0])
    ifid = np.argmax(m2)
    bar_angle = -0.5 * np.angle(surfdensft[gd_i, 2][ifid], deg=deg)
    return bar_angle


def align_bar(snap, max_r=5):
    """Rotates the bar so that it is aligned to the x-axis. Specifically the m=2 mode is rotated to lie along the x-axis
    at its maximum"""
    bar_angle = compute_bar_angle(snap, max_r=5, deg=False)
    _ = snap.rotate_snap([-bar_angle], snap.positions, snap.velocities, deg=False,
                         inplace=True)


def barlen(snap, phaselim=None, fractional_m2=None):
    """Computes the bar length of a snapshot using either the point where the m=2 mode twists by phaselim,
    or the power in m=2/m=0 drops by fractional_m2 of its maximum."""
    barlens = ()
    rmid, surfdensft = bar_cyl_fft(snap)
    m2 = np.abs(surfdensft[:, 2]) / np.abs(surfdensft[:, 0])
    ifid = np.argmax(m2)
    if phaselim is not None:
        # ang = np.cumsum(-0.5*np.angle(surfdensft[:, 2], deg=True)) / np.arange(1, len(surfdensft[:, 2]) + 1)
        ang = -0.5 * np.angle(surfdensft[:, 2], deg=True)
        ang -= ang[ifid]
        barlens += (interplen(rmid, np.abs(ang), phaselim, ifid, 'lt'),)
    if fractional_m2 is not None:
        barlens += (interplen(rmid, m2 / m2[ifid], fractional_m2, ifid, 'gt'),)
    return barlens


def interplen(r, vals, lim, ifid, comp='lt'):
    """Helper function for bar lengths allowing linear interpolation to fin the point where the criteria is crossed.
    Use either comp='lt' or comp='gt' to decide in which direction to fin the crossing. ifid should be an index point
    well into the bar to avoid e.g. finding an inner bar, when you wanted the outer."""
    if comp == 'lt':
        i = (vals > lim)  # bad points
        i = np.min(i.nonzero()[0][i.nonzero()[0] > ifid])
    else:
        i = (vals < lim)  # bad points
        i = np.min(i.nonzero()[0][i.nonzero()[0] > ifid])
    r0, val0 = r[i], vals[i]
    r1, val1 = r[i - 1], vals[i - 1]
    thisbarlen = r1 * (val0 - lim) / (val0 - val1) + r0 * (lim - val1) / (val0 - val1)
    return thisbarlen
