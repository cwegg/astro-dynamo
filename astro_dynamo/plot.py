from typing import Union, Sequence

import astro_dynamo.model
import astro_dynamo.snap
import astro_dynamo.targets
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import SubplotBase
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar


def plot_snap_projections(model: Union[
    astro_dynamo.model.DynamicalModel, astro_dynamo.snap.SnapShot],
                          axs: Sequence[SubplotBase] = None,
                          plotmax: float = 10.,
                          vmin: float = 1e-5,
                          vmax: float = 1e-2,
                          cmap: Union[
                              str, matplotlib.colors.Colormap] = plt.cm.get_cmap(
                              'nipy_spectral')) -> matplotlib.figure.Figure:
    """Plot the projections of a DynamicalModel or SnapShot into the three axes in axs if supplied, otherwise draws
    a new figure. Distances will be physical if a model with a physical distance scale is supplied."""
    if isinstance(model, astro_dynamo.model.MilkyWayModel):
        snap, d_scale = model.snap.cpu(), model.d_scale.cpu()
    elif isinstance(model, astro_dynamo.model.DynamicalModel):
        snap, d_scale = model.snap, 1.0
    elif isinstance(model, astro_dynamo.model.SnapShot):
        snap, d_scale = model, 1.0
    else:
        raise ("Expected a DynamicalModel or SnapShot to plot")

    if axs is None:
        f, axs = plt.subplots(2, 1, sharex='col')
    else:
        f = axs.flatten()[0].figure

    x = snap.x.cpu() * d_scale
    y = snap.y.cpu() * d_scale
    z = snap.z.cpu() * d_scale
    m = snap.masses.detach().cpu()

    projections = ((x, y), (y, z), (x, z))
    projection_labels = (('x', 'y'), ('y', 'z'), ('x', 'z'))
    for ax, projection, projection_label in zip(axs.flatten(), projections,
                                                projection_labels):
        ax.hexbin(projection[0], projection[1], C=m, bins='log',
                  extent=(-plotmax, plotmax, -plotmax, plotmax),
                  reduce_C_function=np.sum,
                  vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xlabel(projection_label[0])
        ax.set_ylabel(projection_label[1])
        ax.set_xlim(-plotmax, plotmax)
        ax.set_ylim(-plotmax, plotmax)

    return f


def plot_disk_kinematics(model: astro_dynamo.model.DynamicalModel,
                         axs: Sequence[
                             SubplotBase] = None) -> matplotlib.figure.Figure:
    """Plots the azimuthally averaged kinematics, both mean and sigma in R_cyl, z, phi directions of a model.
    The model must contain a disk kinematics target as this is what id plotted.

    If supplied plots into the first 2 axes supplied in axs, otherwise creates a new figure."""

    if axs is None:
        f, axs = plt.subplots(2, 1, sharex='col')
    else:
        f = axs[0].figure

    try:
        disk_kinematics_obj = next(target for target in model.targets
                                   if isinstance(target,
                                                 astro_dynamo.targets.DiskKinematics))
    except IndexError:
        raise TypeError("Couldnt find a DiskKinematics target in the model.")

    kin_model = disk_kinematics_obj(model).detach().cpu()
    r = disk_kinematics_obj.rmid.cpu()
    for linestyle, kin, kin_err in zip(('-', '-.'),
                                       (kin_model,),
                                       (None,)):
        if kin_err is None:
            axs[0].plot(r, kin[1, :].detach().cpu(), 'r', linestyle=linestyle,
                        label='sig vphi')
            axs[0].plot(r, kin[3, :].detach().cpu(), 'g', linestyle=linestyle,
                        label='sig vr')
            axs[0].plot(r, kin[5, :].detach().cpu(), 'b', linestyle=linestyle,
                        label='sig vz')
        else:
            axs[0].errorbar(r, kin[1, :].detach().cpu().numpy(),
                            yerr=kin_err[1, :].cpu().numpy(),
                            fmt='o', color='r', ecolor='r')
            axs[0].errorbar(r, kin[3, :].detach().cpu().numpy(),
                            yerr=kin_err[3, :].cpu().numpy(),
                            fmt='o', color='g', ecolor='g')
            axs[0].errorbar(r, kin[5, :].detach().cpu().numpy(),
                            yerr=kin_err[5, :].cpu().numpy(),
                            fmt='o', color='b', ecolor='b')

    for linestyle, kin, kin_err in zip(('-', '-.'),
                                       (kin_model,),
                                       (None,)):
        if kin_err is None:
            axs[1].plot(r, kin[0, :].detach().cpu(), 'r', linestyle=linestyle,
                        label='mean vphi')
            axs[1].plot(r, kin[2, :].detach().cpu(), 'g', linestyle=linestyle,
                        label='mean vr')
            axs[1].plot(r, kin[4, :].detach().cpu(), 'b', linestyle=linestyle,
                        label='mean vz')
        else:
            axs[1].errorbar(r, kin[0, :].detach().cpu().numpy(),
                            yerr=kin_err[0, :].cpu().numpy(),
                            fmt='o', color='r', ecolor='r')
            axs[1].errorbar(r, kin[2, :].detach().cpu().numpy(),
                            yerr=kin_err[2, :].cpu().numpy(),
                            fmt='o', color='g', ecolor='g')
            axs[1].errorbar(r, kin[4, :].detach().cpu().numpy(),
                            yerr=kin_err[4, :].cpu().numpy(),
                            fmt='o', color='b', ecolor='b')

    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[0].set_ylim(0, 0.3 * model.v_scale.cpu())
    axs[1].set_ylim(-0.1 * model.v_scale.cpu(), 1.5 * model.v_scale.cpu())
    axs[1].set_xlabel('r [kpc]')
    axs[0].set_ylabel('[km/s]')
    axs[1].set_ylabel('[km/s]')

    f.subplots_adjust(hspace=0)
    f.tight_layout()
    return f


def plot_surface_density_profile(model: astro_dynamo.model.DynamicalModel,
                                 ax: SubplotBase = None,
                                 target_values: torch.Tensor = None) -> SubplotBase:
    """Plots the azimuthally averaged surface density of a model.
    The model must contain a SurfaceDensity target to be plotted.

    If supplied plots into axis ax, otherwise creates a new figure."""

    if ax is None:
        f, axs = plt.subplots(1, 1)
        axs[-1, -1].axis('off')

    try:
        surface_density_obj = next(target for target in model.targets
                                   if type(target) == astro_dynamo.targets.SurfaceDensity)
    except IndexError:
        raise TypeError("Couldn't find a SurfaceDensity target in the model.")

    ax.semilogy(surface_density_obj.rmid.cpu(),
                surface_density_obj(model).detach().cpu(), label='Model')
    if target_values is not None:
        ax.semilogy(surface_density_obj.rmid.cpu(),
                    target_values.detach().cpu(), label='Target')
    ax.set_xlabel('r')
    ax.set_ylabel(r'$\Sigma$')
    ax.set_ylim(1, 1e4)
    ax.legend()
    return ax


def plot_disk_density_twod(model: astro_dynamo.model.DynamicalModel,
                           axs: Sequence[SubplotBase] = None,
                           target_values: torch.Tensor = None) -> matplotlib.figure.Figure:
    """Plots the azimuthally averaged surface density of a model.
    The model must contain a SurfaceDensity target to be plotted.

    If supplied plots into axis ax, otherwise creates a new figure."""

    if axs is None:
        ncol = 2 if target_values is not None else 1
        f, axs = plt.subplots(2, ncol, sharex=True, sharey=True, squeeze=False)
    else:
        f = axs[0].figure

    try:
        disk_density_obj = next(target for target in model.targets
                                if type(target) == astro_dynamo.targets.DoubleExponentialDisk)
    except IndexError:
        raise TypeError(
            "Couldn't find a DoubleExponentialDisk target in the model.")

    model_disk_density = disk_density_obj(model).detach()
    model_disk_density_normed = model_disk_density / model_disk_density.sum(
        dim=1).unsqueeze(1)

    data = model_disk_density.t().log10().cpu().numpy()
    levels = np.max(data) + np.arange(-10, 1) * np.log10(np.exp(1.0))
    print(levels)
    cs = axs[0, 0].contourf(disk_density_obj.rmid.cpu(),
                            disk_density_obj.zmid.cpu(),
                            data, levels=levels)

    data = model_disk_density_normed.t().log10().cpu().numpy()
    levels = np.max(data) + np.arange(-5, 1) * np.log10(np.exp(1.0))
    cs_normed = axs[1, 0].contourf(disk_density_obj.rmid.cpu(),
                                   disk_density_obj.zmid.cpu(),
                                   data, levels=levels)
    print(levels)

    axs[0, 0].set_ylabel('z [kpc]')
    axs[1, 0].set_ylabel('z [kpc]')
    axs[1, 0].set_xlabel('R [kpc]')
    if target_values is not None:
        target_values_normed = target_values / target_values.sum(
            dim=1).unsqueeze(1)
        axs[0, 1].contourf(disk_density_obj.rmid.cpu(),
                           disk_density_obj.zmid.cpu(),
                           target_values.t().log10().cpu().numpy(),
                           levels=cs.levels)

        axs[1, 1].contourf(disk_density_obj.rmid.cpu(),
                           disk_density_obj.zmid.cpu(),
                           target_values_normed.t().log10().cpu().numpy(),
                           levels=cs_normed.levels)

        axs[1, 1].set_xlabel('R [kpc]')
        axs[0, 0].set_title('Model')
        axs[0, 1].set_title('Target')
    ax_divider = make_axes_locatable(axs[0, -1])
    cax = ax_divider.append_axes("right", size="7%", pad="2%")
    _ = colorbar(cs, cax=cax)

    ax_divider = make_axes_locatable(axs[1, -1])
    cax = ax_divider.append_axes("right", size="7%", pad="2%")
    _ = colorbar(cs_normed, cax=cax)

    return f
