from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from ipywidgets import Layout
import starry
from ylm_rot import get_ylm_coeffs
import matplotlib.pyplot as pl
import numpy as np


vslider = \
widgets.FloatSlider(
    value=0.1,
    min=0.1,
    max=10.0,
    step=0.01,
    description=r'$v_\mathrm{eq}$ [km / s]:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.2f',
    layout=Layout(width='40%')
)

oslider = \
widgets.FloatSlider(
    value=0,
    min=-90,
    max=90.0,
    step=0.1,
    description=r'$\lambda$ [deg]:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
    layout=Layout(width='40%')
)

islider = \
widgets.FloatSlider(
    value=90,
    min=1,
    max=179.0,
    step=0.1,
    description=r'$i$ [deg]:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
    layout=Layout(width='40%')
)

aslider = \
widgets.FloatSlider(
    value=0,
    min=0,
    max=1.0,
    step=0.01,
    description=r'$\alpha$:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.2f',
    layout=Layout(width='40%')
)

u1slider = \
widgets.FloatSlider(
    value=0,
    min=0.0,
    max=2.0,
    step=0.01,
    description=r'$u_1$:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.2f',
    layout=Layout(width='40%')
)

u2slider = \
widgets.FloatSlider(
    value=0.0,
    min=-1.0,
    max=1.0,
    step=0.01,
    description=r'$u_2$:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.2f',
    layout=Layout(width='40%')
)

yslider = \
widgets.FloatSlider(
    value=0,
    min=-1.0,
    max=1.0,
    step=0.01,
    description=r'$b$:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.2f',
    layout=Layout(width='40%')
)

rslider = \
widgets.FloatSlider(
    value=0.1,
    min=0.01,
    max=0.5,
    step=0.001,
    description=r'$r / R_\star$:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.3f',
    layout=Layout(width='40%')
)


# Load RV data for HD 189733 from Bedell, corrected for the baseline
xo_189, rv_189 = np.loadtxt("HD189733_sample.txt", unpack=True)


# Create a global starry map
map = starry.Map(5)


def visualize_func(veq=1, inc=90, obl=0, alpha=0, u1=0, u2=0, yo=0, ro=0.1):
    res = 300
    map[:3, :] = (veq * 1.e3) * get_ylm_coeffs(inc, obl, alpha)
    map[0, 0] = (veq * 1.e3) * 1e-15
    map[1] = u1
    map[2] = u2
    
    if (u1 + u2) > 1 or (u1 + 2 * u2) < 0 or u1 < 0:
        u1slider.style.handle_color = "#FF0000"
        u2slider.style.handle_color = "#FF0000"
    else:
        u1slider.style.handle_color = "#FFFFFF"
        u2slider.style.handle_color = "#FFFFFF"
    
    x, y = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
    img = np.array([map(x=x[j], y=y[j]) for j in range(res)]) / 1.e3
    fig = pl.figure(figsize=(15, 8))
    axim = pl.axes((0, 0.05, 0.3, 0.8))
    axcb = pl.axes((0, 0.85, 0.3, 0.03))
    axrm = pl.axes((0.4, 0.20, 0.6, 0.5))
    im = axim.imshow(img, cmap='RdBu_r', origin='lower', 
                     vmin=-veq, vmax=veq, extent=(-1,1,-1,1))
    cb = pl.colorbar(im, orientation='horizontal', cax=axcb)
    cb.ax.set_xlabel("Radial velocity [km / s]")
    axim.contour(img, origin='lower', levels=np.linspace(-veq, veq, 20), 
                 colors=['k' for i in range(20)], alpha=0.25,
                 extent=(-1,1,-1,1))
    axim.axis('off')
    axim.set_aspect(1)
    axim.axhline(yo, color='k', alpha=0.5)
    axim.axhline(yo + 0.5 * ro, color='k', ls='--', alpha=0.5)
    axim.axhline(yo - 0.5 * ro, color='k', ls='--', alpha=0.5)
    
    xo = np.linspace(-1 - 2 * ro, 1 + 2 * ro, 1000)
    rv = map.flux(xo=xo, yo=yo, ro=ro) / np.pi
    axrm.plot(xo, rv)
    axrm.set_xlabel(r"Occultor x position [$R_\star$]", fontsize=16)
    axrm.set_ylabel("Radial velocity [m /s]", fontsize=16)
    axrm.set_title("The Rossiter-McLaughlin effect", fontsize=20)

    axrm.plot(xo_189, rv_189, '.')


def visualize():
    return interact(visualize_func, veq=vslider, inc=islider, 
                    obl=oslider, alpha=aslider, u1=u1slider, 
                    u2=u2slider, yo=yslider, ro=rslider)