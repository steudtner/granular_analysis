"""Matched particle rendering and fluid slice, cropped to the particle region.

Run with python -m granular_analysis.scripts.render_particle_fluid --help.
The overlay is explicitly a full-depth particle projection over a 2D field;
it is not a particle cross-section. The comparison also provides exact cuts.
"""

from granular_analysis.plotting import plotstyle
from pathlib import Path
import argparse
import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from granular_analysis.paths import CasePaths
from granular_analysis.io.loader import load_frame
from granular_analysis.io.fluid_slice import nearest_frame, latest_common_counter, load_fluid_slice
from granular_analysis.geometry.ellipsoid import ellipsoid_slice


DEFAULT_BOUNDS = (0.0, 50.0, 0.0, 30.0)


def render_front_projection(mesh, bounds, *, width=2400):
    """Orthographic +z view, illuminated only from the front along -z."""
    import pyvista as pv
    xmin, xmax, ymin, ymax = bounds
    height = max(240, round(width * (ymax-ymin) / (xmax-xmin)))
    plotter = pv.Plotter(off_screen=True, window_size=(width, height))
    try:
        plotter.set_background('white')
        plotter.remove_all_lights()
        center = [(xmin+xmax)/2, (ymin+ymax)/2, (mesh.bounds[4]+mesh.bounds[5])/2]
        distance = 3*max(xmax-xmin, ymax-ymin, mesh.bounds[5]-mesh.bounds[4])
        front = (center[0], center[1], center[2]+distance)
        plotter.add_light(pv.Light(position=front, focal_point=center,
                                   light_type='scene light', positional=False, intensity=1.0))
        plotter.add_mesh(mesh, color='#b5bbc2', smooth_shading=True,
                         ambient=0.25, diffuse=0.8, specular=0.18)
        plotter.camera_position = [front, center, (0,1,0)]
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = (ymax-ymin)/2
        plotter.reset_camera_clipping_range()
        return plotter.screenshot(transparent_background=True)
    finally:
        plotter.close()


def run_particle_fluid(case, output, *, counter=None, time=None, z=2.25,
                       component='magnitude', bounds=DEFAULT_BOUNDS, skip_first=None, clim=None, fluid_grid=False):
    """Export combined rendering and exact-slice comparison plus provenance.

    Bounds are (xmin, xmax, ymin, ymax); None enables the legacy automatic zoom.
    Particle and fluid times must match exactly. PyVista is imported only here.
    """
    if counter is not None and time is not None:
        raise ValueError('Select either counter or time, not both')
    if not np.isfinite(z):
        raise ValueError('z must be finite')
    if bounds is not None:
        bounds = tuple(bounds)
        if len(bounds) != 4 or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
            raise ValueError('Bounds must contain finite increasing x/y limits')
    if clim is not None:
        clim = tuple(clim)
        if len(clim) != 2 or not np.all(np.isfinite(clim)) or clim[0] >= clim[1]:
            raise ValueError('Colour limits must be finite and increasing')
    output = Path(output)
    case = case if isinstance(case, CasePaths) else CasePaths.from_input(case)
    if time is not None:
        _, counter = nearest_frame(case, time, component)
    elif counter is None:
        counter = latest_common_counter(case, component)
    fluid = load_fluid_slice(case, counter, z, component)
    particles = load_frame(case.trn, f'ell_trn_{counter:06d}.bin')
    if skip_first is None:
        match = re.search(r'(?im)^\s*Nell_fixed\s*=\s*(\d+)', (case.case_root/'input.dat').read_text())
        if match is None:
            raise ValueError('Cannot determine fixed-particle count; specify --skip-first')
        skip_first = int(match.group(1))
    if not 0 <= skip_first < len(particles):
        raise ValueError('--skip-first must leave at least one particle for automatic bounds')
    def valid(p):
        axes = np.array([p['a'],p['b'],p['c']])
        return np.all(np.isfinite(axes)) and np.all(axes > 0) and np.all(np.isfinite(p['x']))
    mobile = [p for p in particles[skip_first:] if valid(p)]
    particles = [p for p in particles if valid(p)]
    if not mobile:
        raise ValueError('No valid mobile particles')
    centers = np.array([p['x'] for p in mobile])
    radii = np.array([max(p['a'], p['b'], p['c']) for p in mobile])
    if bounds:
        xmin, xmax, ymin, ymax = bounds
    else:
        lower = (centers[:, :2] - radii[:, None]).min(axis=0) - 1.0
        upper = (centers[:, :2] + radii[:, None]).max(axis=0) + 1.0
        xmin, ymin = np.maximum(lower, [fluid['x'][0], fluid['y'][0]])
        xmax, ymax = np.minimum(upper, [fluid['x'][-1], fluid['y'][-1]])
    if not np.all(np.isfinite([xmin,xmax,ymin,ymax])) or xmin >= xmax or ymin >= ymax:
        raise ValueError('Bounds must be finite and increasing')
    ix = np.flatnonzero((fluid['x'] >= xmin) & (fluid['x'] <= xmax))
    iy = np.flatnonzero((fluid['y'] >= ymin) & (fluid['y'] <= ymax))
    if len(ix) < 2 or len(iy) < 2:
        raise ValueError('Bounds must contain at least two fluid cells per direction')
    # Retain neighboring cells for complete coverage at crop edges.
    ix = np.arange(max(0,ix[0]-1), min(len(fluid['x']),ix[-1]+2))
    iy = np.arange(max(0,iy[0]-1), min(len(fluid['y']),iy[-1]+2))
    values = fluid['values'][np.ix_(iy,ix)]
    velocity_component = component in {'magnitude', 'u', 'v', 'w'}
    if velocity_component:
        values = values / plotstyle.VELOCITY_SCALE
    elif component == "q":
        values = values / plotstyle.Q_SCALE
    selected = [p for p in particles if
                p['x'][0]+max(p['a'],p['b'],p['c']) >= xmin and
                p['x'][0]-max(p['a'],p['b'],p['c']) <= xmax and
                p['x'][1]+max(p['a'],p['b'],p['c']) >= ymin and
                p['x'][1]-max(p['a'],p['b'],p['c']) <= ymax]
    print(f'Rendering {len(selected)} particles; t={fluid["time"]:.12g}, particle={counter}, fluid={fluid["counter"]}', flush=True)
    from granular_analysis.plotting.particle_3d_plotter import ellipsoids_to_polydata
    mesh = ellipsoids_to_polydata(selected, nu=36, nv=24)
    if mesh.n_points == 0:
        raise ValueError('No valid particles intersect the requested bounds')
    rgba = render_front_projection(mesh, (xmin,xmax,ymin,ymax))
    output.mkdir(parents=True,exist_ok=True)
    stem = f'particle_fluid_{counter:06d}_z{z:g}_{component}'
    if fluid_grid:
        stem += '_grid'
    sequential = component == 'magnitude'
    vmax = float(np.nanmax(values)) if sequential else float(np.nanmax(np.abs(values)))
    vmax = max(vmax, np.finfo(float).eps)
    cmap = plotstyle.FLUID_MAGNITUDE_CMAP if sequential else 'RdBu_r'
    vmin = -vmax
    if sequential:
        vmin, vmax = plotstyle.visible_fluid_range(
            values, fluid["x"][ix], fluid["y"][iy], (xmin, xmax, ymin, ymax))
    if clim is not None:
        vmin, vmax = map(float, clim)
    labels = {'magnitude':r'$\Vert\mathbf{u}_f\Vert$', 'pressure':r'$p\,\rho_f^{-1} g^{-1} D_{eq}^{-1}$',
              'u':r'$u_f$', 'v':r'$v_f$', 'w':r'$w_f$', 'q':plotstyle.Q_LABEL, 'lambda2':r'$\lambda_2$'}
    if velocity_component:
        labels[component] = plotstyle.velocity_label('f', component)
    plotstyle.configure()
    def field(ax):
        x = fluid['x_edges'][ix[0]:ix[-1]+2] if fluid_grid else fluid['x'][ix]
        y = fluid['y_edges'][iy[0]:iy[-1]+2] if fluid_grid else fluid['y'][iy]
        mappable = ax.pcolormesh(x,y,values,cmap=cmap,
                             vmin=vmin,vmax=vmax,shading='flat' if fluid_grid else 'nearest',rasterized=True)
        if fluid_grid:
            plotstyle.draw_fluid_grid(ax, fluid, (xmin, xmax, ymin, ymax))
        return mappable
    def format_axes(ax):
        ax.set(xlim=(xmin,xmax),ylim=(ymin,ymax),xlabel=r'$x\,D_{eq}^{-1}$',ylabel=r'$y\,D_{eq}^{-1}$')
        tick_step = 2.5 if fluid_grid else 5
        ax.set_yticks(np.arange(np.ceil(ymin/tick_step)*tick_step, ymax+1e-9, tick_step))
        if fluid_grid:
            ax.set_xticks(np.arange(np.ceil(xmin/tick_step)*tick_step, xmax+1e-9, tick_step))
        ax.set_aspect('equal')
    ratio=(ymax-ymin)/(xmax-xmin)
    fig, axes = plt.subplots(2,1,figsize=(plotstyle.figwidth_full, max(5,18*ratio+1.5)),layout='constrained')
    axes[0].imshow(rgba,extent=(xmin,xmax,ymin,ymax),origin='upper')
    axes[0].set_title('(a) Partikelrendering – Blick entlang der z-Achse',loc='left')
    mappable=field(axes[1])
    for p in selected:
        cut=ellipsoid_slice(p,z)
        if cut is not None:
            x,y,a,b,angle=cut
            axes[1].add_patch(Ellipse((x,y),2*a,2*b,angle=np.degrees(angle),facecolor='#d9dce0',edgecolor='#353c43',linewidth=0.35,zorder=3))
    axes[1].set_title(f'(b) Fluid und Partikelschnitte bei z/D$_{{eq}}$ = {z:g}',loc='left')
    for ax in axes: format_axes(ax)
    bar = fig.colorbar(mappable,ax=axes.tolist(),label=labels[component],orientation="horizontal",location="top",shrink=0.9,pad=0.08)
    bar.ax.xaxis.set_label_position("bottom")
    plotstyle.align_slice_legends(axes[1], [bar.ax])
    plotstyle.export_figure(fig, output/f'{stem}_vergleich.png',dpi=300)
    plt.close(fig)
    fig, ax=plt.subplots(figsize=(plotstyle.figwidth_full,max(3,9*ratio+1)),layout='constrained')
    mappable=field(ax)
    ax.imshow(rgba,extent=(xmin,xmax,ymin,ymax),origin='upper',zorder=3)
    format_axes(ax)
    ax.set_title(f'Partikelprojektion vor Fluid-Schnitt bei z/D$_{{eq}}$ = {z:g}',loc='left')
    bar = fig.colorbar(mappable,ax=ax,label=labels[component],orientation="horizontal",location="top",shrink=0.9,pad=0.12)
    bar.ax.xaxis.set_label_position("bottom")
    plotstyle.align_slice_legends(ax, [bar.ax])
    plotstyle.export_figure(fig, output/f'{stem}_kombiniert.png',dpi=300)
    plt.close(fig)
    metadata = dict(case=str(case.case_root),time=fluid['time'],particle_counter=counter,
                    fluid_counter=fluid['counter'],z=z,component=component,
                    bounds=[float(v) for v in (xmin,xmax,ymin,ymax)],clim=[vmin,vmax],
                    color_scale_source='explicit' if clim is not None else 'field',
                    fluid_grid=fluid_grid,
                    q_normalization=(dict(g=plotstyle.VELOCITY_GRAVITY,
                        D_eq=plotstyle.VELOCITY_DIAMETER, scale=plotstyle.Q_SCALE,
                        definition="Q_star=Q*D_eq/g") if component == "q" else None),
                    velocity_normalization=(dict(g=plotstyle.VELOCITY_GRAVITY,
                        D_eq=plotstyle.VELOCITY_DIAMETER, scale=plotstyle.VELOCITY_SCALE)
                        if velocity_component else None),
                    skip_first_for_zoom=skip_first,rendered_particles=len(selected),
                    lighting=dict(type='directional', direction=[0,0,-1], source_side='+z', default_lights=False),
                    note='Full-depth orthographic particle projection. Comparison lower panel uses exact particle cuts. Coordinates assume the existing D_eq=1 convention; fluid values retain simulation units except normalized pressure.')
    (output/f'{stem}.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(output/stem,flush=True)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', required=True)
    frames = parser.add_mutually_exclusive_group()
    frames.add_argument('--time', type=float)
    frames.add_argument('--counter', type=int)
    parser.add_argument('--z', type=float, default=2.25)
    parser.add_argument('--fluid', choices=['magnitude', 'u', 'v', 'w', 'pressure', 'q', 'lambda2'], default='magnitude')
    parser.add_argument('--skip-first', type=int)
    parser.add_argument('--fluid-grid', action='store_true', help='Overlay actual fluid cell boundaries')
    parser.add_argument('--clim', type=float, nargs=2, metavar=('MIN','MAX'), help='Fixed fluid colour limits shared between figures')
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument('--bounds', type=float, nargs=4, default=DEFAULT_BOUNDS,
                        metavar=('XMIN','XMAX','YMIN','YMAX'))
    limits.add_argument('--auto-zoom', action='store_true', help='Crop to the particle region instead of fixed limits')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    return run_particle_fluid(args.path, args.out, counter=args.counter, time=args.time,
                              z=args.z, component=args.fluid, skip_first=args.skip_first,
                              bounds=None if args.auto_zoom else args.bounds, clim=args.clim, fluid_grid=args.fluid_grid)


if __name__ == '__main__':
    main()
