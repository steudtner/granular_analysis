"""Final-deposit orientation histograms in spatial bins."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from granular_analysis.analysis.deposit_orientation import deposit_orientation
from granular_analysis.config import SKIP_FIRST_PARTICLES, USE_PARTICLE_SKIP
from granular_analysis.io.loader import load_frame
from granular_analysis.paths import latest_trn_counter
from granular_analysis.plotting import plotstyle


def plot_orientation_histograms(table, histograms, edges, angle_edges, prefix):
    reference = np.cos(np.radians(angle_edges[:-1])) - np.cos(np.radians(angle_edges[1:]))
    centers = (angle_edges[:-1] + angle_edges[1:]) / 2
    for mode in ('prolate', 'oblate'):
        if not ((table['mode'] == mode) & (table.status == 'valid')).any():
            continue
        n_bins = len(edges) - 1
        with plt.rc_context(plotstyle.PLOT_STYLE):
            fig, axes = plt.subplots(n_bins, 3, squeeze=False, sharex=True, sharey=True,
                                     figsize=(plotstyle.figwidth_full, max(3, n_bins * 1.65)),
                                     layout='constrained')
            maximum = histograms.loc[histograms['mode'] == mode, 'fraction'].max()
            ymax = min(1.05, max(.3, float(maximum) * 1.15, reference.max() * 1.3))
            for i in range(n_bins):
                group = table[(table.bin == i) & (table['mode'] == mode)]
                n_valid = int((group.status == 'valid').sum())
                for j, axis in enumerate('xyz'):
                    ax = axes[i, j]
                    hist = histograms[(histograms['mode'] == mode) & (histograms.bin == i) & (histograms.axis == axis)]
                    ax.bar(centers, hist.fraction.fillna(0), width=np.diff(angle_edges),
                           color='#238b8d' if mode == 'prolate' else '#cc6600', edgecolor='white', linewidth=.3)
                    ax.plot(centers, reference, 'k--', linewidth=.7)
                    ax.set(xlim=(0, 90), ylim=(0, ymax), xticks=[0, 30, 60, 90])
                    if i == 0:
                        axis_name = 'a' if mode == 'prolate' else 'c'
                        ax.text(.5, 1.20, f'Winkel {axis_name} zu {axis}',
                                transform=ax.transAxes, ha='center')
                    if j == 0:
                        ax.set_ylabel('Rel. Häufigkeit')
                    if n_valid == 0:
                        ax.text(.5, .5, 'Keine gültigen Achsen', transform=ax.transAxes, ha='center')
                    if i == n_bins - 1:
                        ax.set_xlabel('Winkel [Grad]')
            axes[0, 2].plot([], [], 'k--', label='Isotrop in 3D')
            axes[0, 2].legend(loc='upper left')
            try:
                plotstyle.export_figure(fig, Path(str(prefix) + f'_{mode}.png'), dpi=300)
            finally:
                plt.close(fig)


def run_deposit_orientation(*, case, out_dir, counter=None, n_bins=6, angle_step=10,
                            mode='auto', xlim=None, ylim=None, skip_first=None, z_plane=None, lz=5.0,
                            renderview_image=None):
    z_plane = lz / 2 if z_plane is None else z_plane
    if not np.isfinite(lz) or lz <= 0 or not np.isfinite(z_plane):
        raise ValueError("Require finite z-plane and positive periodic length")
    counter = latest_trn_counter(case) if counter is None else counter
    if counter is None:
        raise FileNotFoundError(f'No particle frames in {case.trn}')
    skip_first = (SKIP_FIRST_PARTICLES if USE_PARTICLE_SKIP else 0) if skip_first is None else skip_first
    particles = load_frame(case.require_trn(), f'ell_trn_{counter:06d}.bin')
    if renderview_image is not None and xlim is None:
        projection_path = Path(renderview_image).with_suffix('.projection.json')
        if projection_path.exists():
            projection = json.loads(projection_path.read_text())
            if 'particle_bounds' in projection:
                xlim = tuple(projection['particle_bounds'][:2])
    table, histograms, edges, angle_edges = deposit_orientation(
        particles, n_bins=n_bins, angle_step=angle_step, mode=mode,
        xlim=xlim, ylim=ylim, skip_first=skip_first)
    target = Path(out_dir) / 'deposit_orientation'
    target.mkdir(parents=True, exist_ok=True)
    prefix = target / f'deposit_orientation_{mode}_{counter:06d}'
    table.to_csv(str(prefix) + '_particles.csv', index=False)
    histograms.to_csv(str(prefix) + '_histograms.csv', index=False)
    metadata = dict(counter=counter, time=float(particles[0]['t']) if particles else None,
                    mode=mode, skip_first_rows=skip_first, x_edges=edges.tolist(),
                    angle_edges_deg=angle_edges.tolist(), ylim=ylim, slice_z=z_plane, periodic_length=lz,
                    inset_population="all valid particles in each x-bin, across full z domain",
                    definitions={'prolate': 'local a axis (prolate symmetry axis)', 'oblate': 'local c axis (oblate symmetry axis / disc normal)',
                                 'angles': 'acos(abs(axis dot coordinate direction)), 0 to 90 degrees',
                                 'bins': 'particle centers; left closed, right open; final right edge included',
                                 'normalization': 'count divided by valid particles of this shape in this spatial bin',
                                 'isotropic_reference': 'cos(lower angle) - cos(upper angle)',
                                 'auto_shape_tolerance': 'equal semiaxes: rtol=1e-6, atol=0'},
                    status_counts={str(k): int(v) for k, v in table.status.value_counts().items()})
    prefix.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    plot_orientation_histograms(table, histograms, edges, angle_edges, prefix)
    plot_slice_y_insets(particles, table, histograms, edges, angle_edges, prefix,
                        z_plane=z_plane, lz=lz, ylim=ylim)
    if renderview_image is not None:
        plot_slice_y_insets(particles, table, histograms, edges, angle_edges, prefix,
                            z_plane=z_plane, lz=lz, ylim=ylim, renderview_image=renderview_image)
    print(f'Deposit orientation: {sum(table.status == "valid")}/{len(table)} valid axes, {n_bins} x-bins; skipped first {skip_first} rows.')
    print(f'Saved particle CSV, histogram CSV, metadata and PNG/PDF/SVG: {prefix}')
    return table


def plot_slice_y_insets(particles, table, histograms, edges, angle_edges, prefix,
                        *, z_plane=2.5, lz=5.0, ylim=None, renderview_image=None):
    """Exact midplane cut with full-volume y-angle histograms as insets."""
    edges = np.asarray(edges, dtype=float)
    angle_edges = np.asarray(angle_edges, dtype=float)
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Ellipse
    from matplotlib.colors import Normalize
    from granular_analysis.geometry.ellipsoid import ellipsoid_slice
    render = None
    render_extent = None
    if renderview_image is not None:
        from PIL import Image, ImageChops
        with Image.open(renderview_image) as source:
            source = source.convert('RGB')
            # Remove only empty background; retain every rendered particle pixel.
            box = ImageChops.difference(source, Image.new('RGB', source.size, 'white')).getbbox()
            projection_path = Path(renderview_image).with_suffix('.projection.json')
            if projection_path.exists():
                extent = json.loads(projection_path.read_text())['extent']
                x0, x1, z1, z0 = extent
                if box is None:
                    box = (0, 0, source.width, source.height)
                left, top, right, bottom = box
                left, top = max(0, left-8), max(0, top-8)
                right, bottom = min(source.width, right+8), min(source.height, bottom+8)
                box = (left, top, right, bottom)
                render_extent = (x0+(x1-x0)*left/source.width, x0+(x1-x0)*right/source.width,
                                 z0+(z1-z0)*bottom/source.height, z0+(z1-z0)*top/source.height)
            render = np.asarray(source.crop(box) if box is not None else source).copy()
    for mode in ('prolate', 'oblate'):
        selected = table[(table['mode'] == mode) & (table.status == 'valid')]
        if selected.empty:
            continue
        patches, colors = [], []
        for row in selected.itertuples():
            particle = particles[row.particle_row - 1]
            for shift in (-lz, 0.0, lz):
                section = ellipsoid_slice(particle, z_plane, shift)
                if section is None:
                    continue
                x, y, a, b, theta = section
                patches.append(Ellipse((x, y), 2*a, 2*b, angle=np.degrees(theta)))
                colors.append(row.angle_y_deg)
        axis_name = 'a' if mode == 'prolate' else 'c'
        n_bins = len(edges) - 1
        width = edges[-1] - edges[0]
        x_low, x_high = edges[0], edges[-1]
        if render_extent is not None:
            x_low = min(x_low, render_extent[0])
            x_high = max(x_high, render_extent[1])
        for patch in patches:
            patch_bounds = patch.get_path().get_extents(patch.get_transform())
            x_low, x_high = min(x_low, patch_bounds.xmin), max(x_high, patch_bounds.xmax)
        plot_width = x_high-x_low
        actual_high = max(float(p.center[1] + max(p.width, p.height)/2) for p in patches) if patches else float(selected.y.max())
        low = ylim[0] if ylim is not None else min(0, float(selected.y.min()))
        deposit_high = ylim[1] if ylim is not None else actual_high
        high = low + max((deposit_high-low)*1.8, plot_width * .5)
        centers = (angle_edges[:-1] + angle_edges[1:]) / 2
        cmap = plt.get_cmap('viridis')
        norm = Normalize(0, 90)
        with plt.rc_context(plotstyle.PLOT_STYLE):
            slice_height = max(3.2, plotstyle.figwidth_full * (high-low)/plot_width + .9)
            render_height = (plotstyle.figwidth_full * (render_extent[2]-render_extent[3])/plot_width + .45
                             if render_extent is not None else
                             plotstyle.figwidth_full * render.shape[0]/render.shape[1] + .45 if render is not None else 0)
            fig = plt.figure(figsize=(plotstyle.figwidth_full, slice_height + render_height),
                             layout='constrained')
            fig.get_layout_engine().set(w_pad=.02, h_pad=.02, hspace=0, wspace=0)
            if render is None:
                layout = fig.add_gridspec(2, 1, height_ratios=[.4, 5])
                offset = 0
            else:
                layout = fig.add_gridspec(3, 1, height_ratios=[slice_height*.4/5.4, slice_height*5/5.4, render_height])
                render_ax = fig.add_subplot(layout[2])
                image_extent = render_extent if render_extent is not None else (x_low, x_high, lz, 0)
                render_ax.imshow(render, extent=image_extent, origin='upper', interpolation='none')
                render_ax.set(xlim=(x_low, x_high), ylim=(image_extent[2], image_extent[3]),
                              xlabel=r'$x / D_{\mathrm{Eq}}$', ylabel=r'$z / D_{\mathrm{Eq}}$')
                render_ax.set_aspect('equal', adjustable='box')
                render_ax.set_anchor('N')
                offset = 0
            cax = fig.add_subplot(layout[offset])
            ax = fig.add_subplot(layout[offset+1])
            collection = PatchCollection(patches, cmap=cmap, norm=norm, edgecolors='0.2', linewidths=.15)
            collection.set_array(np.asarray(colors))
            ax.add_collection(collection)
            ax.set(xlim=(x_low, x_high), ylim=(low, high), xlabel=r'$x / D_{\mathrm{Eq}}$' if render is None else '', ylabel=r'$y / D_{\mathrm{Eq}}$')
            ax.set_aspect('equal', adjustable='box')
            ax.set_anchor('N')
            ax.set_xticks(np.arange(np.ceil(edges[0]/5)*5, edges[-1] + 1e-9, 5))
            if render is not None:
                render_ax.set_xticks(ax.get_xticks())
            ax.set_yticks(np.arange(np.ceil(low/5)*5, high + 1e-9, 5))
            ax.grid(True)
            for edge in edges[1:-1]:
                ax.axvline(edge, color='0.6', linewidth=.6, linestyle=':')
            for i in range(n_bins):
                inset = ax.inset_axes([(edges[i]-x_low+.18*(edges[i+1]-edges[i]))/plot_width,
                                      .73, .64*(edges[i+1]-edges[i])/plot_width, .18])
                hist = histograms[(histograms['mode'] == mode) & (histograms.bin == i) & (histograms.axis == 'y')]
                inset.bar(centers, hist.fraction.fillna(0), width=np.diff(angle_edges),
                          color=cmap(norm(centers)), edgecolor='white', linewidth=.2)
                inset.set(xlim=(0, 90), ylim=(0, 1), xticks=[0, 45, 90], yticks=[0, 1] if i == 0 else [])
                inset.tick_params(pad=1, length=2)
                count = int(hist.valid_particles.iloc[0])
                inset.text(.5, 1.06, rf'$N_{{\mathrm{{Mob}}}}={count}$',
                           transform=inset.transAxes, va='bottom', ha='center')
                if count == 0:
                    inset.text(.5, .4, 'leer', transform=inset.transAxes, ha='center')
            ax.text(.5, .61, f'Winkel {axis_name} zu y [Grad]; rel. Häufigkeit',
                    transform=ax.transAxes, ha='center')
            colorbar = fig.colorbar(collection, cax=cax, orientation='horizontal', ticks=[0, 30, 60, 90])
            colorbar.set_label(f'Winkel {axis_name} zu y [Grad]')
            colorbar.ax.xaxis.set_ticks_position('top')
            colorbar.ax.xaxis.set_label_position('bottom')
            plotstyle.align_slice_legends(ax, [cax])
            if render is not None:
                fig.canvas.draw()
                slice_position = ax.get_position()
                render_position = render_ax.get_position()
                render_ax.set_position((slice_position.x0, render_position.y0,
                                        slice_position.width, render_position.height))
            try:
                suffix = '_renderview_combined' if render is not None else ''
                plotstyle.export_figure(fig, Path(str(prefix) + f'_{mode}_cut_z{z_plane:.2f}_y_insets{suffix}.png'),
                                        dpi=max(400, render.shape[1]/plotstyle.figwidth_full) if render is not None else 400)
            finally:
                plt.close(fig)
