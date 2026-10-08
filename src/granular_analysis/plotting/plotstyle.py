"""Shared pdfLaTeX/Computer Modern style for granular-analysis plots.

Create figures at their print size; export never rescales the figure or text.
The PDF is for unscaled inclusion, the SVG contains TeX glyphs as paths.
"""
from __future__ import annotations

import locale
import os
from pathlib import Path
import re
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cycler
from matplotlib.path import Path as MarkerPath
from matplotlib.text import Text
from matplotlib.ticker import Formatter, ScalarFormatter, FormatStrFormatter
import numpy as np

figwidth_full = 160 / 25.4
figwidth_half = figwidth_full / 2
VELOCITY_GRAVITY = 0.06
VELOCITY_DIAMETER = 1.0
VELOCITY_SCALE = float(np.sqrt(VELOCITY_GRAVITY * VELOCITY_DIAMETER))
FLUID_MAGNITUDE_CMAP = "inferno"
PARTICLE_SPEED_STEPS = 10
Q_SCALE = VELOCITY_GRAVITY / VELOCITY_DIAMETER
Q_LABEL = r"$Q\,g^{-1} D_{\mathrm{eq}}$"

COLORS = {'Kugel': '#000000', 'Prolat': '#000099', 'Oblat': '#cc6600'}
OUTPUT_DIR = Path(__file__).resolve().parents[4] / 'bilder' / 'generiert'


def _ellipse_marker(horizontal):
    vertices = MarkerPath.unit_circle().vertices.copy()
    vertices[:, 0 if horizontal else 1] *= 1.5
    vertices[:, 1 if horizontal else 0] *= 0.65
    return MarkerPath(vertices, MarkerPath.unit_circle().codes)


MARKERS = {'Kugel': 'o', 'Prolat': _ellipse_marker(True), 'Oblat': _ellipse_marker(False)}
PLOT_STYLE = {
    'text.usetex': True,
    'pgf.texsystem': 'pdflatex',
    'text.latex.preamble': r'\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}',
    'font.family': 'serif', 'font.serif': ['Computer Modern Roman'],
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.titlesize': 10, 'figure.labelsize': 10,
    'lines.linewidth': 0.8, 'axes.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'axes.formatter.use_locale': True, 'legend.frameon': False,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'figure.figsize': (figwidth_full, 3.4),
    'axes.grid': False, 'grid.color': '0.75', 'grid.linestyle': ':', 'grid.linewidth': 0.4,
    'svg.fonttype': 'path', 'svg.hashsalt': 'diplomarbeit-cm-10pt',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.prop_cycle': cycler(color=list(COLORS.values())),
}


def ensure_tex_path():
    """Use system TeX, an explicit TeX bin, or the project's portable installation."""
    if shutil.which('latex') and shutil.which('pdflatex'):
        return
    workspace = Path(__file__).resolve().parents[4]
    candidates = []
    if os.environ.get('GRANULAR_TEX_BIN'):
        candidates.append(Path(os.environ['GRANULAR_TEX_BIN']))
    installation = workspace / 'granular_analysis' / '.tools' / 'texlive'
    if (installation / '.installation_ready').is_file():
        candidates.extend(sorted((installation / 'bin').glob('*')))
    for directory in candidates:
        if (directory / 'latex').is_file() and (directory / 'pdflatex').is_file():
            os.environ['PATH'] = str(directory) + os.pathsep + os.environ.get('PATH', '')
            return


def configure(svg_fonttype='path', font_file=None):
    """Apply the shared style. Legacy font arguments no longer override TeX."""
    if svg_fonttype not in ('none', 'path'):
        raise ValueError("svg_fonttype muss 'none' oder 'path' sein")
    ensure_tex_path()
    for loc in ('de_DE.UTF-8', 'de_DE', 'deu_deu', 'German_Germany.1252'):
        try:
            locale.setlocale(locale.LC_NUMERIC, loc)
            break
        except locale.Error:
            continue
    plt.rcParams.update(PLOT_STYLE)


def draw_fluid_grid(ax, fluid, bounds):
    """Draw actual PRIME cell faces behind the particles, clipped to the slice."""
    from matplotlib.collections import LineCollection
    if not {'x_edges', 'y_edges'} <= fluid.keys():
        raise ValueError('Fluid cell faces missing; reload the slice from the simulation grid.')
    xmin, xmax, ymin, ymax = bounds
    x = np.asarray(fluid['x_edges'])
    y = np.asarray(fluid['y_edges'])
    segments = [[(value, ymin), (value, ymax)] for value in x if xmin <= value <= xmax]
    segments += [[(xmin, value), (xmax, value)] for value in y if ymin <= value <= ymax]
    lines = LineCollection(segments, colors='white', linewidths=.2, alpha=.65, zorder=2)
    ax.add_collection(lines)
    return lines


def visible_fluid_range(values, x, y, bounds):
    """Finite velocity range at grid centres inside the plotted viewport."""
    xmin, xmax, ymin, ymax = bounds
    ix = (np.asarray(x) >= xmin) & (np.asarray(x) <= xmax)
    iy = (np.asarray(y) >= ymin) & (np.asarray(y) <= ymax)
    visible = np.asarray(values)[np.ix_(iy, ix)]
    finite = visible[np.isfinite(visible)]
    if not finite.size:
        raise ValueError('No finite fluid velocities in the visible slice.')
    return float(finite.min()), float(finite.max())


def velocity_label(phase, component='magnitude'):
    """Dimensionless speed/component label using inverse half powers of g and D_eq."""
    numerator = (rf'\Vert\mathbf{{u}}_{{{phase}}}\Vert' if component == 'magnitude'
                 else rf'{component}_{{{phase}}}')
    return '$' + numerator + r'\,g^{-1/2} D_{\mathrm{eq}}^{-1/2}$'


def legend_above(ax):
    """Place a frameless horizontal legend above a slice/profile."""
    import textwrap
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    width = max(12, 75 // len(labels))
    labels = [textwrap.fill(label, width, break_on_hyphens=False) if '$' not in label else label
              for label in labels]
    return ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, 1.02),
                     ncol=len(handles), frameon=False)


def case_style(label, *, markers=True):
    """Use thesis colours/markers when a case label identifies its particle form."""
    lowered = str(label).lower()
    for form, tokens in [('Kugel', ('kugel', 'sphere')), ('Prolat', ('prolat',)),
                         ('Oblat', ('oblat',))]:
        if any(token in lowered for token in tokens):
            result = {'color': COLORS[form]}
            if markers:
                result.update(marker=MARKERS[form], markersize=4)
            return result
    return {}


def check_tex():
    """Fail clearly instead of silently changing fonts when TeX is absent."""
    ensure_tex_path()
    missing = [name for name in ('latex', 'pdflatex') if shutil.which(name) is None]
    if missing:
        raise RuntimeError('LaTeX fehlt im PATH: ' + ', '.join(missing) +
                           '. MiKTeX/TeX Live mit type1cm und cm-super installieren. '
                           'Für PNG-Vorschauen wird zusätzlich dvipng benötigt.')


def german_decimal(value, _position=None):
    if value == 0:
        return '0'
    return '$' + f'{value:g}'.replace('e', r'\mathrm{e}').replace('.', '{,}') + '$'


def latex_text(value):
    """Adapt legacy literal labels to TeX, leaving existing math intact."""
    parts = re.split(r'(?<!\\)(\$)', value)
    in_math = False
    greek = {'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
             'Δ': 'Delta', 'θ': 'theta', 'λ': 'lambda', 'ρ': 'rho',
             'φ': 'phi', 'χ': 'chi', 'σ': 'sigma', 'Π': 'Pi', 'ε': 'epsilon'}
    for index, part in enumerate(parts):
        if part == '$':
            in_math = not in_math
            continue
        part = re.sub(r'(?<=\d)\.(?=\d)', '{,}' if in_math else ',', part)
        for char, command in greek.items():
            part = part.replace(char, '\\' + command + ' ' if in_math else '$\\' + command + '$')
        if not in_math:
            part = re.sub(r'(?<!\\)([%&#])', r'\\\1', part)
            for char, replacement in {'−': '-', '²': '$^2$', '³': '$^3$',
                                      '°': '$^\\circ$', '≈': '$\\approx$',
                                      '≤': '$\\leq$', '≥': '$\\geq$',
                                      '√': '$\\surd$', '∑': '$\\sum$'}.items():
                part = part.replace(char, replacement)
        parts[index] = part
    return ''.join(parts)


class CommaScalarFormatter(ScalarFormatter):
    """Preserve scientific offsets, with a TeX decimal comma without extra space."""
    @staticmethod
    def comma(text):
        return re.sub(r'(?<=\d)[.,](?=\d)', '{,}', text)

    def __call__(self, value, pos=None):
        return self.comma(super().__call__(value, pos))

    def get_offset(self):
        return self.comma(super().get_offset())


class TeXLabelFormatter(Formatter):
    """Escape generated category labels while preserving the delegate's ticks."""
    def __init__(self, delegate):
        self.delegate = delegate

    def set_axis(self, axis):
        super().set_axis(axis)
        self.delegate.set_axis(axis)

    def set_locs(self, locs):
        super().set_locs(locs)
        self.delegate.set_locs(locs)

    def __call__(self, value, pos=None):
        return latex_text(self.delegate(value, pos))

    def get_offset(self):
        return latex_text(self.delegate.get_offset())


def legend_numbers(value):
    """Round decimal values in legends, preserving integer counts and powers."""
    pattern = r'(?<![\w.])([+-]?\d+(?:\.\d+|,\d+|\{,\}\d+))(?:([eE][+-]?\d+))?'
    def rounded(match):
        number = float(match[1].replace('{,}', '.').replace(',', '.'))
        return f'{number:.2f}' + (match[2] or '')
    return re.sub(pattern, rounded, value)


def prepare_figure(fig):
    """Unify labels and ticks, preserving categorical and logarithmic axes."""
    for ax in fig.axes:
        colorbar = getattr(ax, '_colorbar', None)
        if colorbar is not None:
            colorbar.formatter = FormatStrFormatter('%.2f')
            colorbar.update_ticks()
        for loc in ('left', 'center', 'right'):
            ax.set_title('', loc=loc)
        for axis in (ax.xaxis, ax.yaxis):
            for which in ('major', 'minor'):
                formatter = getattr(axis, f'get_{which}_formatter')()
                if isinstance(formatter, ScalarFormatter):
                    comma = CommaScalarFormatter(useOffset=formatter.get_useOffset(),
                                                 useMathText=True, useLocale=True)
                    comma.set_powerlimits(formatter._powerlimits)
                    comma.set_scientific(formatter._scientific)
                    getattr(axis, f'set_{which}_formatter')(comma)
                elif not isinstance(formatter, TeXLabelFormatter):
                    getattr(axis, f'set_{which}_formatter')(TeXLabelFormatter(formatter))
        legend = ax.get_legend()
        if legend is not None:
            legend.set_frame_on(False)
            for text in legend.get_texts():
                text.set_text(legend_numbers(text.get_text()))
        for line in ax.get_xgridlines() + ax.get_ygridlines():
            line.set(color='0.75', linestyle=':', linewidth=0.4)
    for legend in fig.legends:
        for text in legend.get_texts():
            text.set_text(legend_numbers(text.get_text()))
    if fig._suptitle is not None:
        fig._suptitle.set_text('')
    for text in fig.findobj(match=Text):
        text.set_fontfamily('serif')
        text.set_fontsize(10)
        text.set_usetex(True)
        text.set_text(latex_text(text.get_text()))


def align_slice_legends(slice_axis, legend_axes):
    """Align horizontal legends with the actual equal-aspect slice rectangle."""
    fig = slice_axis.figure
    prepare_figure(fig)
    fig.canvas.draw()
    # Freeze the finished layout so PDF/SVG export cannot expand the colour bars.
    fig.set_layout_engine('none')
    bounds = slice_axis.get_position()
    for legend_axis in legend_axes:
        position = legend_axis.get_position()
        legend_axis.set_position((bounds.x0, position.y0, bounds.width, position.height))


def export_figure(fig, output_path, *, dpi=300, **kwargs):
    """Write matching PDF+SVG; keep requested raster output for compatibility.

    Keep the physical canvas instead of cropping with bbox='tight': otherwise
    outside legends can make a 160-mm figure wider than the thesis text block.
    Layout is calculated at the original size; no post-export scaling occurs.
    """
    check_tex()
    width = float(fig.get_size_inches()[0])
    if width > figwidth_full + 0.001:
        raise ValueError(f'Figure width {width:.3g} in exceeds 160 mm; set figsize at creation.')
    prepare_figure(fig)
    # Avoid changing explicitly constrained layouts; compact existing manual layouts.
    if fig.get_layout_engine() is None:
        fig.tight_layout(pad=0.5)
    target = Path(output_path)
    if not target.suffix:
        target = target.with_suffix('.pdf')
    target.parent.mkdir(parents=True, exist_ok=True)
    options = {key: value for key, value in kwargs.items()
               if key not in ('bbox_inches', 'pad_inches', 'format', 'metadata')}
    options.update(bbox_inches=fig.bbox_inches, pad_inches=0, dpi=dpi)
    paths = [target.with_suffix('.pdf'), target.with_suffix('.svg')]
    if target.suffix.lower() not in ('.pdf', '.svg'):
        paths.append(target)
    for path in paths:
        metadata = ({'CreationDate': None, 'ModDate': None} if path.suffix == '.pdf'
                    else {'Date': None} if path.suffix == '.svg' else None)
        fig.savefig(path, metadata=metadata, **options)
    return paths


def save(fig, name):
    """Compatibility entry point for existing thesis plotting routines."""
    if not re.fullmatch(r'(?:kap\d+(?:_\d+)?|anhang)_[a-z0-9_]+', name):
        raise ValueError('Name erwartet: kap7_1_kinenergie (ohne Pfad/Endung)')
    return export_figure(fig, OUTPUT_DIR / f'{name}.pdf')


configure()
