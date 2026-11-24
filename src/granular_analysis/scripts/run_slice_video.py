"""Side-view movies of exact z-plane ellipsoid sections with fixed speed colours."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.patches import Ellipse
import numpy as np

from granular_analysis.geometry.ellipsoid import ellipsoid_slice
from granular_analysis.io.loader import load_frame
from granular_analysis.plotting import plotstyle
from granular_analysis.scripts.run_renderview_video import select_video_frames, playback_schedule, find_ffmpeg


def slice_sections(particles, z=2.25, lz=5):
    rows = []
    for p in particles:
        for shift in (-lz, 0., lz):
            section = ellipsoid_slice(p, z, shift)
            if section is not None:
                speed = np.linalg.norm(p.get('u', [0., 0., 0.])) / plotstyle.VELOCITY_SCALE
                rows.append((*section, speed))
    return np.asarray(rows, dtype=float).reshape(-1, 6)


def draw_frame(rows, path, *, vmax, time_s, z=2.25, xlim=(0,100), ylim=(0,20)):
    cmap = LinearSegmentedColormap.from_list('particle_speed', ['white', 'red'], N=10)
    norm = BoundaryNorm(np.linspace(0, vmax if vmax > 0 else 1, 11), cmap.N, clip=True)
    with plt.rc_context({'text.usetex': False, 'font.family': 'DejaVu Sans', 'font.size': 14, 'savefig.bbox': None}):
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
        ax = fig.add_axes([.065, .22, .90, .62])
        patches = [Ellipse((x,y), 2*a, 2*b, angle=np.degrees(theta)) for x,y,a,b,theta,_ in rows]
        collection = PatchCollection(patches, cmap=cmap, norm=norm, edgecolor='black', linewidth=.4)
        collection.set_array(np.nan_to_num(rows[:,5], nan=0., posinf=0., neginf=0.))
        ax.add_collection(collection)
        ax.set(xlim=xlim, ylim=ylim, aspect='equal', xlabel=r'$x/D_{Eq}$', ylabel=r'$y/D_{Eq}$')
        ax.set_xticks(np.arange(xlim[0], xlim[1]+1, 10))
        ax.set_yticks(np.arange(ylim[0], ylim[1]+1, 5))
        ax.grid(True, color='.8', linestyle=':', linewidth=.5)
        ax.set_axisbelow(True)
        ax.set_title(f'z = {z:g}    t = {time_s:.3f} s')
        cax=fig.add_axes([.25,.76,.50,.025])
        fig.colorbar(collection, cax=cax, orientation='horizontal', label=r'$|u_p|/\sqrt{g D_{Eq}}$')
        fig.savefig(path, dpi=100, facecolor='white', bbox_inches=None)
        plt.close(fig)


def run_slice_video(case, out_dir, *, z=2.25, lz=5, fps=30, speed=1, xlim=(0,100), ylim=(0,20)):
    frames=select_video_frames(case)
    indices,targets,scale=playback_schedule(frames,fps=fps,speed=speed)
    selected=np.unique(indices)
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    output=out/f'slice_side_z{z:.2f}.mp4'
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(prefix='slice-video-',dir=out) as folder:
        folder=Path(folder);vmax=0.
        for j,i in enumerate(selected):
            time,counter=frames[i]
            rows=slice_sections(load_frame(case.require_trn(),f'ell_trn_{counter:06d}.bin'),z,lz)
            np.save(folder/f'sections_{i}.npy',rows)
            finite=rows[:,5][np.isfinite(rows[:,5])]
            if finite.size: vmax=max(vmax,float(finite.max()))
            print(f'Prepared {j+1}/{len(selected)}: {counter:06d}, {len(rows)} sections',flush=True)
        for j,i in enumerate(selected):
            time,counter=frames[i]
            draw_frame(np.load(folder/f'sections_{i}.npy'),folder/f'source_{i}.png',vmax=vmax,time_s=time*scale,z=z,xlim=xlim,ylim=ylim)
            print(f'Rendered {j+1}/{len(selected)}',flush=True)
        for j,i in enumerate(indices): os.link(folder/f'source_{i}.png',folder/f'frame_{j:06d}.png')
        subprocess.run([find_ffmpeg(),'-y','-loglevel','error','-framerate',str(fps),'-i',str(folder/'frame_%06d.png'),'-c:v','libx264','-threads','4','-preset','medium','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/'movie.mp4')],check=True)
        (folder/'movie.mp4').replace(output)
    metadata=dict(case=str(case.case_root),z=z,lz=lz,view='side (xy plane)',xlim=xlim,ylim=ylim,fps=fps,speed=speed,resolution=[1920,1080],seconds_per_t_star=scale,particle_color='full particle speed / sqrt(0.06), 10 white-to-red bins; fixed over entire video',color_limits=[0,vmax],periodic_z=True,physical_duration_s=(frames[-1][0]-frames[0][0])*scale,encoded_duration_s=len(indices)/fps,frames=[dict(counter=frames[i][1],t_star=frames[i][0],source_time_s=frames[i][0]*scale,playback_time_s=float(targets[j])) for j,i in enumerate(indices)])
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Saved',output,flush=True)
    return output
