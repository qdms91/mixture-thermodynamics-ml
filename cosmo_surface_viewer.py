#!/usr/bin/env python3
"""Interactive COSMO screening-charge surfaces using openCOSMO-RS and Plotly.

Put this file in the cosmo_rs_1 project root and run:
    python cosmo_surface_viewer.py

Defaults: newly calculated water and methanol surfaces in calculations/.
Choose molecules: python cosmo_surface_viewer.py water benzene
Published surfaces: python cosmo_surface_viewer.py acetone acetonitrile --source reference
Explicit file paths are also accepted in place of molecule names.
Raw (unaveraged) densities: add --raw

The file provides segment centers and areas, not polygon boundaries. We draw
area-equivalent regular polygons tangent to the parent atom's cavity sphere.
These are approximate display patches, with real segment positions and sigma
values. They can overlap or leave gaps. No interpolated charge field, electronic
structure calculation, or thermodynamic calculation is performed here.

Designed for small neutral molecules and tested against opencosmorspy commit
c2bc186762e4a2a3e6bb547fc9cbc917ed35d9b0 with the published 24a reference files.
"""

import argparse
from html import escape
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from opencosmorspy.input_parsers import SigmaProfileParser
from opencosmorspy.parameterization import openCOSMORS24a

from project_config import MOLECULES, PROJECT, surface_paths


COLORS = {"H": "#f3f4f6", "C": "#424750", "N": "#3050f8", "O": "#ef3e45",
          "F": "#90e050", "S": "#ffd123", "P": "#ff8000", "Cl": "#1ff01f"}
SIGMA_COLORS = [[0, "#2166ac"], [0.5, "#f7f7f7"], [1, "#b2182b"]]


def perpendicular_basis(direction):
    """Two unit vectors spanning a plane normal to direction."""
    normal = np.asarray(direction, dtype=float)
    normal /= np.linalg.norm(normal)
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    first = np.cross(normal, axis)
    first /= np.linalg.norm(first)
    return first, np.cross(normal, first)


def patch_mesh(surface, sigma, sides=12):
    """One flat, area-equivalent polygon per original surface segment."""
    # The surface file gives segment centers and areas, but not their outlines.
    # Here we draw a small polygon tangent to each atom's cavity sphere, choosing
    # its radius so that its area equals the original segment area. The colors
    # use the calculated sigma values; only the patch boundaries are approximate.
    vertices, faces, values, descriptions = [], [], [], []
    angles = np.arange(sides) * 2 * np.pi / sides
    for index, (center, area, atom, value) in enumerate(zip(
        surface["seg_pos"], surface["seg_area"], surface["seg_atm_nr"], sigma
    )):
        u, v = perpendicular_basis(center - surface["atm_pos"][atom])
        # A regular n-gon has area n*r^2*sin(2*pi/n)/2.
        radius = np.sqrt(2 * area / (sides * np.sin(2 * np.pi / sides)))
        ring = center + radius * (np.cos(angles)[:, None] * u
                                 + np.sin(angles)[:, None] * v)
        offset = len(vertices)
        vertices.extend(np.vstack([center, ring]))
        faces.extend((offset, offset + 1 + k, offset + 1 + (k + 1) % sides)
                     for k in range(sides))
        values.extend([value] * (sides + 1))
        descriptions.extend([
            f"Segment {index + 1} · {surface['atm_elmnt'][atom]}{atom + 1}"
            f"<br>σ = {value:.6f} e/Å²<br>Area = {area:.5f} Å²"
        ] * (sides + 1))
    return np.asarray(vertices), np.asarray(faces), values, descriptions


def solid_trace(xyz, color, label):
    """A shaded parametric sphere or cylinder with a constant atom/bond color."""
    return go.Surface(
        x=xyz[0], y=xyz[1], z=xyz[2],
        surfacecolor=np.zeros_like(xyz[0]),
        colorscale=[[0, color], [1, color]], cmin=0, cmax=1,
        showscale=False, hovertemplate=label + "<extra></extra>",
        lighting=dict(ambient=0.5, diffuse=0.85, specular=0.3, roughness=0.6),
        name=label, showlegend=False,
    )


def structure_traces(surface):
    """Atoms as spheres; one stick per connection in the file's adjacency matrix.

    Bond multiplicities are not inferred. There is no distance-based bond guess.
    """
    # atm_pos stores the atomic coordinates, while adjacency_matrix describes
    # which atoms are connected. We represent atoms by small spheres and their
    # connections by cylinders, so the structure remains visible inside the
    # screening-charge surface. A stick indicates connectivity, not bond order.
    xyz = surface["atm_pos"]
    adjacency = surface.get("adjacency_matrix")
    if adjacency is None:
        raise ValueError("This viewer needs an ORCA file containing an adjacency matrix.")
    theta, phi = np.meshgrid(np.linspace(0, 2 * np.pi, 25),
                             np.linspace(0, np.pi, 17))
    sphere = np.array([np.sin(phi) * np.cos(theta),
                       np.sin(phi) * np.sin(theta), np.cos(phi)])
    for index, (element, position) in enumerate(zip(surface["atm_elmnt"], xyz)):
        radius = 0.18 if element == "H" else 0.28
        yield solid_trace(position[:, None, None] + radius * sphere,
                          COLORS.get(element, "#a577bd"), f"{element}{index + 1}")
    angle, length = np.meshgrid(np.linspace(0, 2 * np.pi, 13), [0, 1])
    for first, second in zip(*np.where(np.triu(adjacency, 1) > 0)):
        direction = xyz[second] - xyz[first]
        u, v = perpendicular_basis(direction)
        cylinder = (xyz[first][:, None, None] + direction[:, None, None] * length
                    + 0.065 * (u[:, None, None] * np.cos(angle)
                               + v[:, None, None] * np.sin(angle)))
        yield solid_trace(cylinder, "#8d929b", f"Bond {first + 1}–{second + 1}")


def build_figure(paths, raw=False, limit=0.02):
    """Read once, average once, and use the same sigma values in 3D and 1D."""
    if not 1 <= len(paths) <= 2:
        raise ValueError("Compare one or two molecules per page; run again for others.")
    parameterization = openCOSMORS24a()
    surfaces = []
    # This section reads the surface and averages the segment charge densities
    # with the same radius used in plot_sigma_profiles.py. The 3D colors and the
    # profile below use the same sigma values and original segment areas. The
    # area check verifies that all segments are represented by the profile.
    for path in paths:
        surface = SigmaProfileParser(str(path))
        surface.calculate_averaged_sigmas(averaging_radius=parameterization.r_av)
        key = "seg_sigma_raw" if raw else "seg_sigma_averaged"
        sigma = np.asarray(surface[key])
        if not np.all(np.isfinite(sigma)) or np.any(surface["seg_area"] <= 0):
            raise ValueError(f"Invalid segment densities or areas in {path}")
        if np.max(np.abs(sigma)) > limit:
            raise ValueError(f"Color scale too narrow for {path.name}; increase --limit.")
        grid, areas = surface.cluster_and_create_sigma_profile(sigmas=key)
        if not np.isclose(areas.sum(), np.sum(surface["seg_area"]), atol=1e-7):
            raise ValueError("Profile grid does not capture the complete surface.")
        name = "water" if path.stem == "H2O_c000" else path.stem
        name = name.removesuffix(".solute").removesuffix("_c000").removesuffix("_CnfS1")
        surfaces.append((surface, sigma, grid, areas, name))
        print(f"{name}: {len(sigma)} segments, area = {areas.sum():.4f} A^2, "
              f"sigma = [{sigma.min():.6f}, {sigma.max():.6f}] e/A^2")

    titles = [f"{escape(name)} · {len(sigma)} segments · {areas.sum():.4f} Å²"
              for _, sigma, _, areas, name in surfaces]
    fig = make_subplots(
        rows=2, cols=len(paths), row_heights=[0.71, 0.29],
        specs=[[{"type": "scene"} for _ in paths], [{"type": "xy"} for _ in paths]],
        vertical_spacing=0.08, horizontal_spacing=0.07,
        subplot_titles=titles + ["σ-profile · area per bin" for _ in paths],
    )
    surface_ids, structure_ids = [], []
    # We use equal spatial scales and the same color limits for both molecules.
    # This lets us compare their sizes and charge densities without the viewer
    # automatically rescaling each molecule to fill its individual panel.
    span = max(np.ptp(s[0]["seg_pos"], axis=0).max() for s in surfaces) + 0.7
    ymax = max(s[3].max() for s in surfaces) * 1.12
    for col, (surface, sigma, grid, areas, name) in enumerate(surfaces, 1):
        vertices, faces, values, descriptions = patch_mesh(surface, sigma)
        surface_ids.append(len(fig.data))
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            intensity=values, text=descriptions, hovertemplate="%{text}<extra></extra>",
            coloraxis="coloraxis", opacity=0.65, flatshading=True,
            lighting=dict(ambient=1, diffuse=0, specular=0, fresnel=0),
            name=f"{name} surface", showlegend=False,
        ), row=1, col=col)
        for trace in structure_traces(surface):
            structure_ids.append(len(fig.data))
            fig.add_trace(trace, row=1, col=col)
        fig.add_trace(go.Scatter(
            x=grid, y=areas, mode="lines+markers", line=dict(color="#667085", width=1.5),
            marker=dict(size=5, color=grid, coloraxis="coloraxis"), showlegend=False,
            hovertemplate="σ = %{x:.3f} e/Å²<br>Area = %{y:.4f} Å²<extra></extra>",
        ), row=2, col=col)
        fig.update_xaxes(title_text="σ (e/Å²)", range=[-0.022, 0.022], row=2, col=col)
        fig.update_yaxes(title_text="Area (Å²)", range=[0, ymax], row=2, col=col)
        center = (surface["seg_pos"].min(axis=0) + surface["seg_pos"].max(axis=0)) / 2
        scene = dict(
            aspectmode="cube", bgcolor="#f6f8fb",
            camera=dict(eye=dict(x=0.15, y=-0.35, z=2.2),
                        projection=dict(type="orthographic")),
        )
        for axis, origin in zip("xyz", center):
            scene[f"{axis}axis"] = dict(visible=False, range=[origin - span/2, origin + span/2])
        fig.update_layout(**{f"scene{col if col > 1 else ''}": scene})

    # These buttons change only the display: surface transparency and whether
    # the ball-and-stick structure is visible. They do not recalculate the
    # molecular surface or alter the sigma profile displayed underneath it.
    menus = [dict(type="buttons", direction="right", x=0, y=1.13, xanchor="left",
                  buttons=[dict(label=label, method="restyle", args=[{"opacity": value}, surface_ids])
                           for label, value in [("Transparent", 0.3), ("Balanced", 0.65),
                                                ("Opaque", 1.0), ("Hide surface", 0.0)]], active=1),
             dict(type="buttons", direction="right", x=0.72, y=1.13, xanchor="left",
                  buttons=[dict(label="Toggle structure", method="restyle",
                                args=[{"visible": False}, structure_ids],
                                args2=[{"visible": True}, structure_ids])], showactive=False)]
    fig.update_layout(
        template="plotly_white", height=800, margin=dict(l=65, r=95, t=85, b=55),
        font=dict(family="Arial, sans-serif", color="#26334b", size=12),
        coloraxis=dict(colorscale=SIGMA_COLORS, cmin=-limit, cmax=limit,
                       colorbar=dict(title="σ (e/Å²)", len=0.62, y=0.72,
                                     tickvals=[-limit, 0, limit], tickformat=".3f")),
        updatemenus=menus, uirevision="keep-camera",
    )
    return fig, parameterization.r_av


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("surfaces", nargs="*", help="One or two molecule names or ORCA COSMO file paths")
    parser.add_argument("--source", choices=["calculated", "reference"], default="calculated")
    parser.add_argument("--output", type=Path, help="Optional HTML output path")
    parser.add_argument("--raw", action="store_true", help="Show unaveraged sigma instead")
    parser.add_argument("--limit", type=float, default=0.02, help="Symmetric color limit in e/A^2")
    args = parser.parse_args()
    selected = args.surfaces or ["water", "methanol"]
    if not 1 <= len(selected) <= 2:
        parser.error("Choose one or two molecules for each comparison page.")

    # Molecule names are resolved through the same shared file mapping used by
    # the other scripts. We can also pass an explicit .orcacosmo file to inspect
    # a different molecule. Each selected pair gets its own HTML filename.
    paths = [
        surface_paths(args.source, [item])[item] if item in MOLECULES
        else Path(item).expanduser().resolve()
        for item in selected
    ]
    labels = [path.stem.removesuffix(".solute") for path in paths]
    if args.output is None:
        args.output = PROJECT / "figures" / args.source / ("cosmo_" + "_".join(labels) + ".html")
    if not np.isfinite(args.limit) or args.limit <= 0:
        parser.error("--limit must be finite and positive")
    for path in paths:
        if not path.is_file():
            parser.error(f"Missing surface file: {path}")
    fig, radius = build_figure(paths, args.raw, args.limit)
    mode = "Raw segment σ" if args.raw else f"Averaged σ · openCOSMORS24a r_av = {radius:g} Å"
    fragment = fig.to_html(full_html=False, include_plotlyjs=True,
                           config=dict(responsive=True, displaylogo=False, scrollZoom=True))
    files = " · ".join(escape(p.name) for p in paths)
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>COSMO screening-charge surfaces</title>
<style>
body {{ margin: 0; background: #fff; color: #26334b; font: 15px/1.5 Arial, sans-serif; }}
header, footer {{ max-width: 1280px; margin: auto; padding: 20px 30px; }}
h1 {{ font-size: 26px; margin: 0 0 5px; }}
p {{ margin: 5px 0; }} .muted {{ color: #58677e; font-size: 13px; }}
.blue {{ color: #2166ac; }} .red {{ color: #b2182b; }}
.viewer {{ max-width: 1450px; min-width: 740px; margin: auto; }}
.scroll {{ overflow-x: auto; }}
</style></head><body>
<header><h1>COSMO screening-charge surfaces</h1>
<p>{mode} · Same color and spatial scales across molecules.</p>
<p><b class="blue">Negative σ: donor-side screening</b> · Near zero: weakly polar ·
<b class="red">Positive σ: acceptor-side screening</b></p>
<p class="muted">Drag to rotate · Scroll to zoom · Hover for σ and patch area ·
Oxygen atoms are red, carbon gray, nitrogen blue, hydrogen white.</p></header>
<div class="scroll"><div class="viewer">{fragment}</div></div>
<footer><p><b>Reading the display.</b> Colors represent surface screening-charge density,
not atomic charge or electrostatic potential. Negative screening charge occurs near
the positive molecular regions, such as hydroxyl hydrogens.</p>
<p class="muted">Patch outlines are area-equivalent tangent polygons, not original mesh
boundaries; gaps and overlaps are display approximations. The profiles use the original
segment areas and the same σ values as the 3D colors. The displayed ball sizes are schematic,
and sticks show connectivity rather than bond order. These are individual molecules,
not a simulated arrangement of a mixture.</p>
<p class="muted">Sources: {files}. Renderer: NumPy + Plotly + openCOSMO-RS Python.
This HTML embeds its plotting library and works offline.</p></footer></body></html>"""
    # The HTML contains the Plotly library as well as the molecular data. It can
    # therefore be opened directly in a browser without starting a server or
    # requiring an internet connection. Only the selected molecules are included.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
