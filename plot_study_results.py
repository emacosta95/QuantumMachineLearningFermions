"""Plot JSON reports produced by the root nuclear-structure study scripts.

The report type is detected automatically:

* ``study_gaussian_fidelity.py`` -> isotope comparison (four bar panels);
* legacy Euler-grid reports -> projection-convergence heat maps.

The plotting code deliberately depends only on NumPy and Matplotlib so it also
works in a headless Slurm job.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

# A non-interactive backend works both on laptops and compute nodes.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _values(rows, key, legacy_key=None):
    """Return a float array, representing absent/None entries as NaN."""
    values = []
    for row in rows:
        value = row.get(key)
        if value is None and legacy_key is not None:
            value = row.get(legacy_key)
        values.append(np.nan if value is None else float(value))
    return np.asarray(values)


def _positive(values):
    """Make an array safe for logarithmic plotting."""
    values = np.asarray(values, dtype=float)
    return np.where(values > 0.0, values, np.nan)


def plot_gaussian_fidelity(report):
    """Build the four-panel isotope comparison figure."""
    rows = report.get("results", [])
    if not rows:
        raise ValueError("The Gaussian-fidelity report has no completed results")

    labels = [str(row["nucleus"]) for row in rows]
    x = np.arange(len(rows), dtype=float)
    has_component_fidelities = all(
        row.get("closest_bogoliubov_fidelity") is not None
        and row.get("closest_hartree_fock_fidelity") is not None
        for row in rows
    )
    width = 0.25 if has_component_fidelities else 0.34
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    fidelity_series = (
        (
            "energy-optimized HFB/HF",
            "variational_ground_state_fidelity",
            "variational_ground_state_fidelity_raw",
            "C0",
        ),
        (
            "maximum-overlap Bogoliubov",
            "closest_bogoliubov_fidelity",
            None,
            "C1",
        ),
        (
            "maximum-overlap HF",
            "closest_hartree_fock_fidelity",
            None,
            "C2",
        ),
    ) if has_component_fidelities else (
        ("energy-optimized HFB/HF", "variational_ground_state_fidelity",
         "variational_ground_state_fidelity_raw", "C0"),
        ("maximum-overlap Gaussian", "closest_gaussian_ground_state_fidelity",
         "closest_gaussian_ground_state_fidelity_raw", "C1"),
    )
    for index, (label, key, legacy_key, color) in enumerate(fidelity_series):
        axes[0, 0].bar(
            x + (index - (len(fidelity_series) - 1) / 2) * width,
            _values(rows, key, legacy_key),
            width,
            label=label,
            color=color,
        )
    axes[0, 0].set_title("Intrinsic ground-state fidelity")
    axes[0, 0].set_ylabel(r"$|\langle\Psi_0|\Phi\rangle|^2$")
    axes[0, 0].set_ylim(0.0, 1.05)
    axes[0, 0].legend(fontsize=8)

    energy_width = 0.25
    axes[0, 1].bar(
        x - energy_width,
        _values(rows, "exact_energy"),
        energy_width,
        label="exact",
        color="black",
    )
    axes[0, 1].bar(
        x,
        _values(rows, "variational_energy"),
        energy_width,
        label="HFB/HF",
        color="C0",
    )
    axes[0, 1].bar(
        x + energy_width,
        _values(rows, "closest_gaussian_energy"),
        energy_width,
        label="maximum overlap",
        color="C1",
    )
    axes[0, 1].set_title("Energy comparison")
    axes[0, 1].set_ylabel("energy")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].bar(
        x - width / 2,
        _positive(_values(rows, "variational_energy_relative_error")),
        width,
        label="HFB/HF",
        color="C0",
    )
    axes[1, 0].bar(
        x + width / 2,
        _positive(_values(rows, "closest_gaussian_energy_relative_error")),
        width,
        label="max overlap (diagnostic energy)",
        color="C1",
    )
    axes[1, 0].set_title("Energy relative error")
    axes[1, 0].set_ylabel(r"$|E-E_0|/|E_0|$")
    axes[1, 0].set_yscale("log")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].bar(
        x - width / 2,
        _positive(_values(rows, "variational_stationarity_error")),
        width,
        label=r"HFB/HF $\|G^{20}\|$",
        color="C0",
    )
    axes[1, 1].bar(
        x + width / 2,
        _positive(_values(rows, "closest_gaussian_gradient_norm")),
        width,
        label="max-overlap gradient",
        color="C1",
    )
    axes[1, 1].set_title("Optimization residual")
    axes[1, 1].set_ylabel("gradient norm")
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend(fontsize=8)

    for axis in axes.flat:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)

    interaction = str(report.get("interaction", "unknown")).upper()
    status = str(report.get("status", "unknown"))
    nucleus = f" {labels[0]}" if len(labels) == 1 else " isotope chain"
    fig.suptitle(
        f"{interaction}{nucleus} Gaussian comparison ({status})", fontsize=14
    )
    return fig


def _grid(rows, key, m_values, j_values):
    values = np.full((len(j_values), len(m_values)), np.nan)
    m_index = {value: index for index, value in enumerate(m_values)}
    j_index = {value: index for index, value in enumerate(j_values)}
    for row in rows:
        value = row.get(key)
        if value is not None:
            values[j_index[int(row["J_grid_points"])],
                   m_index[int(row["M_grid_points"])]] = float(value)
    return values


def plot_projection_grid(report):
    """Build four Euler-grid convergence heat maps."""
    rows = [row for row in report.get("grid_rows", []) if "error" not in row]
    if not rows:
        raise ValueError("The projection report has no successful grid rows")
    m_values = sorted({int(row["M_grid_points"]) for row in rows})
    j_values = sorted({int(row["J_grid_points"]) for row in rows})
    panels = (
        ("fidelity", "Projected ground-state fidelity", "viridis", 0.0, 1.0),
        ("projected_energy_relative_error", "Energy relative error", "magma_r", None, None),
        ("effective_J", r"Effective $J$", "cividis", None, None),
        ("faf_difference_from_exact_ground_state", "FAF minus exact-ground-state FAF", "coolwarm", None, None),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    extent = (min(m_values) - 0.5, max(m_values) + 0.5,
              min(j_values) - 0.5, max(j_values) + 0.5)
    for axis, (key, title, cmap, vmin, vmax) in zip(axes.flat, panels):
        values = _grid(rows, key, m_values, j_values)
        if key == "faf_difference_from_exact_ground_state":
            scale = np.nanmax(np.abs(values))
            if np.isfinite(scale) and scale > 0:
                vmin, vmax = -scale, scale
        image = axis.imshow(
            values,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(title)
        axis.set_xlabel(r"$N_\alpha=N_\gamma=M$")
        axis.set_ylabel(r"$N_\beta=J_{\rm grid}$")
        axis.set_xticks(m_values)
        axis.set_yticks(j_values)
        fig.colorbar(image, ax=axis, shrink=0.85)
    interaction = str(report.get("interaction", "unknown")).upper()
    nucleus = str(report.get("nucleus", "unknown"))
    status = str(report.get("status", "unknown"))
    fig.suptitle(f"{interaction} {nucleus} projection-grid convergence ({status})",
                 fontsize=14)
    return fig


def plot_report(report):
    """Detect the JSON schema and return the corresponding figure."""
    if "grid_rows" in report:
        return plot_projection_grid(report)
    if "results" in report:
        return plot_gaussian_fidelity(report)
    raise ValueError(
        "Unknown report format: expected 'results' or 'grid_rows'"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Gaussian-fidelity or projection-grid JSON data."
    )
    parser.add_argument("report", help="JSON report produced by a study script")
    parser.add_argument(
        "--output",
        help="PNG/PDF/SVG path (default: report name with .png extension)",
    )
    parser.add_argument(
        "--per-isotope",
        action="store_true",
        help="also save a separate four-panel figure for every isotope",
    )
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    report_path = Path(args.report).expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else report_path.with_suffix(".png")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = plot_report(report)
    figure.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output_path}")

    if args.per_isotope:
        if "results" not in report:
            parser.error("--per-isotope applies only to Gaussian-fidelity reports")
        for row in report.get("results", []):
            isotope_report = dict(report)
            isotope_report["results"] = [row]
            isotope_figure = plot_gaussian_fidelity(isotope_report)
            isotope_path = output_path.with_name(
                f"{output_path.stem}_{str(row['nucleus']).lower()}"
                f"{output_path.suffix}"
            )
            isotope_figure.savefig(
                isotope_path, dpi=args.dpi, bbox_inches="tight"
            )
            plt.close(isotope_figure)
            print(f"Wrote {isotope_path}")


if __name__ == "__main__":
    main()
