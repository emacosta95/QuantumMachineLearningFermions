"""Study angular-momentum projection as the Euler grid is enlarged.

For every pair M=1..M_MAX and J=1..J_MAX, the Euler quadrature is (M,J,M).
The JSON output stores projected non-Gaussianity (FAF), fidelity, energy error,
<J^2>, effective J, and the exact ground-state non-Gaussianity.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
for directory in (ROOT / "src" / "NSMFermions", ROOT / "benchmarks"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from projection_grid_convergence import main as run_grid_study  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Euler-grid convergence of projected non-Gaussianity."
    )
    parser.add_argument(
        "--interaction", type=str.lower, choices=("cki", "usdb"), required=True
    )
    parser.add_argument(
        "--mass", type=int,
        help="Defaults to Be-8 for CKI and Ne-20 for USDB.",
    )
    parser.add_argument("--m-max", type=int, required=True)
    parser.add_argument("--j-max", type=int, required=True)
    parser.add_argument("--intrinsic-method", choices=("hf", "hfb"), default="hfb")
    parser.add_argument("--starts", type=int, default=4)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--faf-order", type=int, default=2)
    parser.add_argument("--number-grid", nargs=2, type=int, metavar=("LN", "LZ"))
    parser.add_argument("--allow-unconverged-intrinsic", action="store_true")
    parser.add_argument("--output-dir", default=str(ROOT / "results"))
    args = parser.parse_args()
    run_grid_study(
        interaction_name=args.interaction,
        mass=args.mass or (8 if args.interaction == "cki" else 20),
        m_max=args.m_max,
        j_max=args.j_max,
        intrinsic_method=args.intrinsic_method,
        starts=args.starts,
        maxiter=args.maxiter,
        seed=args.seed,
        faf_order=args.faf_order,
        allow_unconverged_intrinsic=args.allow_unconverged_intrinsic,
        number_grid=tuple(args.number_grid) if args.number_grid else None,
        output_dir=args.output_dir,
    )
