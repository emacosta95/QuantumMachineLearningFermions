"""Dedicated multi-start Hartree-Fock driver for USDB Neon isotopes."""

import argparse

from usdb_ne_pav_faf_components import main as run_neon_study


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--isotopes", nargs="+", type=int, default=[20, 22, 24])
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--output-dir")
    arguments = parser.parse_args()
    run_neon_study(
        isotopes=tuple(arguments.isotopes),
        starts=arguments.starts,
        maxiter=arguments.maxiter,
        seed=arguments.seed,
        hfb_only=True,
        hartree_fock=True,
        output_dir=arguments.output_dir,
    )
