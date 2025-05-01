#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import logging
import math
import scipy.stats.qmc as qmc

logger = logging.getLogger(__name__)

VERSION = "0.1.0"

parser = argparse.ArgumentParser(
    description="""
    Calculate rate coefficients from cross sections for a reaction.
    Takes in a file with at least two columns. The first column is the energy in eV, and the second column is the cross section.
    The units of the cross section are assumed to be in square meters, but this can be changed with the --cross-section-scale option.
    The output is a file with the rate coefficients of the reaction in a specified temperature range, again in a two-column format. 
    """
)

parser.add_argument("cross_section_file", type=str, help="Input file containing cross section data.")
parser.add_argument("--energy", "-e", type=float, help="Reaction energy in eV. If not supplied, the first provided energy in the cross section file will be used.")
parser.add_argument("--cross-section-scale", "-s", type=float, default=1.0, help="Scale factor for cross section data. Default is 1.0.")
parser.add_argument("--spacing", "-S", type=str, choices=["linear", "log"], default="linear", help="Spacing of the temperature points. Default is 'linear'.")
parser.add_argument("--output-file", "-o", type=str, default="rate_coeffs.dat", help="Output file for rate coefficients. Default is 'rate_coeffs.dat'.")
parser.add_argument("--skipheader", "-H", type=int, default=0, help="Number of header lines to skip in the input file. Default is 1")
parser.add_argument("--delimiter", "-D", type=str, default=",", help="Delimiter for input file. Default is ','.")
parser.add_argument("--loglevel", "-l", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="WARNING", help="Set the logging level. Default is WARNING.")
parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {VERSION}", help="Show program version and exit.")

q_e = 1.602176634e-19  # J/eV
m_e = 9.10938356e-31  # kg

def compute_rate_coefficients(temperatures_eV, energy_eV, sigma_m2, num_samples=8192):
    """Integrate cross sections over maxwellian VDF to obtain reaction rate coefficients
    Given electron energy in eV (`energy_eV`) and reaction cross sections at those energies (`sigma_m2`),
    this function computes the reaction rate coefficient $k(T_e)$ for maxwellian electrons at
    a provided list of electron temperatures `temperatures_eV`.

    The rate coefficient is given by

    $$
    k(T_e) = \\int \\sigma(c) c dc
    $$

    where the speed $c$ is drawn from a 3D maxwellian distribution function with zero mean velocity and temperature $T_e$.
    We solve this using a quasi-Monte Carlo approach, drawing a large number of low-discrepancy samples from
    the appropriate distribution and obtaining the average of $\\sigma(c) c$.
    """
    thermal_speed_scale = np.sqrt(q_e / m_e)
    k = np.zeros(temperatures_eV.size)

    # obtain low-discrepancy samples of normal dist
    dist = qmc.MultivariateNormalQMC(np.zeros(3), np.eye(3))
    v = dist.random(num_samples)
    speed_squared = (v[:, 0] ** 2 + v[:, 1] ** 2 + v[:, 2] ** 2)
    speed = np.sqrt(speed_squared)

    for i, T in enumerate(temperatures_eV):
        # scale velocities to proper temperature
        # compute energies corresponding to each sampled velocity vector
        e = 0.5 * speed_squared * T
        c = speed * thermal_speed_scale * np.sqrt(T)
        # get cross section by interpolating on table
        sigma = np.interp(e, energy_eV, sigma_m2, left=0)
        k[i] = np.mean(sigma * c)
    return k

def main(args: argparse.Namespace) -> None:
    if args.loglevel:
        logging.basicConfig(level=args.loglevel.upper())
        logger.setLevel(args.loglevel.upper())
        logger.info(f"Logging level set to {args.loglevel.upper()}")

    logger.info("Reading cross section data from {args.cross_section_file}...")

    # Read cross section data
    delimiter = args.delimiter
    if args.delimiter == "tab":
        delimiter = "\t"

    data = np.loadtxt(args.cross_section_file, delimiter=delimiter, skiprows=args.skipheader)
    energy = data[:, 0]
    cross_section = data[:, 1] * args.cross_section_scale

    # Use quase-monte carlo method to calculate rate coefficients
    max_energy = np.max(energy)
    mean_energy_eV = np.linspace(0, max_energy, math.ceil(max_energy + 1), dtype=int)
    temperature_eV = 2/3 * mean_energy_eV

    rate_coefficients = compute_rate_coefficients(temperature_eV, energy, cross_section)

    # Write the rate coefficients to the output file
    logger.info(f"Writing rate coefficients to {args.output_file}...")
    
    energy = args.energy if args.energy is not None else energy[0]
    with open(args.output_file, "w") as f:
        print(f"Ionization energy (eV): {energy}", file=f)
        print("Energy (eV)\tRate coefficient (m^3/s)", file=f)
        for T, k in zip(mean_energy_eV, rate_coefficients):
            f.write(f"{T:.1f}\t{k:.6e}\n")
    logger.info(f"Rate coefficients written to {args.output_file}.")
    logger.info("Done.")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)