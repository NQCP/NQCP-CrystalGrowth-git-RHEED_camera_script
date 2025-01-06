# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 17:06:45 2024

@author: qgh880
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_gaas_atoms(num_atoms):
    # Initialize positions for Ga and As atoms in two interpenetrating FCC lattices
    lattice_constant = 1.0
    lattice_positions_Ga = lattice_constant * np.array([[i, j, k] for i in range(num_atoms) for j in range(num_atoms) for k in range(num_atoms) if (i + j + k) % 2 == 0])
    lattice_positions_As = lattice_constant * np.array([[i, j, k] for i in range(num_atoms) for j in range(num_atoms) for k in range(num_atoms) if (i + j + k) % 2 == 1]) + lattice_constant * np.array([0.5, 0.5, 0.5])

    return lattice_positions_Ga, lattice_positions_As

def plot_gaas_atoms(atom_positions_Ga, atom_positions_As):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(atom_positions_Ga[:, 0], atom_positions_Ga[:, 1], atom_positions_Ga[:, 2], marker='o', s=50, c='b', label='Ga')
    ax.scatter(atom_positions_As[:, 0], atom_positions_As[:, 1], atom_positions_As[:, 2], marker='^', s=50, c='r', label='As')
    ax.view_init(elev = 10, azim = 190)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('GaAs Zincblende Crystal Structure')
    ax.legend()

    plt.show()

def main():
    num_atoms = 3

    # Generate GaAs zincblende crystal structure with 10 atoms
    Ga_positions, As_positions = generate_gaas_atoms(num_atoms)

    # Plot the GaAs zincblende crystal structure
    plot_gaas_atoms(Ga_positions, As_positions)

if __name__ == "__main__":
    main()