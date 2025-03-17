import os
from ase.calculators.vasp import Vasp
from ase.io import read, write, Trajectory
from ase.db import connect
from shutil import copy
import os, subprocess, sys 
from clease.tools import update_db
import numpy as np
import argparse
import json
import toml
from pathlib import Path
from ase.calculators.calculator import Calculator
from ase.calculators.calculator import CalculationFailed
from perqueue.constants import INDEX_KW
import logging

from ase import Atom

from ase.constraints import ExpCellFilter
from ase.optimize.bfgs import BFGS
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.optimize.fire import FIRE
from ase.optimize.lbfgs import LBFGS, LBFGSLineSearch
from ase.optimize.mdmin import MDMin
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase import Atoms, units

from ase.io import Trajectory
from ase.optimize.optimize import Optimizer

from utils import Relaxer, Cathode

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description=" Single relaxation of structures", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--run_path",
        type=str,
        help="Path to run path",
    )
    parser.add_argument(
        "--db_path", 
        type=str, 
        help="Path to the ase database",
    )
    parser.add_argument(
        "--index",
        type=int,
        help="Index of the structure in the ase database",
    )
    parser.add_argument(
        "--cfg_pth",
        type=str,
        help="Path to the toml config file",
    )
    return parser.parse_args(arg_list)


def main():
    args = get_arguments()
    run_path = args.run_path
    db_path = args.db_path
    db_id = args.index
    cfg_pth = args.cfg_pth

    # Load the database:
    db = connect(db_path)

    # Load the structure
    row = db.get(id=db_id)
    atom = row.toatoms()
    name = row.name

    # Load the parameters
    with open(cfg_pth, 'r') as f:
        params = toml.load(f)

    # Remove the vaccancies
    X_indice = [a.index for a in atom if a.symbol == 'X']
    del atom[X_indice]

    # setting and creating the directory for the saved files
    relax_directory = f'{run_path}/relaxation'
    relaxsim_directory =f'{relax_directory}/{name}'
    try:
        os.makedirs(relaxsim_directory)
    except:
        pass

    # Setting up logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    log_path = relaxsim_directory+'/relaxation.log'
    runHandler = logging.FileHandler(log_path, mode='w')
    runHandler.setLevel(logging.DEBUG)
    runHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))
    logger.addHandler(runHandler)
    logger.info(f'Optimizing: {name}')

    #### DFT optimization #####
    if params['method'] == 'VASP':

        #### Cathode materials ####
        try:
            cathode_params = params['cathode']
        except:
            cathode_params = None
        if cathode_params:
            # The amount of the moving ion in the structure
            ion = cathode_params['ion']
            M_ions = cathode_params['M_ions']
            psudo_M_ion = cathode_params['pusdo_M_ion']
            ion_len = np.sum(np.array(atom.get_chemical_symbols()) == ion)

            # Finding all index transition metal ions (M_ion) in the system
            M_ion_index = [a.index for a in atom if a.symbol in M_ions or a.symbol in psudo_M_ion]

            cathode_obj = Cathode()

            # Set magnetic moment
            # Remove Ga or other cheater ions and add the Metalic ion with correct magnetic moment
            # Note that all funcitons are taken from chemical_values.py
            for i, ind in enumerate(M_ion_index):
                M_ion_i = atom.symbols[ind]
                # Set the magnetic moment for the redox atom (+3)
                if M_ion_i in psudo_M_ion: 
                    atom.symbols[ind] = cathode_obj.remove_redox_psudo(M_ion_i) 
                    atom[ind].magmom = cathode_obj.get_magmom(atom.symbols[ind],redox=True) # Set 3+ ion
                # Set the magnetic moment for the other metalic ions (+2)
                else:
                    atom[ind].magmom = cathode_obj.get_magmom(M_ion_i,redox=False)

            # Set U-values for the DFT calculation
            # Define U-value
            ldau_luj = {'ldau_luj':{}}
            if type(M_ions)==str:
                ldau_luj['ldau_luj'][M_ions] = {'L': 2, 'U': cathode_obj.get_U_value(M_ions),'J':0}
                logger.info(f'{name} has L, U, J values: (2, {cathode_obj.get_U_value(M_ions)}, 0)')
            else:
                for m in M_ions:
                    ldau_luj['ldau_luj'][m] = {'L': 2, 'U': cathode_obj.get_U_value(m),'J':0}
                    logger.info(f'{name} has L, U, J values: (2, {cathode_obj.get_U_value(m)}, 0)')
        else:
            ldau_luj = None
        ##############

        # Vasp calculator
        vasp_params = params['VASP']

        # Set total magmom for the structure if relevant
        tot_magmom = 0
        for a in atom:
            tot_magmom += a.magmom
        if tot_magmom != 0:
            vasp_params['nupdown'] = tot_magmom
            logger.info(f'{name} has nupdown {tot_magmom}')

        # Set the VASP calculator
        calc = Vasp(directory=relaxsim_directory,**vasp_params)
        
        # Set the LDAU values if relevant
        if ldau_luj:
            calc.set(**ldau_luj)
            logger.info(f'{name} has LDAU values: {ldau_luj}')

        # Set th VASP calcualtor
        atom.set_calculator(calc)

        # Start the calculation for structure optimization.
        atom.get_potential_energy()
        
        # Check if the relaxation have reached required accuracy
        with open(relaxsim_directory+'/OUTCAR') as file:
            # read all lines using readline()
            lines = file.readlines()
            try:
                lines.index(' reached required accuracy - stopping structural energy minimisation\n')
                var= True
            except:
                var=False
                
        if not var:
            logger.info(f'Relaxation did not converge. Fmax: {max(np.sqrt(np.sum(np.square(atom.get_forces()),axis=1)))}' )
            raise CalculationFailed('Relaxation did not converge')

    #### ML optimization ####
    else:
        # Set parameters 
        calc_name = params['method']
        calc_path = params['calc_path']
        optimizer = params['optimizer']
        relax_cell = params['relax_cell']
        traj_path = relaxsim_directory+'/opt.traj'
        log_path = relaxsim_directory+'/opt.log'
        fmax = np.abs(params['VASP']['ediffg'])
        max_step = params['max_step']
        # get calculator and set it to atoms object 
        relaxer = Relaxer(calc_name=calc_name,calc_paths=calc_path,
                          optimizer=optimizer,relax_cell=relax_cell,device='cuda',
                          fmax=fmax,steps=max_step,traj_file=traj_path,log_file=log_path,interval=1)

        relax_results=relaxer.relax(atom)
        final_structure = relax_results["final_structure"]
        final_energy = final_structure.get_potential_energy() #relax_results["trajectory"].energies[-1]
        force = np.sqrt(np.sum(((final_structure.get_forces())**2),axis=1))
        fmax_relax = np.max(force)

        logger.info(f'Final structure: {final_structure}')
        logger.info(f"The final energy is {float(final_energy):.3f} eV.")
        logger.info(f"The maximum force is {fmax_relax:.3f} eV/Å.")

        # Check if the relaxation have reaxhed required accuracy
        #traj = Trajectory(traj_path)
        if fmax_relax > fmax:
            var = False
            logger.info(f'Relaxation did not converge. Fmax: {fmax_relax}' )
            raise CalculationFailed('Relaxation did not converge')

    # Update database 
    update_db(uid_initial=db_id, final_struct=atom, db_name=db_path)

if __name__ == "__main__":
    main()