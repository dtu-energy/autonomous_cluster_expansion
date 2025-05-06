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

def main(run_path,db_path,run_list,cfg_pth,**kwargs):
    
    # Load perqueue index
    idx, *_ =kwargs[INDEX_KW]
    #return True, {'initial_start':False}

    # Load the database:
    db = connect(db_path)

    # Load the structure
    db_id = run_list[idx] # pq_index to map the correct id
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
        cathode_obj = Cathode()
        cathode_params = params['cathode']
        # The amount of teh moving ion in the structure
        ion = cathode_params['ion']
        M_ions = cathode_params['M_ions']
        ion_len = np.sum(np.array(atom.get_chemical_symbols()) == ion)
        # Finding all index transition metal ions (M_ion) in the system
        M_ion_all = [a.index for a in atom if a.symbol in M_ions ]

        # Set magnetic moment
        N_Na_max = len(M_ion_all) # CHECK!!! Works if N_Na_max= N_M_ion
        N_redox = N_Na_max -ion_len
        tot_magmom = 0
        # Sort the atom:
        atom, i = cathode_obj.sort(atom, key=cathode_obj.redox_sort_func)
        count = 0
        for a in atom:
            if isinstance(M_ions, str):
                if a.symbol == M_ions:
                    if count < N_redox:
                        a.magmom = cathode_obj.get_magmom(M_ions,redox=True)
                        tot_magmom += cathode_obj.get_magmom(M_ions,redox=True)
                    else:
                        a.magmom = cathode_obj.get_magmom(M_ions,redox=False)
                        tot_magmom += cathode_obj.get_magmom(M_ions,redox=False)
                    count += 1
                else:
                    a.magmom = 0
            else:
                if a.symbol in M_ions:
                    if count < N_redox:
                        a.magmom = cathode_obj.get_magmom(a.symbol,redox=True)
                        tot_magmom += cathode_obj.get_magmom(a.symbol,redox=True)
                    else:
                        a.magmom = cathode_obj.get_magmom(a.symbol,redox=False)
                        tot_magmom += cathode_obj.get_magmom(a.symbol,redox=False)
                    count += 1
                else:
                    a.magmom = 0
        
        # Define the Na concentration
        Na_conc = ion_len /len(M_ion_all)

        logger.info(f"Total magnetic moment: {tot_magmom}")
        logger.info(f"Na concentration: {Na_conc}")
        logger.info(f"M_ion: {M_ions}")
        logger.info(f"Na_len: {ion_len}")
        logger.info(f"Na_max: {N_Na_max}")

        # Vasp calculator
        vasp_params = params['VASP']

        vasp_params['nupdown'] = tot_magmom
        logger.info(f'{name} has nupdown {tot_magmom}')
        calc = Vasp(directory=relaxsim_directory,**vasp_params)
        
        # Define U-value
        ldau_luj = {'ldau_luj':{}}
        if type(M_ions)==str:
            ldau_luj['ldau_luj'][M_ions] = {'L': 2, 'U': cathode_obj.get_U_value(M_ion),'J':0}
            logger.info(f'{name} has L, U, J values: (2, {cathode_obj.get_U_value(M_ion)}, 0)')
        else:
            for m in M_ions:
                ldau_luj['ldau_luj'][m] = {'L': 2, 'U': cathode_obj.get_U_value(m),'J':0}
                logger.info(f'{name} has L, U, J values: (2, {cathode_obj.get_U_value(m)}, 0)')
        calc.set(**ldau_luj)

        # Set th VASP calcualtor
        atom.set_calculator(calc)

        # Start the calculation for structure optimization.
        try:
            atom.get_potential_energy()
        except CalculationFailed:
            logger.info(f"Calculation failed for {name}")
            return_parameters = {}
            return True, return_parameters
        
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
            return_parameters = {}
            logger.info(f'Relaxation did not converge. Fmax: {max(np.sqrt(np.sum(np.square(atom.get_forces()),axis=1)))}' )
            return True, return_parameters

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
            return_parameters = {}
            return True, return_parameters

    # Update database 
    update_db(uid_initial=db_id, final_struct=atom, db_name=db_path)
    return_parameters = {}
    return True, return_parameters
if __name__ == "__main__":
    main()