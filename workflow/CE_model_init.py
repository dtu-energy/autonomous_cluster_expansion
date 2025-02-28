
from clease.tools import reconfigure
from clease import  NewStructures,Evaluate
from clease.settings import Concentration, CECrystal, CEBulk
from clease.tools import reconfigure
from clease.regression import PhysicalRidge,LinearRegression
from clease.basis_function import BinaryLinear
from clease.regression.physical_ridge import random_cv_hyper_opt
from clease.settings import settings_from_json
import os
import logging
logging.basicConfig(level=logging.INFO)
from clease.corr_func import CorrFunction
import clease.plot_post_process as pp

import json# from tqdm import tqdm
import itertools
from ase.visualize import view
import shutil
from ase.io import read
import numpy as np
from ase.db import connect
import matplotlib.pyplot as plt
from spglib import get_spacegroup, find_primitive, standardize_cell
from ase.io import read
from ase.visualize import view
from ase import Atoms

import toml
from perqueue.constants import DYNAMICWIDTHGROUP_KEY,CYCLICALGROUP_KEY, ITER_KW

def main(run_path, db_path, cif_file ,cfg_path, random_seed=27,**kwargs):

    # Load the CE model parameters
    with open(cfg_path, 'r') as f:
        CE_model_dict = toml.load(f)['CE']

    ## Setting up logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    log_path = run_path+'/CE_init.log'
    print(log_path)
    runHandler = logging.FileHandler(log_path, mode='w')
    runHandler.setLevel(logging.DEBUG)
    runHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))
    logger.addHandler(runHandler)
    logger.info(f'CE parameters: {CE_model_dict}')
    
    # Load cif file 
    atom_cif = read(cif_file)
    if CE_model_dict['auto_set_cell']:
        cif_cell = (atom_cif.cell, atom_cif.get_scaled_positions(), atom_cif.numbers)
        # Define space group
        space_group_str = get_spacegroup(cif_cell,
                symprec=1e-5)
        logger.info(f'Space group is not defined in the CE model, found space group: {space_group_str}')
        # Set up the new cell
        new_cell = standardize_cell(cif_cell, to_primitive=True, symprec=5e-1)
        new_unit_cell, new_scaled_positions, new_numbers = new_cell
        new_atoms = Atoms(new_numbers, cell=new_unit_cell, scaled_positions=new_scaled_positions)
        atom_cif = new_atoms

        # Define cellpar and space group
        cellpar = atom_cif.cell.cellpar()
        space_group = int(space_group_str.split('(')[-1].split(')')[0])

        # Setup the CE model        
        basis_element = []
        basis_element_dict = CE_model_dict['basis_element']
        basis_coord = []
        basis_index = {}
        for a in atom_cif:
            # Set coordinates
            basis_coord.append(a.position)
            if a.symbol in list(basis_element_dict.keys()):
                basis_element.append(basis_element_dict[a.symbol])
            else:
                basis_element.append([a.symbol])

            # Set index
            if a.symbol not in basis_index.keys():
                basis_index[a.symbol] = []
            basis_index[a.symbol].append(a.index)
        
        # Set basis index list
        basis_index_list = [ basis_index[group] for group in basis_index.keys()]
    else:
        # Define cellpar and space group
        cellpar = atom_cif.cell.cellpar()
        space_group = CE_model_dict['space_group']
        basis_element = CE_model_dict['basis_element']
        basis_index_list = CE_model_dict['basis_index']
        basis_coord = CE_model_dict['basis_coord']


    logger.info(f'Atom: {atom_cif.get_chemical_formula()}, Space group: {space_group}, Cell parameters: {cellpar}')
    logger.info(f'Basis element: {basis_element}')
    logger.info(f'Basis index:{basis_index_list}')
    logger.info(f'Basis coord: {basis_coord}')

    # Define Concentration object
    # Assume that the basis order and basis index are corresponding
    conc = Concentration(basis_elements=basis_element,grouped_basis=basis_index_list)

    # Define concentration range
    conc.A_eq = np.array(CE_model_dict['A_eq'], dtype=int)
    conc.b_eq = np.array(CE_model_dict['b_eq'], dtype=int)
    # Delete db if it exists
    if os.path.isfile(db_path):
        os.remove(db_path)

    # Define the crystal object
    setting = CECrystal(cellpar=cellpar,
                basis=basis_coord,
                concentration=conc,
                spacegroup=space_group,

                db_name=db_path,
                max_cluster_dia=CE_model_dict['max_cluster_dia'])

    setting.basis_func_type=CE_model_dict['basis_func_type']

    # Set the size of the cell to sample from
    if CE_model_dict['size'] is not None:
        setting.size=CE_model_dict['size']
    else:
        setting.supercell_factor = CE_model_dict['supercell_factor']
    
    # Create the initial randomstructures
    db = connect(db_path)
    init_db_len = len(db)
    n_random_struc = CE_model_dict['n_random_struc']
    logger.info(f'Initial start, creating {n_random_struc} random structures')
    ns = NewStructures(setting, struct_per_gen=n_random_struc, generation_number=0)
    ns.generate_random_structures()
    
    # Save initial settings
    setting.save(run_path+'/initial_setting.json')

    # Return the dynamic width group key and the run path
    dmkey = n_random_struc
    run_list = []
    for i in range(dmkey):
        db_id = i + init_db_len +1 # because we index with 1 in ase db
        run_list.append(db_id)
    logger.info(f'Submitting: {run_list}')

    return_parameters = {DYNAMICWIDTHGROUP_KEY: dmkey,'run_list':run_list}
    return True , return_parameters


if __name__ == "__main__":
    main()