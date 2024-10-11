from clease.tools import reconfigure
from clease import  NewStructures,Evaluate
from clease.settings import Concentration, CECrystal, CEBulk
from clease.tools import reconfigure
from clease.regression import PhysicalRidge,LinearRegression
from clease.basis_function import BinaryLinear
from clease.regression.physical_ridge import random_cv_hyper_opt
from clease.settings import settings_from_json

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

def main(run_path, db_path, initial_start, cif_file ,cfg_path, random_seed=27,**kwargs):
    import os
    print(os.environ['CONDA_DEFAULT_ENV'])

    # Load the CE model parameters
    with open(cfg_path, 'r') as f:
        CE_model_dict = toml.load(f)['CE']

    #### Set initial settings for CE model ####
    if initial_start:
       
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
    ################

    #### CE model training ####
    else:
        # Define iteration number
        iter_idx, *_ =kwargs[ITER_KW]
        iter_idx = iter_idx + 1

        ## Setting up logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        log_path = run_path+f'/CE_{iter_idx}.log'
        runHandler = logging.FileHandler(log_path, mode='w')
        runHandler.setLevel(logging.DEBUG)
        runHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))
        logger.addHandler(runHandler)
        
        
        # Load initial settings and set the basis function type and size
        logger.info('Loading initial settings')
        setting = settings_from_json(run_path+'/initial_setting.json')
        setting.basis_func_type=CE_model_dict['basis_func_type'] #: polynomial, trigonometric and binaryLinear.

        try: 
            setting.size=CE_model_dict['size']
        except:
            setting.supercell_factor = CE_model_dict['supercell_factor']

        # Setting the correlation function
        logger.info('Setting up the correlation function')
        cf = CorrFunction(setting)
        cf.reconfigure_db_entries(verbose=True)

        # Set up the evaluation
        try:
            select_cond = CE_model_dict['select_cond']
        except:
            select_cond = None
        scoring_scheme = CE_model_dict['scoring_scheme']
        max_cluster_size = CE_model_dict['max_cluster_size']
        evl=Evaluate(setting,select_cond = select_cond,scoring_scheme=scoring_scheme, max_cluster_size=max_cluster_size)

        # scan different values of alpha and return the value of alpha that yields
        # the lowest CV score
        fiting_scheme = CE_model_dict['fitting_scheme']
        alpha_min = CE_model_dict['alpha_min']
        alpha_max = CE_model_dict['alpha_max']
        num_alpha = CE_model_dict['num_alpha']
        evl.set_fitting_scheme(fitting_scheme=fiting_scheme)
        alpha = evl.plot_CV(alpha_min=alpha_min, alpha_max=alpha_max, num_alpha=num_alpha, savefig=True, fname=run_path+f'/alpha_iter{iter_idx}.png')
        
        # set the alpha value with the one found above, and fit data using it.
        logger.info('Fitting the CE model')
        evl.set_fitting_scheme(fitting_scheme=fiting_scheme, alpha=alpha)
        evl.fit()  # Run the fit with these settings.
        logger.info('Fitting done')
        evl.plot_fit(interactive=False, savefig=True, fname=run_path+f'/fit_new_iter{iter_idx}.png')
        # Plot fit
        fig = pp.plot_fit(evl)#, interactive=True)
        fig.savefig(run_path+f'/fit_iter{iter_idx}.png')
        # plot ECI values
        fig = pp.plot_eci(evl)
        fig.savefig(run_path+f'/eci_iter{iter_idx}.png')

        # save a dictionary containing cluster names and their ECIs
        evl.save_eci(fname=run_path+f'/eci_iter{iter_idx}.json')

        # Evaluating the CE model
        CV_score = evl.get_cv_score()*1e3 # meV/atom
        dE = (np.max(evl.e_dft) - np.min(evl.e_dft) )*1e3 # meV/atom
        CE_model_value = CV_score
        #CE_model_value = CV_score/dE # ratio of CV score to dE
        logger.info(f'CV score[meV/atom]: {CV_score}, dE[meV/atom]: {dE}, CE value: {CE_model_value}')
        with open(run_path+f'/CE_model_result_{iter_idx}.json', 'w') as f:
            json.dump({'CV_score': CV_score, 'dE':dE, 'CE_value': CE_model_value}, f)

        # Check if the CE model has reached convergence
        threshold = CE_model_dict['threshold']
        gs_threshold = CE_model_dict['gs_threshold']
        
        # The CE model has reached convergence
        if CE_model_value < threshold:
            logger.info('CE model has reached convergence')
            # Stop the cyclic run and give the MC parameters
            cyclic = True
            return_parameters = {CYCLICALGROUP_KEY: cyclic,'run_path':run_path,'cfg_path':cfg_path, 
                                 'CE_setting_path':run_path+'/initial_setting.json',
                                 'ECI_path': run_path+f'/eci_iter{iter_idx}.json' }
        
        # If the CE model has reached ground state threshold, we sample ground state structures
        elif CE_model_value< gs_threshold:
            logger.info('CE model has reached ground state threshold')
            # Load database
            db = connect(db_path)
            init_db_len = len(db)
            # Set number of random structures
            gs_n_random_struc = CE_model_dict['gs_n_random_struc']

            # Find the structures chossen to be ground state
            initial_atom_list = []
            for row in db.select([('struct_type','=','initial')]):
                initial_atom_list.append(row.toatoms())

            # If the number of structures to submit is larger than the ones generated, submit all of them
            if gs_n_random_struc > len(initial_atom_list):
                submit_atoms = initial_atom_list

            # if the number of structures to submit is smaller than the ones generated, randomly select gs_n_random_struc
            else:
                random_idx = np.random.randint(0,len(initial_atom_list),gs_n_random_struc)
                submit_atoms = [initial_atom_list[i] for i in random_idx]

            # Generate the ground state structures
            logger.info(f'Generating ground state structures for {gs_n_random_struc} structures')
            logger.info(f'MC parameters: Initial temperature: {CE_model_dict["gs_init_temperature"]}, Final temperature: {CE_model_dict["gs_final_temperature"]}, Number of temperatures: {CE_model_dict["gs_num_temp"]}, Sweeps: {CE_model_dict["gs_sweeps"]}')
            
            # Loading the eci values
            with open(run_path+f'/eci_iter{iter_idx}.json', 'r') as infile:
                eci = json.load(infile)

            # Generate the ground state structures
            ns = NewStructures(setting, struct_per_gen=gs_n_random_struc, generation_number=iter_idx)
            for i, atoms in enumerate(submit_atoms):
                logger.info(f'Generating GS structure for composition {i}: {atoms.get_chemical_formula()}')

                try:
                    ns.generate_gs_structure(atoms=atoms, init_temp=CE_model_dict['gs_init_temperature'],
                            final_temp=CE_model_dict['gs_final_temperature'], num_temp=CE_model_dict['gs_num_temp'],
                            num_steps_per_temp=CE_model_dict['gs_sweeps'],
                            eci=eci, random_composition=False) # do MC aneling to find GS structure for each composition only gives one back
                except:
                    logger.info(f'Failed to generate GS structure in in 10 attempts for composition {i}: {atoms.get_chemical_formula()}')
                    continue
            
            # Return the dynamic width group key and the run path
            cyclic = False
            db_new = db.select(gen=iter_idx)
            run_list = []
            for row in db_new:
                run_list.append(row.id)
            logger.info(f'Submitting: {run_list}')
            dmkey = len(run_list) # number of jobs to submit
            
            return_parameters = {CYCLICALGROUP_KEY: cyclic, DYNAMICWIDTHGROUP_KEY: dmkey, 'run_list':run_list}
        
        # If the CE model has not reached convergence, we continue the iteration with random structures
        else:
            logger.info('CE model needs more random structures to reach convergence')
            # Create randomstructures
            db = connect(db_path)
            init_db_len = len(db)
            n_random_struc = CE_model_dict['n_random_struc']
            logger.info('Initial start, creating {} random structures'.format(n_random_struc))
            ns = NewStructures(setting, struct_per_gen=n_random_struc, generation_number=iter_idx)
            ns.generate_random_structures()

            # Return the dynamic width group key and the run path
            cyclic = False
            db_new = db.select(gen=iter_idx)
            run_list = []
            for row in db_new:
                run_list.append(row.id)
            logger.info('Submitting: ', run_list)
            dmkey = len(run_list)  # number of jobs to submit

            return_parameters = {CYCLICALGROUP_KEY: cyclic, DYNAMICWIDTHGROUP_KEY: dmkey, 'run_list':run_list}
        
        return True, return_parameters

if __name__ == "__main__":
    main()