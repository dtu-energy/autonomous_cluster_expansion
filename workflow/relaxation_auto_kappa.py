import os
from ase.calculators.vasp import Vasp
from ase.db import connect
import os, sys 
from clease.tools import update_db
import numpy as np
import toml
from ase.calculators.calculator import CalculationFailed
from perqueue.constants import INDEX_KW
import logging

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

    # setting and creating the directory for the saved files
    relax_directory = f'{run_path}/relaxation'
    relaxsim_directory =f'{relax_directory}/{name}'
    try:
        os.makedirs(relaxsim_directory)
    except:
        pass

    ### Find the KPTS ###
    # Get standardized unitcell and primitive matrix
    from auto_kappa.structure import change_structure_format
    from auto_kappa.cui.suggest import get_unitcell_and_primitive_matrix
    from auto_kappa.vasp.kmesh import klength2mesh
    from auto_kappa.utils.config import (
        get_vasp_parameters,
        get_vasp_parameters_by_mode,
        get_potcar_setups,
        get_xc,
        load_user_config,
    )
    from auto_kappa.calculators.vasp import get_vasp_calculator
    from auto_kappa.structure import transform_unit2prim
    unitcell, prim_mat = get_unitcell_and_primitive_matrix(atom)
    k_length = 20
    encut_scale_factor = 1.3

    primitive = transform_unit2prim(unitcell, prim_mat, format='ase')
    kpts = klength2mesh(k_length, primitive.cell.array)

    user_config = {'vasp_parameters': {'relax-full': {'ediff': 1e-08, 'ediffg': -1e-07}}}
    vasp_params = get_vasp_parameters(user_config)
    potcar_setups = get_potcar_setups(user_config)
    xc = get_xc(user_config)
    params = get_vasp_parameters_by_mode(vasp_params, mode='relax-full')
    params['nsw'] = 200

    calc = get_vasp_calculator(
        params,
        directory=relaxsim_directory,
        atoms=primitive,
        kpts=kpts,
        encut_scale_factor=encut_scale_factor,
        setups=potcar_setups,
        xc=xc,
    )

    # Load the parameters
    with open(cfg_pth, 'r') as f:
        params = toml.load(f)

    
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

        # Vasp calculator
        #vasp_params = params['VASP']

        # Set the VASP calculator
        #calc = Vasp(directory=relaxsim_directory,**vasp_params)

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
        # append the workflow path and load the ML relaxer class
        sys.path.append(params['workflow_path'])
        from workflow.utils import ML_Relaxer
        # Set parameters 
        calc_name = params['method']
        if 'calc_path' in params.keys():
            calc_path = params['calc_path']
        else:
            calc_path = None
        optimizer = params['optimizer']
        relax_cell = params['relax_cell']
        traj_path = relaxsim_directory+'/opt.traj'
        log_path = relaxsim_directory+'/opt.log'
        fmax = np.abs(params['VASP']['ediffg'])
        max_step = params['max_step']
        # get calculator and set it to atoms object 
        relaxer = ML_Relaxer(calc_name=calc_name,calc_paths=calc_path,
                          optimizer=optimizer,relax_cell=relax_cell,device='cuda')
        relax_results=relaxer.relax(atom, fmax=fmax, steps=max_step,
                                    traj_file=traj_path, log_file=log_path, interval=1)
        final_structure = relax_results["final_structure"]
        final_energy = final_structure.get_potential_energy() #relax_results["trajectory"].energies[-1]
        force = np.sqrt(np.sum(((final_structure.get_forces())**2),axis=1))
        fmax_relax = np.max(force)

        logger.info(f'Final structure: {final_structure}')
        logger.info(f"The final energy is {float(final_energy):.3f} eV.")
        logger.info(f"The maximum force is {fmax_relax:.3f} eV/Å.")

        # Check if the relaxation have reaxhed required accuracy within 20% of the fmax
        if fmax_relax > fmax+fmax*0.2:
            var = False
            logger.info(f'Relaxation did not converge. Fmax: {fmax_relax}' )
            return_parameters = {'initial_start':False}
            return True, return_parameters

    # Update database 
    update_db(uid_initial=db_id, final_struct=atom, db_name=db_path)
    return_parameters = {}
    return True, return_parameters
if __name__ == "__main__":
    main()