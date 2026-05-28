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

def redox_sort_func(x):
    if isinstance(x,tuple):
        x = x[0]
    if x == 'Fe':
        return 1
    elif x == 'Mn':
        return 2
    elif x == 'Co':
        return 3
    elif x == 'Ni':
        return 4
    elif x == 'Na':
        return 5
    else:
        return 10
def sort(atoms, tags=None,key=None):
    if tags is None:
        tags = atoms.get_chemical_symbols()
    else:
        tags = list(tags)
    deco = sorted([(tag, i) for i, tag in enumerate(tags)],key=key)
    indices = [i for tag, i in deco]
    return atoms[indices], indices
def get_U_value(M:str) -> float:
    """
    Get the U value for the metal ion
    Ref: https://docs.materialsproject.org/methodology/materials-methodology/calculation-details/gga+u-calculations/hubbard-u-values
    Following mapping is used:
        Fe -> 5.3
        Mn -> 3.9
        Ni -> 6.2
        Co -> 3.32
        V  -> 3.25
    Args:
        M (str): The metal ion

    Returns:
        float: The U value of the metal ion
    """
    if M=='Mn':
        U_val =3.9
    elif M=='Fe':
        U_val=5.3
    elif M=='Ni':
        U_val=6.2
    elif M=='Co':
        U_val=3.32
    elif M=='V':
        U_val = 3.25
    else:
        raise ValueError(f'U value is not known for {M}')
    return U_val

def get_magmom(M:str,redox:bool) -> int:
    """
    Get the magnetic moment for the metal ions in the cathode materials. Following mapping is used:
        Fe -> 4 (Fe2+)
        Mn -> 5 (Mn2+)
        Ni -> 2 (Ni2+)
        Co -> 3 (Co2+)
        Ga -> 0
        Fe -> 5 (Fe3+)
        Mn -> 4 (Mn3+)
        Ni -> 3 (Ni3+)
        Co -> 4 (Co3+)

    Args:
        M (str): metal ion
        redox (bool): whether the metal ion is in redox state
    Returns:
        int: magnetic moment
    
    """
    if M=='Fe':
        if redox == True:
            magmom = 5 # Fe3+
        else:
            magmom = 4 #Fe2+
    elif M=='Mn':
        if redox == True:
            magmom = 4 #Mn3+
        else:
            magmom = 5 #Mn2+
    elif M=='Ni':
        if redox == True:
            magmom = 3 #Ni3+ # was 1 but it was wrong due to the anions spin
        else:
            magmom = 2 #Ni2+
    elif M=='Co':
        if redox == True:
            magmom = 4 #Co3+
        else:
            magmom = 3 #Co2+
    elif M=='V':
        if redox == True:
            magmom = 2 #V3+
        else:
            magmom = 1 #V4+ 
    elif M=='Ga':
        magmom= 0
    else:
        raise ValueError(f'Magmom is not known for {M}')
    return magmom

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
        #atom, i = sort(atom, key=redox_sort_func)
        count_redox = 0
        print(atom)
        for a in atom:
            # Ga represents oxidized Fe3+
            if a.symbol == 'Ga':
                magmom = get_magmom('Fe',redox=True)
                a.symbol = 'Fe'
                a.magmom = magmom
                tot_magmom += magmom
                count_redox += 1

            elif a.symbol == 'Fe':
                magmom = get_magmom('Fe',redox=False)
                a.symbol = 'Fe'
                a.magmom = magmom
                tot_magmom += magmom
            else:
                a.magmom = 0
        logger.info(f"Number of redox Fe3+: {count_redox}")
        logger.info(f"Number of Fe2+: {len(M_ion_all)-count_redox}")
        # Define the Na concentration
        Na_conc = ion_len /len(M_ion_all)
        
        logger.info(f"Total magnetic moment: {tot_magmom}")
        logger.info(f"Na concentration: {Na_conc}")
        logger.info(f"M_ion: {M_ions}")
        logger.info(f"Na_len: {ion_len}")
        logger.info(f"Na_max: {N_Na_max}")

        # Vasp calculator
        vasp_params = params['VASP']

        # Set the VASP calculator
        calc = Vasp(directory=relaxsim_directory,**vasp_params)

        vasp_params['nupdown'] = tot_magmom
        logger.info(f'{name} has nupdown {tot_magmom}')
        calc = Vasp(directory=relaxsim_directory,**vasp_params)

        # Define U-value
        ldau_luj = {'ldau_luj':{}}
        if type(M_ions)==str:
            if M_ions == 'Ga':
                pass
            else:
                ldau_luj['ldau_luj'][M_ions] = {'L': 2, 'U': get_U_value(M_ions),'J':0}
                logger.info(f'{name} has L, U, J values: (2, {get_U_value(M_ions)}, 0)')
        else:
            for m in M_ions:
                if m == 'Ga':
                    continue
                else:
                    ldau_luj['ldau_luj'][m] = {'L': 2, 'U': get_U_value(m),'J':0}
                    logger.info(f'{name} has L, U, J values: (2, {get_U_value(m)}, 0)')
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
        # append the workflow path and load the ML relaxer class
        sys.path.append(params['workflow_path'])
        from cPaiNN.relax import ML_Relaxer
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
        cell_relaxer = True
        # get calculator and set it to atoms object 
        device_global = 'cuda' if os.getenv('CUDA_VISIBLE_DEVICES') is not None else 'cpu'
        relaxer = ML_Relaxer(calc_name=calc_name,calc_paths=calc_path,
                            device=device_global,optimizer=optimizer)#
        relax_results=relaxer.relax(atom, fmax=fmax, steps=max_step,traj_file=traj_path, log_file=log_path, interval=1)
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