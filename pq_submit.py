from pathlib import Path
import toml, os
import numpy as np
from perqueue import PersistentQueue
from perqueue.task_classes.task import Task
from perqueue.task_classes.task_groups import CyclicalGroup, Workflow, DynamicWidthGroup, StaticWidthGroup
from perqueue.constants import DYNAMICWIDTHGROUP_KEY

### Paths ###
# Setup the paths
workflow_path = Path('.') # path to where the workflow folder and main config file is located
run_path = Path('.') # path to were you run perqueue from

CE_model_path = str(run_path/'NaFeMnPO4_CE_model') # path to were you want the result to be stored
cfg_path = CE_model_path+'/config.toml' # path to new local config file, which will contain all params
db_name = 'structures.db' # name of the ase database

# Create the directory for the CE model
if not os.path.isdir(CE_model_path):
    os.makedirs(CE_model_path)


### Simulation parameters ###
# Define the parameters of the CE model
cif_file = 'cif_files/NaMPO4_olivine.cif' # path to inital cif file

# Define the crystal structure 
# x,y,z for different elements in the cif file
Fe = (0.28619, 0.25, 0.985605)
Na = (0, 0, 0)
O1 = (0.11328, 0.25, 0.749385)
O2 = (0.46909, 0.25, 0.160365)
O3 = (0.17764, 0.05403, 0.313465)
P = (0.108, 0.25, 0.441755)
space_group = 62 # Space group of the structure
basis_coord = [Na,Fe,P,O1,O2,O3] # The coordinates of the basis elements

basis_element = [['Na','X'],['Fe','Mn'],['P'],['O'],['O'],['O']] # The basis element for all the elements
basis_index = [[0],[1],[2],[3,4,5]] # The index of the basis elements

# Define the CE model parameters
# The size of the cell you want to sample from. Either set the supercell_factor or the size
supercell_factor = None
size = [(1,0,0),(0,2,0),(0,0,2)]
if size is None and supercell_factor is None:
    raise ValueError('Either set the supercell_factor or the size')
if size is not None and supercell_factor is not None:
    raise ValueError('Either set the supercell_factor or the size')

CE_model_dict = {
    'auto_set_cell': False, # Automatically find and set the cell size (DOES NOT WORK)
    'space_group': space_group, # The space group of the structure.
    'basis_index': basis_index, # The index of the basis elements
    'basis_coord': basis_coord, # The coordinates of the basis elements
    'basis_element': basis_element, # The basis element for all the elements

    'A_eq': np.array([#Na, X, Fe, Mn, P, O
                      [1, 1, 0, 0, 0, 0], # 4
                      [0, 0, 1, 1, 0, 0], # 4    
                      [0, 0, 0, 0, 1, 0], # 4
                      [0, 0, 0, 0, 0, 1], # 16
                      #[0, -1, 0, 1,  0, 0], # 
                     # [1, 0, 1, 0,  0, 0], 
                      ] ),
    'b_eq': np.array([1, 1, 1, 1]), 
    'size': size, # The size of the supercell
    'supercell_factor': supercell_factor, # The supercell factor
    'basis_func_type': 'polynomial', #: polynomial, trigonometric and binaryLinear.
    'max_cluster_dia': [5.0, 5.0], # The maximum cluster diameter
    
    # Fiting 
    'scoring_scheme': 'loocv', # The scoring scheme
    'max_cluster_size': 4, # The maximum cluster size
    'fitting_scheme': 'l2', # The fitting scheme
    'alpha_min': 1e-7, # The minimum alpha value
    'alpha_max': 1e3, # The maximum alpha value
    'num_alpha': 50, # The number of alpha values
    #'select_cond': None, # selction condition. 

    # Generation of structures
    'n_random_struc': 5, # The number of random structures to be relaxed
    'threshold': 3, # The threshold
    'gs_threshold': 5, # The ground state threshold. When we begin sampling ground state structures
    'gs_init_temperature': 1000, # The initial temperature for the ground state sampling
    'gs_final_temperature': 300, # The final temperature for the ground state sampling
    'gs_num_temp': 10, # The number of temperatures for the ground state sampling
    'gs_sweeps': 10000, # The number of sweeps for the ground state sampling
    'gs_n_random_struc': 20, # The number of random structures for the ground state sampling
}

# Monte Carlo parameters
MC_params = {'system_temp': 300, #K temperature for the system
            'anelling_temps': [1000,900,600,300],     # K list of temperatures for the anneling process.
            'diffusion_elem': 'Na', # Element to be diffused can als be a list of elements
            'vaccancy_elem':'Na', # Element to be removed
            'sweeps':1000,# Sweeps for the MC simulation
            'config_size': (1,1,1), # Size of the supercell for the MC/KMC
            'conc': 0.5, # Concentration of the elemt to be replaced by the vaccancy (vacaancy concentration 1-conc)
            }
# Kinetic Monte Carlo parameters
KMC_params = {'cutoff_radius':7, #Å - radius for possible  diffusion
              'dilute_barrier':{'Na':0.5,'X':0}, # eV - barrier for dilute limit
              'alpha':0.7, # alpha for the BEP barrier model
              'diffusion_elem':'Na', # Element to be diffused can als be a list of elements
              'sweeps':1000 # Sweeps for the MC simulation
}

# If you consider cathode materials 
cathode_params = {'ion':'Na','M_ions':['Fe','Mn'],'pusdo_M_ion':[]} # ion, redox, U-value

# other parameters
other_params = {'cathode': cathode_params, 'cif_file': cif_file, 
                'method':'chgnet', # machine learning force field 
                'calc_path':None, # path to the calculator if you want to use personal calculator
                'optimizer':'FIRE', # Optimizer for the relaxation
                'relax_cell': True, # if the cell should be relaxed. Only relevant for ML-FF with stress tensor prediciton
                'max_step': 1000} # 

### Setup toml with all paremeters ###
# Load the main config file with the VASP parameters
with open(workflow_path/'config.toml', 'r') as f: 
    main_params = toml.load(f)

# Store all parameters in a local config file
with open(cfg_path, "w") as toml_file:
    toml.dump(other_params, toml_file)
    toml.dump({'CE':CE_model_dict}, toml_file)
    toml.dump({'MC':MC_params}, toml_file)
    toml.dump({'KMC':KMC_params}, toml_file)
    toml.dump({'VASP':main_params['VASP']}, toml_file)


##### Define parameters for workflow #####
# Define job parameters
CE_params = {'run_path':CE_model_path, 'db_path':CE_model_path+f'/{db_name}','cfg_path':cfg_path, 'cif_file':cif_file}
relax_params = {'run_path':CE_model_path, 'db_path':CE_model_path+f'/{db_name}', 'cfg_pth':cfg_path}
MC_task_params = {'run_path':CE_model_path, 'cfg_pth':cfg_path}
KMC_task_params = {'run_path':CE_model_path, 'cfg_pth':cfg_path}

#### Workflow ####
# Setup perqueue tasks
CE_task_init = Task(workflow_path/'workflow/CE_model_init.py', CE_params, '1:local:10m')
CE_task= Task(workflow_path/'workflow/CE_model.py', CE_params , '24:1:xeon24el8:2d')

relaxation_task = Task(workflow_path/'workflow/relaxation.py',   relax_params, '8:sm3090el8:3h')

MC_task= Task(workflow_path/'workflow/MC.py', MC_task_params , '24:1:xeon24el8:2d')
KMC_task= Task(workflow_path/'workflow/KMC.py', KMC_task_params , '24:1:xeon24el8:2d')

# Setup groups
dwg =  DynamicWidthGroup([relaxation_task])
cg = CyclicalGroup([dwg, CE_task], max_tries=10)

# Define and submit workflow
wf = Workflow({CE_task_init: [], cg: [CE_task_init], MC_task: [cg], KMC_task: [MC_task]} )

with PersistentQueue() as pq:
    pq.submit(wf)