from ase.io import read
import json
import toml
import logging
from clease.settings import settings_from_json
from clease.calculator import attach_calculator
from clease.montecarlo.observers import Snapshot, ConcentrationObserver

from clease.montecarlo.kmc_events import KMCEventType
from clease.montecarlo import KineticMonteCarlo
from clease.montecarlo.barrier_models import BEPBarrier

from ase.neighborlist import neighbor_list
from typing import List
from ase import Atoms
class NeighbourSwap(KMCEventType):
    """
    KMC event that provides swaps between neighbours

    :param cutoff: Passed to ASE neighbour list. All sites
        within the cutoff are possible swaps
    """

    def __init__(self, atoms: Atoms, cutoff: float,X_sites: list,diffusion_sites: list):
        super().__init__()
        first, second = neighbor_list("ij", atoms, cutoff)
        self.nl = [[] for _ in range(len(atoms))]
        total_sites = X_sites + diffusion_sites
        for f, s in zip(first, second):
            if f in total_sites and s in total_sites:
                self.nl[f].append(s)

    def get_swaps(self, atoms: Atoms, vac_idx: int) -> List[int]:
        return self.nl[vac_idx]

def get_eci(filename):
    with open(filename, 'r') as infile:
        eci = json.load(infile)
    return eci

def main(run_path,cfg_path,CE_setting_path,ECI_path,gs_path,system_temp,conc,config_size,**kwargs):
    
    ### PARAMETERS ###
    # Load the CE model parameters
    with open(cfg_path, 'r') as f:
        KMC_model_dict = toml.load(f)['KMC']

    cutoff_radius = KMC_model_dict['cutoff_radius'] # Å - radius for possible  diffusion
    dilute_barrier = KMC_model_dict['dilute_barrier'] # eV - barrier for dilute limit
    alpha = KMC_model_dict['alpha'] # alpha for the BEP barrier model
    diffusion_elem = KMC_model_dict['diffusion_elem'] # Element to be diffused can als be a list of elements
    sweeps = KMC_model_dict['sweeps'] # Sweeps for the MC simulation

    settings = settings_from_json(CE_setting_path) # Load the CE model settings
    eci = get_eci(ECI_path) # Load the ECI

    # Load the ground state structure
    gs_atom = read(gs_path,index=-1)
    gs_atom = attach_calculator(settings, gs_atom, eci)

    ### LOGGER ###
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    runHandler = logging.FileHandler(f'kmc_c{conc}_s{config_size[0]}{config_size[1]}{config_size[2]}.log', mode='w')
    runHandler.setLevel(logging.DEBUG)
    runHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))
    logger.addHandler(runHandler)
    logger.debug(f'KMC starts for prev. found lowest energy structure for {sweeps} sweeps')

    # Define the important atom sites
    if isinstance(diffusion_elem,list): # if we have multiple diffusion elements
        sites_list = []
        tot_sites = []
        for elem in diffusion_elem:
            sites = [a.index for a in gs_atom if a.symbol == elem]
            sites_list.append(sites)
            tot_sites += sites
        site_diffusion = tot_sites
        sites_X = [a.index for a in gs_atom if a.symbol == 'X']
        sites_list.append(sites_X)
        tot_sites += sites_X

    else:
        site_diffusion = [a.index for a in gs_atom if a.symbol == diffusion_elem]
        sites_X = [a.index for a in gs_atom if a.symbol == 'X']
        tot_sites = site_diffusion + sites_X
        sites_list = [site_diffusion,sites_X]

    ### Set Observers ###
    # Track the concentration of the diffusion element/elements
    if isinstance(diffusion_elem,list):
        obs_list = []
        for elem in diffusion_elem:
            conc_obs = ConcentrationObserver(gs_atom, element=elem)
            obs_list.append(conc_obs)
    else:
        conc_obs = ConcentrationObserver(gs_atom, element=diffusion_elem)    
        obs_list = [conc_obs]

    # Track the KMC events
    snap = Snapshot(fname=run_path+f'/KMC_c{conc}_s{config_size[0]}{config_size[1]}{config_size[2]}.traj', atoms=gs_atom, mode='w')

    ### KMC Simulation ###

    # barrier estimation 
    barrier_model = BEPBarrier(dilute_barrier=dilute_barrier,alpha=alpha)
    # KMC event types
    event = [NeighbourSwap(gs_atom,cutoff=cutoff_radius,X_sites=sites_X,diffusion_sites=site_diffusion)]
    # KMC Simualtions object
    kmc = KineticMonteCarlo(gs_atom, temp=system_temp,barrier=barrier_model, event_types=event)

    # Attach observers
    kmc.attach(snap,interval=1)
    for obs in obs_list:
        kmc.attach(obs)
    
    # Run KMC by tracking a single vaccaancy
    n_steps = sweeps*len(gs_atom)
    kmc.run(num_steps=n_steps, vac_idx=sites_X[0])

    # Return parameters for the next step
    return_parameters = {}
    return True, return_parameters