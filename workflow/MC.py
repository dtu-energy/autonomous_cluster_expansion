from ase.io import read
import numpy as np
import json
import toml
import logging
import pandas as pd
from clease.montecarlo import KineticMonteCarlo
from clease.settings import settings_from_json
from clease.calculator import attach_calculator
from clease.montecarlo.observers import Snapshot, SiteOrderParameter, ConcentrationObserver, LowestEnergyStructure
from clease.montecarlo import Montecarlo, RandomSwap

from ase.io.trajectory import TrajectoryWriter
from clease.montecarlo.observers import MCObserver
from ase.calculators.singlepoint import SinglePointCalculator

class LowestEnergyStructure_snap(MCObserver):
    """Track the lowest energy state visited during an MC run. 
    And save the config. for every new config. with lowest energy,
    meaning it is the last config in the trajectory which is gs.

    atoms: Atoms object
        Atoms object used in Monte Carlo

    verbose: bool
        If `True`, progress messages will be printed
    """

    name = "LowestEnergyStructure"

    def __init__(self, atoms,name='gs_snap.traj', verbose=True):
        super().__init__()
        self.atoms = atoms
        self.lowest_energy = np.inf
        self.lowest_energy_cf = None
        self.emin_atoms = None
        self.verbose = verbose
        self.name = name
        self.traj = TrajectoryWriter(name, mode='w')



    def reset(self):
        self.lowest_energy = np.inf
        self.lowest_energy_cf = None
        self.emin_atoms = None

    @property
    def calc(self):
        return self.atoms.calc

    @property
    def energy(self):
        return self.calc.results['energy']

    def __call__(self, system_changes):
        """
        Check if the current state has lower energy and store the current
        state if it has a lower energy than the previous state.

        system_changes: class
            system_changes: A class of system changes. For example, if the
            occupation of the atomic index 23 is changed from Mg to Al,
            system_change = [(23, Mg, Al)]. If an Mg atom occupying the atomic
            index 26 is swapped with an Al atom occupying the atomic index 12,
            system_change = [(26, Mg, Al), (12, Al, Mg)]
        """

        if self.emin_atoms is None or self.energy < self.lowest_energy:
            dE = self.energy - self.lowest_energy
            self._update_emin()
            self.traj.write(self.emin_atoms)
            msg = "Found new low energy structure. "
            msg += f"New energy: {self.lowest_energy} eV. Change: {dE} eV"
            #logger.info(msg)
            #if self.verbose:
            #    print(system_changes)
            #    print(msg)

    def _update_emin(self):
        self.lowest_energy_cf = self.calc.get_cf()
        self.lowest_energy = self.energy

        # Store emin atoms, and attach a cache calculator
        self.emin_atoms = self.atoms.copy()
        calc_cache = SinglePointCalculator(self.emin_atoms, energy=self.energy)
        self.emin_atoms.calc = calc_cache


def get_eci(filename):
    with open(filename, 'r') as infile:
        eci = json.load(infile)
    return eci


def main(run_path,cfg_path,CE_setting_path,ECI_path,**kwargs):

    ### PARAMETERS ###
    # Load the CE model parameters
    with open(cfg_path, 'r') as f:
        MC_model_dict = toml.load(f)['MC']

    ### PARAMETERS ###
    system_temp = MC_model_dict['system_temp'] #K temperature for the system
    anelling_temps = MC_model_dict['anelling_temps'] # K list of temperatures for the anneling process.
    diffusion_elem = MC_model_dict['diffusion_elem'] # Element to be diffused
    vaccancy_elem = MC_model_dict['vaccancy_elem'] # Element to be removed
    sweeps = MC_model_dict['sweeps']# Sweeps for the MC simulation

    config_size = MC_model_dict['config_size'] # Size of the supercell for the MC/KMC
    conc =MC_model_dict['conc'] # Concentration of the elemt to be replaced by the vaccancy (vacaancy concentration 1-conc)
    
    settings = settings_from_json(CE_setting_path) # Load the CE model settings
    eci = get_eci(ECI_path) # Load the ECI

    #  Cheking if the last temperature is the same as the system temperature
    if anelling_temps[-1] != system_temp:
        anelling_temps.append(system_temp)

    ### LOGGER ###
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    runHandler = logging.FileHandler(f'mc_c{conc}_s{config_size[0]}{config_size[1]}{config_size[2]}.log', mode='w')
    runHandler.setLevel(logging.DEBUG)
    runHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))
    logger.addHandler(runHandler)
    logger.debug(f'Loading the CE model from {CE_setting_path} and the ECI from {ECI_path}')
    print(f'Using the following parameters: system_temp:{system_temp}, anelling_temps:{anelling_temps}, ')
    print(f'diffusion_elem:{diffusion_elem}, vaccancy_elem:{vaccancy_elem}, sweeps:{sweeps}, conc:{conc}')
    ### Initiate atoms object ###
    # Load atom from CE settings
    atoms_i=settings.atoms
    # Scaling up the atom to a specific size
    atoms_i=atoms_i*config_size
    # Set sites to be replaced by vaccancy
    sites_d = [a.index for a in atoms_i if a.symbol == vaccancy_elem]
    # Determine the number of vaccancies from the concentration and replace X with 
    X_len=int(np.rint(conc*len(sites_d)))#to set the concentration of the X
    for i in range(X_len):
        atoms_i.symbols[sites_d[i]]='X'

    # For efficient initialisation of large cells, CLEASE comes with a convenient helper function called attach_calculator
    atoms_i = attach_calculator(settings, atoms_i, eci)
    # Define the important atom sites
    if isinstance(diffusion_elem,list): # if there are more diffusion elements
        sites_list = []
        tot_sites = []
        for elem in diffusion_elem:
            sites = [a.index for a in atoms_i if a.symbol == elem]
            sites_list.append(sites)
            tot_sites += sites
        sites_X = [a.index for a in atoms_i if a.symbol == 'X']
        sites_list.append(sites_X)
        tot_sites += sites_X

    else: # if there is only one diffusion element
        site_diffusion = [a.index for a in atoms_i if a.symbol == diffusion_elem]
        site_X = [a.index for a in atoms_i if a.symbol == 'X']
        tot_sites = site_diffusion + site_X
        sites_list = [site_diffusion,site_X]


    ### Set Observers ###
    # Set concentration observere to track the concentration of the diffusion element/elements
    if isinstance(diffusion_elem,list):
        obs_list = []
        for elem in diffusion_elem:
            conc_obs = ConcentrationObserver(atoms_i, element=elem)
            obs_list.append(conc_obs)
    else:
        conc_obs = ConcentrationObserver(atoms_i, element=diffusion_elem)    
        obs_list = [conc_obs]

    # Saving all the MC steps as well as the lowest energy strutures uncovered during the MC
    snap = Snapshot(fname=run_path+f'/anneling_c{conc}_s{config_size[0]}{config_size[1]}{config_size[2]}.traj', atoms=atoms_i, mode='w')
    gs_path = run_path+f'/gs_snap_c{conc}_s{config_size[0]}{config_size[1]}{config_size[2]}.traj'
    ls=LowestEnergyStructure_snap(atoms_i,name=gs_path)

    ### Anneling MC simulation ###
    # Run anneling 
    df_list = []
    for T in anelling_temps:
        logger.debug(f'Temperture {T}K :')
        # Setup MC with random swap
        mc = Montecarlo(atoms_i, 0, RandomSwap(atoms_i, tot_sites))
        # Attached observers
        mc.attach(snap,interval=1)
        mc.attach(ls)
        for obs in obs_list:
            mc.attach(obs)  
        # Set MC tempereture and start MC  
        mc.T = T
        mc.run(steps=sweeps)
        # Save he thermodynamic quantities and reset the MC
        thermo = mc.get_thermodynamic_quantities()
        logger.info(thermo)
        df_list.append(thermo)
        mc.reset()
        mc.reset_averagers()

    # Save all the thermodynamic quantities in a csv file
    df_tot = pd.DataFrame(df_list)
    df_tot.to_csv(run_path+f'/anneling_{conc}.csv', index=False)

    # Return parameters for the next step
    return_parameters = { 'run_path':run_path,'cfg_path':cfg_path,'CE_setting_path':CE_setting_path,
                         'ECI_path':ECI_path,'gs_path':gs_path,
                        'system_temp':system_temp,'conc':conc, 'config_size':config_size}
    return True, return_parameters
