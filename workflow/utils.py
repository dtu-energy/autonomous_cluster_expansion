
from ase.constraints import ExpCellFilter
from ase.optimize.bfgs import BFGS
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.optimize.fire import FIRE
from ase.optimize.lbfgs import LBFGS, LBFGSLineSearch
from ase.optimize.mdmin import MDMin
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase.optimize.optimize import Optimizer
from pathlib import Path
from ase import Atoms

# All possible optimizers
OPTIMIZERS = {
    "FIRE": FIRE,
    "BFGS": BFGS,
    "LBFGS": LBFGS,
    "LBFGSLineSearch": LBFGSLineSearch,
    "MDMin": MDMin,
    "SciPyFminCG": SciPyFminCG,
    "SciPyFminBFGS": SciPyFminBFGS,
    "BFGSLineSearch": BFGSLineSearch,
}

class Relaxer:
    """Relaxer is a class for structural relaxation."""
    
    def __init__(
        self,
        calc_name: str | str = "mace_large",
        calc_paths: str | None = None,
        optimizer: Optimizer | str = "FIRE",
        device: str = "cuda",
        relax_cell: bool = True,
        fmax: float = 0.1,
        steps: int = 1000,
        traj_file: str | None = None,
        log_file: str = "opt.log",
        interval: int = 1,

    ):
        """
        Args:
            calc_name (str): calculator name. Defaults to "mace_large".
            calc_paths (str): path to the calculator. Defaults to None.
            optimizer (str or ase Optimizer): the optimization algorithm. Defaults to "FIRE".
            device (str): device to use. Defaults to "cuda".
            relax_cell (bool): whether to relax the lattice cell. Defaults to True.
            fmax (float): total force tolerance for relaxation convergence. Defaults to 0.1.
            steps (int): max number of steps for relaxation. Defaults to 500.
            traj_file (str): the trajectory file for saving. Defaults to None.
            log_file (str): the log file for saving. Defaults to "opt.log".
            interval (int): the step interval for saving the trajectories. Defaults to 1.
        """
        if isinstance(optimizer, str):
            optimizer_obj = OPTIMIZERS.get(optimizer, None)
        elif optimizer is None:
            raise ValueError("Optimizer cannot be None")
        else:
            optimizer_obj = optimizer
        
        self.opt_class: Optimizer = optimizer_obj
        self.calc_name = calc_name
        self.calc_paths = calc_paths
        self.device = device
        self.calculator = self.get_calc()    
        self.relax_cell = relax_cell
        self.fmax = fmax
        self.steps = steps
        self.traj_file = traj_file
        self.log_file = log_file
        self.interval = interval
    
    def relax(
        self,
        atoms: Atoms,
        verbose=False,
        **kwargs,
    ):
        """
        Relax an input Atoms.

        Args:
            atoms (Atoms): the atoms for relaxation
            verbose (bool): Whether to have verbose output. Defaults to False.
            kwargs: Kwargs pass-through to optimizer.
        """
        # Set the calculator
        atoms.set_calculator(self.calculator)
        if self.relax_cell:
            atoms = ExpCellFilter(atoms)
        optimizer = self.opt_class(atoms,trajectory=self.traj_file,logfile=self.log_file,**kwargs)
        optimizer.run(fmax=self.fmax, steps=self.steps)
        if isinstance(atoms, ExpCellFilter):
            atoms = atoms.atoms
        return {
            "final_structure": atoms,#
        }
    
    def get_calc(self):
        """ Get calculator from the given name
        
        Args:
            calc_name (str): calculator name
            calc_paths (str): path to the calculator
            device (str): device to use
            
        Returns:
            calc (ase.calculators.calculator.Calculator): calculator object
        """
        if self.calc_name == 'mace_large':
            from mace.calculators import mace_mp
            print('Using MACE large model')
            calc = mace_mp(model="large", dispersion=False, default_dtype="float64", device=self.device)
        elif self.calc_name == 'mace_medium':
            from mace.calculators import mace_mp
            print('Using MACE medium model')
            calc = mace_mp(model="medium", dispersion=False, default_dtype="float64", device=self.device)
        elif self.calc_name == 'mace_small':
            from mace.calculators import mace_mp
            print('Using MACE small model')
            calc = mace_mp(model="small", dispersion=False, default_dtype="float64", device=self.device)
        elif self.calc_name == 'mace_model':
            from mace.calculators import MACECalculator
            print('Using MACE personal model')
            calc =  MACECalculator(model_paths=self.calc_paths,device=self.device, default_dtype="float64")
        elif self.calc_name == 'chgnet':
            from chgnet.model.dynamics import CHGNetCalculator
            from chgnet.model import CHGNet
            print('Using CHGNet model')
            model = CHGNet.load()
            calc = CHGNetCalculator(model=model,use_device=self.device)
        elif self.calc_name == 'painn':
            from PaiNN.model import PainnModel
            from PaiNN.calculator import MLCalculator, EnsembleCalculator
            import torch
            model_pth = Path(self.calc_paths).rglob('*best_model.pth')
            models = []
            for each in model_pth:
                state_dict = torch.load(each, map_location=torch.device(self.device)) 
                model = PainnModel(
                    num_interactions=state_dict["num_layer"], 
                    hidden_state_size=state_dict["node_size"], 
                    cutoff=state_dict["cutoff"],
                    )
                model.to(self.device)
                model.load_state_dict(state_dict["model"],)    
                models.append(model)

            if len(models)==1:
                print('Using single PAINN model')
                ensemble = False
                calc = MLCalculator(models[0])
            elif len(models)>1:
                print('Using ensemble PAINN model')
                ensemble = True
                calc = EnsembleCalculator(models)
        
        elif self.calc_name == 'cpainn':
            from cPaiNN.model import PainnModel
            from cPaiNN.calculator import MLCalculator, EnsembleCalculator
            import torch
            model_pth = Path(self.calc_paths).rglob('*best_model.pth')
            print(self.calc_paths)
            models = []
            for each in model_pth:
                state_dict = torch.load(each, map_location=torch.device(self.device)) 
                model = PainnModel(
                    num_interactions=state_dict["num_layer"], 
                    hidden_state_size=state_dict["node_size"], 
                    cutoff=state_dict["cutoff"],
                    compute_forces=state_dict["compute_forces"],
                    compute_stress=state_dict["compute_stress"],
                    compute_magmom=state_dict["compute_magmom"],
                    compute_bader_charge=state_dict["compute_bader_charge"],
                    )
                model.to(self.device)
                model.load_state_dict(state_dict["model"],)    
                models.append(model)

            if len(models)==1:
                print('Using single PAINN model')
                ensemble = False
                calc = MLCalculator(models[0])
            elif len(models)>1:
                print('Using ensemble PAINN model')
                ensemble = True
                calc = EnsembleCalculator(models)
        else:
            raise RuntimeError('Calculator not found!')
        return calc


class Cathode:
    """
    Cathode is a class containing relavant fucntion for performing structural relaxation on cathode materials.
    """
    def __init__(self):
        pass
        

    def get_magmom(self,M:str,redox:bool) -> int:
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
        elif M=='Ga':
            magmom= 0
        else:
            raise ValueError(f'Magmom is not known for {M}')
        return magmom

    def remove_redox_psudo(self,metal_ion:str) -> str:
        """
        Mapping pseudo redox metal ions to the real metal ions. Following mapping is used:
            Ga -> Fe
            In -> Mn
            Ti -> Ni
            Al -> Co
        
        Args:
            metal_ion (str): pseudo redox metal ion
        Returns:
            str: real metal ion
        """
        if metal_ion =='Ga':
            return 'Fe'
        elif metal_ion == 'In':
            return 'Mn'
        elif metal_ion =='Ti':
            return 'Ni'
        elif metal_ion =='Al':
            return 'Co'
        else:
            raise ValueError(f'Redox_sort is not known for {metal_ion}')

    def get_U_value(self,M:str) -> float:
        """ 
        Get the U value for the metal ion
        Ref: https://docs.materialsproject.org/methodology/materials-methodology/calculation-details/gga+u-calculations/hubbard-u-values
        Following mapping is used:
            Fe -> 5.3
            Mn -> 3.9
            Ni -> 6.2
            Co -> 3.32
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
        else:
            raise ValueError(f'U value is not known for {M}')
        return U_val
    def sort(atoms, tags=None,key=None):
        """
        Sort the atoms based on the tags
        Args:
            atoms (Atoms): The atoms to be sorted
            tags (list): The tags to sort the atoms
            key (function): The function to sort the tags
        Returns:
            Atoms: The sorted atoms
            list: The indices of the sorted
        """
        if tags is None:
            tags = atoms.get_chemical_symbols()
        else:
            tags = list(tags)
        deco = sorted([(tag, i) for i, tag in enumerate(tags)],key=key)
        indices = [i for tag, i in deco]
        return atoms[indices], indices
    
    def redox_sort_func(x):
        """
        Sort the metal ions based on the redox state
        Args:
            x (str): The metal ion
        Returns:
            int: The redox state of the metal ion
        """
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
