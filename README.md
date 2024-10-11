# autonomous_cluster_expansion
![plot](PQ_graphs.pdf)

Autonomouse cluster expansion workflow. This workflow utilizes [PerQueue](https://gitlab.com/asm-dtu/perqueue) wokflow manager to autonomousely train a cluster expansion model from the [CLEASE](https://clease.readthedocs.io/en/stable/index.html) packages based on a particular material. The python version requirement is 3.11 and other package requirement are available in 'requirement.txt'.


To intall required packages:
```bash
pip install -r requirement.txt
```

Afterwards PerQueue needs to be initialized:
```bash
pq init
```

The 'pq_submit.py' file is used to initialize the workflow. All parameters need to train the cluster expansion model needs to be edited in here before starting the workflow. As an example the 'pq_submit.py' is set to train a cluster expansion model for NaFeMnPO4, where Na can be replaced with a vacant site (X) and Fe/Mn can swap site. The example cluster expansion model can be used to model disorder in NaFeMnPO4.

The 'config.toml' file is the main configuration file containing the VASP parameters used for the structure relaxation, if machine learning optimization is not used. This file can be empty but should not be deleted.

The '/workflow' folder contains all scripts needed to run the workflow. If one needs to perform single structure optimizations outside the workflow '/workflow/single_relaxation.py' can be used. As an example case: 'python workflow/single_relaxation.py --run_path="./NaFeMnPO4_CE_model" --db_path="./NaFeMnPO4_CE_model/structures.db" --index=2 --cfg_path="./NaFeMnPO4_CE_model/config.toml"

The 'CE_training_model.ipynb' is used to manually train an cluster expansion model based on the generated data. Some times a human touch is needed to make the cluster expansion model fit perfectly to the dataset and this script provide the ability to do so.

To initialize the workflow:
```bash
python pq_submit.py
```

The wokflow will generate a subfolder where the cluster expansion will be trained. In the folder a local configuration file is created 'config.toml'. The parameters used in the workflow can be changed in this file while runing the workflow, making it easy to change settings during iterations.

When the cluster expansion model is done a monte carlo simualtion can be performed along with a kinetic monte carlo method. THIS NEEDS FURTHER CHANGES BEFORE IT CAN BE USED