import os
import sys
import json
import time
import matplotlib.pyplot as plt
import numpy as np
import argparse
from random import randint
from multiprocessing import cpu_count
from tqdm import tqdm

import torch
from torch.optim import Adam

# from schnetpack import AtomsLoader, AtomsData
# from schnetpack.environment import TorchEnvironmentProvider
# from schnetpack.train import build_mse_loss

from schnetpack.datasets import QM9
from schnetpack.transform import ASENeighborList, CastTo32

# from mlsuite.novelty_models.random_network_distillation import RandomNetworkDistillation
from random_network_distillation import RandomNetworkDistillation
# from mlsuite.utils.torch_utils import weight_reset

from inp_helper import get_target_model, build_kld_loss, build_cossim_loss, build_ce_loss, build_kld_nov_score

PROJECT_PATH = "/home/jfinkbeiner/MasterThesis/projects/rnd_qm9"

def determine_argmax(db_name, property_):
    from ase.db import connect
    max_val = -np.inf
    argmax = 0
    # print(db_name)
    with connect(db_name) as db:
        # print(len(db))
        for row in tqdm(db.select()):
            val = row.data.get(property_)
            if val is None:
                val = row.get(property_)
            # print(val)
            if isinstance(val, np.ndarray):
                val = np.max(val)
            if val > max_val:
                max_val = val
                argmax = row.id
            print(val)
            
    return argmax

def determine_argmax_natoms(db_name):
    from ase.db import connect
    max_val = -np.inf
    argmax = 0
    # print(db_name)
    with connect(db_name) as db:
        # print(len(db))
        for row in tqdm(db.select()):
            if row.natoms > max_val:
                max_val = row.natoms
                argmax = row.id
                print(row.id, row.natoms, row.formula)
    return argmax

from ase.db.row import row2dct

def get_specific_configs(db_name, ids):
    from ase.db import connect
    with connect(db_name) as db:
        # configs = [row for row in tqdm(db.select(selection=ids))]
        configs = [row2dct(row) for row in tqdm(db.select(limit=4))]
        # limit
    return configs





def iter_to_num_epochs(iteration):
    return int(np.sqrt(128 / iteration) * 250)

def main(args):
    system = "qm9"
    rcut = 6.0 # LiCl: 5.5, NaCl: 6.0, KCl: 7.0
    model_type = "large-ext"
    learning_rate = 1e-3
    data_run = args.data_run
    samples_per_iter = 1
    nov_func_name = "cossim"

    model = get_target_model(model_type, rcut)
    optimizer = Adam
    optimizer_kwargs = dict(lr=learning_rate)
    buffer_db_name = os.path.join(PROJECT_PATH, f"qm9.db")

    # crit = torch.nn.KLDivLoss(reduction='none', log_target=True) # TODO correct output shape ?



    if nov_func_name == "kld":
        loss_fn = build_ce_loss(["target"])
        novelty_func = build_kld_nov_score()
    elif nov_func_name == "mse":
        loss_fn = build_mse_loss(["target"])
        crit = torch.nn.MSELoss(reduction='none')
        novelty_func = lambda target, pred: torch.sum(crit(pred, target), dim=-1)
    elif nov_func_name == "cossim":
        loss_fn = build_cossim_loss(["target"])
        crit = torch.nn.CosineSimilarity(dim=2)
        novelty_func = lambda target, pred: -(crit(target, pred)-1) 
    else:
        raise ValueError(f"unknown `nov_func_name`, got '{novelty_func}'")

    
    # TODO change back
    model_path = os.path.join(PROJECT_PATH, "test_distribs", system, f"{system}_{nov_func_name}_{model_type}_run{int(data_run)}_periter{samples_per_iter}_{randint(16**7, 16**8 - 1):x}")
    save_filename = os.path.join(model_path, f"selected_samples.json")

    transforms = [
        ASENeighborList(cutoff=rcut),
        # RemoveOffsets(QM9.U0, remove_mean=True, remove_atomrefs=True),
        CastTo32()
    ]

    rnd = RandomNetworkDistillation(
        model_path=model_path,
        model=model,
        transforms=transforms,
        # buffer_db_name=buffer_db_name,
        loss_fn=loss_fn,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        train_batch_size=8,
        eval_batch_size=128*8,
        epochs_per_iter=iter_to_num_epochs,
        samples_per_iter=samples_per_iter,
        max_iter=128, # TODO
        num_workers=0, #cpu_count(),
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        novelty_func=novelty_func,
        save_distrbs_interval=1, # TODO
    )

    # init_subset = [4000]
    # TODO which one to choose?!
    # init_subset = [determine_argmax(buffer_db_name, "energy_U0")-1] # TODO tmp inedx fix

    # init_subset = [determine_argmax_natoms(buffer_db_name)-1]
    init_subset = [1094-1] # longest thing, C9H20 or sth...
    # init_subset = [0] # CH4

    # import pprint
    # pprint.pprint(get_specific_configs(buffer_db_name, [0,1,2,3,4]))


    print(init_subset)
    # sys.exit()
    # init_subset = [150385-147249+1]
    target_subset=None

    # print(init_subset)
    # from ase.db import connect
    # with connect(buffer_db_name) as db:
    #    print(db.get(init_subset[0]-1).energy)
    #    print(db.get(init_subset[0]).energy)
    #    print(db.get(init_subset[0]+1).energy)

    # sys.exit()

    buffer_atomsdatamodule = QM9(
        '../qm9.db', 
        batch_size=10,
        test_batch_size=11,
        num_train=110000,
        num_val=10000,
        transforms=transforms
    )
    buffer_atomsdatamodule.prepare_data()
    buffer_atomsdatamodule.setup()

    iter_selections = rnd.select_iteratively(
        buffer_atomsdatamodule=buffer_atomsdatamodule, 
        init_subset=init_subset.copy(), 
        target_subset=target_subset, 
        return_iter_selections=True,
        save_selection_filename=save_filename,
    )

    with open(save_filename, "w") as jsonfile:
        json.dump(iter_selections, jsonfile)


def try_argmax():

    buffer_db_name = os.path.join("/home/jfinkbeiner/MasterThesis/projects/classical_proofs/data/Ar/Ar.db")
    name = "sss-energies-npt-4096samples-run3"
    
    from ase.db import connect
    property_ = "energies"
    max_val = -np.inf
    argmax = 0
#     print(buffer_db_name)
    with connect(buffer_db_name) as db:
#         print(db.get(1))
#         print(len(db))
        for row in db.select(name=name):
            val = row.data.get(property_)
            if val is None:
                val = row.get(property_)
#             print(val)
            if isinstance(val, np.ndarray):
                val = np.max(val)
            if val > max_val:
                argmax = row.id
                max_val = val
            print(row.id, val, row.energy)
    print(argmax)
    return argmax



if __name__ == "__main__":
    # try_argmax()

    print("\ncuda")
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    # sys.exit()


    parser = argparse.ArgumentParser(description='Select configurations via the Random Network Distillation method.')
    parser.add_argument('--data_run', type=int, default=1, help='which data-creation-run to use.')
    args = parser.parse_args()

    main(args)
