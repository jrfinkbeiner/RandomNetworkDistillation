import os
import sys
import copy
import json
import pickle
import warnings
from multiprocessing import cpu_count
from typing import Optional, List
import torch
import numpy as np
from tqdm import tqdm 

from ase.db import connect

import schnetpack as spk
import schnetpack.properties as structure
from schnetpack.data import AtomsLoader, ASEAtomsData
from schnetpack.transform import ASENeighborList, RemoveOffsets, CastTo32
from schnetpack.data.loader import _atoms_collate_fn
# from schnetpack1.0.evaluation import DBEvaluator, DBEvaluatorStats

# from mlsuite.novelty_models.novelty_model import NoveltyModel
from mlsuite.datacreation.data_selection.select_samples import select_configs_uniform_quantity
from mlsuite.ase_io.db_utils import read_properties_from_db
from mlsuite.utils.torch_utils import weight_reset, optimizer_to
from mlsuite.utils.statistics import combine_mean_var_from_subsets

from utils.profiler import profile # TODO to be del

# TODO change model_path to tmp_path and 

class RandomNetworkDistillation:
    """
    Class to iteratively select new configurations based on a novelty score. 
    Just call `select_iteratively` which internally interatively calls the `update` method, which selects new condifurations and retrains the model.
    """
    def __init__(self, 
            model_path: str,
            model: torch.nn.Module,
            transforms,
            loss_fn,
            optimizer,
            optimizer_kwargs,
            train_batch_size: int,
            eval_batch_size: int,
            epochs_per_iter: int,
            samples_per_iter: int,
            max_iter: Optional[int],
            num_workers: Optional[int] = None,
            device=None,
            novelty_func=None,
            contrib_mask:str=None,
            target_model: Optional[torch.nn.Module] = None,
            save_distrbs_interval: int = 0,
            # target_name: str = None,
        ):

        # if target_name is None: # TODO just implement like this
        #     target_name = "target"
        print("WARNING: experimental use only! Implementations is not finished yet.") # TODO 
        assert model.output_modules[0].per_atom_output_key is not None
        self._model_path = model_path
        self._pred_model = model
        if target_model is None:
            target_model = copy.deepcopy(model)
            target_model.apply(weight_reset)
        self._target_model = target_model
        self._db_name = os.path.join(model_path, "targets.db")       # TODO not two databases but a changing subset!
        # self._selected_db_name = os.path.join(model_path, "selected.db")    #

        self.transforms = transforms
        self.loss_fn = loss_fn
        self._optimizer = optimizer
        self._optimizer_kwargs = optimizer_kwargs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self._num_epochs_per_iter = epochs_per_iter
        self.samples_per_iter = samples_per_iter
        if max_iter is None:
            max_iter
        self.max_iter = max_iter if max_iter is not None else sys.maxsize
        self.num_workers = num_workers if num_workers is not None else cpu_count()
        if novelty_func is None:
            kld = torch.nn.KLDivLoss(reduction='none') # TODO correct output shape ?
            novelty_func = lambda target, pred: torch.sum(kld(target, pred), dim=-1) # TODO check correct dim
        self.novelty_func = novelty_func
        if contrib_mask is None:
            contrib_mask = "_atom_mask"
        self._contrib_mask = contrib_mask
        self.save_distrbs_interval = save_distrbs_interval
        self._save_distrbs = bool(save_distrbs_interval)
    @property
    def model_path(self):
        return self._model_path

    @property
    def target_model(self):
        return self._target_model
    
    @property
    def pred_model(self):
        return self._pred_model

    @property
    def db_name(self):
        return self._db_name

    @property
    def selected_subset(self):
        return self._selected_subset

    def epochs_per_iter(self, iteration: Optional[int] = None):
        self._num_epochs_per_iter 
        if isinstance(self._num_epochs_per_iter, (int, float)):
            return self._num_epochs_per_iter
        else:
            assert iteration is not None, "Iteration argument has to be given, if epochs func is used."
            return self._num_epochs_per_iter(iteration)

    def _create_target_db(self, buffer_atomsdatamodule):
        # TODO assuming property name is `"target"`...

        buffer_db_name = buffer_atomsdatamodule.datapath
        
        self.target_model.to(self.device)
        
        # create ase database 


                
        # dataset = ASEAtomsData(
        #     buffer_db_name, # TODO make sure deprecation_update is not triggered
        #     subset=subset,
        #     load_properties=[],
        #     # environment_provider=TorchEnvironmentProvider(cutoff=self.cutoff, device="cpu"), # device), # TODO figure out what env provider and which device to use...
        # )
        # import time
        # start = time.time()
        # dataloader = AtomsLoader(dataset, batch_size=self.eval_batch_size, num_workers=self.num_workers, shuffle=False)
        # # dataloader = self.get_dataloader(train=False)
        # os.makedirs(os.path.dirname(self.db_name), exist_ok=True)
        # # dbevaluator = DBEvaluator(self.target_model, dataloader, self.db_name)
        # dbevaluator = DBEvaluatorStats(self.target_model, dataloader, self.db_name)
        # means, stddevs = dbevaluator.evaluate(self.device, ["target"])
        # end = time.time()
        # print(f"predictions took: {end-start}s")

        # stats_dir = os.path.join(self.model_path, "target_stats")
        # os.makedirs(stats_dir, exist_ok=True)
        # with open(os.path.join(stats_dir, "means.pkl"), "wb") as f_:
        #     pickle.dump(means["target"], f_)
        # with open(os.path.join(stats_dir, "stddevs.pkl"), "wb") as f_:
        #     pickle.dump(stddevs["target"], f_)
        
        # # np.savez(os.path.join(stats_dir, "means.npz"), **means["target"])
        # # np.savez(os.path.join(stats_dir, "stddevs.npz"), **stddevs["target"])


        # with connect(buffer_db_name) as conn:
        #     metadata = conn.metadata

        metadata = buffer_atomsdatamodule.train_dataset.metadata
        property_unit_dict = {**metadata["_property_unit_dict"], "target": None}
        atomrefs = {**metadata["atomrefs"], "target": [0.0]*len(metadata["atomrefs"]["energy_U0"])}

        # print(property_unit_dict.keys())
        # sys.exit()


        # # print(self.db_name)
        # # sys.exit()
        os.makedirs(os.path.dirname(self.db_name), exist_ok=True)
        # with connect(self.db_name) as conn:
        #     conn.metadata = {
        #         "_property_unit_dict": property_unit_dict,
        #         "_distance_unit": metadata["_distance_unit"],
        #         "atomrefs": atomrefs,
        #     }

        # target_database = ASEAtomsData(self.db_name)
        target_dataset = ASEAtomsData.create(self.db_name, distance_unit=metadata["_distance_unit"], property_unit_dict=property_unit_dict, atomrefs=atomrefs)
        target_dataset.transforms = self.transforms

        # batchsize = buffer_atomsdatamodule.test_batch_size
        dataloader = AtomsLoader(
            buffer_atomsdatamodule.train_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=buffer_atomsdatamodule._pin_memory,
            collate_fn=lambda x: x,
        )

        torch.save(self.target_model, os.path.join(self.model_path, "target_model"))

        for list_batch in tqdm(dataloader):
            batch = _atoms_collate_fn(list_batch)
            num_atoms_cumsum = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(batch[structure.n_atoms])])
            batch = {k: v.to(self.device) for k, v in batch.items()}
            result = self.target_model(batch)
            targets = result["target"].detach().cpu().numpy()

            for l in list_batch:
                del l["_idx"]
                del l["_n_atoms"]
                del l["_idx_i"]
                del l["_idx_j"]
                del l["_offsets"]
                l[structure.cell] = l[structure.cell].squeeze()

            dict_list_results = [{**{key:val.numpy() for key,val in sample_dict.items()}, "target": targets[num_atoms_cumsum[i]:num_atoms_cumsum[i+1]]} for i,sample_dict in enumerate(list_batch)]
            # # dict_list_results = [{k: v.squeeze().numpy() for k, v in sample.items()} for sample in dict_list_results]
            # # batched_systems = {k: v.cpu() for k, v in batch.items()}
            # import pprint
            # pprint.pprint(list_batch)
            # # print({k: v.shape for k, v in batch.items()})
            # pprint.pprint([{k: v.squeeze().shape for k, v in sample.items()} for sample in dict_list_results])
            # pprint.pprint([{k: type(v.squeeze()) for k, v in sample.items()} for sample in dict_list_results])
            # # sys.exit()
            # # dict_list_results = [{k: v[i:i+1] for k, v in batched_systems.items()} for i in range(batchsize)]
            # # probably have to change structure
            dict_list_results
            target_dataset.add_systems(dict_list_results)
            sys.exit()


    def compute_novelty_scores(self, dataloader, return_all: bool = False):
        novelty_scores = np.empty(len(dataloader.dataset))
        novelty_means = np.empty_like(novelty_scores)
        novelty_vars = np.empty_like(novelty_scores)
        novelty_num_contribs = np.empty_like(novelty_scores)
        contrib_ids = np.empty(len(dataloader.dataset), dtype=np.uint32)
        start_i = 0
        end_i = 0
        all_novs = []
        self.pred_model.to(self.device)
        with torch.no_grad():
            for ibatch, batch in enumerate(dataloader):
                # build batch for prediction
                batch = {k: v.to(self.device) for k, v in batch.items()}
                # predict
                target = batch[self.target_model.output_modules[0].per_atom_output_key]
                pred_target = self.pred_model(batch)[self.pred_model.output_modules[0].per_atom_output_key].detach() # TODO deepcopy ?
                novelty_arr = self.novelty_func(target, pred_target) # apply atom_mask already here?
            
                start_i = end_i
                end_i += novelty_arr.shape[0]
            
                atom_mask = batch[self._contrib_mask].byte()

                # argmax_ids = torch.argmax(novelty_arr, dim=1)
                max_vals, max_ids = torch.max(novelty_arr, dim=1)
                novelty_scores[start_i:end_i] = max_vals.cpu()
                contrib_ids[start_i:end_i] = max_ids.cpu()
                # novelty_scores[start_i:end_i] = novelty_arr[:, argmax_ids].cpu()
                novelty_means[start_i:end_i] = torch.mean(novelty_arr, dim=-1).cpu()
                novelty_vars[start_i:end_i] = torch.var(novelty_arr, dim=-1, unbiased=False).cpu()
                # TODO figure out how to use with atom_mask...
                # argmax_ids = novelty_arr[atom_mask].argmax(axis=1)
                # novelty_scores[start_i:end_i] = novelty_arr[atom_mask][:,argmax_ids]
                # novelty_means[start_i:end_i] = torch.mean(novelty_arr[atom_mask], dim=-1)
                # novelty_vars[start_i:end_i] = torch.var(novelty_arr[atom_mask], dim=-1, unbiased=False)
                novelty_num_contribs[start_i:end_i] = torch.sum(atom_mask, dim=-1).cpu()
                # contrib_ids[start_i:end_i] = argmax_ids

                if return_all or self._save_distrbs:
                    all_novs.append(novelty_arr.cpu().numpy()) # TODO deepcopy ?
                    # all_novs.append(novelty_arr[atom_mask].cpu().numpy()) # TODO deepcopy ? # TODO atommask ?

        stats_dict = {
            "mean": novelty_means,
            "var": novelty_vars,
            "num_contribs": novelty_num_contribs,
        }

        if return_all or self._save_distrbs:
            return novelty_scores, contrib_ids, stats_dict, np.concatenate(all_novs, axis=0) # TODO how to handle 
        else:
            return novelty_scores, contrib_ids, stats_dict

    def init_database(self, buffer_atomsdatamodule, init_subset: List[int], target_subset: Optional[List[int]] = None):
        if os.path.exists(self.db_name):
            warnings.warn("Target databse already exists. Make sure nothing wrong happens...", UserWarning)
        else:
            self._create_target_db(buffer_atomsdatamodule)
        self._selected_subset = init_subset

    def get_dataloader(self, train: bool = True, batch_size: int = None):
        if batch_size is None:
            if train:
                num = len(self.selected_subset)
                batch_size = min(self.train_batch_size, max(round(num**(1/2)), int(num/round(num**(1/2))))) if num != 2 else 1
            else:
                batch_size = self.eval_batch_size

        if train:
            subset = self.selected_subset
            shuffle=True
        else:
            shuffle=False
            subset=None

        print(self.pred_model.output_modules[0].per_atom_output_key)
        # sys.exit()
        dataset = ASEAtomsData(
            self.db_name, 
            load_properties=["target"], # TODO new what does this do ?
            transforms=self.transforms,
            # environment_provider=TorchEnvironmentProvider(cutoff=self.cutoff, device="cpu"), # device), # TODO figure out what env provider and which device to use...
            subset_idx=subset,
        )

        dataloader = AtomsLoader(dataset, batch_size=batch_size, num_workers=self.num_workers, shuffle=shuffle)
        return dataloader

    def update(self, iter_id=None):

        if self._save_distrbs:
            assert iter_id is not None

        # run training
        self.pred_model.apply(weight_reset) # reset weights to ensure good convergence
        save_file = os.path.join(self.model_path, "training", f"losses_iter{str(iter_id).zfill(4)}.txt")
        self.train(save_file=save_file)
 

        # compute novelty scores
        dataloader = self.get_dataloader(train=False)
        if self._save_distrbs:
            novelty_scores, contrib_ids, stats_dict, all_nov_scores = self.compute_novelty_scores(dataloader)
        else:
            novelty_scores, contrib_ids, stats_dict = self.compute_novelty_scores(dataloader)
        

        # self._selected_subset = [0,1,2,3,4,5,6,7]
        # novelty_scores, contrib_ids = np.random.random(16)+0.7, np.random.randint(16, size=16)
        # novelty_scores[self._selected_subset] -= 0.7
        # print()
        ids = np.arange(len(novelty_scores))

        train_nov_mean, train_nov_var = combine_mean_var_from_subsets(
            stats_dict["mean"][self.selected_subset],
            stats_dict["var"][self.selected_subset],
            stats_dict["num_contribs"][self.selected_subset],
            ddof=1,
         ) 
        train_nov_std = np.sqrt(train_nov_var)

        not_train_mask = np.ones(len(novelty_scores), dtype=bool)
        not_train_mask[self.selected_subset] = False

        unseen_novelty_scores = novelty_scores[not_train_mask]
        unseen_contrib_ids = contrib_ids[not_train_mask]
        unseen_ids = ids[not_train_mask]

        # # TODO number of samples based approach
        # sort_ids = np.argsort(unseen_novelty_scores)
        # new_selection_ids = unseen_ids[sort_ids][-100*samples_per_iter:]

        # TODO novelty magnitude/stddev based approach
        threshhold = 0.0 # train_nov_mean+1*train_nov_std # TODO
        nov_ids = np.argwhere(unseen_novelty_scores > threshhold).flatten()
        new_selection_ids = unseen_ids[nov_ids]
       
        if self._save_distrbs:
            self._create_novelty_hists(novelty_scores, all_nov_scores, iter_id+1, threshhold=threshhold)

        if len(new_selection_ids) > self.samples_per_iter:
            converged = False
            if self.samples_per_iter == 1:
                new_nov_scores = unseen_novelty_scores[nov_ids]
                max_id = np.argmax(new_nov_scores)
                selected_ids = np.array(new_selection_ids[max_id]) # np.array sto be consistent in selected_ids type
            else: # TODO not tested
                # TODO rewrite sample selector to choose samples here
                all_energies_list = read_properties_from_db(self.db_name, ["energies"], read_ids=new_selection_ids)["energies"]
                energies = [eners[contrib_id] for contrib_id,eners in zip(unseen_contrib_ids[new_selection_ids],all_energies_list)]
                selected_ids = select_configs_uniform_quantity(self.samples_per_iter, energies, unseen_ids)
        else:
            converged = True
            selected_ids = new_selection_ids
        
        selected_ids_list = selected_ids.astype(int).tolist()

        if not isinstance(selected_ids_list, list):
            selected_ids_list = [int(selected_ids_list)]

        self._selected_subset.extend(selected_ids_list.copy())
        return selected_ids_list, converged

    def train(self, save_file: Optional[str] = None):
        dataloader = self.get_dataloader(train=True)

        self.pred_model.to(self.device)
        optimizer = self._optimizer(self.pred_model.parameters(), **self._optimizer_kwargs)
        optimizer_to(optimizer, self.device)

        dataset_size = len(dataloader.dataset)
        loss_list = []
        self.pred_model.to(self.device)
        
        for epoch in range(self.epochs_per_iter(dataset_size)):

            running_loss = 0
            for train_batch in dataloader:
                optimizer.zero_grad()

                # move input to gpu, if needed
                train_batch = {k: v.to(self.device) for k, v in train_batch.items()}

                result = self.pred_model(train_batch)
                loss = self.loss_fn(train_batch, result)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            loss_list.append(running_loss/dataset_size)
        
        if save_file is not None:
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            np.savetxt(save_file, loss_list)


        return loss_list

    def _create_novelty_hists(self, max_novelty_scores, all_novelty_scores, iter_id, threshhold=None):
        path = os.path.join(self.model_path, "novelty_distribs")
        os.makedirs(path, exist_ok=True)
        
        # TODO remove
        np.save(os.path.join(path, f"novelty_distrb_all_iter{str(iter_id).zfill(3)}.npy"), all_novelty_scores)
        np.save(os.path.join(path, f"novelty_distrb_max_iter{str(iter_id).zfill(3)}.npy"), all_novelty_scores)
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        not_train_mask = np.ones(len(max_novelty_scores), dtype=bool)
        not_train_mask[self.selected_subset] = False

        train_novs = all_novelty_scores[self.selected_subset].flatten()
        eval_novs = all_novelty_scores[not_train_mask].flatten()

        eval_scores_max = max_novelty_scores[not_train_mask].flatten()
        train_scores_max = max_novelty_scores[self.selected_subset].flatten()

        plt.figure()
        plt.title("all novelty scores")
        plt.hist(eval_novs, bins=int(np.sqrt(len(eval_novs))), label="eval")
        plt.hist(train_novs, bins=int(np.sqrt(len(train_novs))), label=f"train ({len(train_scores_max)} configs)")
        if threshhold is not None:
            # ylim = plt.gca().get_ylim()
            ylim = 2e4 # TODO
            plt.vlines(threshhold, 0, ylim, linestyles="dashed")
        plt.yscale("log")
        plt.legend()
        plt.savefig(os.path.join(path, f"novelty_distrb_all_iter{str(iter_id).zfill(3)}.pdf"))
        plt.close()

        plt.figure()
        plt.title("max novelty scores")
        plt.hist(eval_scores_max, bins=int(np.sqrt(len(eval_scores_max))), label="eval")
        plt.hist(train_scores_max, bins=int(np.sqrt(len(train_scores_max))), label=f"train ({len(train_scores_max)} configs)")
        if threshhold is not None:
            # ylim = plt.gca().get_ylim()
            ylim = 1e3 # TODO
            plt.vlines(threshhold, 0, ylim, linestyles="dashed")
        plt.yscale("log")
        plt.legend()
        plt.savefig(os.path.join(path, f"novelty_distrb_max_iter{str(iter_id).zfill(3)}.pdf"))
        plt.close()

    # @profile(output_file="/auto.eland/home/jfinkbeiner/MasterThesis/projects/novelty_methods/scripts/profile_select_iter.txt", sort_by='cumulative', lines_to_print=None, strip_dirs=False)
    def select_iteratively(self, buffer_atomsdatamodule, init_subset: List[int], target_subset: Optional[List[int]] = None, return_iter_selections=False, save_selection_filename: Optional[str] = None):
        
        self.init_database(buffer_atomsdatamodule, init_subset, target_subset)
        self._save_distrbs = bool(self.save_distrbs_interval)

        if self._save_distrbs:
            dataloader = self.get_dataloader(train=False)
            novelty_scores, _, _, all_nov_scores = self.compute_novelty_scores(dataloader)
            self._create_novelty_hists(novelty_scores, all_nov_scores, 0)

        if return_iter_selections:
            iter_selections = [self.selected_subset.copy()]

        for iiter in range(self.max_iter):
            if bool(self.save_distrbs_interval) and (((iiter+1)/self.save_distrbs_interval % 1 == 0 and iiter!=1) or iiter==0):
                self._save_distrbs = True
            else:
                self._save_distrbs = False
            selected_ids, converged = self.update(iiter)
            if return_iter_selections:
                iter_selections.append(selected_ids)

            with open(save_selection_filename, "w") as save_file:
                if return_iter_selections:
                    save_vals = iter_selections
                else:
                    save_vals = iter_selections
                json.dump(save_vals, save_file)

            if converged:
                break

        if return_iter_selections:
            return iter_selections
        else:
            return self.selected_subset
