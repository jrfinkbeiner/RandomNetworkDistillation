import numpy as np
from copy import deepcopy
from ase.db import connect
from ase import Atoms
import schnetpack.properties as Properties


class Evaluator:
    def __init__(self, model, dataloader):
        """
        Base class for model predictions.

        Args:
            model (torch.nn.Module): trained model
        """
        self.model = model
        self.dataloader = dataloader

    def _get_predicted(self, device):
        """
        Calculate the predictions for the dataloader.

        Args:
            device (str): cpu or cuda

        Returns:

        """
        predicted = {}
        for ibatch, batch in enumerate(self.dataloader):
            # build batch for prediction
            batch = {k: v.to(device) for k, v in batch.items()}
            # predict
            result = self.model(batch)
            # store prediction batches to dict
            for p in result.keys():
                value = deepcopy(result[p].cpu().detach().numpy())
                if p in predicted.keys():
                    predicted[p].append(value)
                else:
                    predicted[p] = [value]
            # store positions, numbers and mask to dict
            for p in [Properties.R, Properties.Z, Properties.cell]:
                value = deepcopy(batch[p].cpu().detach().numpy())
                if p in predicted.keys():
                    predicted[p].append(value)
                else:
                    predicted[p] = [value]

        max_shapes = {
            prop: max([list(val.shape) for val in values])
            for prop, values in predicted.items()
        }
        for prop, values in predicted.items():
            max_shape = max_shapes[prop]
            predicted[prop] = np.vstack(
                [
                    np.lib.pad(
                        batch,
                        [
                            [0, add_dims]
                            for add_dims in max_shape - np.array(batch.shape)
                        ],
                        mode="constant",
                    )
                    for batch in values
                ]
            )

        return predicted

    def evaluate(self, device):
        raise NotImplementedError


class NPZEvaluator(Evaluator):
    def __init__(self, model, dataloader, out_file):
        self.out_file = out_file
        super(NPZEvaluator, self).__init__(model=model, dataloader=dataloader)

    def evaluate(self, device):
        predicted = self._get_predicted(device)
        np.savez(self.out_file, **predicted)


class DBEvaluator(Evaluator):
    def __init__(self, model, dataloader, out_file):
        self.dbpath = dataloader.dataset.dbpath
        self.out_file = out_file
        super(DBEvaluator, self).__init__(model=model, dataloader=dataloader)

    def evaluate(self, device):
        predicted = self._get_predicted(device)
        positions = predicted.pop(Properties.R)
        atomic_numbers = predicted.pop(Properties.Z)
        atom_masks = predicted.pop(Properties.atom_mask).astype(bool)
        cell = predicted.pop(Properties.cell) # TODO really always in there ?

        with connect(self.out_file) as conn:
            for i, mask in enumerate(atom_masks):
                z = atomic_numbers[i, mask]
                r = positions[i, mask]
                ats_kwargs = {
                    "numbers": z, 
                    "positions": r,
                    "cell": None if cell is None else cell[i],
                    "pbc": None if cell is None else True # TODO not always correct...
                }
                ats = Atoms(**ats_kwargs)
                data = {
                    prop: self._unpad(mask, values[i])
                    for prop, values in predicted.items()
                }
                conn.write(ats, data=data)

    def _unpad(self, mask, values):
        if len(values.shape) == 1:
            return values
        return values[mask]


class DBEvaluatorStats(Evaluator):
    def __init__(self, model, dataloader, out_file):
        self.dbpath = dataloader.dataset.dbpath
        self.out_file = out_file
        super().__init__(model=model, dataloader=dataloader)

    def evaluate(self, device, stat_properties):
        predicted = self._get_predicted(device)
        positions = predicted.pop(Properties.R)
        atomic_numbers = predicted.pop(Properties.Z)
        atom_masks = predicted.pop(Properties.atom_mask).astype(bool)
        cell = predicted.pop(Properties.cell) # TODO really always in there ?

        unique_numbers = np.unique(atomic_numbers)
        atomic_num_masks = {}
        means = {stat_prop: {} for stat_prop in stat_properties}
        stddevs = {stat_prop: {} for stat_prop in stat_properties}

        for unique_num in unique_numbers:
            atomic_num_masks[int(unique_num)] = atomic_numbers == unique_num
            for stat_prop in stat_properties:
                masked_values = predicted[stat_prop][atomic_num_masks[int(unique_num)], :]
                means[stat_prop][int(unique_num)] = np.mean(masked_values, axis=0)
                stddevs[stat_prop][int(unique_num)] = np.std(masked_values, axis=0, ddof=1)

                predicted[stat_prop][atomic_num_masks[int(unique_num)], :] -= means[stat_prop][int(unique_num)]
                predicted[stat_prop][atomic_num_masks[int(unique_num)], :] /= stddevs[stat_prop][int(unique_num)]

        with connect(self.out_file) as conn:
            for i, mask in enumerate(atom_masks):
                z = atomic_numbers[i, mask]
                r = positions[i, mask]
                ats_kwargs = {
                    "numbers": z,
                    "positions": r,
                    "cell": None if cell is None else cell[i],
                    "pbc": None if cell is None else True # TODO not always correct...
                }
                ats = Atoms(**ats_kwargs)
                data = {
                    prop: self._unpad(mask, values[i])
                    for prop, values in predicted.items()
                }
                conn.write(ats, data=data)

        return means, stddevs

    def _unpad(self, mask, values):
        if len(values.shape) == 1:
            return values
        return values[mask]

