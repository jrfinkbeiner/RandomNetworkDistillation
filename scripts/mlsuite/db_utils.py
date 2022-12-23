from random import randint
from typing import List, Union, Dict, Iterable, Optional
import itertools
import numpy as np
from ase.db import connect
from ase.io.trajectory import Trajectory
from ase import Atoms

from mlsuite.utils.helper import check_file_exist_ask_overwrite

import sys
import os

from typing import Iterable

def find_string_from_partial(partial_str, strings: Iterable[str]):
    for string in strings:
        if partial_str in string:
            return string
    return None


def query_yes_no(question, default=None):
    """Ask a yes/no question via input() and return their answer.

    Args:
        question (str): string that is presented to the user.
        default (str): the presumed answer if the user just hits <Enter>.
            It must be "yes", "no" or None (the default) (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def check_file_exist_ask_overwrite(filename: str, overwrite_default="no"):
    if os.path.exists(filename):
        overwrite = query_yes_no(f"Choice of destination trajectory-file already exists. Got {filename}. \nDo you want to override?", default=overwrite_default)
        # raise ValueError(f"Choice for destination trajectory-file already exists. Choose a different one, or delete the old one. Got {filename}")
        if not overwrite:
            print("As destination trajectory-file already exists and the file should not be overwritten, process will be exited without saving data to a file.")
            sys.exit()


def write_atoms_to_db(
        atoms: Union[Atoms, List[Atoms]], 
        open_db: Optional[str] = None, 
        db_name: Optional[str] = None,
        data: Optional[Dict] = None,
        key_value_pairs: Optional[Union[Dict, List[Dict]]] = None, 
        unique_key_value_pairs: bool = False,
    ):
    """
    Writes 
    """
    if isinstance(atoms, Atoms):
        atoms = [atoms]
        # if isinstance(key_value_pairs, dict):
        #     key_value_pairs = [key_value_pairs]
    if isinstance(key_value_pairs, dict): # and len(atoms) > 1:
        key_value_pairs = itertools.repeat(key_value_pairs, len(atoms))
    elif key_value_pairs is None:
        key_value_pairs = itertools.repeat({}, len(atoms))
    else:
        assert isinstance(key_value_pairs, list)
        assert len(atoms) == len(key_value_pairs)
    if data is None:
        data = itertools.repeat({}, len(atoms))
    else:
        if isinstance(data, dict):
            data = [data]
        assert len(data) == len(atoms)
    if open_db is not None:
        ids = _write_atoms_keys_to_db(open_db, atoms, data, key_value_pairs, unique_key_value_pairs)
    else:
        assert db_name is not None, "If no already opened database `open_db` is given, `db_name` has to be specified."
        with connect(db_name) as db:
            ids = _write_atoms_keys_to_db(db, atoms, data, key_value_pairs, unique_key_value_pairs)
    return ids

def _write_atoms_keys_to_db(db, atoms, data, key_value_pairs, unique_key_value_pairs):
    ids = [None]*len(atoms)
    for iat, (at,keyp, dat) in enumerate(zip(atoms, key_value_pairs, data)):
        if unique_key_value_pairs:
            id_ = db.reserve(key_value_pairs=keyp)
            kwargs = dict(id=id_, key_value_pairs=keyp)
        else:
            kwargs = dict(key_value_pairs=keyp)

        if ("energies" in at.calc.results.keys()) and ("energies" not in dat):
            dat["energies"] = at.get_potential_energies()

        ids[iat] = db.write(at, **kwargs, data=dat)
    return ids
        
def write_ase_trajectory(
        traj_filename: str,
        system: List[Atoms],
        mode='a',
        overwrite=False,
    ):

    if mode == 'w' and not overwrite:
        check_file_exist_ask_overwrite(filename=traj_filename, overwrite_default="no")

    traj = Trajectory(
        filename=traj_filename,
        mode=mode,
    )

    for atoms in system:
        traj.write(atoms)


def read_properties_from_db(db_name: str, properties: Union[str, List[str]], read_ids: Optional[Iterable[int]] = None, **read_kwargs):
    if isinstance(properties, str):
        properties = [properties]
    properties_dict = {prop: [] for prop in properties}
    
    def append_prop_dict_from_row(row):
        for prop in properties:
            val = row.data.get(prop)
            if val is None:
                val = row.get(prop)
            properties_dict[prop].append(val)
    
    with connect(db_name) as db:
        if read_ids is None:
            for row in db.select(**read_kwargs):
                append_prop_dict_from_row(row)
        else:
            for read_id in read_ids:
                row = db.get(id=int(read_id), **read_kwargs) # TODO what are **read_kwargs doing if id is specified...?!
                append_prop_dict_from_row(row)
    return properties_dict


def get_properties_from_row(row, properties: List[str]):
    data = {}
    for prop in properties:
        val = row.data.get(prop)
        if val is None:
            val = getattr(row, prop)    
        if isinstance(val, (float, int)):
            data[prop] = np.array([val], dtype=np.float64)
        else:
            data[prop] = np.asarray(val, dtype=np.float64)
    return data

def transfer_db_data(
        db_name_read: str,
        db_name_write: str,
        read_ids: Iterable[int] = None,
        read_kwargs: Optional[Dict] = None,
        write_kwargs: Optional[Dict] = None,
        properties_to_data: Optional[List["str"]] = None,
        reroll_unique_ids: bool = False,
    ):

    if read_kwargs is None:
        read_kwargs = {}
    if write_kwargs is None:
        write_kwargs = {}
    with connect(db_name_read) as db_read:
        with connect(db_name_write) as db_write:
            transfer_ids = []
            if read_ids is None:
                rows = db_read.select(**read_kwargs)
                for row in rows:
                    if properties_to_data is not None:
                        kwargs = {"data": get_properties_from_row(row, properties_to_data)}
                    else:
                        kwargs = {}
                    if reroll_unique_ids:
                        row.unique_id = '%x' % randint(16**31, 16**32 - 1)
                    transfer_ids.append(db_write.write(row, key_value_pairs=write_kwargs, **kwargs))
            else:
                for read_id in read_ids:
                    row = db_read.get(id=int(read_id), **read_kwargs) # TODO what are **read_kwargs doing if id is specified...?!
                    if reroll_unique_ids:
                        row.unique_id = '%x' % randint(16**31, 16**32 - 1)

                    if properties_to_data is not None:
                        kwargs = {"data": get_properties_from_row(row, properties_to_data)}
                    else:
                        kwargs = {}
                    transfer_ids.append(db_write.write(row, key_value_pairs=write_kwargs, **kwargs)) # TODO possibly merge needed due to row object having kwargs...? or overwrite of attributes?
    return transfer_ids

