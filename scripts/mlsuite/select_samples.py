import numpy as np
from scipy import stats

def select_configs_uniform_quantity(num_samples: int, quantities: np.ndarray, system_ids: np.ndarray):

    assert len(quantities) == len(system_ids)

    selected_system_ids = np.empty(num_samples, dtype=np.uint32)
    num_selected_samples = 0

    trys = 0
    while num_samples > num_selected_samples:
        
        bin_edges = np.histogram_bin_edges(quantities, bins=num_samples-num_selected_samples)
        hist, bins, binnumbers = stats.binned_statistic(quantities, quantities, 'count', bins=bin_edges)

        for ibin in range(len(hist)):

            # check for empty bin
            if hist[ibin] == 0:
                continue

            mask = binnumbers == ibin+1
            glob_sys_ids = system_ids[mask]
            chosen_sys_id = np.random.choice(glob_sys_ids, 1)[0]

            if chosen_sys_id not in selected_system_ids[:num_selected_samples]: # TODO could be optimized
                
                selected_system_ids[num_selected_samples] = chosen_sys_id
                num_selected_samples+=1
        trys += 1
        if trys > 100:
            raise RuntimeError(f"Unable to select {num_samples} samples. Maybe too little configurations are given, got {len(set(system_ids))}.")
    return selected_system_ids