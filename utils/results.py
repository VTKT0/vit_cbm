import os
import json
import numpy as np
import pandas as pd
from vit_cbm.utils import get_subdirs, get_config_info, get_results

log_path = '/scratch/prj/ccc_vit_finetuning/vit_cbm/results'

c_index_list = []

# get a list of all the experiment runs
all_dirs = get_subdirs(log_path)

## exclude debug runs etc
exclusions = [x for x in all_dirs if 'debug' in x]


exclusions = list(set(exclusions))

for d in exclusions:
    print(f'Excluding {d}')


# the experiement name
reqd_dirs = [x for x in all_dirs if x not in exclusions]

# results are structured xp/run_1 etc so need to get subdirs

# traverse to get the subruns
xps_to_include =[]

for d in reqd_dirs:
    subdirs = get_subdirs(os.path.join(log_path, d))
    for s in subdirs:
        xps_to_include.append(os.path.join(d, s))


results_list = []

for d in xps_to_include:
    assert os.path.isdir(os.path.join(log_path, d)), f'{d} not found'

    print(f'Processing {d}')

    #log the experiment config setup
    config_info = get_config_info(d)

    # log the experiment results
    results = get_results(d)

    # ingore these fields - not needed
    for info in config_info.keys():
        if info not in ['cpt_ids', 'concept_states']:
            results[info] = config_info[info]
   
   # get the list of concepts 
    concepts = config_info['cpt_ids']

    # add the specific concept results
    for n, concept in enumerate(concepts):
        results = results.rename(columns=lambda c: c.replace(f"concept_{n}", f"{concept}"))

    results_list.append(results)

results_df = pd.concat(results_list)
results_df.to_csv('/scratch/prj/ccc_vit_finetuning/vit_cbm/output/results.csv', index=False)


