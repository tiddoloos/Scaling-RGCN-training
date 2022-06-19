from typing import Dict, Union
from os import listdir
from os.path import isfile, join


def check_sum_map_files(sum_path: str, map_path: str) -> int:
    sum_files = [f for f in listdir(sum_path) if not f.startswith('.') if isfile(join(sum_path, f))]
    map_files = [f for f in listdir(map_path) if not f.startswith('.') if isfile(join(map_path, f))]
    assert len(sum_files) == len(map_files), f'for every summary file there needs to be a map file. \n {len(sum_files)} sum files found and {len(map_files)} map files found'
    return len(sum_files)

def check_emb_dim(configs: Dict[str, Union[int, str]], num_sum_files: int) -> Dict[str, Union[int, str]]:
    emb_dim = configs['emb']
    new_emb = round(emb_dim/num_sum_files) * num_sum_files
    configs['emb'] = new_emb
    if new_emb != emb_dim:
        print(f'updated embedding dimension for attention experiment: new emb_dim is {new_emb}, was {emb_dim}')
    return configs

def do_checks(configs: Dict[str, Union[int, str]], sum_path: str, map_path: str) -> Dict[str, Union[int, str]]:
    num_sum_files = check_sum_map_files(sum_path, map_path)
    updated_configs = check_emb_dim(configs, num_sum_files)
    return updated_configs
