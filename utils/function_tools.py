import os
import json
import logging
import random
import torch
import numpy as np
import torch.nn as nn
from argparse import Namespace
import copy
from typing import Any, Optional, Sequence

_DEFAULT_SEED = 42


def _to_int_seed(seed_like: Any, default: int = _DEFAULT_SEED) -> int:
    if seed_like is None:
        return int(default)
    if isinstance(seed_like, Sequence) and not isinstance(seed_like, (str, bytes)):
        if len(seed_like) == 0:
            return int(default)
        return int(seed_like[0])
    return int(seed_like)


def resolve_run_seed(cfgs: Any, default: int = _DEFAULT_SEED) -> int:
    if hasattr(cfgs, "seed"):
        return _to_int_seed(getattr(cfgs, "seed"), default=default)
    return int(default)


def resolve_master_seed(cfgs: Any) -> Optional[int]:
    master_seed = getattr(cfgs, "master_seed", None)
    if master_seed is None:
        return None
    return int(master_seed)


def _stream_to_int(stream: Any) -> int:
    if isinstance(stream, int):
        return int(stream)

    table = {
        "global": 0,
        "train": 1,
        "valid": 2,
        "val": 2,
        "test": 3,
        "split": 4,
        "food_split": 5,
        "avmnist_split": 6,
        "kinetic_split": 7,
    }
    stream_str = str(stream)
    if stream_str in table:
        return table[stream_str]
    return sum((i + 1) * ord(ch) for i, ch in enumerate(stream_str)) % 100000


def build_seed_sequence(seed: int, master_seed: Optional[int] = None, stream: Any = "global") -> np.random.SeedSequence:
    run_seed = int(seed)
    stream_id = _stream_to_int(stream)
    if master_seed is None:
        entropy = [run_seed, stream_id]
    else:
        entropy = [int(master_seed), run_seed, stream_id]
    return np.random.SeedSequence(entropy)


def derive_seed(seed: int, master_seed: Optional[int] = None, stream: Any = "global") -> int:
    seq = build_seed_sequence(seed=seed, master_seed=master_seed, stream=stream)
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def derive_seed_from_cfg(cfgs: Any, stream: Any = "global", default: int = _DEFAULT_SEED) -> int:
    run_seed = resolve_run_seed(cfgs, default=default)
    master_seed = resolve_master_seed(cfgs)
    return derive_seed(seed=run_seed, master_seed=master_seed, stream=stream)


def build_numpy_rng(cfgs: Any, stream: Any = "global", default: int = _DEFAULT_SEED) -> np.random.Generator:
    run_seed = resolve_run_seed(cfgs, default=default)
    master_seed = resolve_master_seed(cfgs)
    seq = build_seed_sequence(seed=run_seed, master_seed=master_seed, stream=stream)
    return np.random.default_rng(seq)


def build_dataloader_generator(cfgs: Any, split: str) -> torch.Generator:
    seed_value = derive_seed_from_cfg(cfgs, stream=split)
    generator = torch.Generator()
    generator.manual_seed(seed_value)
    return generator


def seed_worker(worker_id: int):
    _ = worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed, master_seed: Optional[int] = None):
    seed_seq = build_seed_sequence(seed=int(seed), master_seed=master_seed, stream="global")
    py_seed, np_seed, torch_seed, cuda_seed = [int(v) for v in seed_seq.generate_state(4, dtype=np.uint32)]

    os.environ["PYTHONHASHSEED"] = str(py_seed)
    random.seed(py_seed)
    np.random.seed(np_seed)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cuda_seed)
        torch.cuda.manual_seed_all(cuda_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True 
    
def save_config(cfg, save_dir, fname=None):
    if isinstance(cfg, Namespace):
        cfg_dict = vars(cfg)
    elif isinstance(cfg, object):
        cfg_dict = {}
        cfg_dict.update(cfg.__dict__)
    else:
        assert isinstance(cfg, dict)
        cfg_dict = cfg

    if fname is None:
        fpath = os.path.join(save_dir, 'config.json')
    else:
        assert fname.endswith(".json")
        fpath = os.path.join(save_dir, fname)
    
    with open(fpath, "w") as output:
        json.dump(cfg_dict, output, indent=4)

def load_config(load_dir, to_Namespace=True):
    files = os.listdir(load_dir)
    flist = list(filter(lambda x: x.endswith(".json"), files))
    try:
        assert len(flist) == 1
    except:
        print(f"Existing multiple cfg files: {flist}", flush=True)

    fname = flist[0]
    with open(os.path.join(load_dir, fname), "r") as input:
        cfg = json.load(input)
    
    assert isinstance(cfg, dict)

    if to_Namespace:
        cfg = Namespace(**cfg)
    
    return cfg

def get_logger(logger_name, logger_dir=None, log_name=None, is_mute_logger=False):
    logger = logging.getLogger(logger_name)
    logger.handlers.clear() 

    if is_mute_logger:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    hterm = logging.StreamHandler()
    hterm.setFormatter(formatter)
    hterm.setLevel(logging.INFO)
    logger.addHandler(hterm)
    
    if not os.path.exists(logger_dir):
        os.makedirs(logger_dir, exist_ok=True)

    if logger_dir is not None:
        if log_name is None:
            logger_path = os.path.join(logger_dir, f"{logger_name}.log")
        else:
            logger_path = os.path.join(logger_dir, log_name)
        hfile = logging.FileHandler(logger_path) 
        hfile.setFormatter(formatter)
        hfile.setLevel(logging.INFO)
        logger.addHandler(hfile)
    return logger

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device("cpu")

def boolean_string(s):
    if s not in {'False', 'True', '0', '1'}:
        raise ValueError('Not a valid boolean string.')
    return (s == 'True') or (s == '1')

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
def load_weights(model, params):
    for mp, p in zip(model.parameters(), params):
        mp.data = copy.deepcopy(p.data)


def mean_std(values):
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, variance ** 0.5


def resolve_seed_list(cfgs):
    if getattr(cfgs, "master_seed", None) is None:
        raise ValueError("master_seed is required. Please provide --master_seed.")

    num_exp = int(cfgs.num_exp)
    if num_exp <= 0:
        raise ValueError(f"num_exp must be positive, got {num_exp}")

    seed_min = 0
    seed_span = 10000
    if num_exp > seed_span:
        raise ValueError(f"num_exp={num_exp} exceeds available unique 4-digit seeds ({seed_span}).")

    master_seed_generator = np.random.SeedSequence(int(cfgs.master_seed))
    exp_seed_sequences = master_seed_generator.spawn(num_exp)

    used = set()
    seed_list = []
    for seq in exp_seed_sequences:
        raw = int(seq.generate_state(1, dtype=np.uint32)[0])
        candidate = raw % seed_span
        while candidate in used:
            candidate = (candidate + 1) % seed_span
        used.add(candidate)
        seed_list.append(seed_min + candidate)

    return seed_list
