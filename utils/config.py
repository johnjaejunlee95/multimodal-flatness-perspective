import argparse


def boolean_string(s):
    if s not in {'False', 'True', '0', '1'}:
        raise ValueError('Not a valid boolean string.')
    return (s == 'True') or (s == '1')

# Dataset-specific default options
CREMAD_CONFIG = {'mode': 'train', 'num_classes': 6}
AVMNIST_CONFIG = {'mode': 'train'}
KINETICS_CONFIG = {'num_classes': 31}
FOOD101_CONFIG = {'num_classes': 101}

DATASET_CONFIGS = {
    'CREMAD': CREMAD_CONFIG,
    'AVMNIST': AVMNIST_CONFIG,
    'Kinetics': KINETICS_CONFIG,
    'Food101': FOOD101_CONFIG,
}

# Keep runner dataset names intact while resolving config keys cleanly.
DATASET_CONFIG_ALIASES = {
    'kinetic': 'Kinetics',
    'KineticSound': 'Kinetics',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--modality', type=str, default='Multimodal') #, choices=['Audio', 'Visual', 'Text', 'Multimodal']
    parser.add_argument('--master_seed', type=int, default=None, help='Optional master seed for SeedSequence-based reproducibility')
    parser.add_argument('--num_exp', type=int, default=1, help='Number of trials spawned from master_seed')
    parser.add_argument('--expt_dir', type=str, default="checkpoint")
    parser.add_argument('--expt_name', type=str, default='test_experiment')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--used_frames', type=int, default=1, help='Number of frames to use for each sample')
    parser.add_argument('--learning_rate', type=float, default=0.00001)
    parser.add_argument('--dataset', default='AVMNIST', type=str)
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='degree of Gradient Modulation')
    parser.add_argument('--lr_decay_ratio', type=float, default=0.1)
    parser.add_argument('--lr_decay_step', type=int, default=70)
    parser.add_argument('--save_checkpoint', type=boolean_string, default=False, help='whether to save checkpoint or not.')
    parser.add_argument('--optim', default='sgd', type=str, choices=['sgd', 'adamw', 'adam'], help='the type of the optimizer')
    parser.add_argument('--lr_scheduler', default='lrstep', type=str, choices=['lrstep', 'cosinestep', 'cosinestepwarmup'], help='the type of the step learning rate')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--shuffling', action='store_true', help='whether to shuffle the dataset')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='label smoothing value')
    parser.add_argument('--transform', type=str,  default=None, help='transform function for dataset') #,
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    args = parser.parse_args()
    return args


class Config():
    def __init__(self):
        args = parse_args()
        args_dict = vars(args)
        self.add_args(args_dict)
        self.normalize_runtime_args()
        self.select_model_params()
        self.normalize_runtime_args()
    def add_args(self, args_dict):
        for arg, value in args_dict.items():
            setattr(self, arg, value)
    def normalize_runtime_args(self):
        return
    def _resolve_dataset_config_key(self):
        if self.dataset in DATASET_CONFIGS:
            return self.dataset
        return DATASET_CONFIG_ALIASES.get(self.dataset, None)
    def select_model_params(self):
        dataset_key = self._resolve_dataset_config_key()
        if dataset_key is None:
            return
        self.add_args(DATASET_CONFIGS[dataset_key])
