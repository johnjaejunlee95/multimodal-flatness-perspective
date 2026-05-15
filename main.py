import os
from utils.config import Config
from utils.function_tools import get_logger, mean_std, resolve_seed_list
import importlib


DATASET_RUNNERS = {
    "AVMNIST": ("run.AVMNIST_main", "AVMNIST_main"),
    "CREMAD": ("run.CREMAD_main", "CREMAD_main"),
    "kinetic": ("run.KineticSound_main", "KineticSound_main"),
    "KineticSound": ("run.KineticSound_main", "KineticSound_main"),
    "Food101": ("run.Food_main", "Food101Main"),
}


def run_single_seed(cfgs, seed):
    if cfgs.dataset not in DATASET_RUNNERS:
        supported = ", ".join(sorted(DATASET_RUNNERS.keys()))
        raise ValueError(f"Unsupported dataset: {cfgs.dataset}. Supported datasets: {supported}")

    module_name, runner_name = DATASET_RUNNERS[cfgs.dataset]
    module = importlib.import_module(module_name)
    runner = getattr(module, runner_name)
    return runner(cfgs, seed)



def main():
    cfgs = Config()

    save_dir = os.path.join(cfgs.expt_dir, cfgs.dataset)
    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger(f"{cfgs.expt_name}_{cfgs.mode}_summary", logger_dir=save_dir)

    seed_list = resolve_seed_list(cfgs)
    acc_list = []
    best_acc_list = []

    logger.info(
        f"Running mode={cfgs.mode}, dataset={cfgs.dataset}, seeds={seed_list}, "
        f"master_seed={getattr(cfgs, 'master_seed', None)}"
    )
    for seed in seed_list:
        print(f"Set random seed to {seed}")
        cfgs.seed = int(seed)
        test_acc, best_acc = run_single_seed(cfgs, seed)

        acc_list.append(test_acc)
        best_acc_list.append(best_acc)
        logger.info(
            f"{cfgs.mode.title()} accuracy on {cfgs.expt_name} with seed {seed} "
            f"- Last Acc: {test_acc*100:.2f} Best Acc: {best_acc*100:.2f}"
        )

    acc_avg, acc_std = mean_std(acc_list)
    best_acc_avg, best_acc_std = mean_std(best_acc_list)

    logger.info(
        f"Average {cfgs.mode} accuracy on {cfgs.expt_name} - "
        f"Last Acc: {acc_avg*100:.2f} Std: {acc_std*100:.2f} "
        f"Best Acc: {best_acc_avg*100:.2f} Std: {best_acc_std*100:.2f}"
    )


if __name__ == '__main__':
    main()
