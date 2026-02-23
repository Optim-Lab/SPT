import os
import json 
import dataset
import numpy as np
import argparse
import importlib

import torch
from torch.utils.data import DataLoader
from models.ema import ExponentialMovingAverage
import losses as losses
from models import utils as mutils

from models.utils import set_random_seed

import sys
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("./wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

project = "" # put your WANDB project name
entity = "" # put your WANDB username

run = wandb.init(
    project=project, 
    entity=entity, 
    tags=["train"], # put tags of this python project
)
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--seed', type=int, default=0, 
                        help='seed for repeatable results')
    parser.add_argument("--model", type=str, default="STaSy")
    parser.add_argument('--dataset', type=str, default='whitewine', 
                        help="""
                        Dataset options: 
                        abalone, adult, banknote, breast, concrete, covtype,
                        kings, letter, loan, redwine, whitewine
                        """)
    parser.add_argument('--test_size', type=float, default=0.2)
    
    parser.add_argument('--data.image_size',type = int, default=77)
    parser.add_argument('--data.centered', type=bool, default=False, help='Data centered')

    parser.add_argument('--model.name', type=str, default="ncsnpp_tabular")
    parser.add_argument('--model.layer_type', type=str, default="concatsquash")
    parser.add_argument('--model.scale_by_sigma', type=bool, default=False)
    parser.add_argument('--model.ema_rate', type=float, default=0.9999)
    parser.add_argument('--model.activation', type=str, default="elu")
    parser.add_argument('--model.nf', type=int, default=64)
    parser.add_argument('--model.hidden_dims', type=list, default=[1024, 2048, 1024, 1024])
    parser.add_argument('--model.conditional', type=bool, default=True)
    parser.add_argument('--model.embedding_type', type=str, default="fourier")
    parser.add_argument('--model.fourier_scale', type=int, default=16)
    parser.add_argument('--model.conv_size', type=int, default=3)

    parser.add_argument('--model.sigma_min', type=float, default=0.01, help='Minimum sigma')
    parser.add_argument('--model.sigma_max', type=float, default=10., help='Maximum sigma')
    parser.add_argument('--model.num_scales', type=int, default=50)
    parser.add_argument('--model.alpha0', type=float, default=0.3)
    parser.add_argument('--model.beta0', type=float, default=0.95)
    parser.add_argument('--test.n_iter', type=int, default=1)

    parser.add_argument('--optim.lr', type=float, default=2e-3)
    parser.add_argument('--optim.eps', type=float, default=1e-8, help='Epsilon value')
    parser.add_argument('--optim.beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--optim.weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--optim.optimizer', type=str, default='Adam', help='Optimizer type')
    parser.add_argument('--optim.warmup', type=int, default=5000)
    parser.add_argument('--optim.grad_clip', type=float, default=1.0)
    
    parser.add_argument('--training.epoch', type=int, default=10000) #10000 for paper setting
    parser.add_argument('--training.snapshot_freq', type=int, default=300)
    parser.add_argument('--training.eval_freq', type=int, default=100)
    parser.add_argument('--training.snapshot_freq_for_preemption', type=int, default=100)
    parser.add_argument('--training.snapshot_sampling', type=bool, default=True)
    parser.add_argument('--training.likelihood_weighting', type=bool, default=False)
    parser.add_argument('--training.continuous', type=bool, default=True)
    parser.add_argument('--training.eps', type=float, default=1e-05)
    parser.add_argument('--training.loss_weighting', type=bool, default=False)
    parser.add_argument('--training.spl', type=bool, default=True)
    parser.add_argument('--training.lambda_', type=float, default=0.5)

    parser.add_argument('--training.sde', type=str, default="vesde")
    parser.add_argument('--training.reduce_mean', type=bool, default=True)
    parser.add_argument('--training.n_iters', type=int, default=100000)
    parser.add_argument('--training.tolerance', type=float, default=1e-03)
    parser.add_argument('--training.hutchinson_type', type=str, default="Rademacher")
    parser.add_argument('--training.retrain_type', type=str, default="median")
    parser.add_argument('--training.batch_size', type=int, default=1000)

    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()
#%%
def main():
    config = vars(get_args(debug=False)) # default configuration
    set_random_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Current device is', device)
    config['device'] = device
    wandb.config.update(config)
    
    dataset_module = importlib.import_module(f"datasets.preprocess")
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset
    train_dataset = CustomDataset(config, train=True)
    train_z = np.concatenate([train_dataset.X_num, train_dataset.X_cat], axis=1)

    config["data.image_size"] = train_z.shape[1]
    train_dataloader = DataLoader(
        train_z, batch_size=config['training.batch_size'], shuffle=True)

    score_model = mutils.create_model(config)
    num_params = sum(p.numel() for p in score_model.parameters())
    print("the number of parameters", num_params)
    wandb.log({"the number of parameters": num_params})

    ema = ExponentialMovingAverage(score_model.parameters(), decay=config["model.ema_rate"])
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0, epoch=0)

    initial_step = int(state['epoch'])

    # Setup SDEs
    import sde_lib as sde_lib
    if config["training.sde"].lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config["model.beta_min"], beta_max=config["model.beta_max"], N=config["model.num_scales"])
        sampling_eps = 1e-3
    elif config["training.sde"].lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config["model.beta_min"], beta_max=config["model.beta_max"], N=config["model.num_scales"])
        sampling_eps = 1e-3
    elif config["training.sde"].lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config["model.sigma_min"], sigma_max=config["model.sigma_max"], N=config["model.num_scales"])
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config['training.sde']} unknown.")
        logging.info(score_model)

    optimize_fn = losses.optimization_manager(config)
    continuous = config["training.continuous"]
    reduce_mean = config["training.reduce_mean"]
    likelihood_weighting = config["training.likelihood_weighting"]

    base_name = f"{config['dataset']}_{config['model.sigma_min']}_{config['model.sigma_max']}_{config['optim.lr']}_{config['model.beta0']}_{config['model.alpha0']}"
    model_dir = f"./assets/models/{base_name}/"
    os.makedirs(model_dir, exist_ok=True)
    model_name = f"{base_name}_{config['seed']}"
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting, workdir=model_dir, spl=config["training.spl"], 
                                    alpha0=config["model.alpha0"], beta0=config["model.beta0"])

    best_loss = np.inf

    for epoch in range(initial_step, config["training.epoch"]+1):
        state['epoch'] += 1
        batch_loss = 0
        batch_num = 0
        for iteration, batch in enumerate(train_dataloader): 
            batch = batch.to(config["device"]).float()

            num_sample = batch.shape[0]
            batch_num += num_sample
            loss = train_step_fn(state, batch)

            batch_loss += loss.item() * num_sample

        batch_loss = batch_loss / batch_num
        wandb.log({'loss' : batch_loss})
        if epoch % 200 == 0:
            print("epoch: %d, iter: %d, training_loss: %.5e" % (epoch, iteration, batch_loss))

        from utils import save_checkpoint, restore_checkpoint, apply_activate
        if batch_loss < best_loss:
            best_loss = batch_loss
            best_state = save_checkpoint(os.path.join(model_dir, f"{model_name}.pth"), state)

    config_path = f"{model_dir}/config_{model_name}.json"
    def make_json_serializable(d):
        new_d = {}
        for k, v in d.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                new_d[k] = v
            else:
                new_d[k] = str(v) 
        return new_d

    json_config = make_json_serializable(config)

    with open(config_path, "w") as f:
        json.dump(json_config, f, indent=2)

    artifact = wandb.Artifact(
        "_".join(model_name.split("_")[:-1]), 
        type='model',
        metadata=config
    )

    artifact.add_file(config_path)

    artifact.add_file('./main.py')
    artifact.add_file('./datasets/preprocess.py')

    wandb.log_artifact(artifact)    
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%
    
    