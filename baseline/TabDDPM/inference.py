from dython.nominal import associations
import torch
import argparse
import importlib

import numpy as np

from modules.utils import get_model
from modules.utils import set_random_seed
from synthetic_eval import evaluation
from prdc import compute_prdc
from evaluation.clipped_coverage import ClippedDensityCoverage 
import warnings
warnings.filterwarnings('ignore')
import sys
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("../wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

project = "" # put your WANDB project name
entity = "" # put your WANDB username

run = wandb.init(
    project=project, 
    entity=entity, 
    tags=["inference"], # put tags of this python project
)
#%%
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--ver', type=int, default=0, 
                        help='model version number')
    parser.add_argument('--dataset', type=str, default='adult', 
                        help="""
                        Tabular dataset options: 
                        abalone, anuran, banknote, breast, concrete,
                        kings, letter, loan, redwine, shoppers, whitewine
                        """)
    parser.add_argument("--model", type=str, default="TabDDPM")
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.002, 
                        help="Learning rate")
    parser.add_argument("--num_timesteps", type=int, default=1000, 
                        help="Number of timesteps")
    parser.add_argument("--num_layers", type=int, default=2, 
                        help="the number of mlp layers for TabDDPM")
    parser.add_argument("--dim_embed", type=int, default=256, 
                        help="embedding dimension of TabDDPM")
    parser.add_argument("--weight_decay", type=float, default=1e-5, 
                        help="embedding dimension of TabDDPM")
    parser.add_argument("--coverage_k", default=5, type=int,
                        help="The nearest neighbor in inference procedure")
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    config = vars(get_args(debug=False)) 
    
    """model load"""
    base_name = f"TabDDPM_{config['dataset']}_{config['weight_decay']}_{config['lr']}_{config['num_layers']}_{config['dim_embed']}_{config['batch_size']}"
    model_name = f"{base_name}"
    try:
        print(f"Attempting v0...")
        artifact = wandb.use_artifact(f"{project}/{model_name}_{config['ver']}:v0", type='model')
    except Exception as e:
        try:
            print(f"v0 failed, attempting v1...")
            artifact = wandb.use_artifact(f"{project}/{model_name}_{config['ver']}:v1", type='model')
        except Exception as e:
            print(f"v1 failed, attempting v2...")
            artifact = wandb.use_artifact(f"{project}/{model_name}_{config['ver']}:v2", type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()
    
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    set_random_seed(config["seed"])
    wandb.config.update(config)
   
    dataset_module = importlib.import_module('dataset.preprocess')
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset
    
    """dataset"""
    train_dataset = CustomDataset(
        config,
        train=True
    )
    
    test_dataset = CustomDataset(
        config,
        train=False,
        cont_scalers=train_dataset.cont_scalers
    )
    
    """model"""
    model = get_model(config)
    model.to(device)
   
    model_module = importlib.import_module('modules.model')
    importlib.reload(model_module)
    diffusion = model_module.GaussianMultinomialDiffusion(
        EncodedInfo=train_dataset.EncodedInfo,
        denoise_fn=model,
        config=config,
        device=device
    ).to(device)

    if config["cuda"]:
        diffusion.load_state_dict(
            torch.load(
                f"./assets/models/{model_name}/{model_name}_{config['seed']}.pth"
            )
        )
    else:
        diffusion.load_state_dict(
            torch.load(
                model_dir + "/" + model_name,
                f"./assets/models/{model_name}/{model_name}_{config['seed']}.pth"
            )
        )
    diffusion.eval()
    
    count_parameters = lambda model: sum(p.numel() for p in model.parameters())
    num_params = count_parameters(model)
    print(f"Number of Parameters: {num_params / 1000000:.6f}M")
    wandb.log({"Number of Parameters": num_params / 1000000})
    
    n = len(train_dataset.raw_data)
    syndata = diffusion.generate_synthetic_data(
        num_samples=n, 
        train_dataset=train_dataset,
        ddim=False)
    
    syndata[train_dataset.ClfTarget] = syndata[train_dataset.ClfTarget].astype(int)
     
    # syndata.to_csv(f"{config['dataset']}_syndata.csv", index=False)
    results = evaluation.evaluate(
        syndata, train_dataset.raw_data, test_dataset.raw_data, 
        train_dataset.ClfTarget, train_dataset.continuous_features, train_dataset.categorical_features + [train_dataset.ClfTarget], device
        )
    #%%
    print("Coverage for Diversity Evaluation ...")
    coverage_k = config['coverage_k']
    coverage = compute_prdc(real_features=train_dataset.raw_data.astype('float32'),
                            fake_features=syndata,
                            nearest_k=coverage_k)
    print("Coverage : ",coverage["coverage"])
    wandb.log({"Coverage":coverage["coverage"]})
    
    print("Clipped Coverage(Train Set) for Diversity Evaluation ...")
    CDC = ClippedDensityCoverage(train_dataset.raw_data.to_numpy(dtype='float32'),
                                K=coverage_k,
                                n_jobs=8,)
    
    print("Clipped_Coverage(Train Set) : ", CDC.ClippedCoverage(syndata.to_numpy(dtype='float32')))
    wandb.log({"Clipped_Coverage(Train Set)" : CDC.ClippedCoverage(syndata.to_numpy(dtype='float32'))})
    #%%
    print("Pairwise correlation difference (PCD)...")
    syn_asso = associations(
        syndata, nominal_columns=train_dataset.categorical_features,
        compute_only=True)
    true_asso = associations(
        train_dataset.raw_data, nominal_columns=train_dataset.categorical_features,
        compute_only=True)
    pcd_corr = np.linalg.norm(true_asso["corr"] - syn_asso["corr"])
    print("Pairwise correlation difference (PCD) : ",pcd_corr)
    wandb.log({"PCD":pcd_corr})

    for x, y in results._asdict().items():
        print(f"{x}: {y:.3f}")
        wandb.log({f"{x}": y})
        
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
    #%% 
if __name__ == "__main__":
    main()
#%%