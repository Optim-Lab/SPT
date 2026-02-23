#%%
from dython.nominal import associations
from synthetic_eval import evaluation
import pandas as pd
import numpy as np
import argparse
import importlib
import torch
from datasets.preprocess import build_num_inverse_fn, build_cat_inverse_fn
from modules.utils import set_random_seed, sample, recover_data, split_num_cat_target
from prdc import compute_prdc
from evaluation.clipped_coverage import ClippedDensityCoverage 
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
    tags=["inference"], # put tags of this python project
)

def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--ver', type=int, default=0, 
                        help='ver for repeatable results')
    parser.add_argument("--model", type=str, default="AutoDiff")
    parser.add_argument('--dataset', type=str, default='anuran', 
                        help="""
                        Tabular dataset options: 
                        abalone, adult, anuran, banknote, breast, concrete,
                        kings, letter, loan, redwine, shoppers, whitewine
                        """)
    parser.add_argument('--test_size', default=0.2, type=float, 
                        help="Proportion of the dataset to include in the test split")
    
    # Stage 1 Model
    parser.add_argument('--batch_size1', default=512, type=int, 
                        help="Batch size for stage 1 training")
    parser.add_argument('--lr1', default=0.001, type=float,
                        help='Learning rate for stage 1 training')
    parser.add_argument('--weight_decay1', default=0, type=float,
                        help='weight decay for stage 1')
    parser.add_argument('--d_token', default=4, type=int,
                        help='Latent dimension')
    parser.add_argument('--num_layers', default=2, type=int,
                        help='The number of layer in transformer')
    parser.add_argument('--factor', default=32, type=int,
                        help='FACTOR')
    parser.add_argument('--n_head', default=1, type=int,
                        help='N_HEAD')
    parser.add_argument('--bias', default=True, type=bool,
                        help='Token Bias')        

    # Stage 2 Model
    parser.add_argument('--batch_size2', default=512, type=int,
                        help='Batch size for stage 2 training' )
    parser.add_argument('--scheduler', type=str, default='linear', 
                        help="Options for beta scheduling: linear and cosine")
    parser.add_argument('--weight_decay2', default=1e-4, type=float,
                        help='Weight decay for AdamW in training stage 2')
    parser.add_argument("--denoising_dim", default=1024, type=int,
                        help="Size of latent dimension for stage 2")
    
    # Inference factor
    parser.add_argument("--coverage_k", default=5, type=int,
                        help="The nearest neighbor in inference procedure")
    
    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()
    
#%%
def main():
    config = vars(get_args(debug=False)) # default configuration
    model_name = f"{config['dataset']}_{config['lr1']}_{config['d_token']}_{config['denoising_dim']}"
    model_name += f"_{config['batch_size1']}_{config['batch_size2']}"
    
    artifact = wandb.use_artifact(f"{project}/{model_name}_{config['ver']}:v0", type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()
    info = artifact.metadata.get("info", {})
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    set_random_seed(config["seed"])
    wandb.config.update(config)

    dataset_module = importlib.import_module(f"datasets.preprocess")
    importlib.reload(dataset_module)

    CustomDataset = dataset_module.CustomDataset
    train_dataset = CustomDataset(
        config, train=True)
    test_dataset = CustomDataset(
        config, train=False, cont_scalers=train_dataset.cont_scalers, cat_scalers=train_dataset.cat_scalers )        
    
    num_inverse = build_num_inverse_fn(train_dataset.cont_scalers)
    cat_inverse = build_cat_inverse_fn(train_dataset.cat_scalers)

    train_z = np.load(f'./assets/models/{model_name}/{model_name}_{config["seed"]}_train_z.npy')
    train_z = torch.tensor(train_z).float()
    train_z = train_z[:, 1:, :]

    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    train_z = train_z.view(B, in_dim)

    denoise_fn_module = importlib.import_module(f"modules.model2")
    importlib.reload(denoise_fn_module)
    denoise_fn = denoise_fn_module.MLPDiffusion(
        in_dim,
        config['denoising_dim'], ## fixed
    ).to(device)
    model2 = denoise_fn_module.Model(
        denoise_fn = denoise_fn, hid_dim = in_dim
    ).to(device) 
    if config["cuda"]:
        model2.load_state_dict(
            torch.load(
                f"./assets/models/{model_name}/stage2_{model_name}_{config['seed']}.pth"
            )
        )
    else:
        model2.load_state_dict(
            torch.load(
                f"./{model_dir}/stage2_{model_name}.pth",
                map_location=torch.device("cpu"),
            )
        )
    info['model_dir'] = f"./assets/models/{model_name}/stage1_{model_name}_{config['seed']}.pth"
    model2.eval()

    num_samples = B
    x_next = sample(model2.denoise_fn_D, num_samples, in_dim)
    x_next = x_next * 2 + train_z.mean(0).to(device)

    syn_data = x_next.float().cpu().numpy()
    syn_data = syn_data.astype(np.float32)
    #%%
    df_synth = pd.DataFrame(syn_data, columns=[f"dim_{i}" for i in range(syn_data.shape[1])])
    output_path = f"./syn_data/synthetic_prior_{model_name}.csv"
    df_synth.to_csv(output_path, index=False)
    #%%
    syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse, config, device) 
    
    syn_df = recover_data(syn_num, syn_cat, syn_target, info)
    
    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}
    
    syn_df.rename(columns = idx_name_mapping, inplace=True)
    
    syn_df[train_dataset.categorical_features] = syn_df[train_dataset.categorical_features].astype(int)
    syn_df[train_dataset.integer_features] = syn_df[train_dataset.integer_features].astype(int)
    syn_df[train_dataset.continuous_features] = syn_df[train_dataset.continuous_features].astype(np.float32)
    
    """ Synthetic Eval packages """
    results = evaluation.evaluate(
        syn_df,train_dataset.raw_data.astype('float32'), test_dataset.raw_data.astype('float32'), 
        train_dataset.ClfTarget, train_dataset.continuous_features, train_dataset.categorical_features, device
        )
    '''print results'''
    for x, y in results._asdict().items():
        print(f"{x}: {y:.3f}")
        wandb.log({f"{x}": y})
    syn_df.to_csv(f"./syn_data/{config['dataset']}_{config['model']}_ver{config['ver']}")
    #%%
    print("Coverage for Diversity Evaluation ...")
    coverage_k = config['coverage_k']
    coverage = compute_prdc(real_features=train_dataset.raw_data.astype('float32'),
                            fake_features=syn_df,
                            nearest_k=coverage_k)
    print("Coverage : ",coverage["coverage"])
    wandb.log({"Coverage":coverage["coverage"]})
    
    print("Clipped Coverage(Train Set) for Diversity Evaluation ...")
    CDC = ClippedDensityCoverage(train_dataset.raw_data.to_numpy(dtype='float32'),
                                K=coverage_k,
                                n_jobs=8,)
    
    print("Clipped_Coverage(Train Set) : ", CDC.ClippedCoverage(syn_df.to_numpy(dtype='float32')))
    wandb.log({"Clipped_Coverage(Train Set)" : CDC.ClippedCoverage(syn_df.to_numpy(dtype='float32'))})
    #%%
    print("Pairwise correlation difference (PCD)...")
    syn_asso = associations(
        syn_df, nominal_columns=train_dataset.categorical_features,
        compute_only=True)
    true_asso = associations(
        train_dataset.raw_data, nominal_columns=train_dataset.categorical_features,
        compute_only=True)
    pcd_corr = np.linalg.norm(true_asso["corr"] - syn_asso["corr"])
    print("Pairwise correlation difference (PCD) : ",pcd_corr)
    
    wandb.log({"PCD":pcd_corr})
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
    
if __name__ == '__main__':
    main()
    #%%