import argparse
import os
import shutil

import numpy as np
import torch
import torch.utils.tensorboard
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

import utils.misc as misc
import utils.train as utils_train
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_diffusion')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--train_report_iter', type=int, default=200)
    parser.add_argument('--classifier_path', type=str, default='/work/jiaqi/DiffMol/synth_log/balanced150-0324-0203')
    args = parser.parse_args()

    # Load configs
    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ]
    if config.data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)

    # Datasets and loaders
    print('Loading dataset...')
    dataset, subsets = get_dataset(
        config=config.data,
        transform=transform,
    )
    train_set, val_set = subsets['train'], subsets['test']
    print(f'Training: {len(train_set)} Validation: {len(val_set)}')
    print(train_set[0])
    # print(train_set[0].protein_atom_feature)
    # print(train_set[0].protein_atom_feature.shape)
    print(train_set[0].ligand_element)
    print(len(train_set))
    # input()
    
    save_path = '/work/jiaqi/DiffMol/molecule_synth_data/crossdocked100k.pt'
    smile_save_path = '/work/jiaqi/DiffMol/molecule_synth_data/crossdocked100k_cid_smiles.txt'
    smile_split_dir = '/work/jiaqi/DiffMol/molecule_synth_data/crossdocked100k_smile_splits/'
    os.makedirs(smile_split_dir, exist_ok=True)
    
    synth_data = []
    id_smiles = []
    for idx, data in tqdm(enumerate(train_set), total=len(train_set)):
        mol_data = {}
        mol_data['element'] = data.ligand_element
        mol_data['pos'] = data.ligand_pos
        mol_data['bond_index'] = data.ligand_bond_index
        mol_data['bond_type'] = data.ligand_bond_type
        mol_data['center_of_mass'] = data.ligand_center_of_mass
        mol_data['atom_feature'] = data.ligand_atom_feature
        mol_data['hybridization'] = data.ligand_hybridization
        mol_data['atom_feature_full'] = data.ligand_atom_feature_full
        mol_data['smiles'] = data.ligand_smiles
        cid = f'cd-{idx}'
        mol_data['cid'] = cid
        id_smiles.append([cid, data.ligand_smiles])
        mol_data['timestep'] = torch.tensor([0])
        synth_data.append(mol_data)
    
    print(f'Total entries: {len(synth_data)}')
    torch.save(synth_data, save_path)
    print(f'save data to {save_path}')
    with open(smile_save_path, 'w') as f:
        for cid, smiles in id_smiles:
            f.write(f'{cid} {smiles}\n')
    print(f'save cid-smiles to {smile_save_path}')
    batch_size = 2000
    num_batches = int(np.ceil(len(id_smiles) / batch_size))
    for i in range(num_batches):
        with open(os.path.join(smile_split_dir, f'{i}.txt'), 'w') as f:
            for cid, smiles in id_smiles[i * batch_size: (i + 1) * batch_size]:
                f.write(f'{cid} {smiles}\n')
    
        