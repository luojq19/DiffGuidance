
import argparse
import os

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from glob import glob
from collections import Counter

from utils.evaluation import eval_atom_type, scoring_func, analyze, eval_bond_length

from utils import misc

from utils import reconstruct

from utils import transforms

from utils.evaluation.docking_qvina import QVinaDockingTask

from utils.evaluation.docking_vina import VinaDockingTask

from utils.visualize import MolTo3DView

torch.set_num_threads(1)
def print_dict(d, logger):
    for k, v in d.items():
        if v is not None:
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')


def print_ring_ratio(all_ring_sizes, logger):
    for ring_size in range(3, 10):
        n_mol = 0
        for counter in all_ring_sizes:
            if ring_size in counter:
                n_mol += 1
        logger.info(f'ring size: {ring_size} ratio: {n_mol / len(all_ring_sizes):.3f}')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('sample_path', type=str)
    # parser.add_argument('--verbose', type=eval, default=False)
    # parser.add_argument('--eval_step', type=int, default=-1)
    # parser.add_argument('--eval_num_examples', type=int, default=None)
    # parser.add_argument('--save', type=eval, default=True)
    # parser.add_argument('--protein_root', type=str, default='./data/crossdocked_v1.1_rmsd1.0')
    # parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    # parser.add_argument('--docking_mode', type=str, choices=['qvina', 'vina_score', 'vina_dock', 'none'])
    # parser.add_argument('--exhaustiveness', type=int, default=16)
    # args = parser.parse_args()
    
    sample_path = 'results_guided/outputs_guided_crsd_1e-3_500'
    verbose = False
    eval_num_examples = 1
    eval_step = -1
    atom_enc_mode = 'add_aromatic'
    print(1)

    result_path = os.path.join(sample_path, 'eval_results')
    os.makedirs(result_path, exist_ok=True)
    logger = misc.get_logger('evaluate', log_dir=result_path)
    if not verbose:
        RDLogger.DisableLog('rdApp.*')

    # Load generated data
    results_fn_list = glob(os.path.join(sample_path, '*result_*.pt'))
    results_fn_list = sorted(results_fn_list, key=lambda x: int(os.path.basename(x)[:-3].split('_')[-1]))
    if eval_num_examples is not None:
        results_fn_list = results_fn_list[:eval_num_examples]
    num_examples = len(results_fn_list)
    logger.info(f'Load generated data done! {num_examples} examples in total.')

    num_samples = 0
    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
    n_recon_success, n_eval_success, n_complete = 0, 0, 0
    results = []
    all_pair_dist, all_bond_dist = [], []
    all_atom_types = Counter()
    success_pair_dist, success_atom_types = [], Counter()
    # print(results_fn_list, len(results_fn_list))
    for example_idx, r_name in enumerate(tqdm(results_fn_list, desc='Eval')):
        r = torch.load(r_name)  # ['data', 'pred_ligand_pos', 'pred_ligand_v', 'pred_ligand_pos_traj', 'pred_ligand_v_traj']
        all_pred_ligand_pos = r['pred_ligand_pos_traj']  # [num_samples, num_steps, num_atoms, 3]
        all_pred_ligand_v = r['pred_ligand_v_traj']
        num_samples += len(all_pred_ligand_pos)
        # print(len(all_pred_ligand_pos), len(all_pred_ligand_pos[0]), len(all_pred_ligand_pos[0][0]), len(all_pred_ligand_pos[0][0][0]), len(all_pred_ligand_v), len(all_pred_ligand_v[0]), (all_pred_ligand_v[0][0]))
        # input()
        for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_ligand_pos, all_pred_ligand_v)):
            pred_pos, pred_v = pred_pos[eval_step], pred_v[eval_step]

            # stability check
            pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode=atom_enc_mode)
            all_atom_types += Counter(pred_atom_type)
            r_stable = analyze.check_stability(pred_pos, pred_atom_type)
            all_mol_stable += r_stable[0]
            all_atom_stable += r_stable[1]
            all_n_atom += r_stable[2]

            pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
            all_pair_dist += pair_dist

            # reconstruction
            try:
                pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=atom_enc_mode)
                mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
                smiles = Chem.MolToSmiles(mol)
                # print(smiles)
            except reconstruct.MolReconsError:
                if verbose:
                    logger.warning('Reconstruct failed %s' % f'{example_idx}_{sample_idx}')
                continue
            n_recon_success += 1

            if '.' in smiles:
                print('. in smiles:', smiles)
                continue
            n_complete += 1
            results.append(mol)
    print(len(results))
    view = MolTo3DView(results[0])
    