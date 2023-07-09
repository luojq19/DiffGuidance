import torch
import os
import torch_cluster
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit.Chem.Descriptors

device = 'cuda:0'

def get_classifier(classifier_path, device):
    os.system(f'cp {os.path.join(classifier_path, "models/synth_model.py")} /work/jiaqi/targetdiff/models/classifier.py')
    from models.classifier import SynthEGNN
    classifier = SynthEGNN()
    state_dict = torch.load(os.path.join(classifier_path, 'best_checkpoint.pt'))
    classifier.load_state_dict(state_dict)
    classifier.to(device)
    classifier.eval()
    print(f'Successfully loaded classifier from {classifier_path}')
        
    return classifier

def run_classifier(classifier, ligand_pos, ligand_v, timestep=torch.tensor([0]), batch=None, classifier_scale=0.1):
    # TODO: here we set a fixed threshold r=1.6 to infer edges from coordinates, but the actually edges here are chemical bonds, so later we need to use more sophisticated way to infer edges, like molecule reconstruction
    edge_index = torch_cluster.radius_graph(ligand_pos, r=1.6, batch=batch).to(device)
    atom_feature_full = torch.nn.functional.one_hot(ligand_v, classifier.atom_feature_dim).to(device)
    # print(batch.dtype, ligand_pos.dtype, ligand_v.dtype, edge_index.dtype, atom_feature_full.dtype)
    logits = classifier(ligand_pos, atom_feature_full, edge_index, t=timestep.to(device), batch=batch)
    log_probs = F.log_softmax(logits, dim=-1)
    selected = log_probs[range(len(logits)), 1]
    return logits, log_probs, selected

def compute_mol_density(mol):
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    volume = AllChem.ComputeMolVolume(mol)
    mass = Chem.Descriptors.MolWt(mol)
    density = mass / volume
    
    return density

def eval_molecules(data_list):
    results = {'avg_atom_num': 0,
               'atom_nums': [],
               'has_aromatic': [],
               'aromatic_rate': 0,
               'density': [],
               'avg_density': 0}
    for mol in data_list:
        atom_num = mol['pred_pos'].shape[0]
        results['atom_nums'].append(atom_num)
        rdmol = Chem.MolFromSmiles(mol['smiles'])
        aromatic_atoms = rdmol.GetAromaticAtoms()
        results['has_aromatic'].append(1 if aromatic_atoms else 0)
        # print(1)
        # results['density'].append(compute_mol_density(rdmol))
    results['avg_atom_num'] = int(np.mean(results['atom_nums']))
    results['aromatic_rate'] = np.sum(results['has_aromatic']) / len(data_list)
    # results['avg_density'] = np.mean(results['density'])
    
    return results

classifier = get_classifier('/work/jiaqi/DiffMol/3d_synth_noised_log/exp_crsd_noise0_no_edge-20230408-0622', device)
results_dir = 'results_guided/outputs_guided_crsd_1e-3_500_noise0_TDsample_balanced'
# results_dir = 'outputs/'
outputs = []

all_data = torch.load(os.path.join(results_dir, 'eval_results/metrics_-1.pt'))['all_results']
print(f'num of entries: {len(all_data)}')
retro_res_file = os.path.join(results_dir, 'eval_results/smiles_synth_results.txt')
with open(retro_res_file) as f:
    lines = f.readlines()
retro_res = []
for line in lines:
    _, res = line.strip().split()
    retro_res.append(int(res))
predicts = []
for data in all_data:
    ligand_pos = torch.tensor(data['pred_pos'], dtype=torch.float32).to(device)
    ligand_v = torch.tensor(data['pred_v']).to(device)
    batch = torch.zeros(len(ligand_pos), dtype=torch.int64).to(device)
    logits, log_probs, selected = run_classifier(classifier, ligand_pos, ligand_v, batch=batch)
    pred = torch.argmax(logits)
    predicts.append(pred.item())
retro_res = torch.tensor(retro_res)
predicts = torch.tensor(predicts)
print(results_dir)
print(f'acc: {(retro_res == predicts).sum() / len(predicts):.4f}')
cfm = confusion_matrix(retro_res, predicts)
print(cfm)

tp, tn, fp, fn = [], [], [], []
for i in range(len(predicts)):
    if predicts[i] == retro_res[i] == 1:
        tp.append(i)
    elif predicts[i] == retro_res[i] == 0:
        tn.append(i)
    elif predicts[i] == 1 and retro_res[i] == 0:
        fp.append(i)
    elif predicts[i] == 0 and retro_res[i] == 1:
        fn.append(i)
    else:
        raise NotImplementedError
print(f'tp: {len(tp)}\ntn: {len(tn)}\nfp: {len(fp)}\nfn: {len(fn)}')
gt_pos_idx, gt_neg_idx = [], []
for i in range(len(retro_res)):
    if retro_res[i] == 1:
        gt_pos_idx.append(i)
    else:
        gt_neg_idx.append(i)

tp_res = eval_molecules([all_data[i] for i in tp])
tn_res = eval_molecules([all_data[i] for i in tn])
fp_res = eval_molecules([all_data[i] for i in fp])
fn_res = eval_molecules([all_data[i] for i in fn])
pred_pos_res = eval_molecules([all_data[i] for i in tp + fp])
pred_neg_res = eval_molecules([all_data[i] for i in tn + fn])
all_res = eval_molecules(all_data)
gt_pos_res = eval_molecules([all_data[i] for i in gt_pos_idx])
gt_neg_res = eval_molecules([all_data[i] for i in gt_neg_idx])
print(f'avg atom nums:\ntp: {tp_res["avg_atom_num"]}\ntn: {tn_res["avg_atom_num"]}\nfp: {fp_res["avg_atom_num"]}\nfn: {fn_res["avg_atom_num"]}\nall: {all_res["avg_atom_num"]}\ngt_pos: {gt_pos_res["avg_atom_num"]}\ngt_neg: {gt_neg_res["avg_atom_num"]}\npred_pos: {pred_pos_res["avg_atom_num"]}\npred_neg: {pred_neg_res["avg_atom_num"]}')
# print(f'atom nums:\ntp: {tp_res["atom_nums"][:30]}\ntn: {tn_res["atom_nums"][:30]}\nfp: {fp_res["atom_nums"][:30]}\nfn: {fn_res["atom_nums"][:30]}\ngt_pos: {gt_pos_res["atom_nums"][:30]}\ngt_neg: {gt_neg_res["atom_nums"][:30]}\npred_pos: {pred_pos_res["atom_nums"][:30]}\npred_neg: {pred_neg_res["atom_nums"][:30]}\nall: {all_res["atom_nums"]}')
print(f'aromatic rate:\ntp: {tp_res["aromatic_rate"]}\ntn: {tn_res["aromatic_rate"]}\nfp: {fp_res["aromatic_rate"]}\nfn: {fn_res["aromatic_rate"]}\nall: {all_res["aromatic_rate"]}\ngt_pos: {gt_pos_res["aromatic_rate"]}\ngt_neg: {gt_neg_res["aromatic_rate"]}\npred_pos: {pred_pos_res["aromatic_rate"]}\npred_neg: {pred_neg_res["aromatic_rate"]}')
# print(f'avg density:\ntp: {tp_res["avg_density"]}\ntn: {tn_res["avg_density"]}\nfp: {fp_res["avg_density"]}\nfn: {fn_res["avg_density"]}\nall: {all_res["avg_density"]}\ngt_pos: {gt_pos_res["avg_density"]}\ngt_neg: {gt_neg_res["avg_density"]}\npred_pos: {pred_pos_res["avg_density"]}\npred_neg: {pred_neg_res["avg_density"]}')

