import torch
import os
import torch_cluster
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

torch.set_num_threads(1)
device = 'cuda:3'

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
    edge_index = torch_cluster.radius_graph(ligand_pos, r=1.6, batch=batch).to('cuda:3')
    atom_feature_full = torch.nn.functional.one_hot(ligand_v, classifier.atom_feature_dim).to('cuda:3')
    # print(batch.dtype, ligand_pos.dtype, ligand_v.dtype, edge_index.dtype, atom_feature_full.dtype)
    logits = classifier(ligand_pos, atom_feature_full, edge_index, t=timestep.to('cuda:3'), batch=batch)
    log_probs = F.log_softmax(logits, dim=-1)
    selected = log_probs[range(len(logits)), 1]
    return logits, log_probs, selected

def eval_one_dir(classifier, results_dir):
    sample_data = []
    
    all_data = torch.load(os.path.join(results_dir, 'eval_results/metrics_-1.pt'))['all_results']
    print(f'num of entries: {len(all_data)}')
    retro_res_file = os.path.join(results_dir, 'eval_results/smiles_synth_results.txt')
    for root, dirs, files in os.walk(os.path.join(results_dir, 'eval_results')):
        for file in files:
            if file.endswith('.txt') and not file.startswith('log'):
                retro_res_file = os.path.join(root, file)
                break
    # retro_res_file = os.path.join(results_dir, 'eval_results/smiles_synth_results.txt')
    with open(retro_res_file) as f:
        lines = f.readlines()
    retro_res = []
    for line in lines:
        _, res = line.strip().split()
        retro_res.append(int(res))
    assert len(retro_res) == len(all_data), print('retro res mismatch!', len(retro_res), len(all_data), results_dir)
    predicts = []
    for idx, data in enumerate(all_data):
        ligand_pos = torch.tensor(data['pred_pos'], dtype=torch.float32).to(device)
        ligand_v = torch.tensor(data['pred_v']).to(device)
        batch = torch.zeros(len(ligand_pos), dtype=torch.int64).to('cuda:3')
        logits, log_probs, selected = run_classifier(classifier, ligand_pos, ligand_v, batch=batch)
        pred = torch.argmax(logits)
        predicts.append(pred.item())
        if int(pred) != int(retro_res[idx]):
            mol_data = {'label': int(retro_res[idx]),
                        'pos': ligand_pos.detach().cpu(),
                        'atom_feature_full': ligand_v.detach().cpu(),
                        'timestep': torch.tensor([0])}
            sample_data.append(mol_data)
    retro_res = torch.tensor(retro_res)
    predicts = torch.tensor(predicts)
    cfm = confusion_matrix(retro_res, predicts)
    print(f'acc: {(cfm[0,0]+cfm[1,1])/len(predicts):.4f}; fp: {cfm[0,1]/len(predicts):.4f}; fn: {cfm[1,0]/len(predicts):.4f}')
    
    return sample_data

if __name__ == '__main__':
    cls_path = '/work/jiaqi/DiffMol/3d_synth_noised_log/exp_crsd_noise0-20230407-0623'
    classifier = get_classifier(cls_path, device=device)
    results_dir = 'results_guided/'
    entries = os.listdir(results_dir)
    dirs = [os.path.join(results_dir, entry) for entry in entries if os.path.isdir(os.path.join(results_dir, entry))]
    print(f'number of result dirs: {len(dirs)}')
    sample_data = []
    for res_dir in tqdm(dirs, dynamic_ncols=True):
        try:
            part_data = eval_one_dir(classifier, res_dir)
        except:
            part_data = []
            print(res_dir)
        sample_data.extend(part_data)
    
    print(f'number of sample data: {len(sample_data)}')
    for i in range(len(sample_data)):
        sample_data[i]['cid'] = f'sd-{i}'
    
    save_path = f'/work/jiaqi/DiffMol/molecule_synth_data/sampled_data.pt'
    torch.save(sample_data, save_path)
    print(f'save data to {save_path}')
        
    
    
    
    