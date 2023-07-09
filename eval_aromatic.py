from rdkit import Chem
import argparse

def has_aromatic(smiles, mode='aromatic'):
    rdmol = Chem.MolFromSmiles(smiles)
    if mode == 'aromatic':
        aromatic_atoms = rdmol.GetAromaticAtoms()
        
        return 1 if aromatic_atoms else 0
    elif mode == 'benzene':
        benzene_pattern = Chem.MolFromSmarts('c1ccccc1')
        if rdmol.HasSubstructMatch(benzene_pattern):
            return 1
        else:
            return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--smiles', '-s', type=str, help='smiles file')
    parser.add_argument('--mode', '-m', type=str, default='aromatic', choices=['aromatic', 'benzene'])
    
    args = parser.parse_args()
    all_smiles = []
    with open(args.smiles) as f:
        lines = f.readlines()
    for line in lines:
        all_smiles.append(line.strip().split()[-1])
    print(f'number of smiles: {len(all_smiles)}')
    aromatic_res = []
    for smi in all_smiles:
        aromatic_res.append(has_aromatic(smi, mode=args.mode))
    print(f'has aromatic rate: {sum(aromatic_res)} / {len(aromatic_res)} = {sum(aromatic_res)/len(aromatic_res):.4f}')