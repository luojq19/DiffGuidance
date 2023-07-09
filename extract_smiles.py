import torch
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input', type=str, default='outputs/eval_results/metrics_-1.pt')
    parser.add_argument('--output', type=str, default=None)
    
    args = parser.parse_args()
    
    data = torch.load(args.input)
    results = data['all_results']
    if args.output is None:
        output = os.path.join(os.path.dirname(args.input), 'smiles.smi')
    else:
        output = args.output
    with open(output, 'w') as f:
        for i in range(len(results)):
            f.write(f'{i} {results[i]["smiles"]}\n')
    print(f'Number of smiles: {len(results)}')
    
    