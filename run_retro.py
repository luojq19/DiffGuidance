from retro_star.api import RSPlanner
from retro_star.common import args
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse
import sys
import torch

torch.set_num_threads(1)

planner = RSPlanner(
    gpu=-1,
    use_value_fn=True,
    iterations=100,
    expansion_topk=50
)

if __name__ == '__main__':
    input_file = args.input
    output = args.output
    print(f'Input: {input_file}')
    print(f'Output: {output}')
    data = torch.load(input_file)
    print(0)
    results = data['all_results']
    synthsizable = 0
    print(1)
    with open(output, 'w') as f:
        for i in tqdm(range(len(results))):
            smile = results[i]['smiles']
            flag = 0
            try:
                res = planner.plan(smile)
            except:
                res = None
            if res is not None:
                synthsizable += 1
                flag = 1
            f.write(f'{smile} {flag}\n')
    print(f'Total smiles: {len(results)}')
    print(f'Number of synthsizable: {synthsizable}')
    print(f'Synthsizable rate: {synthsizable / len(results) * 100:.2f}%')