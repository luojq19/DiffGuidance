import os
from tqdm.auto import tqdm
import sys
import time

classifier_scale = sys.argv[1]
guide_timestep = sys.argv[2]
classifier_path = sys.argv[3]
device = sys.argv[4]
noise = sys.argv[5]
tag = sys.argv[6]

print(f'argument check:\nclassifier_scale: {classifier_scale}\nguide_timestep: {guide_timestep}\nclassifier_path: {classifier_path}\ncls trainset noise: {noise}\ntag: {tag}')
time.sleep(10)

result_path = f'results_guided/outputs_guided_crsd_offset_{classifier_scale}_{guide_timestep}_noise{noise}_{tag}/'
os.makedirs(result_path, exist_ok=True)
with open(os.path.join(result_path, 'log.txt'), 'w') as f:
    f.write(f'classifier_scale: {classifier_scale}\nguide_timestep: {guide_timestep}\nclassifier_path: {classifier_path}\ncls trainset noise: {noise}\ntag: {tag}')
    

for i in tqdm(range(100), dynamic_ncols=True, desc=f'{classifier_scale} {guide_timestep}'):
    command = f'python sample_diffusion_2.py configs/sampling.yml --data_id {i} --result_path {result_path} --classifier_scale {classifier_scale} --device cuda:{device} --guide_timestep {guide_timestep} --classifier_path {classifier_path}'
    print(command)
    os.system(command)

os.system(f'python evaluate_diffusion.py {result_path}  --docking_mode vina_score --protein_root data/test_set --verbose 1')

os.system(f'python extract_smiles.py --input {os.path.join(result_path, "eval_results/metrics_-1.pt")}')