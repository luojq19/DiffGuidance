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
mode = sys.argv[7]

add_mode = ''
if mode == 'pos_only':
    add_mode = '--pos_only'
elif mode == 'type_only':
    add_mode = '--type_only'
elif mode == 'protein':
    add_mode = '--cls_mode protein'
else:
    raise NotImplementedError

print(f'argument check:\nclassifier_scale: {classifier_scale}\nguide_timestep: {guide_timestep}\nclassifier_path: {classifier_path}\ncls trainset noise: {noise}\ntag: {tag}')
time.sleep(10)

result_path = f'results_guided/outputs_guided_crsd_{classifier_scale}_{guide_timestep}_noise{noise}_{tag}_{mode}/'
os.makedirs(result_path, exist_ok=True)
with open(os.path.join(result_path, 'log.txt'), 'w') as f:
    f.write(f'classifier_scale: {classifier_scale}\nguide_timestep: {guide_timestep}\nclassifier_path: {classifier_path}\ncls trainset noise: {noise}\ntag: {tag}')
    

for i in tqdm(range(100), dynamic_ncols=True, desc=f'{classifier_scale} {guide_timestep}'):
    command = f'python sample_diffusion.py configs/sampling.yml --data_id {i} --result_path {result_path} --classifier_scale {classifier_scale} --device cuda:{device} --guide_timestep {guide_timestep} --classifier_path {classifier_path} {add_mode}'
    print(command)
    os.system(command)

os.system(f'python evaluate_diffusion.py {result_path}  --docking_mode vina_score --protein_root data/test_set --verbose 1')

os.system(f'python extract_smiles.py --input {os.path.join(result_path, "eval_results/metrics_-1.pt")}')