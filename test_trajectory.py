import os
import yaml
from os.path import join
import inspect
import glob
import re

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
with open(join(currentdir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

models = cfg['eval']['models']

data_dir = join(cfg['mvo']['multimodal_data_dir'], 'test')
test_files = glob.glob(join(data_dir, '*.h5'))
print(test_files)

seqs = [re.search('seq_(.+?).h5', file).group(1) for file in test_files]

for model in models:
    print('Test Model {}'.format(model))
    model_dir = join(cfg['mvo']['model_dir'], model)
    max_epochs = max([int(x) for x in os.listdir(model_dir) if str.isdigit(x[0])])
    epochs = sorted([str(x) for x in os.listdir(model_dir) if str.isdigit(x[0])])
    print(epochs)

    for epoch in epochs:
        str_seq = ','.join([str(elem) for elem in seqs])
        print(epoch)
        cmd = 'python -W ignore ' + 'utility/test_double.py' + ' ' + '--seqs ' + str_seq + ' ' + '--model ' + \
              model + ' --epoch ' + epoch + ' --data_dir ' + data_dir
        print(cmd)
        os.system(cmd)
        continue

## quantitively evaluate pose errors
# cmd = 'python -W ignore ' + 'os_evaluate_robot.py'
# os.system(cmd)
