import os
import inspect
import glob
import re
import yaml
from os.path import join, dirname
import csv
import argparse

DESCRIPTION = """This script computes a dataset mean for particular modality."""
parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument('--dim', help='evaluate on 2D or 3D spaces', default='2')
args = parser.parse_args()

# !!! Please re-generate your network output together with timestamp
max_diff = str(10000000) # default - 10000000

######################
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
with open(join(currentdir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

models = cfg['eval']['models']

data_dir = join(cfg['mvo']['multimodal_data_dir'], 'test')
test_files = glob.glob(join(data_dir, '*.h5'))
print(test_files)

seqs = [re.search('seq_(.+?).h5', file).group(1) for file in test_files]
dataroot_dir = cfg['dataset_creation']['dataroot']

for seq_idx in seqs:
      sub_dir = cfg['dataset_creation']['all_exp_files'][(int(seq_idx) - 1)]
      sequence_dir = join(dataroot_dir, sub_dir)
      time_name = 'time_seq' + seq_idx  # format: 'time_seq' + args.seq

      ate_dir = join('./error_comp/', args.dim + 'D', 'ate')
      rte_dir = join('./error_comp/', args.dim + 'D', 'rte')

      if not os.path.exists(ate_dir):
          os.makedirs(ate_dir)

      if not os.path.exists(rte_dir):
          os.makedirs(rte_dir)

      # the stats to log
      stats_tokens = ['rmse', 'mean', 'median', 'std', 'min', 'max']
      rte_header = ['MODEL'] + ['trans_' + x for x in stats_tokens] + ['rot_' + x for x in stats_tokens]
      ate_headers = ['MODEL'] + ['trans_' + x for x in stats_tokens]

      for model in models:
            model_dir = join(cfg['mvo']['model_dir'], model)
            try:
                  epochs = sorted([str(x) for x in os.listdir(model_dir) if str.isdigit(x[0])])
            except:
                  max_epochs = 100
                  epochs = [str(x) for x in range(0, max_epochs, 5)]

            csv_meta_name = 'seq' + seq_idx + '_' + model + '.csv'
            # rte csv
            rte_csv_path = join(rte_dir, csv_meta_name)
            print(rte_header)
            with open(rte_csv_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(rte_header)
                file.close()
            # ate csv
            ate_csv_path = join(ate_dir, csv_meta_name)
            with open(ate_csv_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(ate_headers)
                file.close()

            for epoch in epochs:
                  pose_name = model+'_ep'+epoch+'_seq' + seq_idx  # format: args.model + '_ep' + args.epoch + '_seq' + args.seq
                  #===================#
                  ### 1 - Convert the output poses and GT to quaternion format
                  # A: Convert the network output pose to quaternion format
                  pose_path = join('./results', pose_name)
                  time_path = join('./results', time_name)
                  q_prediction_path = join('./results', 'q_' + pose_name)
                  cmd = 'python -W ignore ' + 'eval/convert_pose_to_quat.py' + ' ' + '--pose ' + pose_path + ' ' + '--timestamp ' + time_path \
                        + ' ' + '--save_path ' + q_prediction_path + ' ' + '--is_gt ' + '0'
                  os.system(cmd)

                  # in case you havent generate the gmapping full traj for particular sequence (recommended: generate all gmapping first)
                  delta_gmapping_path = sequence_dir + '/true_delta_gmapping.csv'
                  cmd = 'python -W ignore ' + 'eval/composing_delta_pose.py' + ' ' + '--delta_path ' + delta_gmapping_path
                  os.system(cmd)

                  # B: Convert gmapping pose to quaternion format
                  full_traj_gmapping_path = sequence_dir + '/full_traj_gmapping.csv'
                  q_gmapping_path = join('./results', 'q_gmapping_seq' + seq_idx)
                  cmd = 'python -W ignore ' + 'eval/convert_pose_to_quat.py' + ' ' + '--pose ' + full_traj_gmapping_path + ' ' + '--timestamp ' \
                        + delta_gmapping_path + ' ' + '--save_path ' + q_gmapping_path + ' ' + '--is_gt ' + '1'
                  os.system(cmd)

                  #===================#
                  ### 2 - Evaluate using TUM RGB-D SLAM approach
                  # You can choose either one or both

                  # A. Evaluate RPE
                  # I set delta unit to f:frame, there are many options available
                  print('********* RELATIVE Pred. Error **********')
                  # cmd = 'python2 -W ignore ' + 'eval/evaluate_rpe.py ' + q_gmapping_path + ' ' + q_prediction_path + ' ' + '--delta_unit ' + 'f' \
                  #       + ' ' + '--fixed_delta ' + ' --err_comp ' + rte_csv_path + \
                  #       ' --model ' + pose_name + ' --verbose'
                  # os.system(cmd)

                  # Evauate RPE without alignment
                  if args.dim == '2':
                      cmd = 'python2 -W ignore ' + 'eval/evaluate_std_rpe_2D.py ' + q_gmapping_path + ' ' + q_prediction_path \
                            + ' ' + '--max_difference ' + max_diff + ' --err_comp ' + rte_csv_path + \
                            ' --model ' + pose_name + ' --verbose'
                  elif args.dim == '3':
                      cmd = 'python2 -W ignore ' + 'eval/evaluate_std_rpe.py ' + q_gmapping_path + ' ' + q_prediction_path \
                            + ' ' + '--max_difference ' + max_diff + ' --err_comp ' + rte_csv_path + \
                            ' --model ' + pose_name + ' --verbose'
                  print(cmd)
                  os.system(cmd)

                  # B. Evaluate ATE
                  print('********* ABSOLUTE Pred. Error **********')
                  if args.dim == '2':
                      cmd = 'python2 -W ignore ' + 'eval/evaluate_std_ate_2D.py ' + q_gmapping_path + ' ' + q_prediction_path + ' ' + '--max_difference ' + \
                            max_diff + ' ' + ' --err_comp ' + ate_csv_path + ' --model ' + pose_name + ' --verbose'
                  elif args.dim == '3':
                      cmd = 'python2 -W ignore ' + 'eval/evaluate_std_ate.py ' + q_gmapping_path + ' ' + q_prediction_path + ' ' + '--max_difference ' + \
                            max_diff + ' ' + ' --err_comp ' + ate_csv_path + ' --model ' + pose_name + ' --verbose'
                  os.system(cmd)