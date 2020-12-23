import numpy as np
import matplotlib.pylab as plt
plt.switch_backend('agg')
plt.style.use('ggplot')

lw = 6
linestyle_ls = [':', '-', '-', '-.', ':', '-', '-.', '--']

def plot2d(output_pred, output_gt, fig_path):
    plt.rcParams['axes.facecolor'] = 'w'
    plt.rcParams['savefig.facecolor'] = 'w'
    plt.rcParams['grid.color'] = '#C0C0C0'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.size'] = 18

    pred_x, pred_y = output_pred[:, 3], output_pred[:, 7]
    plt.plot(pred_x, pred_y, linestyle=linestyle_ls[0], linewidth=lw, label='Prediction')
    gt_x, gt_y = output_gt[:, 3], output_gt[:, 7]
    plt.plot(gt_x, gt_y, linestyle=linestyle_ls[1], linewidth=lw, label='Ground_truth')

    # plt.gca().set_aspect("equal")
    plt.legend(loc='best', ncol=1)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    # plt.show()
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()


def plot2d_multi(output_dict, fig_path, labels, color_sink, lw=5, ncol=2):
    plt.rcParams['axes.facecolor'] = 'w'
    plt.rcParams['savefig.facecolor'] = 'w'
    plt.rcParams['grid.color'] = '#C0C0C0'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.size'] = 20

    counter = 0
    for label in labels:
        output = output_dict[label]
        print(counter)
        print(label)
        print(output.shape)
        pred_x, pred_y = output[:, 3], output[:, 7]
        if counter < 2:
            plt.plot(pred_x, pred_y, color_sink[counter], linestyle=linestyle_ls[counter], linewidth=lw*1.8, label=label)
        else:
            plt.plot(pred_x, pred_y, color_sink[counter], linestyle=linestyle_ls[counter], linewidth=lw, label=label)
        counter += 1

    # plt.gca().set_aspect("equal")
    if ncol:
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow = True, ncol = ncol)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    # plt.show()
    plt.savefig(fig_path + '.png', bbox_inches='tight')
    plt.savefig(fig_path + '.pdf', bbox_inches='tight')
    plt.close()