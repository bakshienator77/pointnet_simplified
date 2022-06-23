import numpy as np
import argparse

import torch
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg
from train import test
from data_loader import get_data_loader
from matplotlib import pyplot as plt
import pytorch3d

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')
    parser.add_argument('--batch_size', type=int, default=32, help='The number of images in a batch.')
    parser.add_argument('--main_dir', type=str, default='./data/')
    parser.add_argument('--task', type=str, default="cls", help='The task: cls or seg')
    parser.add_argument('--num_workers', type=int, default=4, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--exit_early', type=bool, default=False, help='Important for Q3')
    parser.add_argument('--azim_angle', type=int, default=180, help='Important for Q3')
    parser.add_argument('--elev_angle', type=int, default=0, help='Important for Q3')

    return parser

def main(args):
    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model().cuda()

    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    if not args.exit_early:
        print("successfully loaded checkpoint from {}".format(model_path))

    # Sample Points per Object
    ind = np.random.choice(10000, args.num_points, replace=False)
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=3.0, elev=args.elev_angle, azim=args.azim_angle, device=args.device)
    test_data = torch.from_numpy((np.load(args.test_data))[:, ind, :]) @ R.cpu()
    test_label = torch.from_numpy((np.load(args.test_label))[:, ind])

    # ------ TO DO: Make Prediction ------

    pred_label = torch.cat(
        [torch.argmax(model(test_data[row_idx:row_idx + args.batch_size].cuda()), dim=2).cpu().detach() for row_idx in
         range(0, len(test_data), args.batch_size)], dim=0)
    # test_accuracy, pred_label = test(test_dataloader, model, 0, args, None)
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.reshape((-1, 1)).size()[0])
    if args.exit_early:
        # viz_seg(test_data[args.i], test_label[args.i], "{}/numpoints_{}_gt_{}_{}.gif".format(args.output_dir, args.num_points,
        #                                                                                     args.exp_name, args.i),
        #         args.device)
        # viz_seg(test_data[args.i], pred_label[args.i], "{}/numpoints_{}_pred_{}_{}.gif".format(args.output_dir, args.num_points,
        #                                                                                 args.exp_name, args.i),
        #         args.device)
        viz_seg(test_data[args.i], test_label[args.i],
                "{}/rotation_e{}_a{}_gt_{}.gif".format(args.output_dir, args.elev_angle, args.azim_angle, args.i),
                args.device)
        viz_seg(test_data[args.i], pred_label[args.i],
                "{}/rotation_e{}_a{}_pred_{}.gif".format(args.output_dir, args.elev_angle, args.azim_angle,  args.i),
                args.device)
        return test_accuracy
    print("test accuracy: {}".format(test_accuracy))

    # ConfusionMatrixDisplay.from_predictions(test_label.data, pred_label.data, display_labels=["Chairs", "Vases", "Lamps"])
    # plt.savefig("{}/Confusion_matrix_{}".format(args.output_dir, args.task))

    # goal is to find failure cases
    cnt = 0
    corrects = torch.sum(pred_label.eq(test_label.data), dim=-1)
    idxs = torch.argsort(corrects)
    print("Worst example's accuracy is ", torch.min(corrects) / args.num_points)
    for data, true_label, p_label in zip(test_data[idxs[:10]], test_label[idxs[:10]], pred_label[idxs[:10]]):
        cnt += 1
        # Visualize Segmentation Result (Pred VS Ground Truth)
        print("This example's accuracy is ", p_label.eq(true_label.data).sum().item() / args.num_points)
        viz_seg(data, true_label, "{}/worst_gt_{}_{}.gif".format(args.output_dir, args.exp_name, cnt),
                args.device)
        viz_seg(data, p_label, "{}/worst_pred_{}_{}.gif".format(args.output_dir, args.exp_name, cnt),
                args.device)
        if cnt > 10:
            break
    print("Best example's accuracy is ", torch.max(corrects) / args.num_points)
    cnt = 0
    for data, true_label, p_label in zip(test_data[idxs[-10:]], test_label[idxs[-10:]], pred_label[idxs[-10:]]):
        cnt += 1
        # Visualize Segmentation Result (Pred VS Ground Truth)
        print("This example's accuracy is ", p_label.eq(true_label.data).sum().item() / args.num_points)
        viz_seg(data, true_label, "{}/best_gt_{}_{}.gif".format(args.output_dir, args.exp_name, cnt),
                args.device)
        viz_seg(data, p_label,
                "{}/best_pred_{}_{}.gif".format(args.output_dir, args.exp_name, cnt), args.device)
        if cnt > 10:
            break


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    main(args)

# epoch: 214   train loss: 12.7632   test accuracy: 0.9072
# best model saved at epoch 214