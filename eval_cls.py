from statistics import mode
import numpy as np
import argparse

import torch
from models import cls_model
from utils import create_dir, viz_seg
from train import test
from data_loader import get_data_loader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import pytorch3d

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
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



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args):
    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    model = cls_model().to(args.device).eval()

    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    if not args.exit_early:
        print("successfully loaded checkpoint from {}".format(model_path))
    # print("Number of params in the modelwa: ", count_parameters(model))

    # Sample Points per Object
    ind = np.random.choice(10000, args.num_points, replace=False)
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=3.0, elev=args.elev_angle, azim=args.azim_angle, device=args.device)
    test_data = torch.from_numpy((np.load(args.test_data))[:, ind, :]) @ R.cpu() # uncomment for Q3 robustness to rotation
    # print(R)
    test_label = torch.from_numpy(np.load(args.test_label)).reshape([-1])
    # test_dataloader = get_data_loader(args=args, train=False)
    # print("Am i able to get the data into mem?")
    # ------ TO DO: Make Prediction ------
    with torch.no_grad():
        if not args.exit_early:
            pred_label = torch.cat(
                [torch.argmax(model(test_data[row_idx:row_idx + args.batch_size].cuda()), dim=-1).cpu().detach() for row_idx
                 in range(0, len(test_data), args.batch_size)], dim=0)
        else:
            pred_label = torch.cat(
                [torch.argmax(model(test_data[row_idx:row_idx + args.batch_size].cuda()), dim=-1).cpu().detach() for row_idx
                 in range(0, len(test_data), args.batch_size)], dim=0)

        # pred_label = torch.argmax(model(test_data), dim=1)

    # Compute Accuracy
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
    # test_accuracy = test(test_dataloader, model, 0, args, None)[0]

    if args.exit_early:
        # print(test_data.shape)
        # print(pred_label[args.i])
        # viz_seg(test_data[args.i], pred_label[args.i] + 1,
        #         "{}/numpoints_{}_true_{}_Pred_{}_{}_{}.gif".format(args.output_dir, args.num_points, test_label[args.i], pred_label[args.i], args.task, args.i),
        #         args.device)
        viz_seg(test_data[args.i], pred_label[args.i] + 1,
                "{}/rotation_e{}_a{}_true_{}_Pred_{}_{}_{}.gif".format(args.output_dir, args.elev_angle, args.azim_angle,
                                                                       test_label[args.i], pred_label[args.i], args.task, args.i),
                args.device)
        return test_accuracy

    ConfusionMatrixDisplay.from_predictions(test_label.data, pred_label.data,
                                            display_labels=["Chairs", "Vases", "Lamps"])
    plt.savefig("{}/Confusion_matrix_{}".format(args.output_dir, args.task))
    print("test accuracy: {}".format(test_accuracy))

    for clsss in [0, 1, 2]:
        # goal is to find failure cases
        cnt = 0
        print("Number of misclassification of class ", clsss, " are ",
              len(test_data[~pred_label.eq(test_label.data) & test_label.data.eq(clsss)]))
        for data, true_label, p_label in zip(test_data[~pred_label.eq(test_label.data) & test_label.data.eq(clsss)],
                                             test_label[~pred_label.eq(test_label.data) & test_label.data.eq(clsss)],
                                             pred_label[~pred_label.eq(test_label.data) & test_label.data.eq(clsss)]):
            cnt += 1
            # print("class: ", clsss)
            viz_seg(data, p_label + 1,
                    "{}/true_{}_Pred_{}_{}_{}.gif".format(args.output_dir, true_label, p_label, args.task, cnt),
                    args.device)
            if cnt > 10:
                break
        print("Number of correct classification of class ", clsss, " are ",
              len(test_data[pred_label.eq(test_label.data) & test_label.data.eq(clsss)]))
        cnt = 0
        for data, true_label, p_label in zip(test_data[pred_label.eq(test_label.data) & test_label.data.eq(clsss)],
                                             test_label[
                                                 pred_label.eq(test_label.data) & test_label.data.eq(clsss)],
                                             pred_label[
                                                 pred_label.eq(test_label.data) & test_label.data.eq(clsss)]):
            cnt += 1
            # print("class: ", clsss)
            viz_seg(data, p_label + 1,
                    "{}/true_{}_Pred_{}_{}_{}.gif".format(args.output_dir, true_label, p_label, args.task, cnt),
                    args.device)
            if cnt > 10:
                break


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    main(args)
            # viz_seg(test_data[args.i], pred_label[args.i], "{}/pred_{}_{}_{}.gif".format(args.output_dir, args.exp_name, args.i, args.task), args.device)


# epoch: 52   train loss: 49.2060   test accuracy: 0.9717
# best model saved at epoch 52
