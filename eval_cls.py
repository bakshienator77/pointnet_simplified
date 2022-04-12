from statistics import mode
import numpy as np
import argparse

import torch
from models import cls_model
from utils import create_dir
from train import test
from data_loader import get_data_loader


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

    return parser



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    model = cls_model().to(args.device).eval()
    
    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))
    # print("Number of params in the modelwa: ", count_parameters(model))

    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    # test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:]).to(args.device)
    # test_label = torch.from_numpy(np.load(args.test_label)).reshape([-1])
    test_dataloader = get_data_loader(args=args, train=False)
    print("Am i able to get the data into mem?")
    # ------ TO DO: Make Prediction ------
    # with torch.no_grad():
        # pred_label = torch.argmax(model(test_data), dim=1)

    # Compute Accuracy
    # test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
    test_accuracy = test(test_dataloader, model, 0, args, None)[0]
    print ("test accuracy: {}".format(test_accuracy))

# epoch: 52   train loss: 49.2060   test accuracy: 0.9717
# best model saved at epoch 52
