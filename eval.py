import os
import torch
import argparse
import seaborn as sns
import numpy as np
from utils.dataset import load_mat_hsi
from models.get_model import get_model
from train import test
from utils.utils import metrics, show_results
import imageio
import spectral
from scipy import io
def color_results(arr2d, palette):
    arr_3d = np.zeros((arr2d.shape[0], arr2d.shape[1], 3), dtype=np.uint8)
    for c, i in palette.items():
        m = arr2d == c
        arr_3d[m] = i
    return arr_3d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HSI classification evaluation")
    parser.add_argument("--model", type=str, default='NEWdlst')
    parser.add_argument("--dataset_name", type=str, default="Ori")
    parser.add_argument("--dataset_dir", type=str, default="./dataset_new")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--patch_size", type=int, default=7)
    parser.add_argument("--weights", type=str, default="./checkpoints/NEWdlst/Ori/1")
    parser.add_argument("--outputs", type=str, default="./results_new")

    opts = parser.parse_args()

    device = torch.device("cuda:{}".format(opts.device))

    print("dataset: {}".format(opts.dataset_name))
    print("patch size: {}".format(opts.patch_size))
    print("model: {}".format(opts.model))

    image, gt, labels = load_mat_hsi(opts.dataset_name, opts.dataset_dir)

    num_classes = len(labels)
    num_bands = image.shape[-1]

    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", num_classes + 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))

    # load model and weights
    model = get_model(opts.model, opts.dataset_name, opts.patch_size)
    print('loading weights from %s' % opts.weights + '/model_best.pth')
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(opts.weights, 'model_best.pth')))
    model.eval()

    # testing model: metric for the whole HSI, including train, val, and test
    probabilities = test(model, opts.weights, image, opts.patch_size, num_classes, device=device)
    prediction = np.argmax(probabilities, axis=-1)

    run_results = metrics(prediction, gt, n_classes=num_classes)

    # prediction[gt < 0] = -1

    # color results
    colored_gt = color_results(gt+1, palette)
    colored_pred = color_results(prediction+1, palette)
    outputs = prediction+1

    outfile = os.path.join(opts.outputs, opts.dataset_name,  opts.model)
    os.makedirs(outfile, exist_ok=True)

    # imageio.imsave(os.path.join(outfile, opts.dataset_name + '_gt_NEWdlst.tif'), colored_gt)  # eps or png
    imageio.imsave(os.path.join(outfile, opts.dataset_name+'_' + opts.model + 'NEWdlst.tif'), colored_pred)  # or png
