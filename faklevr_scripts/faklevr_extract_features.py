# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse, os, json  # noqa: E401, F401
import h5py
import numpy as np
from PIL import Image
import imageio

import torch
import torchvision

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--input_image_dir", required=True)
parser.add_argument("--output_h5_file", required=True)

parser.add_argument("--image_height", default=224, type=int)
parser.add_argument("--image_width", default=224, type=int)

parser.add_argument("--model", default="resnet101", type=str)
parser.add_argument("--model_stage", default=3, type=int)
parser.add_argument("--batch_size", default=128, type=int)


def build_model(args):
    if args.model.lower() == "none":
        return None
    if not hasattr(torchvision.models, args.model):
        raise ValueError('Invalid model "%s"' % args.model)
    if "resnet" not in args.model:
        raise ValueError("Feature extraction only supports ResNets")
    cnn = getattr(torchvision.models, args.model)(pretrained=True)
    layers = [
        cnn.conv1,
        cnn.bn1,
        cnn.relu,
        cnn.maxpool,
    ]
    for i in range(args.model_stage):
        name = "layer%d" % (i + 1)
        layers.append(getattr(cnn, name))
    model = torch.nn.Sequential(*layers)
    model.to(device)  # Send model to the MPS (or CPU) device.
    model.eval()
    return model


def run_batch(cur_batch, model):
    if model is None:
        image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
        return image_batch / 255.0  # Scale pixel values to [0, 1]
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)

    image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
    image_batch = (image_batch / 255.0 - mean) / std
    image_batch = torch.FloatTensor(image_batch).to(device)

    # Disable gradient calculations for inference.
    with torch.no_grad():
        feats = model(image_batch)

    feats = feats.cpu().numpy()
    return feats


def main(args):
    input_paths = []
    idx_set = set()
    for fn in os.listdir(args.input_image_dir):
        if not fn.endswith(".png"):
            continue
        idx = int(os.path.splitext(fn)[0].split("_")[-1])
        input_paths.append((os.path.join(args.input_image_dir, fn), idx))
        idx_set.add(idx)
    input_paths.sort(key=lambda x: x[1])
    assert len(idx_set) == len(input_paths)
    assert min(idx_set) == 0 and max(idx_set) == len(idx_set) - 1
    print(input_paths[0])
    print(input_paths[-1])

    model = build_model(args)

    img_size = (args.image_height, args.image_width)
    with h5py.File(args.output_h5_file, "w") as f:
        feat_dset = None
        i0 = 0
        cur_batch = []
        for i, (path, idx) in enumerate(input_paths):
            img = imageio.imread(path, pilmode="RGB")
            im = Image.fromarray(img)
            im_resized = im.resize((img_size[1], img_size[0]), resample=Image.BICUBIC)
            img = np.array(im_resized)
            img = img.transpose(2, 0, 1)[None]
            cur_batch.append(img)
            if len(cur_batch) == args.batch_size:
                feats = run_batch(cur_batch, model)
                if feat_dset is None:
                    N = len(input_paths)
                    _, C, H, W = feats.shape
                    feat_dset = f.create_dataset(
                        "features", (N, C, H, W), dtype=np.float32
                    )
                i1 = i0 + len(cur_batch)
                feat_dset[i0:i1] = feats
                i0 = i1
                print("Processed %d / %d images" % (i1, len(input_paths)))
                cur_batch = []
        if len(cur_batch) > 0:
            feats = run_batch(cur_batch, model)
            i1 = i0 + len(cur_batch)
            feat_dset[i0:i1] = feats
            print("Processed %d / %d images" % (i1, len(input_paths)))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)