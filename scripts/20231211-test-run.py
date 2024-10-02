import pathlib
import argparse

import torch
import numpy as np
import einops
import poregen.models


MAINFOLDER = pathlib.Path(".")
RAWDATAFOLDER = MAINFOLDER/"saveddata"/"raw"/"gravity_packing"


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, scale_factor=4):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset[idx].unsqueeze(0)
        return x


def extract_subvolumes(voxel, width=128, height=128, depth=16, stride=32):
    # Precomputing the number of steps along each dimension
    steps_x = (voxel.shape[0] - width) // stride + 1
    steps_y = (voxel.shape[1] - height) // stride + 1
    steps_z = (voxel.shape[2] - depth) // stride + 1

    # Initializing an empty list to hold the sub-volumes
    subvolumes = []

    # Nested loops to traverse the voxel and extract sub-volumes
    for i in range(steps_x):
        for j in range(steps_y):
            for k in range(steps_z):
                # Computing the indices for the sub-volume
                x_start, x_end = i * stride, i * stride + width
                y_start, y_end = j * stride, j * stride + height
                z_start, z_end = k * stride, k * stride + depth

                # Extracting and appending the sub-volume to the list
                subvolume = voxel[x_start:x_end, y_start:y_end, z_start:z_end]
                subvolumes.append(subvolume)

    # Converting the list of sub-volumes to a 4D numpy array
    result = np.stack(subvolumes, axis=0)

    return result


def main(nepochs, batch_size, window_size):
    name = "voxel_gravity_packing_p480_r10_n25000_extended_001.npy"
    voxel = np.load(RAWDATAFOLDER/name)
    result = extract_subvolumes(voxel,
                                window_size,
                                window_size,
                                16,
                                stride=16)
    result = einops.rearrange(result, "b x y z -> (b z) x y")
    dataset = CustomDataset(torch.tensor(result, dtype=torch.float))

    batch_size = batch_size
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                [train_size,
                                                                 test_size])

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = poregen.models.HFNetUncond([16, 32, 64], norm_num_groups=8)
    model = model.to(device)
    scheduler = poregen.models.DDPMScheduler()
    trainer = poregen.models.UncondDDPMTrainer(model, scheduler,
                                               train_dataloader,
                                               test_dataloader=test_dataloader,
                                               device=device,
                                               loss_scale_factor=100.0)
    trainer.set_optimizer_and_scheduler()
    trainer.train(nepochs=100)
    trainer.save_model("savedmodels/121109_testscript.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--nepochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--length', type=int, default=32,
                        help='Length of window (default: 32)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')

    # Parse arguments
    args = parser.parse_args()
    main(args.nepochs, args.batch_size, args.length)
