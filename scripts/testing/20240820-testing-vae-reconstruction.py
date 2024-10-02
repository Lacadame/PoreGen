import pathlib

import torch
import lightning

import poregen.data
import poregen.models


CURRENTPATH = pathlib.Path(__file__).parent.absolute()
MAINPATH = CURRENTPATH.parent.parent
RAWDATAPATH = MAINPATH/"saveddata"/"raw"


def main():
    voxel_size = 64
    number_of_samples_per_sandstone = 16

    checkpoint_path = (MAINPATH/
        'savedmodels/experimental/[pore]-[2024-08-21]-[dfn]-[64-noattn-vae-1-2-4-4-8-16]/sample-epoch=28-val/rec_loss=0.094070.ckpt')

    lossconfig = poregen.models.nets.autoencoderldm3d.lossconfig(kl_weight=1e-4)
    ddconfig = poregen.models.nets.autoencoderldm3d.ddconfig(
        resolution=voxel_size,
        has_mid_attn=False,
        ch_mult=[1, 2, 4, 4, 8, 16])

    vae_module = poregen.models.nets.autoencoderldm3d.AutoencoderKL.load_from_checkpoint(
        checkpoint_path, ddconfig=ddconfig, lossconfig=lossconfig
        )
    vae_module.eval();

    sandstones = ['Berea',
                  'BUG',
                  'Leopard',
                  'Bentheimer',
                  'Parker',
                  'BanderaBrown',
                  'Kirby',
                  'BanderaGray',
                  'CastleGate',
                  'BSG',
                  'BB']

    for sandstone in sandstones:
        datapath = RAWDATAPATH/f'eleven_sandstones/{sandstone}_2d25um_binary.raw'
        voxel = poregen.data.load_binary_from_eleven_sandstones(datapath)
        dataset = poregen.data.VoxelToSubvoxelDataset(
            voxel,
            voxel_size
        )


        print(f"Voxel size : {voxel_size}")
        with torch.inference_mode():
            vae_module.to("cpu")

            x_orig = torch.stack([
                dataset[i].to("cpu")
                for i in range(number_of_samples_per_sandstone)
            ], axis=0)
    
            x_encoded = vae_module.encode(x_orig)
            
            x_decoded = vae_module.decode(x_encoded)

            x_orig_bin = (x_orig > x_orig.mean(axis=(1, 2, 3, 4), keepdim=True))
            x_decoded_bin = (x_decoded > x_decoded.mean(axis=(1, 2, 3, 4), keepdim=True))

            score = (x_decoded_bin != x_orig_bin).float().mean(axis=(1, 2, 3, 4))

            print(sandstone)
            print(score.tolist())
            print('-'*10)


if __name__ == "__main__":
    main()