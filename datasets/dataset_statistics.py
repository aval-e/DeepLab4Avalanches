""" This script calculates the dataset input value distribution for each band. It was used for analysing the
distributions of variables to ensure they were suitable for deep learning
"""


import numpy as np
from torch.utils.data import DataLoader
from datasets.avalanche_dataset_points import AvalancheDatasetPoints
import matplotlib.pyplot as plt
from tqdm import tqdm


def main():
    my_dataset = AvalancheDatasetPoints(data_folder,
                                        ava_file,
                                        region_file,
                                        means=[986.3, 1028.3, 1023.9, 949.9],
                                        stds=[1014.3, 955.9, 823.4, 975.5],
                                        tile_size=512,
                                        dem_path=dem_path,
                                        random=False
                                        )
    dataloader = DataLoader(my_dataset, batch_size=4, shuffle=False, num_workers=8)

    # Show bins from -3, 3 as this should cover most standardised values
    bins = np.linspace(-3, 3, no_bins + 1)
    cum_hist = np.zeros([no_channels, no_bins])
    for k, batch in enumerate(tqdm(iter(dataloader), desc='Computing histrogram')):
        x, y = batch
        x = x.clamp(-3, 3)  # clamp to the histogram range to ensure no values are omitted
        for i in range(no_channels):
            hist, _ = np.histogram(x[:, i, :, :], bins=bins)
            cum_hist[i, :] += hist

    # plot distributions with y axis representing the amount as percentage of all data
    fig, a = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle('Dataset Input Distributions')
    a[0, 0].set_title('Blue Band')
    a[0, 0].hist(bins[1:], bins, weights=cum_hist[0, :] / cum_hist[0, :].sum() * 100)
    a[0, 1].set_title('Green Band')
    a[0, 1].hist(bins[1:], bins, weights=cum_hist[1, :] / cum_hist[1, :].sum() * 100)
    a[1, 0].set_title('Red Band')
    a[1, 0].hist(bins[1:], bins, weights=cum_hist[2, :] / cum_hist[2, :].sum() * 100)
    a[1, 1].set_title('Infrared Band')
    a[1, 1].hist(bins[1:], bins, weights=cum_hist[3, :] / cum_hist[3, :].sum() * 100)
    a[2, 0].set_title('DEM')
    a[2, 0].hist(bins[1:], bins, weights=cum_hist[4, :] / cum_hist[4, :].sum() * 100)
    # a[2, 1].set_title('DEM grad y')
    # a[2, 1].hist(bins[1:], bins, weights=cum_hist[5, :])
    plt.show()
    fig.savefig('data_distribution.png')


if __name__ == "__main__":
    no_channels = 5
    no_bins = 100

    # pfpc
    data_folder = '/home/pf/pfstud/bartonp/slf_avalanches/2018'
    ava_file = 'avalanches0118_endversion.shp'
    region_file = 'Val_area_2018.shp'
    dem_path = '/home/pf/pfstud/bartonp/dem_ch/swissalti3d_2017_ESPG2056_packbits_tiled.tif'

    main()
