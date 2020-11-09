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
                                        tile_size=256,
                                        dem_path=dem_path,
                                        random=False
                                        )
    dataloader = DataLoader(my_dataset, batch_size=4, shuffle=False, num_workers=8)

    cum_hist = np.zeros([no_channels, no_bins])
    bins = no_channels * [None]
    for batch in tqdm(iter(dataloader), desc='Computing histrogram'):
        x, y = batch
        for i in range(no_channels):
            hist, bins[i] = np.histogram(x[:, :, :, i], bins=no_bins)
            cum_hist[i,:] += hist

    fig, a = plt.subplots(3, 2, figsize=(15,10))
    fig.suptitle('Dataset Distributions')
    a[0, 0].set_title('Channel 0')
    a[0, 0].hist(bins[0][1:], bins[0], weights=cum_hist[0, :])
    a[0, 1].set_title('Channel 1')
    a[0, 1].hist(bins[1][1:], bins[1], weights=cum_hist[1, :])
    a[1, 0].set_title('Channel 2')
    a[1, 0].hist(bins[2][1:], bins[2], weights=cum_hist[2, :])
    a[1, 1].set_title('Channel 3')
    a[1, 1].hist(bins[3][1:], bins[3], weights=cum_hist[3, :])
    a[2, 0].set_title('DEM grad x')
    a[2, 0].hist(bins[4][1:], bins[4], weights=cum_hist[4, :])
    a[2, 1].set_title('DEM grad y')
    a[2, 1].hist(bins[5][1:], bins[5], weights=cum_hist[5, :])
    plt.show()
    fig.savefig('data_distribution.png')


if __name__ == "__main__":
    no_channels = 6
    no_bins = 100

    # home
    data_folder = '/home/patrick/ecovision/data/2018'
    ava_file = 'avalanches0118_endversion.shp'
    region_file = 'Region_Selection.shp'
    dem_path = '/home/patrick/ecovision/data/2018/avalanches0118_endversion.tif'

    # pfpc
    # data_folder = '/home/pf/pfstud/bartonp/slf_avalanches/2018'
    # ava_file = 'avalanches0118_endversion.shp'
    # region_file = 'Val_area_2018.shp'
    # dem_path="" #'/home/pf/pfstud/bartonp/dem_ch/swissalti3d_2017_ESPG2056.tif'

    main()