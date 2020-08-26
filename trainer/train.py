from pytorch_lightning import Trainer
from experiments.easy_experiment import EasyExperiment
from datasets.AvalancheDataset import AvalancheDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


# data_folder = '/media/patrick/Seagate Expansion Drive/SLF_Avaldata/2019'
data_folder = '/home/patrick/ecovision/data'
tif_folder = 'Spot6_Ortho_2_3_3_4_5'
ava_file = 'avalanches0119_endversion.shp'
region_file = 'Region_Selection.shp'
# region_file = 'Multiple_regions.shp'

my_dataset = AvalancheDataset(data_folder, [tif_folder], ava_file, region_file, transform=ToTensor(), tile_size=(600,600))
dataloader = DataLoader(my_dataset, batch_size=2, shuffle=True, num_workers=1, drop_last=True)

model = EasyExperiment()
trainer = Trainer(gpus=1)

trainer.fit(model, dataloader)