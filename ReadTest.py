import os.path
import matplotlib.pyplot as plt
from osgeo import gdal, ogr
import numpy as np

# data_folder = '/media/patrick/Seagate Expansion Drive/SLF_Avaldata/2019'
data_folder = '/home/patrick/ecovision/data'
tif_folder = 'Spot6_Ortho_2_3_3_4_5'
tif = 'ortho_2_dim_spot6_pms_201901161003537_sen_37024891011.TIF'
shp = 'avalanches0119_endversion.shp'

# get satellite image
tif_path = os.path.join(os.path.join(data_folder, tif_folder), tif)
dataset = gdal.Open(tif_path, gdal.GA_ReadOnly)

offset = (5000,5000) # x, y
res = (5000, 5000) # x, y
bands = []
for band in range(3):
    rb = dataset.GetRasterBand(band+1)
    bands.append(rb.ReadAsArray(offset[0], offset[1], res[0], res[1]))

# plot image
image = np.stack(bands, axis=2)
image = (image / image.max())


# get shapefile in raster format
geo_transform = dataset.GetGeoTransform()
x_min = geo_transform[0]
y_max = geo_transform[3]
x_max = x_min + geo_transform[1] * dataset.RasterXSize
y_min = y_max + geo_transform[5] * dataset.RasterYSize
x_res = dataset.RasterXSize
y_res = dataset.RasterYSize

shape_path = os.path.join(data_folder, shp)
mb_v = ogr.Open(shape_path)
mb_l = mb_v.GetLayer()
pixel_width = geo_transform[1]

target_ds = gdal.GetDriverByName('MEM').Create('', res[0], res[1], 1, gdal.GDT_Byte)
target_ds.SetGeoTransform((x_min+offset[0]*pixel_width, pixel_width, 0, y_min+(-offset[1]+y_res)*pixel_width, 0, -pixel_width))
err = gdal.RasterizeLayer(target_ds, [1], mb_l, burn_values=[1])
if err != 0:
    print('Rasterising error: ', err)

shp_image = target_ds.ReadAsArray()
image[:,:,0] += 0.2*shp_image
image[:,:,1] -= 0.2*shp_image
image[:,:,2] -= 0.2*shp_image

plt.figure(dpi=500)
plt.imshow(image)
plt.show()
