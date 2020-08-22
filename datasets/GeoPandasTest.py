import geopandas as gpd
import matplotlib.pyplot as plt
import os
import numpy as np
from utils import data_utils
from shapely.geometry import Point, GeometryCollection

# data_folder = '/media/patrick/Seagate Expansion Drive/SLF_Avaldata/2019'
data_folder = '/home/patrick/ecovision/data'
tif_folder = 'Spot6_Ortho_2_3_3_4_5'
ava_file = 'avalanches0119_endversion.shp'
region_file = 'Region_Selection.shp'

region = gpd.read_file(os.path.join(data_folder, region_file))

# generate sample points
bb = region.bounds
no_points = 10
minx = bb.at[0,'minx']
maxx = bb.at[0,'maxx']
miny = bb.at[0,'miny']
maxy = bb.at[0,'maxy']
spacing = ((maxx - minx) / no_points, (maxy - miny) / no_points)

X, Y = np.mgrid[minx:maxx:spacing[0], miny:maxy:spacing[1]]
X, Y = X.ravel(), Y.ravel()
points = gpd.GeoSeries(map(Point, zip(X, Y)))

mask = points.within(region.loc[0, 'geometry'])
points = points.loc[mask]


fig, ax = plt.subplots()
region.plot(ax=ax, facecolor='red')
# ax.scatter(X.ravel(), Y.ravel(), color='b')
points.plot(ax=ax, color='b')
plt.show()