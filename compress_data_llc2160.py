import xarray as xr
import numpy as np
import glob 
import xgcm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap, cm, shiftgrid
import dask.array as da
import pandas as pd

#from dask.distributed import Client, LocalCluster
#cluster = LocalCluster(n_workers=3, threads_per_worker=40)
#client = Client(cluster)


uname = "/HOME/users/fcampos/outputs/global/U/"
vname = "/HOME/users/fcampos/outputs/global/V/"
tuname = "/HOME/users/fcampos/outputs/global/TAUX/"
tvname = "/HOME/users/fcampos/outputs/global/TAUY/"
gridname = "/HOME/users/fcampos/outputs/global/grid/"
nx, ny = 8640, 6480

XC = np.memmap(gridname+'XC.data',dtype='>f4',shape=(ny,nx),mode='r+')
YC = np.memmap(gridname+'YC.data',dtype='>f4',shape=(ny,nx),mode='r+')
XG = np.memmap(gridname+'XG.data',dtype='>f4',shape=(ny,nx),mode='r+')
YG = np.memmap(gridname+'YG.data',dtype='>f4',shape=(ny,nx),mode='r+')
dxC = np.memmap(gridname+'dxC.data',dtype='>f4',shape=(ny,nx),mode='r+')
dyC = np.memmap(gridname+'dyC.data',dtype='>f4',shape=(ny,nx),mode='r+')
dxG = np.memmap(gridname+'dxG.data',dtype='>f4',shape=(ny,nx),mode='r+')
dyG = np.memmap(gridname+'dyG.data',dtype='>f4',shape=(ny,nx),mode='r+')
drF = np.memmap(gridname+'drF.data',dtype='>f4',shape=(ny,nx),mode='r+')
rAZ = np.memmap(gridname+'rAz.data',dtype='>f4',shape=(ny,nx),mode='r+')
rA = np.memmap(gridname+'rA.data',dtype='>f4',shape=(ny,nx),mode='r+')
hfacs=np.memmap(gridname+'hFacS.data',dtype='>f4',shape=(ny,nx),mode='r+')
hfacw=np.memmap(gridname+'hFacW.data',dtype='>f4',shape=(ny,nx),mode='r+')
hfacc=np.memmap(gridname+'hFacC.data',dtype='>f4',shape=(ny,nx),mode='r+')

coords={
    "i": (["i"], np.arange(nx), {"axis": "X"}, ),
    "j": (["j"], np.arange(ny), {"axis": "Y"}, ), 
    "i_g": (["i_g"], np.arange(nx), {"axis": "X", "c_grid_axis_shift": -0.5}, ),
    "j_g": (["j_g"], np.arange(ny), {"axis": "Y", "c_grid_axis_shift": -0.5}, ),
    "XC": (["j", "i"], XC, ),
    "XG": (["j_g", "i_g"], XG, ),
    "YC": (["j", "i"], YC, ),
    "YG": (["j_g", "i_g"], YG, ),
    "dxC": (["j", "i_g"], dxC, ),
    "dyC": (["j_g", "i"], dyC, ),    
    "dxG": (["j_g", "i"], dxG, ),
    "dyG": (["j", "i_g"], dyG, ),    
    "rAz": (["j_g", "i_g"], rAZ, ),
    "rA": (["j", "i"], rA, ),    
    "hFacS": (["j_g", "i"], hfacs, ),
    "hFacW": (["j", "i_g"], hfacw, ),
    "hFacC": (["j", "i"], hfacc, ),
    }

#ds = xr.Dataset(data_vars=None, coords=coords,).astype("float32")
#ds.to_netcdf("/HOME/users/fcampos/outputs/global/grid/grid.nc")

time = pd.date_range("2012-07-01", periods=91, freq="d")

for j in range(len(time)):
    print(time[j])
    file= str(time[j])
    uf = np.sort(glob.glob(uname+"jas/U_8640x6480_"+file[0:4]+file[5:7]+file[8:10]+"T*"))
    vf = np.sort(glob.glob(vname+"jas/V_8640x6480_"+file[0:4]+file[5:7]+file[8:10]+"T*"))
    #txf = np.sort(glob.glob(tuname+"jfm/TAUX_8640x6480_"+file[0:4]+file[5:7]+file[8:10]+"T*"))
    #tyf = np.sort(glob.glob(tuname+"jfm/TAUY_8640x6480_"+file[0:4]+file[5:7]+file[8:10]+"T*"))
    nt = len(uf)
    u = np.zeros((nt, ny, nx))
    v = np.zeros((nt, ny, nx))
    #tx = np.zeros((nt, ny, nx))
    #ty = np.zeros((nt, ny, nx))
    for i in range(nt):
        u[i,:,:] = np.squeeze(np.memmap(uf[i],dtype='>f4',shape=(ny,nx),mode='r+'))
        v[i,:,:] = np.squeeze(np.memmap(vf[i],dtype='>f4',shape=(ny,nx),mode='r+'))
        #tx[i,:,:] = np.squeeze(np.memmap(txf[i],dtype='>f4',shape=(ny,nx),mode='r+'))
        #ty[i,:,:] = np.squeeze(np.memmap(tyf[i],dtype='>f4',shape=(ny,nx),mode='r+'))
    data_vars = dict(
        u = (["time","j","i_g"], u, {"units":"m/s"}),
        v = (["time","j_g","i"], v, {"units":"m/s"}),
        #tx = (["time","j","i_g"], tx, {"units":"N/m^2"}),
        #ty = (["time","j_g","i"], ty, {"units":"N/m^2"}),
        )
    del u, v, uf, vf
    ds = xr.Dataset(data_vars=data_vars, 
                    coords=coords, 
                    ).astype("float32").reset_coords(drop=True)
    encoding = {"u": {'zlib': True,}, "v": {"zlib": True}}  
    ds.to_netcdf("/HOME/users/fcampos/outputs/global/compressed_data/currents/vel_compress_"+
                 file[0:4]+file[5:7]+file[8:10]+".nc", encoding = encoding)

#var = xr.open_mfdataset("./velocity_comp_2012010*.nc", parallel=True)



