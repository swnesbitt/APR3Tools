"""
=================

A few tools to view and read APR3 data

Developed by Randy J. Chase (Feb 2017) at The University of Illinois, Urbana-Champaign 

=================
"""
from pyhdf.SD import SD, SDC
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib import dates
import pyresample as pr
from pyresample import geometry, data_reduce
from pylab import rc
from scipy.spatial import cKDTree

##plot attributes 
rc('axes', linewidth=2.5)

def apr3plot(filename,scan,band,fontsize=14,fontsize2=12,savefig=False,cmap = 'seismic',figsize=[10,5],mask=False):
    """
    ===========
    
    This function is designed to produce high quality quick plots of the apr3 radar data. 
    
    ===========
    
    filename = filename of the apr3 file, str
    scan = the scan within the desired data (12 is nadir), int
    band = the wavelength of radar you would like. Options: Ku, Ka, W, ldr, vel., str
    fontsize = fontsize for labels of the plot, int
    fontsize2 = fontsize for tick labels of the plot, int
    savefig = True will save figure, False will not, Bool
    cmap = color map for plot, str 
    figsize = size of the plot size, vector of 2 nums [length,width]
    
    """
    fig,ax = plt.subplots(1,1,figsize = figsize)
    
    if band == 'vel':
        
        colorbarlabel = 'Velocity, $[m s^{-1}]$'
        vmin = -10 
        vmax = 10
        
    elif band == 'DFR_1':
        
        colorbarlabel = 'DFR_Ku-Ka, $[dB]$'
        vmin = 0 
        vmax = 10

    elif band == 'DFR_2':
        
        colorbarlabel = 'DFR_Ka-W, $[dB]$'
        vmin = -20 
        vmax = 20
        
    elif band == 'DFR_3':
        
        colorbarlabel = 'DFR_Ku-W, $[dB]$'
        vmin = -20 
        vmax = 20
    elif band == 'W' :
        
        colorbarlabel = 'Reflectivity, $[dBZ]$'
        vmin = -20 
        vmax = 20
        
    else:
        
        colorbarlabel = 'Reflectivity, $[dBZ]$'
        vmin = 0 
        vmax = 40
        
    if mask:
        color_p = 'k'
    else:
        color_p = 'w'
        
    apr = apr3read(filename,scan,mask=mask)
    
    if (apr['W'].size==0 and band=='W') or (apr['W'].size==0 and band=='DFR_2') or (apr['W'].size==0 and band=='DFR_3'):
        print()
    else:
        time = apr['time_gate']
        time_dates = apr['time_dates']
        alt = apr['alt_gate']
        plane = apr['alt_plane']
        radar = apr[band]
        surface = apr['surface']
        pm = ax.pcolormesh(time,alt/1000.,radar,vmin=vmin,vmax=vmax,cmap=cmap)
        ax.fill_between(time_dates,surface/1000.,-8.,color='k',edgecolor='w')
        ax.plot(time_dates,plane/1000.,'-.',lw=3,color=color_p)
        hfmt = dates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(hfmt)
        ax.set_ylim([0,max(plane/1000.)+0.1])
        ax.set_ylabel('Altitude, [km]',fontsize=fontsize)
        ax.set_xlabel('Time, [UTC]',fontsize=fontsize)
        ax.set_title(time_dates[0],fontsize=fontsize+2)
        ax.tick_params(axis='both',direction='in',labelsize=fontsize2,width=2,length=5,color='k')
        cbar = plt.colorbar(pm,aspect = 10)
        cbar.set_label(colorbarlabel,fontsize=fontsize)
        cax = cbar.ax
        cax.tick_params(labelsize=fontsize2)

        if savefig:
            print('Save file is: ' + str(time_dates[0])+'_'+band+'.png')
            plt.savefig(str(time_dates[0])+'_'+band+'.png',dpi=300)
        plt.show()
    return

def apr3flighttrack(filename):
    
    """
    ===========
    
    This is for creating a quick map of the flight for spatial reference (currently just for OLYMPEX domain)
    
    ===========
    
    filename = filename of the apr3 file
    """
        
    fig,ax = plt.subplots(1,1,figsize = [10,5])
    apr = apr3read(filename,12)
    lat = apr['lat']
    lon = apr['lon']
    area_def = pr.geometry.AreaDefinition('areaD', 'IPHEx', 'areaD',
                                      {'a': '6378144.0', 'b': '6356759.0',
                                       'lat_0': '47.6', 'lat_ts': '47.6','lon_0': '-124.5', 'proj': 'stere'},
                                      400, 400,
                                      [-400000., -400000.,
                                       400000., 400000.])
    bmap = pr.plot.area_def2basemap(area_def,resolution='l')
    bmap.drawcoastlines(linewidth=2)
    bmap.drawstates(linewidth=2)
    bmap.drawcountries(linewidth=2)
    parallels = np.arange(-90.,90,2)
    bmap.drawparallels(parallels,labels=[1,0,0,0],fontsize=12)
    meridians = np.arange(180.,360.,2)
    bmap.drawmeridians(meridians,labels=[0,0,0,1],fontsize=12)
    bmap.drawmapboundary(fill_color='aqua')
    bmap.fillcontinents(color='coral',lake_color='aqua')
    x,y = bmap(lon,lat)
    plt.plot(x,y,'k--',lw=3)
    plt.show()

def apr3read(filename,scan,mask=False,vmin=0,vmax=40):
    
    """
    ===========
    
    This is for reading in apr3 hdf files from OLYMPEX and return them all in one dictionary
    
    ===========
    
    filename = filename of the apr3 file
    scan = the scan within the desired data (12 is nadir)
    """
    
    apr = {}
    flag = 0
    ##Radar varibles in hdf file found by hdf.datasets
    radar_freq = 'zhh14' #Ku
    radar_freq2 = 'zhh35' #Ka
    radar_freq3 = 'zvv95' #W
    radar_freq4 = 'ldr14' #LDR
    vel_str = 'vel14' #Doppler
    ##

    hdf = SD(filename, SDC.READ)
    
    listofkeys = hdf.datasets().keys()
    if 'zvv95' in listofkeys:
        radar3 = hdf.select(radar_freq3)
        radar_n3 = radar3.get()
        radar_n3 = radar_n3[:,scan,:]/100.
    else:
        radar_n3 = np.array([])
        flag = 1
        print('No W band')
    
    
    alt = hdf.select('alt3D')
    lat = hdf.select('lat')
    lon = hdf.select('lon')
    time = hdf.select('scantime')
    surf = hdf.select('surface_index')
    isurf = hdf.select('isurf')
    plane = hdf.select('alt_nav')
    radar = hdf.select(radar_freq)
    radar2 = hdf.select(radar_freq2)
    radar4 = hdf.select(radar_freq4)
    vel = hdf.select(vel_str)

    alt = alt.get()
    alt = alt[:,scan,:]
    lat = lat.get()
    lat = lat[scan,:]
    lon = lon.get()
    lon = lon[scan,:]
    time = time[scan,:]
    surf = surf[scan,:]
    isurf = isurf[scan,:]
    plane = plane[scan,:]
    radar_n = radar.get()
    radar_n = radar_n[:,scan,:]/100.
    radar_n2 = radar2.get()
    radar_n2 = radar_n2[:,scan,:]/100.

    radar_n4 = radar4.get()
    radar_n4 = radar_n4[:,scan,:]/100.
    vel_n = vel.get()
    vel_n = vel_n[:,scan,:]/100.

"""
=================

A few tools to view and read APR3 data

Developed by Randy J. Chase (Feb 2017) at The University of Illinois, Urbana-Champaign 

=================
"""
from pyhdf.SD import SD, SDC
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib import dates
import pyresample as pr
from pyresample import geometry, data_reduce
from pylab import rc
from pyart.core.radar import Radar

##plot attributes 
rc('axes', linewidth=2.5)

def apr3plot(filename,scan,band,fontsize=14,fontsize2=12,savefig=False,cmap = 'seismic',figsize=[10,5],mask=False):
    """
    ===========
    
    This function is designed to produce high quality quick plots of the apr3 radar data. 
    
    ===========
    
    filename = filename of the apr3 file, str
    scan = the scan within the desired data (12 is nadir), int
    band = the wavelength of radar you would like. Options: Ku, Ka, W, ldr, vel., str
    fontsize = fontsize for labels of the plot, int
    fontsize2 = fontsize for tick labels of the plot, int
    savefig = True will save figure, False will not, Bool
    cmap = color map for plot, str 
    figsize = size of the plot size, vector of 2 nums [length,width]
    
    """
    fig,ax = plt.subplots(1,1,figsize = figsize)
    
    if band == 'vel':
        
        colorbarlabel = 'Velocity, $[m s^{-1}]$'
        vmin = -10 
        vmax = 10
        
    elif band == 'DFR_1':
        
        colorbarlabel = 'DFR_Ku-Ka, $[dB]$'
        vmin = -10 
        vmax = 10

    elif band == 'DFR_2':
        
        colorbarlabel = 'DFR_Ka-W, $[dB]$'
        vmin = -20 
        vmax = 20
        
    elif band == 'DFR_3':
        
        colorbarlabel = 'DFR_Ku-W, $[dB]$'
        vmin = -20 
        vmax = 20
    elif band == 'W' :
        
        colorbarlabel = 'Reflectivity, $[dBZ]$'
        vmin = -20 
        vmax = 20
        
    else:
        
        colorbarlabel = 'Reflectivity, $[dBZ]$'
        vmin = 0 
        vmax = 40
        
    if mask:
        color_p = 'k'
    else:
        color_p = 'w'
        
    apr = apr3read(filename)
    
    if (apr['W'].size==0 and band=='W') or (apr['W'].size==0 and band=='DFR_2') or (apr['W'].size==0 and band=='DFR_3'):
        print()
    else:
        time_dates = apr['timedates'][scan,:]
        alt = apr['alt_gate'][:,scan,:]
        plane = apr['alt_plane'][scan,:]
        radar = apr[band][:,scan,:]
        surface = apr['surface'][scan,:]
        pm = ax.pcolormesh(time_dates,alt/1000.,radar,vmin=vmin,vmax=vmax,cmap='seismic')
        ax.fill_between(time_dates,alt[surface,0]/1000.,color='k',edgecolor='w')
        ax.plot(time_dates,plane/1000.,'-.',lw=3,color='w')
        ax.set_ylim([0,max(plane/1000.)+0.1])
        hfmt = dates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(hfmt)
        ax.set_ylim([0,max(plane/1000.)+0.1])
        ax.set_ylabel('Altitude, [km]',fontsize=fontsize)
        ax.set_xlabel('Time, [UTC]',fontsize=fontsize)
        ax.set_title(time_dates[0],fontsize=fontsize+2)
        ax.tick_params(axis='both',direction='in',labelsize=fontsize2,width=2,length=5,color='k')
        cbar = plt.colorbar(pm,aspect = 10)
        cbar.set_label(colorbarlabel,fontsize=fontsize)
        cax = cbar.ax
        cax.tick_params(labelsize=fontsize2)

        if savefig:
            print('Save file is: ' + str(time_dates[0])+'_'+band+'.png')
            plt.savefig(str(time_dates[0])+'_'+band+'.png',dpi=300)
        plt.show()
    return

def apr3flighttrack(filename):
    
    """
    ===========
    
    This is for creating a quick map of the flight for spatial reference (currently just for OLYMPEX domain)
    
    ===========
    
    filename = filename of the apr3 file
    """
        
    fig,ax = plt.subplots(1,1,figsize = [10,5])
    apr = apr3read(filename)
    lat = apr['lat'][12,:]
    lon = apr['lon'][12,:]
    area_def = pr.geometry.AreaDefinition('areaD', 'IPHEx', 'areaD',
                                      {'a': '6378144.0', 'b': '6356759.0',
                                       'lat_0': '47.6', 'lat_ts': '47.6','lon_0': '-124.5', 'proj': 'stere'},
                                      400, 400,
                                      [-400000., -400000.,
                                       400000., 400000.])
    bmap = pr.plot.area_def2basemap(area_def,resolution='l')
    bmap.drawcoastlines(linewidth=2)
    bmap.drawstates(linewidth=2)
    bmap.drawcountries(linewidth=2)
    parallels = np.arange(-90.,90,2)
    bmap.drawparallels(parallels,labels=[1,0,0,0],fontsize=12)
    meridians = np.arange(180.,360.,2)
    bmap.drawmeridians(meridians,labels=[0,0,0,1],fontsize=12)
    bmap.drawmapboundary(fill_color='aqua')
    bmap.fillcontinents(lake_color='aqua')
    x,y = bmap(lon,lat)
    plt.plot(x,y,'k--',lw=3)
    plt.show()

def apr3read(filename):
    
    """
    ===========
    
    This is for reading in apr3 hdf files from OLYMPEX and return them all in one dictionary
    
    ===========
    
    filename = filename of the apr3 file
    """
    
    apr = {}
    flag = 0
    ##Radar varibles in hdf file found by hdf.datasets
    radar_freq = 'zhh14' #Ku
    radar_freq2 = 'zhh35' #Ka
    radar_freq3 = 'zvv95' #W
    radar_freq4 = 'ldr14' #LDR
    vel_str = 'vel14' #Doppler
    ##

    hdf = SD(filename, SDC.READ)
    
    listofkeys = hdf.datasets().keys()
    if 'zvv95' in listofkeys:
        radar3 = hdf.select(radar_freq3)
        radar_n3 = radar3.get()
        radar_n3 = radar_n3/100.
    else:
        radar_n3 = np.array([])
        flag = 1
        print('No W band')
    
    
    alt = hdf.select('alt3D')
    lat = hdf.select('lat')
    lon = hdf.select('lon')
    time = hdf.select('scantime').get()
    surf = hdf.select('surface_index').get()
    isurf = hdf.select('isurf').get()
    plane = hdf.select('alt_nav').get()
    radar = hdf.select(radar_freq)
    radar2 = hdf.select(radar_freq2)
    radar4 = hdf.select(radar_freq4)
    vel = hdf.select(vel_str)
    lon3d = hdf.select('lon3D')
    lat3d = hdf.select('lat3D')
    alt3d = hdf.select('alt3D')
    lat3d_scale = hdf.select('lat3D_scale').get()[0][0]
    lon3d_scale = hdf.select('lon3D_scale').get()[0][0]
    alt3d_scale = hdf.select('alt3D_scale').get()[0][0]
    lat3d_offset = hdf.select('lat3D_offset').get()[0][0]
    lon3d_offset = hdf.select('lon3D_offset').get()[0][0]
    alt3d_offset = hdf.select('alt3D_offset').get()[0][0]
 
    alt = alt.get()
    ngates = alt.shape[0]
    #alt = alt[:,scan,:]
    lat = lat.get()
    #lat = lat[scan,:]
    lon = lon.get()
    #lon = lon[scan,:]
    
    lat3d = lat3d.get()
    lat3d = (lat3d/lat3d_scale) + lat3d_offset
    lon3d = lon3d.get()
    lon3d = (lon3d/lon3d_scale) + lon3d_offset
    alt3d = alt3d.get()
    alt3d = (alt3d/alt3d_scale) + alt3d_offset
    
    #time = time[scan,:]
    #surf = surf[scan,:]
    #isurf = isurf[scan,:]
    #plane = plane[scan,:]
    radar_n = radar.get()
    radar_n = radar_n/100.
    radar_n2 = radar2.get()
    radar_n2 = radar_n2/100.

    radar_n4 = radar4.get()
    radar_n4 = radar_n4/100.
    vel_n = vel.get()
    vel_n = vel_n/100.

    ##convert time to datetimes
    time_dates = np.empty(time.shape,dtype=object)
    for i in np.arange(0,time.shape[0]):
        for j in np.arange(0,time.shape[1]):
            tmp = datetime.datetime.utcfromtimestamp(time[i,j])
            time_dates[i,j] = tmp

    apr['Ku'] = radar_n
    apr['Ka'] = radar_n2
    apr['W'] = radar_n3
    apr['DFR_1'] = radar_n - radar_n2 #Ku - Ka
    
    if flag == 0:
        apr['DFR_2'] = radar_n2 - radar_n3 #Ka - W
        apr['DFR_3'] = radar_n - radar_n3 #Ku - W
        apr['info'] = 'The shape of these arrays are: Radar[Vertical gates,Time/DistanceForward]'
    else:
        apr['DFR_2'] = np.array([]) #Ka - W
        apr['DFR_3'] = np.array([]) #Ku - W
        apr['info'] = 'The shape of these arrays are: Radar[Vertical gates,Time/DistanceForward], Note No W band avail'
        
    apr['ldr'] = radar_n4
    apr['vel'] = vel_n
    apr['lon'] = lon
    apr['lat'] = lat
    apr['alt_gate'] = alt3d
    apr['alt_plane'] = plane
    apr['surface'] = isurf 
    apr['time']= time
    apr['timedates']= time_dates
    apr['lon_gate'] = lon3d
    apr['lat_gate'] = lat3d
    
    fileheader = hdf.select('fileheader')
    roll = hdf.select('roll').get()
    pitch = hdf.select('pitch').get()
    drift = hdf.select('drift').get()
    
    apr['fileheader'] = fileheader
    apr['ngates'] = ngates
    apr['roll'] = roll
    apr['pitch'] = pitch
    apr['drift'] = drift
    
    _range = np.arange(15,550*30,30)
    _range = np.asarray(_range,float)
    ind = np.where(_range >= plane.mean())
    _range[ind] = np.nan
    apr['range'] = _range
    
    return apr

def R_ob(filename):
    
    """ 
    ===========
    This function converts APR3 data to a Pyart Radar Object. This is intented so that one could use the
    awot.util.matcher.RadarMatch code to co-located microphysical measurements to the radar gate. 
    ===========
    """
    
    data = apr3read(filename)
    time_d = data['timedates']
    lon_gate = data['lon_gate']
    lat_gate = data['lat_gate']
    alt_gate = data['alt_gate']
    Z_Ku = data['Ku']
    Z_Ka = data['Ka']
    Z_W = data['W']
    Z_DFR1 = data['DFR_1']
    Z_DFR2 = data['DFR_2']
    Z_DFR3 = data['DFR_3']
    plane = data['alt_plane']
    rrange = data['range']
    lon_ = data['lon']
    lat_ = data['lat']

    time = {}
    pitch = {}
    roll = {}
    drift = {}
    fields = {}
    Ku = {}
    Ka = {} 
    W = {}
    DFR1 = {}
    DFR2 = {}
    DFR3 = {}
    _range = {}
    metadata = {}
    longitude = {}
    latitude = {}
    altitude = {}
    sweep_number = {}
    sweep_mode = {}
    fixed_angle = {}
    sweep_start_ray_index = {}
    sweep_end_ray_index = {}
    rays_per_sweep = {}
    azimuth = {} 
    elevation = {}
    gate_altitude = {}
"""
=================

A few tools to view and read APR3 data

Developed by Randy J. Chase (Feb 2017) at The University of Illinois, Urbana-Champaign 

=================
"""
from pyhdf.SD import SD, SDC
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib import dates
import pyresample as pr
from pyresample import geometry, data_reduce
from pylab import rc
from pyart.core.radar import Radar
import scipy.signal as scisig
import netCDF4
import metpy.plots as mp
from metpy.calc import get_wind_components


##plot attributes 
rc('axes', linewidth=2.5)

def apr3plot(filename,scan,band,fontsize=14,fontsize2=12,savefig=False,cmap = 'seismic',figsize=[10,5],mask=False):
    """
    ===========
    
    This function is designed to produce high quality quick plots of the apr3 radar data. 
    
    ===========
    
    filename = filename of the apr3 file, str
    scan = the scan within the desired data (12 is nadir), int
    band = the wavelength of radar you would like. Options: Ku, Ka, W, ldr, vel., str
    fontsize = fontsize for labels of the plot, int
    fontsize2 = fontsize for tick labels of the plot, int
    savefig = True will save figure, False will not, Bool
    cmap = color map for plot, str 
    figsize = size of the plot size, vector of 2 nums [length,width]
    
    """
    fig,ax = plt.subplots(1,1,figsize = figsize)
    
    if band == 'vel':
        
        colorbarlabel = 'Velocity, $[m s^{-1}]$'
        vmin = -10 
        vmax = 10
        
    elif band == 'DFR_1':
        
        colorbarlabel = 'DFR_Ku-Ka, $[dB]$'
        vmin = -10 
        vmax = 10

    elif band == 'DFR_2':
        
        colorbarlabel = 'DFR_Ka-W, $[dB]$'
        vmin = -20 
        vmax = 20
        
    elif band == 'DFR_3':
        
        colorbarlabel = 'DFR_Ku-W, $[dB]$'
        vmin = -20 
        vmax = 20
    elif band == 'W' :
        
        colorbarlabel = 'Reflectivity, $[dBZ]$'
        vmin = -20 
        vmax = 20
        
    else:
        
        colorbarlabel = 'Reflectivity, $[dBZ]$'
        vmin = 0 
        vmax = 40
        
    if mask:
        color_p = 'k'
    else:
        color_p = 'w'
        
    apr = apr3read(filename)
    
    if (apr['W'].size==0 and band=='W') or (apr['W'].size==0 and band=='DFR_2') or (apr['W'].size==0 and band=='DFR_3'):
        print()
    else:
        time_dates = apr['timedates'][scan,:]
        alt = apr['alt_gate'][:,scan,:]
        plane = apr['alt_plane'][scan,:]
        radar = apr[band][:,scan,:]
        surface = apr['surface'][scan,:]
        pm = ax.pcolormesh(time_dates,alt/1000.,radar,vmin=vmin,vmax=vmax,cmap='seismic')
        ax.fill_between(time_dates,alt[surface,0]/1000.,color='k',edgecolor='w')
        ax.plot(time_dates,plane/1000.,'-.',lw=3,color='w')
        ax.set_ylim([0,max(plane/1000.)+0.1])
        hfmt = dates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(hfmt)
        ax.set_ylim([0,max(plane/1000.)+0.1])
        ax.set_ylabel('Altitude, [km]',fontsize=fontsize)
        ax.set_xlabel('Time, [UTC]',fontsize=fontsize)
        ax.set_title(time_dates[0],fontsize=fontsize+2)
        ax.tick_params(axis='both',direction='in',labelsize=fontsize2,width=2,length=5,color='k')
        cbar = plt.colorbar(pm,aspect = 10)
        cbar.set_label(colorbarlabel,fontsize=fontsize)
        cax = cbar.ax
        cax.tick_params(labelsize=fontsize2)

        if savefig:
            print('Save file is: ' + str(time_dates[0])+'_'+band+'.png')
            plt.savefig(str(time_dates[0])+'_'+band+'.png',dpi=300)
        plt.show()
    return

def apr3flighttrack(filename):
    
    """
    ===========
    
    This is for creating a quick map of the flight for spatial reference (currently just for OLYMPEX domain)
    
    ===========
    
    filename = filename of the apr3 file
    """
        
    fig,ax = plt.subplots(1,1,figsize = [10,5])
    apr = apr3read(filename)
    lat = apr['lat'][12,:]
    lon = apr['lon'][12,:]
    area_def = pr.geometry.AreaDefinition('areaD', 'IPHEx', 'areaD',
                                      {'a': '6378144.0', 'b': '6356759.0',
                                       'lat_0': '47.6', 'lat_ts': '47.6','lon_0': '-124.5', 'proj': 'stere'},
                                      400, 400,
                                      [-400000., -400000.,
                                       400000., 400000.])
    bmap = pr.plot.area_def2basemap(area_def,resolution='l')
    bmap.drawcoastlines(linewidth=2)
    bmap.drawstates(linewidth=2)
    bmap.drawcountries(linewidth=2)
    parallels = np.arange(-90.,90,2)
    bmap.drawparallels(parallels,labels=[1,0,0,0],fontsize=12)
    meridians = np.arange(180.,360.,2)
    bmap.drawmeridians(meridians,labels=[0,0,0,1],fontsize=12)
    bmap.drawmapboundary(fill_color='aqua')
    bmap.fillcontinents(lake_color='aqua')
    x,y = bmap(lon,lat)
    plt.plot(x,y,'k--',lw=3)
    plt.show()

def apr3read(filename):
    
    """
    ===========
    
    This is for reading in apr3 hdf files from OLYMPEX and return them all in one dictionary
    
    ===========
    
    filename = filename of the apr3 file
    """
    
    apr = {}
    flag = 0
    ##Radar varibles in hdf file found by hdf.datasets
    radar_freq = 'zhh14' #Ku
    radar_freq2 = 'zhh35' #Ka
    radar_freq3 = 'zvv95' #W
    radar_freq4 = 'ldr14' #LDR
    vel_str = 'vel14' #Doppler
    ##

    hdf = SD(filename, SDC.READ)
    
    listofkeys = hdf.datasets().keys()
    if 'zvv95' in listofkeys:
        radar3 = hdf.select(radar_freq3)
        radar_n3 = radar3.get()
        radar_n3 = radar_n3/100.
    else:
        radar_n3 = np.array([])
        flag = 1
        print('No W band')
    
    
    alt = hdf.select('alt3D')
    lat = hdf.select('lat')
    lon = hdf.select('lon')
    time = hdf.select('scantime').get()
    surf = hdf.select('surface_index').get()
    isurf = hdf.select('isurf').get()
    plane = hdf.select('alt_nav').get()
    radar = hdf.select(radar_freq)
    radar2 = hdf.select(radar_freq2)
    radar4 = hdf.select(radar_freq4)
    vel = hdf.select(vel_str)
    lon3d = hdf.select('lon3D')
    lat3d = hdf.select('lat3D')
    alt3d = hdf.select('alt3D')
    lat3d_scale = hdf.select('lat3D_scale').get()[0][0]
    lon3d_scale = hdf.select('lon3D_scale').get()[0][0]
    alt3d_scale = hdf.select('alt3D_scale').get()[0][0]
    lat3d_offset = hdf.select('lat3D_offset').get()[0][0]
    lon3d_offset = hdf.select('lon3D_offset').get()[0][0]
    alt3d_offset = hdf.select('alt3D_offset').get()[0][0]
 
    alt = alt.get()
    ngates = alt.shape[0]
    #alt = alt[:,scan,:]
    lat = lat.get()
    #lat = lat[scan,:]
    lon = lon.get()
    #lon = lon[scan,:]
    
    lat3d = lat3d.get()
    lat3d = (lat3d/lat3d_scale) + lat3d_offset
    lon3d = lon3d.get()
    lon3d = (lon3d/lon3d_scale) + lon3d_offset
    alt3d = alt3d.get()
    alt3d = (alt3d/alt3d_scale) + alt3d_offset
    
    #time = time[scan,:]
    #surf = surf[scan,:]
    #isurf = isurf[scan,:]
    #plane = plane[scan,:]
    radar_n = radar.get()
    radar_n = radar_n/100.
    radar_n2 = radar2.get()
    radar_n2 = radar_n2/100.

    radar_n4 = radar4.get()
    radar_n4 = radar_n4/100.
    vel_n = vel.get()
    vel_n = vel_n/100.
    
    #Quality control
    radar_n[radar_n <= -99] = np.nan
    radar_n2[radar_n2 <= -99] = np.nan
    radar_n4[radar_n4 <= -99] = np.nan
    
    ##convert time to datetimes
    time_dates = np.empty(time.shape,dtype=object)
    for i in np.arange(0,time.shape[0]):
        for j in np.arange(0,time.shape[1]):
            tmp = datetime.datetime.utcfromtimestamp(time[i,j])
            time_dates[i,j] = tmp
            
    #Create a time at each gate (assuming it is the same down each ray, there is a better way to do this)      
    time_gate = np.empty(lat3d.shape,dtype=object)
    for k in np.arange(0,550):
        for i in np.arange(0,time_dates.shape[0]):
            for j in np.arange(0,time_dates.shape[1]):
                time_gate[k,i,j] = time_dates[i,j]        

    apr['Ku'] = radar_n
    apr['Ka'] = radar_n2
    apr['W'] = radar_n3
    apr['DFR_1'] = radar_n - radar_n2 #Ku - Ka
    
    if flag == 0:
        apr['DFR_2'] = radar_n2 - radar_n3 #Ka - W
        apr['DFR_3'] = radar_n - radar_n3 #Ku - W
        apr['info'] = 'The shape of these arrays are: Radar[Vertical gates,Time/DistanceForward]'
    else:
        apr['DFR_2'] = np.array([]) #Ka - W
        apr['DFR_3'] = np.array([]) #Ku - W
        apr['info'] = 'The shape of these arrays are: Radar[Vertical gates,Time/DistanceForward], Note No W band avail'
        
    apr['ldr'] = radar_n4
    apr['vel'] = vel_n
    apr['lon'] = lon
    apr['lat'] = lat
    apr['alt_gate'] = alt3d
    apr['alt_plane'] = plane
    apr['surface'] = isurf 
    apr['time']= time
    apr['timedates']= time_dates
    apr['time_gate'] = time_gate
    apr['lon_gate'] = lon3d
    apr['lat_gate'] = lat3d
    
    fileheader = hdf.select('fileheader')
    roll = hdf.select('roll').get()
    pitch = hdf.select('pitch').get()
    drift = hdf.select('drift').get()
    
    apr['fileheader'] = fileheader
    apr['ngates'] = ngates
    apr['roll'] = roll
    apr['pitch'] = pitch
    apr['drift'] = drift
    
    _range = np.arange(15,550*30,30)
    _range = np.asarray(_range,float)
    ind = np.where(_range >= plane.mean())
    _range[ind] = np.nan
    apr['range'] = _range
    
    return apr

def R_ob(filename):
    
    """ 
    ===========
    This function converts APR3 data to a Pyart Radar Object. This is intented so that one could use the
    awot.util.matcher.RadarMatch code to co-located microphysical measurements to the radar gate. 
    ===========
    """
    
    data = apr3read(filename)
    time_d = data['timedates']
    lon_gate = data['lon_gate']
    lat_gate = data['lat_gate']
    alt_gate = data['alt_gate']
    Z_Ku = data['Ku']
    Z_Ka = data['Ka']
    Z_W = data['W']
    Z_DFR1 = data['DFR_1']
    Z_DFR2 = data['DFR_2']
    Z_DFR3 = data['DFR_3']
    plane = data['alt_plane']
    rrange = data['range']
    lon_ = data['lon']
    lat_ = data['lat']

    time = {}
    pitch = {}
    roll = {}
    drift = {}
    fields = {}
    Ku = {}
    Ka = {} 
    W = {}
    DFR1 = {}
    DFR2 = {}
    DFR3 = {}
    _range = {}
    metadata = {}
    longitude = {}
    latitude = {}
    altitude = {}
    sweep_number = {}
    sweep_mode = {}
    fixed_angle = {}
    sweep_start_ray_index = {}
    sweep_end_ray_index = {}
    rays_per_sweep = {}
    azimuth = {} 
    elevation = {}
    gate_altitude = {}
    gate_longitude = {}
    gate_latitude = {}

    metadata['info'] = 'This is radar data from the APR3 insturment aboard the NASA DC-8 during the OLYMPEX Field Campain. This data contains matched Ku, Ka and W band radar data'
    projection =  'pyart_aeqd'
    Ku['data'] = Z_Ku
    Ka['data'] = Z_Ka
    W ['data'] = Z_W
    DFR1['data'] = Z_DFR1
    DFR2['data'] = Z_DFR2
    DFR3['data'] = Z_DFR3
    _range['data'] = rrange
    fields['Ku'] = Ku
    fields['Ka'] = Ka
    fields['W'] = W
    fields['DFR1'] = DFR1
    fields['DFR2'] = DFR2
    fields['DFR3'] = DFR3
    
    time['data'] = time_d
    time['units'] = 'EPOCH_UNITS'
    ##
    #
    #THIS IS WHERE THE ISSUE LIES.....plane radar is moving...ground radar is not...
    #Pyart can deal? probs not. 
    longitude['data'] = lon_
    latitude['data'] = lat_
    altitude['data'] = plane
    #
    #
    ##
    gate_altitude['data'] = alt_gate
    gate_longitude['data'] =lon_gate
    gate_latitude['data'] = lat_gate
    sweep_number['data'] = np.arange(0,24,1)
    sweep_mode['data'] = [np.nan]
    fixed_angle['data'] = [np.nan]
    pitch['data'] = data['pitch']
    roll['data'] = data['roll']
    drift['data'] = data['drift']
    scan_type = 'apr3 scan'
    sweep_start_ray_index['data'] = [0]
    sweep_end_ray_index['data'] = [23]
    rays_per_sweep['data'] = [time_d.shape[1]]
    azimuth['data'] = [np.nan]
    elevation['data'] = [np.nan]
    ngates = int(_range['data'].shape[0])
    nrays = int(time_d.shape[1])
    nsweeps = int(time_d.shape[0])
    APR = Radar(time, _range, fields, metadata, scan_type, latitude, longitude, altitude, sweep_number,
                sweep_mode, fixed_angle, sweep_start_ray_index, sweep_end_ray_index, azimuth, elevation,
                altitude, target_scan_rate=None, rays_are_indexed=None, ray_angle_res=None,
                scan_rate=None, antenna_transition=None, instrument_parameters=None,
                radar_calibration=None, rotation=None, tilt=None, roll=roll, drift=drift,
                heading=None, pitch=pitch, georefs_applied=None)
    
    APR.gate_altitude.update(gate_altitude)
    APR.gate_longitude.update(gate_longitude)
    APR.gate_latitude.update(gate_latitude)
    
    return APR

def HB(Z,T_profile):
    
    """
    ================
    
    A function to implement the HB method of attenuation correction
    
    ================
    """
    T_ind = np.where(T < -20)
    
    
    #Table of alpha from Seto et al. (2013)
    a_A_s = 0.3124
    a_A_c = 0.4814
    a_B_s = 1.2651
    a_B_s2 = 3.1409
    a_C_s = 5.0167
    a_C_s2 = 4.0639
    a_D_s = 3.111
    a_D_c = 4.2864
    #
    #Table of beta from Seto et al. (2013)
    b_s = 0.78069
    b_c = 0.75889
    #
    return

def find_node(date,temperature):
    
    """ 
    ==============
    
    A function to find the nodes from Seto et al. (2013) using the sounding and BB
    
    ==============
    """
    filename = '/data/gpm/a/shared/snesbitt/olympex/sounding/kuil/OLYMPEX_upaL4.0_kuil.nc'
    data = netCDF4.Dataset(filename)
    data2 = data.variables
    time = data2['launch_time']
    T = data2['T']
    a = data2['alt']
    t = netCDF4.date2num(date,time.units)
    ind = find_nearest(time[:],t)
    print('Sounding time & date: ' + str(netCDF4.num2date(time[ind],time.units)))
    
    T1 = T[ind,:] 
    a1 = a[ind,:]
    
    ind = find_nearest(T1,temperature)
    alt = a1[ind]
    
    return alt

def findBB(filename,Envi=False):
    
    """
    ==============
    
    A function to return the bright band location (if one) at each ray. This uses the idea that the brightband with is typically 
    + or - 0.5 km from the peak in the reflectivity (at nadir)
    
    ==============
    """
    data = apr3read(filename)
    alt = data['alt_gate']
    Ku = data['Ku']
    Ka = data['Ka']
    surface = data['surface'][12,:]
    date = data['timedates'][12,0]
    z = find_node(date,0)
    z_pad = z + 500
    z_pad2 = z - 500
    
    if Envi:
        BBarray = np.ones(Ku.shape[2])*z
        BBtoparray = BBarray + 500
        BBbottomarray = BBarray - 500
    else:

        BBtoparray = np.array([])
        BBbottomarray = np.array([])
        BBarray = np.array([])

        for i in np.arange(0,Ku.shape[2]):

            surface_2 = surface[i]
            #QC the data 
            ind = np.where(alt[:,12,i] < 0)
            ind2 = np.where(Ku[:,12,i] > 40)
            ind3 = np.where(Ku[:,12,i] < 0)
            ind4 = np.arange(surface_2,550,1)

            Ku[ind,12,i]  = np.nan
            Ku[ind2,12,i] = np.nan
            Ku[ind3,12,i] = np.nan
            Ku[ind4,12,i] = np.nan

            ind = scisig.argrelextrema(Ku[:,12,i],np.greater)

            #plt.plot(Ku[:,12,i],alt[:,12,i],'k.',ms=1)

            if ind[0].shape[0] == 0 or np.nanmean(Ku[:,12,i]) <= 14:
                BBtoparray = np.append(BBtoparray,np.nan)
                BBbottomarray = np.append(BBbottomarray,np.nan)
                BBarray = np.append(BBarray,np.nan)
                print(i)
                continue

            K = Ku[ind,12,i][0]
            a = alt[ind,12,i][0]
            dK = np.diff(K)

            indind = scisig.argrelextrema(K,np.greater)
            K2 = K[indind]
            a2 = a[indind]

            ind4 = np.where(K == K.max())

            if K.shape[0] > 1:

                ind5 = np.where(dK == dK.max())
                ind5 = ind5[0][0]

                if a[ind5+1] > z_pad or a[ind5+1] < z_pad2:
                    ind6 = np.where(K2 == K2.max())
                    BB = a2[ind6]
                    BBtop = a2[ind6] + 500
                    BBbottom = a2[ind6] - 500
                else:
                    BB = a[ind5+1]
                    BBtop = a[ind5+1] + 500
                    BBbottom = a[ind5+1] - 500

            else:
                BB = a[ind4]
                BBtop = a[ind4] + 500
                BBbottom = a[ind4] - 500



            BBarray = np.append(BBarray,BB)
            BBtoparray = np.append(BBtoparray,BBtop)
            BBbottomarray = np.append(BBbottomarray,BBbottom)

    BB = {}
    BB['top'] = BBtoparray
    BB['bottom'] = BBbottomarray
    BB['bb'] = BBarray
    return BB
 
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def apr3plot2(filename,scan,band,fontsize=14,fontsize2=12,savefig=False,cmap = 'seismic',figsize=[10,5],mask=False):
    """
    ===========
    
    This function is designed to produce high quality quick plots of the apr3 radar data. 
    
    ===========
    
    filename = filename of the apr3 file, str
    scan = the scan within the desired data (12 is nadir), int
    band = the wavelength of radar you would like. Options: Ku, Ka, W, ldr, vel., str
    fontsize = fontsize for labels of the plot, int
    fontsize2 = fontsize for tick labels of the plot, int
    savefig = True will save figure, False will not, Bool
    cmap = color map for plot, str 
    figsize = size of the plot size, vector of 2 nums [length,width]
    
    """
    fig,ax = plt.subplots(1,1,figsize = figsize)
    
    if band == 'vel':
        
        colorbarlabel = 'Velocity, $[m s^{-1}]$'
        vmin = -10 
        vmax = 10
        
    elif band == 'DFR_1':
        
        colorbarlabel = 'DFR_Ku-Ka, $[dB]$'
        vmin = -10 
        vmax = 10

    elif band == 'DFR_2':
        
        colorbarlabel = 'DFR_Ka-W, $[dB]$'
        vmin = -20 
        vmax = 20
        
    elif band == 'DFR_3':
        
        colorbarlabel = 'DFR_Ku-W, $[dB]$'
        vmin = -20 
        vmax = 20
    elif band == 'W' :
        
        colorbarlabel = 'Reflectivity, $[dBZ]$'
        vmin = -20 
        vmax = 20
        
    else:
        
        colorbarlabel = 'Reflectivity, $[dBZ]$'
        vmin = 0 
        vmax = 40
        
    if mask:
        color_p = 'k'
    else:
        color_p = 'w'
        
    apr = apr3read(filename)
    

   
    if (apr['W'].size==0 and band=='W') or (apr['W'].size==0 and band=='DFR_2') or (apr['W'].size==0 and band=='DFR_3'):
        print()
    else:
        #Pick sounding site
        #filename2 = '/data/gpm/a/shared/snesbitt/olympex/sounding/kuil/OLYMPEX_upaL4.0_kuil.nc'
        filename2 = '/data/gpm/a/shared/snesbitt/olympex/sounding/knpol/OLYMPEX_upaL4.0_npol.nc'
        data = netCDF4.Dataset(filename2)
        data2 = data.variables

    
        time_dates = apr['timedates'][scan,:]
        
        
        
        #Search for closest T sounding
        time = data2['launch_time']
        t = netCDF4.date2num(time_dates[0],time.units)
        ind = find_nearest(time[:],t)
        
        T = data2['T'][ind,:]
        a = data2['alt'][ind,:]/1000.
        
        #Hallett-Mossop
        ind2 = find_nearest(T,-3)
        a2 = a[ind2]
        ind3 = find_nearest(T,-8)
        a3 = a[ind3]
        #
        #Dendritic growth zone
        ind4 = find_nearest(T,-12)
        a4 = a[ind4]
        ind5 = find_nearest(T,-18)
        a5 = a[ind5]
        #
        
        alt = apr['alt_gate'][:,scan,:]
        plane = apr['alt_plane'][scan,:]
        radar = apr[band][:,scan,:]
        surface = apr['surface'][scan,:]
        pm = ax.pcolormesh(time_dates,alt/1000.,radar,vmin=vmin,vmax=vmax,cmap='seismic')
        ax.fill_between(time_dates,alt[surface,0]/1000.,color='k',edgecolor='w')
        ax.plot(time_dates,plane/1000.,'-.',lw=3,color='w')
     
        ax.fill_between(time_dates,a2,a3,color='y',alpha=0.5)
        ax.fill_between(time_dates,a4,a5,color=[.5,.5,.5],alpha=0.5)
            
        ax.set_ylim([0,max(plane/1000.)+0.1])
        hfmt = dates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(hfmt)
        ax.set_ylim([0,max(plane/1000.)+0.1])
        
        
        
        ax.set_ylabel('Altitude, [km]',fontsize=fontsize)
        ax.set_xlabel('Time, [UTC]',fontsize=fontsize)
        ax.set_title(time_dates[0],fontsize=fontsize+2)
        ax.tick_params(axis='both',direction='in',labelsize=fontsize2,width=2,length=5,color='k')
        cbar = plt.colorbar(pm,aspect = 10)
        cbar.set_label(colorbarlabel,fontsize=fontsize)
        cax = cbar.ax
        cax.tick_params(labelsize=fontsize2)

        if savefig:
            print('Save file is: ' + str(time_dates[0])+'_'+band+'.png')
            plt.savefig(str(time_dates[0])+'_'+band+'.png',dpi=300)
        plt.show()
    return
    
    
def apr3plot3(filename,scan,band,fontsize=14,fontsize2=12,savefig=False,cmap = 'seismic',figsize=[10,5],mask=False,site='npol'):
    """
    ===========
    
    This function is designed to produce high quality quick plots of the apr3 radar data. 
    
    ===========
    
    filename = filename of the apr3 file, str
    scan = the scan within the desired data (12 is nadir), int
    band = the wavelength of radar you would like. Options: Ku, Ka, W, ldr, vel., str
    fontsize = fontsize for labels of the plot, int
    fontsize2 = fontsize for tick labels of the plot, int
    savefig = True will save figure, False will not, Bool
    cmap = color map for plot, str 
    figsize = size of the plot size, vector of 2 nums [length,width]
    
    """
    f1 = plt.figure(figsize=(10,10))
    ax2 = plt.subplot2grid((2,2), (0,0), colspan=2)
    ax4 = plt.subplot2grid((2,2), (1, 0))
    ax5 = plt.subplot2grid((2,2), (1, 1))
    
    if band == 'vel':
        
        colorbarlabel = 'Velocity, $[m s^{-1}]$'
        vmin = -10 
        vmax = 10
        
    elif band == 'DFR_1':
        
        colorbarlabel = 'DFR_Ku-Ka, $[dB]$'
        vmin = -10 
        vmax = 10

    elif band == 'DFR_2':
        
        colorbarlabel = 'DFR_Ka-W, $[dB]$'
        vmin = -20 
        vmax = 20
        
    elif band == 'DFR_3':
        
        colorbarlabel = 'DFR_Ku-W, $[dB]$'
        vmin = -20 
        vmax = 20
    elif band == 'W' :
        
        colorbarlabel = 'Reflectivity, $[dBZ]$'
        vmin = -20 
        vmax = 20
        
    else:
        
        colorbarlabel = 'Reflectivity, $[dBZ]$'
        vmin = 0 
        vmax = 40
        
    if mask:
        color_p = 'k'
    else:
        color_p = 'w'
        
    apr = apr3read(filename)
    

   
    if (apr['W'].size==0 and band=='W') or (apr['W'].size==0 and band=='DFR_2') or (apr['W'].size==0 and band=='DFR_3'):
        print()
    else:
        #Pick sounding site
        if site=='npol':
            filename2 = '/data/gpm/a/shared/snesbitt/olympex/sounding/knpol/OLYMPEX_upaL4.0_npol.nc'
        else:
            filename2 = '/data/gpm/a/shared/snesbitt/olympex/sounding/kuil/OLYMPEX_upaL4.0_kuil.nc'
        
        data = netCDF4.Dataset(filename2)
        data2 = data.variables

    
        time_dates = apr['timedates'][scan,:]
        
        
        
        #Search for closest T sounding
        time = data2['launch_time']
        t = netCDF4.date2num(time_dates[0],time.units)
        ind = find_nearest(time[:],t)
        #
        #Load in data from sounding
        T = data2['T'][ind,:]
        Td = data2['Td'][ind,:]
        time_title = time[ind]
        time_title = netCDF4.num2date(time_title,time.units)
        p = data2['p'][ind,:]
        a = data2['alt'][ind,:]/1000.
        spd = data['wspd'][ind,:]
        direc = data['wdir'][ind,:]
        u, v = get_wind_components(spd, np.deg2rad(direc))
        #
        
        #Hallett-Mossop
        ind2 = find_nearest(T,-3)
        a2 = a[ind2]
        ind3 = find_nearest(T,-8)
        a3 = a[ind3]
        #
        #Dendritic growth zone
        ind4 = find_nearest(T,-12)
        a4 = a[ind4]
        ind5 = find_nearest(T,-18)
        a5 = a[ind5]
        #
        
        ##Top subplot
        ax = ax2
        
        alt = apr['alt_gate'][:,scan,:]
        plane = apr['alt_plane'][scan,:]
        radar = apr[band][:,scan,:]
        surface = apr['surface'][scan,:]
        pm = ax.pcolormesh(time_dates,alt/1000.,radar,vmin=vmin,vmax=vmax,cmap='seismic')
        ax.fill_between(time_dates,alt[surface,0]/1000.,color='k',edgecolor='w')
        ax.plot(time_dates,plane/1000.,'-.',lw=3,color='w')
     
        ax.fill_between(time_dates,a2,a3,color='y',alpha=0.5)
        ax.fill_between(time_dates,a4,a5,color=[.5,.5,.5],alpha=0.5)
            
        ax.set_ylim([0,max(plane/1000.)+0.1])
        hfmt = dates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(hfmt)
        ax.set_ylim([0,max(plane/1000.)+0.1])
        
        
        
        ax.set_ylabel('Altitude, [km]',fontsize=fontsize)
        ax.set_xlabel('Time, [UTC]',fontsize=fontsize)
        ax.set_title(time_dates[0],fontsize=fontsize+2)
        ax.tick_params(axis='both',direction='in',labelsize=fontsize2,width=2,length=5,color='k')
        cbar = plt.colorbar(pm,aspect = 10,ax=ax)
        cbar.set_label(colorbarlabel,fontsize=fontsize)
        cax = cbar.ax
        cax.tick_params(labelsize=fontsize2)
        
        
        #Bottom Left subplot
        ax = ax4
        apr = apr3read(filename)
        lat3d = apr['lat_gate']
        lon3d = apr['lon_gate']
        alt3d = apr['alt_gate']
        radar_n = apr['Ku']

        lon_s = np.empty(alt3d.shape[1:])
        lat_s = np.empty(alt3d.shape[1:])
        swath = np.empty(alt3d.shape[1:])
        for i in np.arange(0,alt3d.shape[2]):
            for j in np.arange(0,alt3d.shape[1]):
                ind = np.where(alt3d[:,j,i]/1000. > 3)
                ind2 = np.where(alt3d[:,j,i]/1000. < 3.1)
                ind3 = np.intersect1d(ind,ind2)
                ind3= ind3[0]
                l1 = lat3d[ind3,j,i]
                l2 = lon3d[ind3,j,i]
                k1 = radar_n[ind3,j,i]
                lon_s[j,i] = l2
                lat_s[j,i] = l1
                swath[j,i] = k1

        area_def = pr.geometry.AreaDefinition('areaD', 'IPHEx', 'areaD',
                                          {'a': '6378144.0', 'b': '6356759.0',
                                           'lat_0': '47.7998', 'lat_ts': '47.7998','lon_0': '-123.7066', 'proj': 'stere'},
                                          400, 400,
                                          [-70000., -70000.,
                                           70000., 70000.])
        bmap = pr.plot.area_def2basemap(area_def,resolution='l',ax=ax)
        bmap.drawcoastlines(linewidth=2)
        bmap.drawstates(linewidth=2)
        bmap.drawcountries(linewidth=2)
        parallels = np.arange(-90.,90,4)
        bmap.drawparallels(parallels,labels=[1,0,0,0],fontsize=12)
        meridians = np.arange(180.,360.,4)
        bmap.drawmeridians(meridians,labels=[0,0,0,1],fontsize=12)
        bmap.drawmapboundary(fill_color='aqua')
        bmap.fillcontinents(lake_color='aqua')

        x,y = bmap(lon_s,lat_s)
        swath[np.where(swath < 0)] = np.nan
        pm1 = bmap.pcolormesh(x,y,swath,vmin=0,vmax=40,zorder=11,cmap='seismic')
        cbar1 = plt.colorbar(pm1,label='$Z_m, [dBZ]$')
        
        #
        
        #Bottom Right
        ax = ax5
        ax.plot(T,a,'r-')
        ax.plot(Td,a,'g-')
        ax.invert_yaxis()
        ax.set_xlabel('Temperature, $[^{o}C]$',fontsize=fontsize)
        ax.set_ylabel('Altitude, $[km]$',fontsize=fontsize)
        ax.set_ylim(0, 8)
        ax.set_xlim(-40, 10)
        ax.tick_params(axis='both',direction='in',labelsize=fontsize2,width=2,length=5,color='k')
        ax.set_title(str(time_title),fontsize=fontsize)
        ax.grid()
        
        plt.tight_layout()
        
        if savefig:
            print('Save file is: ' + str(time_dates[0])+'_'+band+'.png')
            plt.savefig(str(time_dates[0])+'_'+band+'.png',dpi=300)
            
        plt.show()
    return

def apr3tocit(apr3filename,cit_awot_fl,sphere_size,query_k = 1,plotson=False):
    
    """
    =================
    
    This function finds either the closest gate or averages over a number of gates (query_k) nearest to 
    the citation aircraft in the radar volume of the APR3. It will return a dict of the original full length
    arrays and the matched arrays. 
    
    =================
    """

    cit_time = fl['time']['data']
    apr = apr3read(apr3filename)
    time_dates = apr['timedates'][:,:]
    
    if time_dates[12,:].shape[0] < 50:
        print('Limited radar gates in time')
        return
    
    fontsize=14
    #Varibles needed for the kdtree
    leafsize = 16
    query_eps = 0
    query_p=2
    query_distance_upper_bound = sphere_size
    query_n_jobs =1
    Barnes = True
    K_d = 1e3
    #


    #Pre-Determine arrays
    Ku_gate = np.array([])
    Ka_gate = np.array([])
    DFR_gate = np.array([])
    lon_c = np.array([])
    lat_c = np.array([])
    alt_c = np.array([])
    t_c = np.array([])
    lon_r = np.array([])
    lat_r = np.array([])
    alt_r = np.array([])
    t_r = np.array([])
    dis_r = np.array([])
    ind_r = np.array([])
    conc_hvps3 = np.array([])
    #

    
    #Set reference point (Currently Mount Olympus, Washington)
    lat_0 = 47.7998
    lon_0 = -123.7066
    #Set up map projection to calculate distances for this radar
    p = Proj(proj='laea', zone=10, ellps='WGS84',
             lat_0=lat_0,
             lon_0=lon_0)


    td = np.ravel(time_dates)
    datestart = td[0]
    dateend = td[td.shape[0]-1] 

    #Constrain Citation data to radar time
    ind = np.where(cit_time > datestart)
    ind2 = np.where(cit_time < dateend)
    ind3 = np.intersect1d(ind,ind2)
    #

    cit_time2 = fl['time']['data'][ind3]
    cit_lon = fl['longitude']['data'][ind3]
    cit_lat = fl['latitude']['data'][ind3]
    cit_alt = fl['altitude']['data'][ind3]
    bigins = fl['HVPS3_hor']['data'][ind3]
    
    #Print out number of potential points
    print(cit_time2.shape)
    #
    

    #Make 1-D arrays of radar spatial files
    apr_x = np.ravel(apr['lon_gate'][:,:,:])
    apr_y = np.ravel(apr['lat_gate'][:,:,:])
    apr_alt = np.ravel(apr['alt_gate'][:,:,:])
    apr_t = np.ravel(apr['time_gate'][:,:,:])
    #
    
    ##Make 1-D arrays of radar 
    apr_ku = np.ravel(apr['Ku'][:,:,:])
    apr_ka = np.ravel(apr['Ka'][:,:,:])
    apr_DFR = apr_ku - apr_ka
    ##
    
    #Use projection to get cartiesian distances
    apr_x2,apr_y2 = p(apr_x,apr_y)
    cit_x2,cit_y2 = p(cit_lon,cit_lat)
    #
    
    #Kdtree things
    kdt = cKDTree(zip(apr_x2, apr_y2, apr_alt), leafsize=leafsize)

    prdistance, prind1d = kdt.query(zip(cit_x2,cit_y2,cit_alt),k=query_k, eps=query_eps, p=query_p,
                            distance_upper_bound=query_distance_upper_bound,n_jobs=query_n_jobs)
    
    #
    
    
    #if query_k >1 means you are considering more than one gate and an average is needed
    
    if query_k > 1:
        
        #Issue with prind1d being the size of apr_ku...
        
        ind = np.where(prind1d == apr_ku.shape[0])
        if len(ind[0]) > 0 or len(ind[1]) > 0:
            print('gate was outside distance upper bound, eliminating those instances')
            #mask values outside search area
            prind1d[ind] = np.ma.masked
            prdistance[ind] = np.ma.masked
            
        #Barnes weighting     
        W_d_k = np.ma.array(np.exp(-1*prdistance**2./K_d**2.))
        W_d_k2 = np.ma.masked_where(np.ma.getmask(apr_ku[prind1d]), W_d_k.copy())
        w1 = np.ma.sum(W_d_k2 * 10. **(apr_ku[prind1d] / 10.),axis=1)
        w2 = np.ma.sum(W_d_k2, axis=1)
        ku_temp = 10. * np.ma.log10(w1/w2)
        
        W_d_k = np.ma.array(np.exp(-1*prdistance**2./K_d**2.))
        W_d_k2 = np.ma.masked_where(np.ma.getmask(apr_ka[prind1d]), W_d_k.copy())
        w1 = np.ma.sum(W_d_k2 * 10. **(apr_ka[prind1d] / 10.),axis=1)
        w2 = np.ma.sum(W_d_k2, axis=1)
        ka_temp = 10. * np.ma.log10(w1/w2)

        W_d_k2 = np.ma.masked_where(np.ma.getmask(apr_DFR[prind1d]), W_d_k.copy())
        w1 = np.ma.sum(W_d_k2 * 10. **((apr_DFR[prind1d]) / 10.),axis=1)
        w2 = np.ma.sum(W_d_k2, axis=1)
        dfr_temp = 10. * np.ma.log10(w1/w2)

        W_d_k = np.ma.array(np.exp(-1*prdistance**2./K_d**2.))
        W_d_k2 = np.ma.masked_where(np.ma.getmask(prdistance), W_d_k.copy())
        w1 = np.ma.sum(W_d_k2 * 10. **(prdistance/ 10.),axis=1)
        w2 = np.ma.sum(W_d_k2, axis=1)
        dis_temp = 10. * np.ma.log10(w1/w2)
    
        Ku_gate = ku_temp
        Ka_gate = ka_temp
        DFR_gate = dfr_temp

        #append current lat,lon and alt of the citation plane
        lat_c = np.append(lat_c,cit_lat)
        lon_c = np.append(lon_c,cit_lon)
        alt_c = np.append(alt_c,cit_alt)
        t_c = np.append(t_c,cit_time2)
        #

        #Use plane location for barnes averaged radar value
        lat_r = cit_lat
        lon_r = cit_lon
        alt_r = cit_alt
        t_r = cit_time2
        #
        dis_r = dis_temp
        ind_r = np.nan
        
        
        t_tiled = np.empty([t_c.shape[0],query_k],dtype=object)
        for i in np.arange(0,t_c.shape[0]):
            t_tiled[i,:] = t_c[i]
        diftime = apr_t[prind1d] - t_tiled
        diftime2 = np.empty(diftime.shape)
        for i in np.arange(0,diftime.shape[0]-1):
            for j in np.arange(0,diftime.shape[1]-1):
                diftime2[i,j] = diftime[i,j].total_seconds()
        
        W_d_k = np.ma.array(np.exp(-1*prdistance**2./K_d**2.))
        W_d_k2 = np.ma.masked_where(np.ma.getmask(diftime2), W_d_k.copy())
        w1 = np.ma.sum(W_d_k2 * 10. **(diftime2/ 10.),axis=1)
        w2 = np.ma.sum(W_d_k2, axis=1)
        dif_temp = 10. * np.ma.log10(w1/w2)
        
        dif_t = dif_temp
        
   
    else:
            
        #If gate outside sphere will need to remove flaged data == apr_ku.shape[0]
        ind = np.where(prind1d == apr_ku.shape[0])
        if len(ind[0]) > 0:
            print('gate was outside distance upper bound, eliminating those instances')
            #mask ind and distances that are outside the search area
            prind1d[ind] = np.ma.masked
            prdistance[ind] = np.ma.masked
               
        ku_temp = apr_ku[prind1d]
        ka_temp = apr_ka[prind1d]
        dfr_temp = ku_temp - ka_temp
        Ku_gate = np.append(Ku_gate,ku_temp)
        Ka_gate = np.append(Ka_gate,ka_temp)
        DFR_gate = np.append(DFR_gate,dfr_temp)
        #

        #append current lat,lon and alt of the citation plane
        lat_c = np.append(lat_c,cit_lat)
        lon_c = np.append(lon_c,cit_lon)
        alt_c = np.append(alt_c,cit_alt)
        t_c = np.append(t_c,cit_time2)
        conc_hvps3 = np.append(conc_hvps3,bigins)
        #

        #Get radar gate info and append it
        lat_r = np.append(lat_r,apr_y[prind1d])
        lon_r = np.append(lon_r,apr_x[prind1d])
        alt_r = np.append(alt_r,apr_alt[prind1d])
        t_r = np.append(t_r,apr_t[prind1d])
        dis_r = np.append(dis_r,prdistance)
        ind_r = np.append(ind_r,prind1d)

        dif_t = np.nan

    
    #Make lists
    matcher = {}
    Cit = {}
    APR = {}
    matched = {}
    kdtree = {}
    
    #Pack values in lists for export
    kdtree['prind1d'] = prind1d
    kdtree['prdistance'] = prdistance
    kdtree['query_k'] = query_k
    
    Cit['lat'] = cit_lat
    Cit['lon'] = cit_lon
    Cit['alt'] = cit_alt
    Cit['time'] = cit_time2
    APR['lat'] = apr_y
    APR['lon'] = apr_x
    APR['alt'] = apr_alt
    APR['Ku'] = apr_ku
    APR['Ka'] = apr_ka
    APR['DFR'] = apr_ku - apr_ka
    APR['time'] = apr_t

    matched['Ku'] = Ku_gate
    matched['Ka'] = Ka_gate
    matched['DFR'] = DFR_gate
    matched['lat_r'] = lat_r
    matched['lon_r'] = lon_r
    matched['alt_r'] = alt_r
    matched['lat_c'] = lat_c
    matched['lon_c'] = lon_c
    matched['alt_c'] = alt_c
    matched['time_r'] = t_r
    matched['time_c'] = t_c
    matched['dist'] = dis_r
    matched['dif_t'] = dif_t
    matched['array index'] = ind_r
    matched['conc_hvps3'] = conc_hvps3
    
    matcher['Cit'] = Cit
    matcher['APR'] = APR
    matcher['matched'] = matched
    matcher['kdtree'] = kdtree
    
    #Several plots to visualize data
    if plotson:
        fontsize=fontsize
        matched = matcher
        
        if query_k <= 1:
            diftime = matched['matched']['time_r'] - matched['matched']['time_c']
            diftime2 = np.array([])
            for i in np.arange(0,diftime.shape[0]):
                diftime2 = np.append(diftime2,diftime[i].total_seconds())
        else:
            diftime2= matched['matched']['dif_t']
            

        fig1,axes = plt.subplots(1,2,)
        
        #ax1 is the histogram of times
        ax1 = axes[0]
        ax1.hist(diftime2/60.,facecolor='b',edgecolor='k')
        ax1.set_xlabel('$t_{gate} - t_{Cit}, [min]$')
        ax1.set_ylabel('Number of gates')
        ax1.set_title(matched['matched']['time_r'][0])
        #ax2 is the histogram of distances
        ax2 = axes[1]
        distances = matched['matched']['dist']
        ax2.hist(distances,facecolor='r',edgecolor='k')
        ax2.set_xlabel('Distance, $[m]$')
        ax2.set_ylabel('Number of gates')
        ax2.set_title(matched['matched']['time_r'][0])

        plt.tight_layout()

        #Print some quick stats
        print(distances.shape[0],np.nanmean(diftime2)/60.,np.nanmean(distances))
        #
        
        fig = plt.figure()
        #ax3 is the swath plot to show radar and plane location
        ax3 = plt.gca()
        apr = apr3read(apr3filename)
        lat3d = apr['lat_gate']
        lon3d = apr['lon_gate']
        alt3d = apr['alt_gate']
        radar_n = apr['Ku']

        lon_s = np.empty(alt3d.shape[1:])
        lat_s = np.empty(alt3d.shape[1:])
        swath = np.empty(alt3d.shape[1:])
        for i in np.arange(0,alt3d.shape[2]):
            for j in np.arange(0,alt3d.shape[1]):
                ind = np.where(alt3d[:,j,i]/1000. > 2.3)
                ind2 = np.where(alt3d[:,j,i]/1000. < 2.6)
                ind3 = np.intersect1d(ind,ind2)
                ind3= ind3[0]
                l1 = lat3d[ind3,j,i]
                l2 = lon3d[ind3,j,i]
                k1 = radar_n[ind3,j,i]
                lon_s[j,i] = l2
                lat_s[j,i] = l1
                swath[j,i] = k1

        area_def = pr.geometry.AreaDefinition('areaD', 'IPHEx', 'areaD',
                                          {'a': '6378144.0', 'b': '6356759.0',
                                           'lat_0': '47.7998', 'lat_ts': '47.7998','lon_0': '-123.7066', 'proj': 'stere'},
                                          400, 400,
                                          [-70000., -70000.,
                                           70000., 70000.])
        bmap = pr.plot.area_def2basemap(area_def,resolution='l',ax=ax3)
        bmap.drawcoastlines(linewidth=2)
        bmap.drawstates(linewidth=2)
        bmap.drawcountries(linewidth=2)
        parallels = np.arange(-90.,90,4)
        bmap.drawparallels(parallels,labels=[1,0,0,0],fontsize=12)
        meridians = np.arange(180.,360.,4)
        bmap.drawmeridians(meridians,labels=[0,0,0,1],fontsize=12)
        bmap.drawmapboundary(fill_color='aqua')
        bmap.fillcontinents(lake_color='aqua')

        x,y = bmap(lon_s,lat_s)
        swath[np.where(swath < 0)] = np.nan
        pm1 = bmap.pcolormesh(x,y,swath,vmin=0,vmax=40,zorder=11,cmap='seismic')
        cbar1 = plt.colorbar(pm1,label='$Z_m, [dBZ]$')

        x2,y2 = bmap(matched['matched']['lon_c'],matched['matched']['lat_c'])
        pm2 = bmap.scatter(x2,y2,c=diftime2/60.,marker='o',zorder=12,cmap='PuOr',edgecolor=[],vmin=-10,vmax=10)
        cbar2 = plt.colorbar(pm2,label = '$\Delta{t}, [min]$')

        ax3.set_ylabel('Latitude',fontsize=fontsize,labelpad=20)
        ax3.set_xlabel('Longitude',fontsize=fontsize,labelpad=20)

        plt.tight_layout()
        plt.show()
        
        #Plot timeseries of barnes averaged or closest gate.
        plt.figure()
        plt.plot(matched['matched']['time_c'],matched['matched']['Ku'],'b',label='Ku')
        plt.plot(matched['matched']['time_c'],matched['matched']['Ka'],'r',label='Ka')
        plt.xlabel('Time')
        plt.ylabel('Z, [dBZ]')
        plt.legend()
        

    print('done')
    return matcher
