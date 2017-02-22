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

