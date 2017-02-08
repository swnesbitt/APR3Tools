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

    ##Make time a 2-D matrix to match radar and alt shape. Also creates surface alts
    time_new = np.ones((alt.shape))
    surfalt = np.array([])
    for i in range(0,time.shape[0]):
        time_new[:,i] = time[i]
        surfalt = np.append(surfalt,alt[isurf[i],i])
    ##

    zero_i = np.where(surf==1)
    newzero = np.abs(surfalt[zero_i[0][0]])
    alt_shift = alt + newzero
    ind = np.where(alt_shift < 0)
    ##Mask radar below surface
    radar_n[ind] = np.inf
    radar_n2[ind] = np.inf
    
    if flag == 0:
        radar_n3[ind] = np.inf
        radar_n3 = np.ma.masked_invalid(radar_n3, copy=True)
    
    if mask:
        print('masking')
        ind = np.where(radar_n < vmin)
        radar_n[ind] = np.inf
        
        ind = np.where(radar_n2 < vmin)
        radar_n2[ind] = np.inf
        
        if flag == 0:
            ind = np.where(radar_n3 < vmin)
            radar_n3[ind] = np.inf
        
#    radar_n4[ind] = np.inf
    radar_n = np.ma.masked_invalid(radar_n, copy=True)
    radar_n2 = np.ma.masked_invalid(radar_n2, copy=True)
    radar_n4 = np.ma.masked_invalid(radar_n4, copy=True)

    ##convert time to datetimes
    time_dates = np.array([])
    for i in range(0,len(time)):
        tmp = datetime.datetime.utcfromtimestamp(time[i])
        time_dates = np.append(time_dates,tmp)
    time_new = np.ones((alt.shape),dtype=object)

    for i in range(0,time.shape[0]):
        time_new[:,i] = time_dates[i]

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
    apr['alt_gate'] = alt_shift
    apr['alt_plane'] = plane
    apr['surface'] = surfalt + newzero
    apr['time_gate']= time_new
    apr['time_dates'] = time_dates

    
    return apr
