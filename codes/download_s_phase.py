'''
Search for available stations in the region and download available events for each station
Author: Utpal Kumar
BSL, UCB @ 2022-06-28
'''
from obspy.clients.fdsn import Client
import sys, os, shutil
from obspy.clients.fdsn.header import URL_MAPPINGS
import pandas as pd
from obspy import UTCDateTime as UTC
from obspy import read_inventory
from obspy.taup import TauPyModel
import glob
import time
import threading
import logging
import numpy as np
import pickle
from obspy.geodetics.base import gps2dist_azimuth

#Creating and Configuring Logger

Log_Format = "%(asctime)s - %(levelname)s - %(message)s"

logging.basicConfig(filename = "logfile_2.log",
                    filemode = "w",
                    format = Log_Format, 
                    level = logging.INFO)

logger = logging.getLogger()

class DownloadData:
    def __init__(self,
        clientlist,
        minlongitude,
        maxlongitude,
        minlatitude,
        maxlatitude,
        inventorytxtfile,
        catalogxmlloc,
        catalogtxtloc,
        channel = "BHE,BHN,BHZ"):   #    "BHZ,BHE,BHN,LHZ,LHE,LHN,VHE,VHN,VHZ"   # try SHZ instead for DBIC  
        self.minlongitude = minlongitude
        self.maxlongitude = maxlongitude
        self.minlatitude = minlatitude
        self.maxlatitude = maxlatitude
        self.client = clientlist
        self.inventorytxtfile = inventorytxtfile
        self.channel = channel
        self.inventoryfile = self.inventorytxtfile.split(".")[0]+ ".xml"
        self.method = 'S'
        self.catalogxmlloc = catalogxmlloc
        self.catalogtxtloc = catalogtxtloc
        self.out_inventorytxtfile = self.inventorytxtfile.split(".")[0]+'_combined'+'.txt'
        self.dl_rec_pkl = 'run_record.pkl'

    def get_stnxml(self,network='*', station="*", level = 'channel'):
        inventory = None
        logger.info("\n")
        logger.info('Retrieving station information...')
        ninvt=0
        while ninvt < len(self.client):
            try:
                client = Client(self.client[ninvt])
            except Exception as e:
                logger.exception("No FDSN services could be discovered for {}! Try again after some time.".format(self.client[ninvt]))
                # sys.exit()
            logger.info('-- from {}'.format(self.client[ninvt]))
            try:
                invt = client.get_stations(network=network, station=station, channel=self.channel, level=level,minlongitude=self.minlongitude, maxlongitude=self.maxlongitude,minlatitude=self.minlatitude, maxlatitude=self.maxlatitude)
                inventory = invt
                break
            except Exception as exception:
                logger.exception("---- No stations found for the given parameters for client {}".format(self.client[ninvt]))
            ninvt+=1
            # sys.exit()
        if len(self.client)>1:
            for cl in self.client[ninvt+1:]:
                logger.info('-- from {}'.format(cl))
                try:
                    client = Client(cl)
                    invt = client.get_stations(network=network, station=station, channel=self.channel, level=level,minlongitude=self.minlongitude, maxlongitude=self.maxlongitude,minlatitude=self.minlatitude, maxlatitude=self.maxlatitude)
                    inventory +=invt
                except Exception as exception:
                    logger.exception("---- FDSNNoDataException for {}".format(cl))
        # logger.info(inventory)
        if inventory is not None:
            inventory.write(self.inventoryfile, 'STATIONXML')
            inventory.write(self.inventorytxtfile, 'STATIONTXT',level='station')
            self.organize_inventory()

    ## inventory_catalog
    def obtain_events(self,minmagnitude=5.5,maxmagnitude=9.5,minradius = 90, maxradius=120, minimum_operation_time_station = 5):
        '''
        Find all the events for each stations
        '''
        self.minradius, self.maxradius = minradius, maxradius
        os.makedirs(self.catalogtxtloc, exist_ok=True)
        os.makedirs(self.catalogxmlloc, exist_ok=True)

        logger.info("\nObtaining event info...")
        self.inv = None

        ## Check for the station information
        if os.path.exists(self.inventorytxtfile):
            invent_df = pd.read_csv(self.inventorytxtfile,sep="|",keep_default_na=False, na_values=[""])
            total_stations = invent_df.shape[0]
            if invent_df.shape[0]==0:
                logger.info("-- No data available, exiting...")
                sys.exit()
        else:
            logger.error("-- No data available, exiting...")
            sys.exit()


        tot_evnt_stns = 0
        if not self.inv:
            logger.info("-- Reading station inventory to obtain events catalog")
            try:
                # Read the station inventory
                self.inv = read_inventory(self.inventoryfile, format="STATIONXML")
            except Exception as exception:
                logger.warning("-- No available data: {}".format(str(exception)))
                sys.exit()
        # list all the events during the station active time
        self.staNamesNet,staLats,staLons=[],[],[]
        count = 1
        for net in self.inv:
            for sta in net:
                try:
                    network = net.code #network name
                    station = sta.code #station name
                    logger.info("\n")
                    logger.info("-- {}/{} Retrieving event info for {}-{}".format(count,total_stations,network,station ))
                    count+=1
                    self.staNamesNet.append("{}_{}".format(network, station))

                    sta_lat = sta.latitude #station latitude
                    staLats.append(sta_lat)

                    sta_lon = sta.longitude #station longitude
                    staLons.append(sta_lon)

                    sta_sdate = sta.start_date #station start date
                    sta_edate = sta.end_date #station end date
                    
                    # sta_edate_str = sta_edate
                    if not sta_edate:
                        sta_edate = UTC("2599-12-31T23:59:59")
                        # sta_edate_str = "2599-12-31T23:59:59"
                    if (sta_edate.year - sta_sdate.year) < minimum_operation_time_station:
                        # logger.info((sta_edate.year - sta_sdate.year), sta_edate, sta_sdate)
                        logger.info("---- Station operation time too small")
                        continue
                    stime, etime = self.date2time(sta_sdate,sta_edate) #station start and end time in UTC


                    catalogxml = os.path.join(self.catalogxmlloc,'{}-{}-{}-{}-{}-{}_events.xml'.format(network,station,sta_sdate.year,sta_edate.year,self.method,self.method)) #xml catalog
                    # self.allcatalogxml.append(catalogxml)
                    catalogtxt = os.path.join(self.catalogtxtloc,'{}-{}-{}-{}-events-info-{}.txt'.format(network,station,sta_sdate.year,sta_edate.year,self.method)) #txt catalog
                    if not os.path.exists(catalogxml) and not os.path.exists(catalogtxt):
                        logger.info("---- Obtaining catalog: {}: {}-{}-{}-{}".format(self.method, network, station, sta_sdate.year, sta_edate.year))
                        kwargs = {'starttime': stime, 'endtime': etime, 
                                        'latitude': sta_lat, 'longitude': sta_lon,
                                        'minradius': self.minradius, 'maxradius': self.maxradius,
                                        'minmagnitude': minmagnitude, 'maxmagnitude': maxmagnitude}
                        client = Client('IRIS')

                        try:
                            catalog = client.get_events(**kwargs)
                        except KeyboardInterrupt:
                            sys.exit()
                        except:
                            logger.warning("---- ConnectionResetError while obtaining the events from the client - IRIS")
                            continue
                        catalog.write(catalogxml, 'QUAKEML') #writing xml catalog

                        
                        tot_evnt_stns += len(catalog)

                        evtimes,evlats,evlons,evdps,evmgs,evmgtps,evphases=[],[],[],[],[],[],[]
                        logger.info("---- Writing the event data into a text file")

                        with open(catalogtxt, 'w') as f:
                            f.write('evtime,evlat,evlon,evdp,evmg,evphase\n')
                            for cat in catalog:
                                try:
                                    try:
                                        evtime,evlat,evlon,evdp,evmg,evphase=cat.origins[0].time,cat.origins[0].latitude,cat.origins[0].longitude,cat.origins[0].depth/1000,cat.magnitudes[0].mag,self.method
                                        sys.exit()
                                    except:
                                        evtime,evlat,evlon,evdp,evmg,evphase=cat.origins[0].time,cat.origins[0].latitude,cat.origins[0].longitude,cat.origins[0].depth/1000,cat.magnitudes[0].mag,self.method
                                    evtimes.append(str(evtime))
                                    evlats.append(float(evlat))
                                    evlons.append(float(evlon))
                                    evdps.append(float(evdp))
                                    evmgs.append(float(evmg))
                                    # evmgtps.append(str(evmgtp))
                                    evphases.append(str(evphase))
                                    f.write('{},{:.4f},{:.4f},{:.1f},{:.1f},{}\n'.format(evtime,evlat,evlon,evdp,evmg,evphase)) #writing txt catalog

                                    
                                except KeyboardInterrupt:
                                    sys.exit()
                                except Exception as exception:
                                    logger.warning("-- Unable to write for {}: {}".format(evtime, str(exception)))
                        logger.info("---- Finished writing the event data into a text and xml file")
                    else:
                        logger.info("---- {} and {} already exists!".format(catalogxml.split('/')[-1], catalogtxt.split('/')[-1]))
                except KeyboardInterrupt:
                    sys.exit()
    
    def organize_inventory(self):
        '''
        create inventory text file containing information of stations with it's start and end time by merging info together
        '''
        
        inv_df = pd.read_csv(self.inventorytxtfile,sep="|",keep_default_na=False, na_values=[""])
    
        inv_df['EndTime'].fillna('2599-12-31T23:59:59',inplace=True)
        net_sta_set = set(inv_df['#Network']+'_'+inv_df['Station']) #get rid of repeated stations by joining rows info


        inv_df['StartTimeNum'] = inv_df['StartTime'].apply(lambda x: int(x.split("-")[0]+x.split("-")[1]+x.split("-")[2][0:2]))
        inv_df['EndTimeNum'] = inv_df['EndTime'].apply(lambda x: int(x.split("-")[0]+x.split("-")[1]+x.split("-")[2][0:2]))
        all_dicts=[]
        for net_sta in net_sta_set:
            net = net_sta.split("_")[0]
            sta = net_sta.split("_")[1]
            # finding all rows with same net and sta
            row = inv_df[(inv_df['#Network']==net) & (inv_df['Station']==sta)]
            row=row.reset_index()
            rowtimemax = row.loc[row['EndTimeNum'].idxmax()]
            rowtimemin = row.loc[row['StartTimeNum'].idxmin()]

            dict_row = {'#Network':net,'Station':sta,'Latitude':row.loc[0,'Latitude'],'Longitude':row.loc[0,'Longitude'],'Elevation':row.loc[0,'Elevation'],'SiteName':row.loc[0,'SiteName'],'StartTime':rowtimemin['StartTime'],'EndTime':rowtimemax['EndTime']}
            all_dicts.append(dict_row)
        new_inv_df=pd.DataFrame(all_dicts)
        # os.remove(inventorytxtfile)
        new_inv_df.to_csv(self.out_inventorytxtfile, index=False,sep="|")


    def date2time(self, sta_sdate,sta_edate):
        smonth = '0{}'.format(sta_sdate.month) if sta_sdate.month < 10 else '{}'.format(sta_sdate.month)
        emonth = '0{}'.format(sta_edate.month) if sta_edate.month < 10 else '{}'.format(sta_edate.month)
        sday = '0{}'.format(sta_sdate.day) if sta_sdate.day < 10 else '{}'.format(sta_sdate.day)
        eday = '0{}'.format(sta_edate.day) if sta_edate.day < 10 else '{}'.format(sta_edate.day)
        stime = '{}-{}-{}'.format(sta_sdate.year, smonth, sday)
        etime = '{}-{}-{}'.format(sta_edate.year, emonth, eday)

        return UTC(stime), UTC(etime)

    def download_all_data(self,
        datafileloc,
        locations=[""], 
        mergedCatLoc = 'merged_catalogs', 
        earthmodel='iasp91',
        filter = True, 
        freqmin = 0.01, 
        freqmax = 0.5, 
        resampleRate = 20.0, 
        phase = 'S',
        waveform_offset = 80.0):

        self.model = TauPyModel(earthmodel)
        os.makedirs(datafileloc, exist_ok=True)
        os.makedirs(mergedCatLoc, exist_ok=True)
        all_events_data = glob.glob(os.path.join(self.catalogtxtloc,"*-events-info-{}.txt".format(self.method))) ##events catalog for each station
        tot_evnt_stns = len(all_events_data)
        logger.info("Total data files to download: {}".format(tot_evnt_stns))

        succ_dl,num_try = 0, 0 
        rf_stalons,sks_stalons = [],[]
        rf_stalats, sks_stalats = [], []
        rf_staNetNames, sks_staNetNames = [],[]

        all_stns_df = pd.read_csv(self.inventorytxtfile,sep="|")
        print(self.inventoryfile)
        all_sta_lats=all_stns_df['Latitude'].values
        all_sta_lons=all_stns_df['Longitude'].values
        all_sta_nms=all_stns_df['Station'].values
        all_sta_nets=all_stns_df['#Network'].values
        
        sta_str_list = []

        #Retrive waveform data for the events
        for slat,slon,stn,net in zip(all_sta_lats,all_sta_lons,all_sta_nms,all_sta_nets):
            # loop to remove duplication of catalog file for one station
            sta_str = "{}-{}-{:.4f}-{:.4f}".format(net, stn, slon, slat)
            if sta_str in sta_str_list:
                # logger.info("Skipping {}".format(sta_str))
                continue
            else:
                sta_str_list.append(sta_str)
                

            search_string = os.path.join(self.catalogtxtloc,"{}-{}-*-events-info-{}.txt".format(net, stn, self.method))
            all_cats = glob.glob(search_string)
            if len(all_cats)>0:
                logger.info("\n")

                ## new catalog name
                catfile = os.path.join(mergedCatLoc,"{}-{}-events-info-{}.txt".format(net, stn, self.method))

                self.merge_catalogs(catfile, all_cats) #merge catalogs into one file
                # logger.info(catfile)

                ## available events catalog name
                cattxtnew = os.path.join(self.catalogtxtloc,"{}-{}-events-info-available-{}.txt".format(net, stn, self.method))

                
                logger.info("-- Searching and downloading data for {}; {}-{}".format(self.method, net, stn))
                print('done1')
                sksdatafile_prefix = os.path.join(datafileloc, "{}-{}".format(net, stn))
                print(sksdatafile_prefix)
                if (os.path.exists(catfile)) and (len(glob.glob(sksdatafile_prefix+"*.mseed"))==0) and (tot_evnt_stns > 0) :
                    print('done2')
                    logger.info("---- Reading events catalog file")
                    dff_event = pd.read_csv(catfile)
                    tasks = []
                    for inum, evtime, evdp, elat, elon in zip(range(len(dff_event['evtime'])),dff_event['evtime'].values, dff_event['evdp'].values, dff_event['evlat'].values, dff_event['evlon'].values):
                        record = "{}-{}-{}".format(net, stn, evtime)
                        if self.write_record(record):
                            # print('done3')
                            try:
                                logger.info("{}/{} Submitting download for {}".format(inum+1, len(dff_event), record))            
                                pharr, tstart, tend = self.get_phase_arrival_time(evdp,
                                                slat,
                                                slon,
                                                elat,
                                                elon, 
                                                evtime,
                                                phase = self.method,
                                                waveform_offset = waveform_offset)
                                
                                t1 = threading.Thread(target=self.download_waveform, args=[net, 
                                                                            stn, 
                                                                            locations[0], 
                                                                            evtime,
                                                                            tstart, 
                                                                            tend,
                                                                            self.channel, 
                                                                            datafileloc,
                                                                            True, 
                                                                            freqmin, 
                                                                            freqmax, 
                                                                            resampleRate,])

                                t1.start()
                                tasks.append(t1)
                            
                                if len(tasks)> 20:
                                    for task in tasks:
                                        task.join()
                                    tasks = []

                            except Exception as exception:
                                logger.error("get_phase_arrival_time did not work, continue")

                            # t1.start()
                            # tasks.append(t1)
                            
                            # if len(tasks)> 20:
                            #     for task in tasks:
                            #         task.join()
                            #     tasks = []
                    # sys.exit()

    def load_records(self):
        with open(self.dl_rec_pkl, 'rb') as handle:
            datalist = pickle.load(handle)
        return datalist

    def write_record(self,record):
        data_list_final = []
        try:

            ## check if the file is in the pickle already
            if os.path.exists(self.dl_rec_pkl):
                data_list_final = [] #self.load_records()           #empty this if pickle isnt working 
                if record in data_list_final:
                    return False
                else:
                    data_list_final.append(record)
            else:
                data_list_final.append(record)
                

            with open(self.dl_rec_pkl, 'wb') as handle:
                pickle.dump(data_list_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return True

        except:
            logger.exception("Failed to write record into a file")
            return False

        return data_list_final

    def merge_catalogs(self, newfile, filelist):
        with open(newfile, "w") as wfd:
            with open(filelist[0], "r") as fd:
                Lines = fd.readlines()
                count = 0
                for line in Lines:
                    count += 1
                    wfd.write(line)
            for f in filelist[1:]:
                with open(f, "r") as fd:
                    Lines = fd.readlines()
                    count = 0
                    for line in Lines:
                        count += 1
                        if count == 1: #skip the header line
                            continue
                        wfd.write(line)

    def get_phase_arrival_time(self,
                            evdp,
                            slat,
                            slon,
                            elat,
                            elon, 
                            evtime,
                            phase = 'S',
                            waveform_offset = 80.0):
        arrivals = self.model.get_travel_times_geo(float(evdp),slat,slon,float(elat),float(elon),phase_list=[phase])
        ##### my code ######
        try: 
            pharr = UTC(str(evtime)) + arrivals[0].time #SKS arrival time
            tstart = pharr - waveform_offset
            tend = pharr + waveform_offset
            return pharr, tstart, tend

        except Exception as exception:
            logger.error("Arrival index of [0] is out of range, phase not calculated.")

        # ####### fixing error saying arrivals[0] is out of range #####
        # pharr = UTC(str(evtime)) + arrivals[0].time #SKS arrival time
        # tstart = pharr - waveform_offset
        # tend = pharr + waveform_offset

        # return pharr, tstart, tend

    def download_waveform(self, 
        net, 
        stn, 
        loc, 
        evtime,
        tstart, 
        tend,
        channel, 
        datafileloc,
        filter = True, 
        freqmin = 0.01, 
        freqmax = 0.5, 
        resampleRate = 20.0):
        # try:
        ## these clients except IRIS dont provide data
        for cll in ['USGS','IRIS','ISC','EMSC']:
            if cll in self.client:
                self.client.remove(cll)

        ## get waveforms from server
        logger.info("---- Fetching waveform data for {}-{} for {}".format(net, stn, evtime))
        st = None
        for cl in ['IRIS']+self.client:
            try:
                client = Client(cl)
                st = client.get_waveforms(net, stn, loc, channel, tstart, tend, attach_response=True)
                st.remove_response(output="VEL")                                                                                          #COME BACK TO THIS#
                if st: 
                    break
                time.sleep(0.5)
            except Exception as err:
                pass
                logger.warning("------ client {}; {}".format(cl, str(err).split("\n")[0].rstrip()))


        if not st:
            logger.info("------  Failed to retrieve waveform data for {}-{} for {}".format(net, stn, evtime))
            return 'failure'

        ## Check for the continuity of stream
        if len(st) < 3:
            logger.warning('------ Need 3 channels {}, only {} components found'.format(channel, [tr.stats.channel for tr in st]))
            # logger.info("channel: {}".format(channel))
            return 'failure'

        for tr in st:
            if any(isinstance(tr.data, np.ma.masked_array) for tr in st):
                logger.warning('------ Gaps or overlaps detected in data')
                return 'failure'

        # ## check if the stream has enough length of data
        # for tr in st:
        #     sttime = tr.stats.starttime
        #     edtime = tr.stats.endtime
        #     if (edtime - sttime) < (tend - tstart):
        #         return 'failure'

        tr = st[0]
        if tr.stats.sampling_rate < resampleRate:
            logger.warning("------ Sampling rate too low: {}, required >= {}Hz".format(tr.stats.sampling_rate, resampleRate))
            return 'failure'

        elif tr.stats.sampling_rate >= resampleRate:
            if tr.stats.sampling_rate % resampleRate == 0:
                factor = int(tr.stats.sampling_rate / resampleRate)
                logger.warning("------ Downsampling to {} Hz, current sr: {}, factor: {}".format(resampleRate, tr.stats.sampling_rate, factor))
                for trr in st:
                    trr.decimate(factor, strict_length=False, no_filter=True) 
                
            else:
                logger.info("------ Resampling traces; New sampling rate: {}".format(tr.stats.sampling_rate))
                for trr in st:
                    trr.resample(resampleRate)

        if filter:
            st.detrend('demean')
            st.detrend('linear')
            # st.filter("bandpass",freqmin=freqmin,freqmax=freqmax)

        if "," in channel:
            channel_str = "_".join(channel.split(","))
        else:
            channel_str = channel

        datafile = os.path.join(datafileloc,'{}-{}-{}.mseed'.format(net, stn, evtime))
        logger.info("------ Writing waveform data to disk: {}".format(datafile))
        st.write(datafile, format="MSEED")
        del st
        return 'success'
        # except Exception:
        #     return 'failure'

directory = 'senior_thesis/data/complete/swp2' 
station = 'swp2' 


if __name__=="__main__":
    downloadclass = DownloadData(
        # clientlist = ['http://ws.ipgp.fr','http://service.iris.edu'],
        # clientlist = ['http://service.iris.edu'],
        # clientlist = ['NCEDC'],
        # clientlist = ['IRIS','GEOFON','AUSPASS'],
        # clientlist = ['IRIS'],
        # clientlist = ['https://geofon.gfz.de','http://service.iris.edu','http://ws.ipgp.fr','https://eida.ethz.ch'],
        clientlist = ['https://geofon.gfz.de','http://service.iris.edu','http://ws.ipgp.fr','https://eida.ethz.ch'] + list(URL_MAPPINGS.keys()),



        # clientlist = list(URL_MAPPINGS.keys()),
        minlongitude = 100, # 120
        maxlongitude = 180, # 160
        minlatitude = -50,  # -40    
        maxlatitude = 20,   #  50
        inventorytxtfile = '../../'+directory+'/'+station+'_stations_info.txt',
        catalogxmlloc='../../'+directory+'/'+station+'_event_info_xml',
        catalogtxtloc='../../'+directory+'/'+station+'_event_info/',
    )

    # # ## download the stations information
    # downloadclass.get_stnxml(network='*', station='*', level = 'response')

    # ## download event catalog corresponding to each station   #make sure to check the assigned epicentral distances!!!
    # downloadclass.obtain_events(
    #         minmagnitude    = 6,
    #         maxmagnitude    = 8,
    #         minradius       = 0, 
    #         maxradius       = 90,
    #         minimum_operation_time_station = 3)
   
    downloadclass.download_all_data(
        datafileloc='../../'+directory+'/'+station+'_S_waveforms',
        locations=[""], 
        mergedCatLoc = 'merged_catalogs', 
        earthmodel='iasp91',
        filter = False, 
        freqmin = 0.01, 
        freqmax = 0.125,  # have done 0.5 in the past 
        resampleRate = 20.0, 
        phase = 'S',
        waveform_offset = 60.0)