import argparse 
import datetime
from itertools import repeat
import json
import logging
from multi_key_dict import multi_key_dict
from multiprocessing import Pool, Value
import numpy as np
import pandas as pd
from pathlib import Path
import re
from typing import Any, List

class GatewayLogIngestor:
    gw_id: int
    log = pd.DataFrame()
    
    def __init__(self, gw_log: Path) -> None:
        self.gw_id = int(str(gw_log.stem).split('-')[1])       # extracts the integer 0 from ajs_station-0.log filename
        print(f'Ingesting {gw_log} ...')

        # Import file
        self.log = pd.read_csv(
            gw_log, 
            sep=f'\s+\[0{self.gw_id}\]\s+', 
            names=['Timestamp', 'Data'],
            header=0,
            engine='python'
            )
        
        # Ignore unimportant rows
        self.log = self.log[~self.log.Data.str.contains("GPS -- time gps:")]
        self.log = self.log[~self.log.Data.str.contains("A transmission is already scheduled (!overflow error!)")]
        self.log = self.log[~self.log.Data.str.contains("Aborting TX was OK")]
        self.log = self.log[~self.log.Data.str.contains("STATS: max gps timeref")]
        self.log = self.log[~self.log.Data.str.contains("checking prev airtime")]
        self.log = self.log[~self.log.Data.str.contains("checking next airtime")]
        self.log = self.log[~self.log.Data.str.contains("We received a PONG")]
        self.log = self.log[~self.log.Data.str.contains("beacon is being sent on PPM")]

        # Set index to timestamp
        self.log['Timestamp'] = self.log.iloc[:,0].str.extract('(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d{3})', expand=False)
        self.log['Timestamp'] = pd.to_datetime(self.log['Timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
        self.log['Timestamp'] = self.log['Timestamp'] - pd.DateOffset(hours=8)
        self.log.set_index('Timestamp')

        # Extract data
        self.log[['Event', 'FCnt', 'DevEui', 'DevAddr', 'pdu', 'seqno']] = None
        self.log[['Event', 'FCnt', 'DevEui', 'DevAddr', 'pdu', 'seqno']] = self.log['Data'].apply(self._parse_data_string)
        self.log.drop(columns=['Data'])

    def _parse_data_string(self, data: str):
        """Parse single line from log file
        """
        if 'UP: ' in data:
            data = json.loads(data.split('UP: ')[1])
            msgtype = data.get('msgtype', None)
            if msgtype:
                msgtype = msgtype.upper()
            return pd.Series([
                msgtype,
                data.get('FCnt', None),
                data.get('DevEui', None),
                data.get('DevAddr', None),
                data.get('pdu', None),
                None
            ])
        elif 'DN: [On-Air] ' in data:
            data = json.loads(data.split('DN: [On-Air] ')[1])
            return pd.Series([
                'DN: [On-Air]',
                None,
                data.get('DevEui', None),
                None,
                None,
                data.get('seqno', None),
            ])
        elif 'DN: [Scheduled] ' in data:
            pdu = re.search('[0-9a-fA-F]{14,}', data).group()
            event = 'DN: [Scheduled] MAC'
            if pdu[:2] == '20':
                event = 'DN: [Scheduled] JACC'
            return pd.Series([
                event,
                None,
                None,
                None,
                pdu,
                None,
            ])

        return pd.Series([
            'UNKNOWN',
            None,
            None,
            None,
            None
        ])


class JoinsLogIngestor:
    log = pd.DataFrame()
    
    def __init__(self, joins_log: Path) -> None:
        print(f'Ingesting {joins_log} ...')

        # Import file
        self.log = pd.read_csv(
            joins_log, 
            sep=f'\s+joins ERRO:\s+', 
            names=['Timestamp', 'Data'],
            header=0,
            engine='python'
            )

        # Set index to timestamp
        self.log['Timestamp'] = self.log.iloc[:,0].str.extract('(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d{3})', expand=False)
        self.log['Timestamp'] = pd.to_datetime(self.log['Timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
        self.log.set_index('Timestamp')

        # Extract data
        self.log[['Event', 'DevEui']] = None
        self.log[['Event', 'DevEui']] = self.log['Data'].apply(self._parse_data_string)
        self.log = self.log.drop(columns=['Data'])

    def _parse_data_string(self, data: str):
        """Parse single line from log file
        """
        deveui = re.search('([0-9A-F]{2}[:-]){7}[0-9A-F]{2}', data).group()
        if 'Join request for unprovisioned device' in data:
            return pd.Series([
                data.split(' - ')[1],
                deveui
            ])
        elif 'Verify of join request failed' in data:
            return pd.Series([
                'Verify of join request failed',
                deveui
            ])
        elif 'Not accepted (in time)' in data:
            return pd.Series([
                'Not accepted (in time)',
                deveui
            ])

        return pd.Series([
            'UNKNOWN',
            deveui
        ])


class NwksLogIngestor:
    log = pd.DataFrame()
    
    def __init__(self, nwks_log: Path) -> None:
        print(f'Ingesting {nwks_log} ...')

        # Import file
        self.log = pd.read_csv(
            nwks_log, 
            sep=f'\s+nwks', 
            names=['Timestamp', 'Data'],
            header=0,
            engine='python'
            )

        # Set index to timestamp
        self.log['Timestamp'] = self.log.iloc[:,0].str.extract('(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d{3})', expand=False)
        self.log['Timestamp'] = pd.to_datetime(self.log['Timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
        self.log.set_index('Timestamp')

        # Extract data
        self.log[['Event', 'DevEui', 'DevAddr', 'NewDiid', 'OldDiid']] = None
        self.log[['Event', 'DevEui', 'DevAddr', 'NewDiid', 'OldDiid']] = self.log['Data'].apply(self._parse_data_string)
        self.log = self.log.drop(columns=['Data'])

    def _parse_data_string(self, data: str):
        """Parse single line from log file
        """
        deveui = re.search('([0-9A-F]{2}[:-]){7}[0-9A-F]{2}', data).group()
        if 'Overwriting dninfo (lost dntxed/abandoned dn msg)' in data:
            newdiid = re.search('diid=\d{1,}', data)
            newdiid = int(newdiid.group().split('=')[1])
            olddiid = re.search("'diid': \d{1,}", data)
            olddiid = int(olddiid.group().split(': ')[1])
            return pd.Series([
                'Overwriting dninfo (lost dntxed/abandoned dn msg)',
                deveui,
                None,
                newdiid,
                olddiid
            ])
        elif 'jacc overwrites pending session' in data:
            return pd.Series([
                'jacc overwrites pending session',
                deveui,
                None,
                None,
                None
            ])
        elif 'Suppressing sending of empty frame while suppressing FOPtsDn' in data:
            return pd.Series([
                'Suppressing sending of empty frame while suppressing FOPtsDn',
                deveui,
                None,
                None,
                None
            ])
        elif 'ADR blocked (temporarily) by pending DN option' in data:
            return pd.Series([
                'ADR blocked (temporarily) by pending DN option',
                deveui,
                None,
                None,
                None
            ])
        elif 'Spurious LinkADRAns' in data:
            return pd.Series([
                'Spurious LinkADRAns',
                deveui,
                None,
                None,
                None
            ])
        elif 'Messages to unknown device (dropped)' in data:
            devaddr = re.search("'DevAddr': \d{1,}", data)
            devaddr = int(devaddr.group().split(': ')[1])
            return pd.Series([
                'Messages to unknown device (dropped)',
                deveui,
                devaddr,
                None,
                None
            ])

        return pd.Series([
            'UNKNOWN',
            deveui,
            None,
            None,
            None
        ])


class DoorLogIngestor:
    log = pd.DataFrame()
    
    def __init__(self, door_log: Path) -> None:
        print(f'Ingesting {door_log} ...')

        # Import file
        self.log = pd.read_csv(
            door_log, 
            sep=f'\s+door INFO:\s+', 
            names=['Timestamp', 'Data'],
            header=0,
            engine='python'
            )

        # Set index to timestamp
        self.log['Timestamp'] = self.log.iloc[:,0].str.extract('(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d{3})', expand=False)
        self.log['Timestamp'] = pd.to_datetime(self.log['Timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
        self.log.set_index('Timestamp')

        # Extract data
        self.log[['Event', 'DevEui', 'pdu', 'diid']] = None
        self.log[['Event', 'DevEui', 'pdu', 'diid']] = self.log['Data'].apply(self._parse_data_string)
        self.log = self.log.drop(columns=['Data'])

    def _parse_data_string(self, data: str):
        """Parse single line from log file
        """
        if 'SENDING Muxs...' in data:
            data = data.replace("\'", "\"")
            data = eval(data.split('SENDING Muxs... ')[1])
            return pd.Series([
                'SENDING Muxs...',
                data.get('DevEui', None),
                data.get('pdu', None),
                data.get('diid', None),
            ])

        deveui = re.search('([0-9A-F]{2}[:-]){7}[0-9A-F]{2}', data).group()
        return pd.Series([
            'UNKNOWN',
            deveui,
            None,
            None
        ])



class MuxsLogIngestor:
    log = pd.DataFrame()
    
    def __init__(self, muxs_log: Path) -> None:
        print(f'Ingesting {muxs_log} ...')

        # Import file
        self.log = pd.read_csv(
            muxs_log, 
            sep=f'\s+muxs INFO:\s+', 
            names=['Timestamp', 'Data'],
            header=0,
            engine='python'
            )

        # Set index to timestamp
        self.log['Timestamp'] = self.log.iloc[:,0].str.extract('(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d{3})', expand=False)
        self.log['Timestamp'] = pd.to_datetime(self.log['Timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
        self.log.set_index('Timestamp')

        # Extract data
        self.log[['Event', 'DevAddr']] = None
        self.log[['Event', 'DevAddr']] = self.log['Data'].apply(self._parse_data_string)
        self.log = self.log.drop(columns=['Data'])

    def _parse_data_string(self, data: str):
        """Parse single line from log file
        """
        if data:
            if 'Unknown DevAddr' in data:
                result = re.search('\d{5,}/0x', data)
                devaddr = None
                if result:
                    devaddr = int(result.group().split('/')[0])
                return pd.Series([
                    'Unknown DevAddr',
                    devaddr
                ])

        return pd.Series([
            'UNKNOWN',
            None,
        ])


class DeviceTimeline:
    timeline = pd.DataFrame(columns=['Timestamp'])
    deveui: str
    devaddrs = list()
    diids = list()
    pdus = list()
    start_time: datetime.datetime

    def __init__(self, deveui: str, start_time: datetime.datetime) -> None:
        if '-' not in deveui:
            deveui = '-'.join([
                deveui[:2],
                deveui[2:4],
                deveui[4:6],
                deveui[6:8],
                deveui[8:10],
                deveui[10:12],
                deveui[12:14],
                deveui[14:16]
            ])
        self.deveui = deveui
        self.start_time = start_time
        self.output_dir = Path('./output')
        self.output_dir.mkdir(exist_ok=True)

    def __str__(self) -> str:
        return self.deveui

    def extract(self, ingestors: List[Any]) -> None:
        # Extract data based on DevEui
        for ingestor in ingestors:
            if isinstance(ingestor, GatewayLogIngestor):
                self._extract_from_gateway(ingestor)
            if isinstance(ingestor, JoinsLogIngestor):
                self._extract_from_joins(ingestor)
            if isinstance(ingestor, NwksLogIngestor):
                self._extract_from_nwks(ingestor)
            if isinstance(ingestor, DoorLogIngestor):
                self._extract_from_door(ingestor)
            if isinstance(ingestor, MuxsLogIngestor):
                self._extract_from_muxs(ingestor)

        # Extract based on metadata associated with this device
        for ingestor in ingestors:
            if isinstance(ingestor, GatewayLogIngestor):
                self._extract_from_gateway_meta(ingestor)
            if isinstance(ingestor, MuxsLogIngestor):
                self._extract_from_muxs_meta(ingestor)

        self.cleanup_timeline()


    def _extract_from_gateway(self, gw_ingestor: GatewayLogIngestor):
        new_col = f'GW{gw_ingestor.gw_id}'
        relevant = gw_ingestor.log.loc[gw_ingestor.log['DevEui'] == self.deveui]
        relevant.loc[:,new_col] = relevant['Event']
        self.timeline[new_col] = None
        self.timeline = pd.concat([self.timeline, relevant[['Timestamp', new_col]]], join='outer')
        self.cleanup_timeline()


    def _extract_from_joins(self, joins_ingestor: JoinsLogIngestor):
        new_col = 'JOINS'
        relevant = joins_ingestor.log.loc[joins_ingestor.log['DevEui'] == self.deveui]
        relevant.loc[:,new_col] = relevant['Event']
        self.timeline[new_col] = None
        self.timeline = pd.concat([self.timeline, relevant[['Timestamp', new_col]]], join='outer')
        self.cleanup_timeline()


    def _extract_from_nwks(self, nwks_ingestor: JoinsLogIngestor):
        new_col = 'NWKS'
        relevant = nwks_ingestor.log.loc[nwks_ingestor.log['DevEui'] == self.deveui]
        relevant.loc[:,new_col] = relevant['Event']
        self.timeline[new_col] = None
        self.timeline = pd.concat([self.timeline, relevant[['Timestamp', new_col]]], join='outer')
        self.cleanup_timeline()

        # Get associated devaddrs and diids
        self.devaddrs += relevant.loc[relevant['DevAddr'].notna(), 'DevAddr'].to_list()
        self.diids += relevant.loc[relevant['NewDiid'].notna(), 'NewDiid'].to_list()
        self.diids += relevant.loc[relevant['OldDiid'].notna(), 'OldDiid'].to_list()


    def _extract_from_door(self, door_ingestor: DoorLogIngestor):
        new_col = 'DOOR'
        relevant = door_ingestor.log.loc[(door_ingestor.log['DevEui'] == self.deveui)]
        relevant.loc[:,new_col] = relevant['Event']
        self.timeline[new_col] = None
        self.timeline = pd.concat([self.timeline, relevant[['Timestamp', new_col]]], join='outer')
        self.cleanup_timeline()

        # Get associated pdus and diids
        self.pdus += relevant.loc[relevant['pdu'].notna(), 'pdu'].to_list()
        self.diids += relevant.loc[relevant['diid'].notna(), 'diid'].to_list()


    def _extract_from_muxs(self, muxs_ingestor: MuxsLogIngestor):
        new_col = 'MUXS'
        self.timeline[new_col] = None
        if self.devaddrs:
            checks = [muxs_ingestor.log['DevAddr'] == devaddr for devaddr in self.devaddrs]
            found = muxs_ingestor.log['DevAddr'] > -1e15
            for check in checks:
                found |= check
            relevant = muxs_ingestor.log.loc[found]
            relevant.loc[:,new_col] = relevant['Event']
            self.timeline = pd.concat([self.timeline, relevant[['Timestamp', new_col]]], join='outer')
            self.cleanup_timeline()


    def _extract_from_gateway_meta(self, gw_ingestor: GatewayLogIngestor):
        new_col = f'GW{gw_ingestor.gw_id}'
        found = gw_ingestor.log['DevAddr'].isin(self.devaddrs) | \
                gw_ingestor.log['pdu'].isin(self.pdus)# | \
                # gw_ingestor.log['seqno'].isin(self.diids)     # don't need to do this - already captured by deveui
        relevant = gw_ingestor.log.loc[found]
        relevant.loc[:,new_col] = relevant['Event']
        self.timeline = pd.concat([self.timeline, relevant[['Timestamp', new_col]]], join='outer')
        self.cleanup_timeline()


    def _extract_from_muxs_meta(self, muxs_ingestor: MuxsLogIngestor):
        new_col = 'MUXS'
        found = muxs_ingestor.log['DevAddr'].isin(self.devaddrs)
        relevant = muxs_ingestor.log.loc[found]
        relevant.loc[:,new_col] = relevant['Event']
        self.timeline = pd.concat([self.timeline, relevant[['Timestamp', new_col]]], join='outer')
        self.cleanup_timeline()


    def check_for_errors(self):
        """Analyze the sequence of events for operational errors
        """
        normal_join_note = 'Normal Join: LNS carried out the normal join process.'
        device_not_joined_note = 'Device Error: Device tried to join again after successful join process.'
        gw_missed_dn_jacc = 'LNS Error: Gateway scheduled JACC but did not transmit it.'

        self.timeline['Notes'] = ''
        for i in range(1, self.timeline.shape[0]+1):
            if i == 3:
                if self._normal_join_process(self.timeline.loc[(i-3):i]):
                    self.timeline.loc[i,'Notes'] = normal_join_note
            if i > 3:
                if self._normal_join_process(self.timeline.loc[(i-4):i]):
                    self.timeline.loc[i,'Notes'] = normal_join_note
            if self.timeline.loc[i-1,'Notes'] == normal_join_note:
                if self._device_not_joined(self.timeline.loc[(i-1):i]):
                    self.timeline.loc[i,'Notes'] = device_not_joined_note
            if i < self.timeline.shape[0]:
                if self._missing_on_air(self.timeline.loc[(i-1):i]):
                    self.timeline.loc[i,'Notes'] = gw_missed_dn_jacc


    def _normal_join_process(self, tl: pd.DataFrame):
        """Returns true if the timeline shows correct join behavior from the LNS point of view.

        Parameters
        ----------
        tl : pd.DataFrame
            Timeline of events, length 4 or 5
        """
        gw_got_jreq = (tl.iloc[-4]['GW0'] == 'JREQ') or (tl.iloc[-4]['GW1'] == 'JREQ')
        if tl.shape[0] > 4:
            gw_got_jreq = gw_got_jreq or (tl.iloc[-5]['GW0'] == 'JREQ') or (tl.iloc[-5]['GW1'] == 'JREQ')
        door_sending_muxs = tl.iloc[-3]['DOOR'] == 'SENDING Muxs...'
        gw_sent_jacc =  (tl.iloc[-2]['GW0'] == 'DN: [Scheduled] JACC' and tl.iloc[-1]['GW0'] == 'DN: [On-Air]') or \
                        (tl.iloc[-2]['GW1'] == 'DN: [Scheduled] JACC' and tl.iloc[-1]['GW1'] == 'DN: [On-Air]')
        return gw_got_jreq and door_sending_muxs and gw_sent_jacc
        

    def _device_not_joined(self, tl: pd.DataFrame):
        """Returns true if device sends new JREQ after LNS sends JACC

        Parameters
        ----------
        tl : pd.DataFrame
            Timeline of events, length 2
        """
        return  ((tl.iloc[0]['GW0'] == 'DN: [On-Air]') or (tl.iloc[0]['GW1'] == 'DN: [On-Air]')) and \
                ((tl.iloc[1]['GW0'] == 'JREQ') or (tl.iloc[1]['GW1'] == 'JREQ'))


    def _missing_on_air(self, tl: pd.DataFrame):
        """Returns true if gateway schedules a transmission but doesn't send it

        Parameters
        ----------
        tl : pd.DataFrame
            Timeline of events, length 2
        """
        return  ((tl.iloc[1]['GW0'] != 'DN: [On-Air]') and (tl.iloc[0]['GW0'] == 'DN: [Scheduled] JACC')) or \
                ((tl.iloc[1]['GW1'] != 'DN: [On-Air]') and (tl.iloc[0]['GW1'] == 'DN: [Scheduled] JACC'))

    def cleanup_timeline(self):
        """Print the timeline to screen
        """
        self.timeline = self.timeline.fillna('')        # Remove all None and NaN
        self.timeline = self.timeline.sort_values(by='Timestamp', axis=0)
        self.timeline = self.timeline.reset_index(drop=True)


    def to_xlsx(self):
        filename = self.output_dir / f'{self.deveui}.xlsx'
        self.cleanup_timeline()
        self.timeline.to_excel(filename)



class DeviceStats:
    def __init__(self, devices: List[DeviceTimeline]) -> None:
        stat_names = ['Normal Join', 'Device Error', 'LNS Error']
        deveuis = [d.deveui for d in devices]
        stats = pd.DataFrame(np.zeros([len(devices), len(stat_names)]), deveuis, stat_names)
        for stat in stat_names:
            for device in devices:
                found = device.timeline['Notes'].str.contains(stat)
                stats.loc[device.deveui, stat] = device.timeline[found].shape[0]

        filename = devices[0].output_dir / 'stats.xlsx'
        stats.to_excel(filename)



def ingest(logtype: str, logfolder: Path):
    """Ingest a log file of a certain type (station, joins, nwks, ...)

    Parameters
    ----------
    type : str
        one of ['*station-0.log', '*station-1.log', '*joins.log', '*nwks.log', '*door.log', '*muxs.log']
    """
    allowed_types = ['*station-0.log', '*station-1.log', '*joins.log', '*nwks.log', '*door.log', '*muxs.log']
    if logtype not in allowed_types:
        raise ValueError('Invalid log type')
    logs = [f for f in logfolder.glob(logtype)]
    if len(logs) > 1:
        raise FileExistsError(f'There should only be one *station-0.log file. Instead found: {logs}')
    if len(logs) == 1:
        if logtype == allowed_types[0] or logtype == allowed_types[1]:
            return GatewayLogIngestor(logs[0])
        if logtype == allowed_types[2]:
            return JoinsLogIngestor(logs[0])
        if logtype == allowed_types[3]:
            return NwksLogIngestor(logs[0])
        if logtype == allowed_types[4]:
            return DoorLogIngestor(logs[0])
        if logtype == allowed_types[5]:
            return MuxsLogIngestor(logs[0])
    return None

def main(args):
    correctlogfiles = ['*station-0.log', '*station-1.log', '*joins.log', '*nwks.log', '*door.log', '*muxs.log']
    
    # Multi-processed version
    with Pool() as p:
        ingestors = p.starmap(ingest, zip(correctlogfiles, repeat(args.logfolder)))
    
    # Single-processed version
    # ingestors = [ingest(correctlogfile, args.logfolder) for correctlogfile in correctlogfiles]

    # Parse specified devices
    devices = []
    if not args.devices:
        # Parse all devices
        # devices = discover_all_devices(ingestors)
        pass
    if '.xlsx' in args.devices:
        inputs = pd.read_excel(args.devices)
        devices = [DeviceTimeline(params[0], params[1]) for _, params in inputs.iterrows()]
    
    for device in devices:
        device.extract(ingestors)
        device.check_for_errors()
        device.to_xlsx()

    stats = DeviceStats(devices)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='timeline',
        description='Analyze device behavior from LNS logs.',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-d', '--devices', type=str, default='',
        help=
'''Devices to focus on. Can be a single device, comma-separated list, 
CSV, or XLSX. Leave blank for all devices.
Examples:
00-00-00-00-00-00-00-01
00-00-00-00-00-00-00-01,00-00-00-00-00-00-00-02,00-00-00-00-00-00-00-03
devs.csv
devs.xlsx
Default: blank'''
        )

    parser.add_argument(
        '-l', '--logfolder',
        type=Path,
        default=Path(__file__).absolute().parent / "logs",
        help=
f'''Path to the logs folder containing any or all of these log files:
*door.log
*joins.log
*muxs.log
*nwks.log
*station-0.log
*station-1.log
Default: {Path(__file__).absolute().parent / "logs"}
'''
    )
    args = parser.parse_args()
    main(args)