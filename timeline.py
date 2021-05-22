import argparse 
import datetime
import json
import logging
from multi_key_dict import multi_key_dict
import pandas as pd
from pathlib import Path
import re
from typing import Any, List

logger = logging.getLogger('TIMELINE')
logger.setLevel(logging.INFO)

class GatewayLogIngestor:
    gw_id: int
    log = pd.DataFrame()
    
    def __init__(self, gw_log: Path) -> None:
        self.gw_id = int(str(gw_log.stem).split('-')[1])       # extracts the integer 0 from ajs_station-0.log filename
        logger.info(f'Ingesting {gw_log} ...')

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
        self.log = self.log[~self.log.Data.str.contains("STATS: max gps timeref")]
        self.log = self.log[~self.log.Data.str.contains("checking prev airtime")]
        self.log = self.log[~self.log.Data.str.contains("checking next airtime")]
        self.log = self.log[~self.log.Data.str.contains("We received a PONG")]
        # self.log = self.log[~self.log.Data.str.contains("Missed a transmission by")]

        # Set index to timestamp
        self.log['Timestamp'] = self.log.iloc[:,0].str.extract('(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d{3})', expand=False)
        self.log['Timestamp'] = pd.to_datetime(self.log['Timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
        self.log.set_index('Timestamp')

        # Extract data
        self.log[['Event', 'FCnt', 'DevEui', 'DevAddr', 'pdu']] = None
        self.log[['Event', 'FCnt', 'DevEui', 'DevAddr', 'pdu']] = self.log['Data'].apply(self._parse_data_string)
        self.log.drop(columns=['Data'])

    def _parse_data_string(self, data: str):
        """Parse single line from log file
        """
        type = data.split(': ')[0]
        if type == 'UP':
            data = json.loads(data.split('UP: ')[1])
            return pd.Series([
                data.get('msgtype', None),
                data.get('FCnt', None),
                data.get('DevEui', None),
                data.get('DevAddr', None),
                data.get('pdu', None),
            ])
        elif type == 'DN':
            pass

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
        logger.info(f'Ingesting {joins_log} ...')

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
        logger.info(f'Ingesting {nwks_log} ...')

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
        logger.info(f'Ingesting {door_log} ...')

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
        deveui = re.search('([0-9A-F]{2}[:-]){7}[0-9A-F]{2}', data).group()
        if 'SENDING Muxs...' in data:
            data = data.replace("\'", "\"")
            data = eval(data.split('SENDING Muxs... ')[1])
            return pd.Series([
                'SENDING Muxs...',
                data.get('DevEui', None),
                data.get('pdu', None),
                data.get('diid', None),
            ])

        return pd.Series([
            'UNKNOWN',
            None,
            None,
            None
        ])



class MuxsLogIngestor:
    log = pd.DataFrame()
    
    def __init__(self, muxs_log: Path) -> None:
        logger.info(f'Ingesting {muxs_log} ...')

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
                devaddr = int(re.search('\d{5,}/0x', data).group().split('/')[0])
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

    def __str__(self) -> str:
        return self.deveui

    def extract(self, ingestors: List[Any]) -> None:
        for ingestor in ingestors:
            if isinstance(ingestor, GatewayLogIngestor):
                self._extract_from_gateway(ingestor)
            if isinstance(ingestor, JoinsLogIngestor):
                self._extract_from_joins(ingestor)
            if isinstance(ingestor, NwksLogIngestor):
                self._extract_from_nwks(ingestor)
            if isinstance(ingestor, MuxsLogIngestor):
                self._extract_from_muxs(ingestor)

    def _extract_from_gateway(self, gw_ingestor: GatewayLogIngestor):
        new_col = f'GW{gw_ingestor.gw_id}'
        relevant = gw_ingestor.log.loc[gw_ingestor.log['DevEui'] == self.deveui]
        relevant.loc[:,new_col] = relevant['Event']
        self.timeline[new_col] = None
        self.timeline = pd.concat([self.timeline, relevant[['Timestamp', new_col]]], join='outer')
        self.timeline = self.timeline.sort_values(by='Timestamp', axis=0)
        self.timeline = self.timeline.reset_index(drop=True)

    def _extract_from_joins(self, joins_ingestor: JoinsLogIngestor):
        new_col = 'JOINS'
        relevant = joins_ingestor.log.loc[joins_ingestor.log['DevEui'] == self.deveui]
        relevant.loc[:,new_col] = relevant['Event']
        self.timeline[new_col] = None
        self.timeline = pd.concat([self.timeline, relevant[['Timestamp', new_col]]], join='outer')
        self.timeline = self.timeline.sort_values(by='Timestamp', axis=0)
        self.timeline = self.timeline.reset_index(drop=True)

    def _extract_from_nwks(self, nwks_ingestor: JoinsLogIngestor):
        new_col = 'NWKS'
        relevant = nwks_ingestor.log.loc[nwks_ingestor.log['DevEui'] == self.deveui]
        relevant.loc[:,new_col] = relevant['Event']
        self.timeline[new_col] = None
        self.timeline = pd.concat([self.timeline, relevant[['Timestamp', new_col]]], join='outer')
        self.timeline = self.timeline.sort_values(by='Timestamp', axis=0)
        self.timeline = self.timeline.reset_index(drop=True)

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
        self.timeline = self.timeline.sort_values(by='Timestamp', axis=0)
        self.timeline = self.timeline.reset_index(drop=True)

        # Get associated pdus and diids
        self.pdus += relevant.loc[relevant['pdu'].notna(), 'pdu'].to_list()
        self.diids += relevant.loc[relevant['diid'].notna(), 'diid'].to_list()

    def _extract_from_muxs(self, muxs_ingestor: MuxsLogIngestor):
        new_col = 'MUXS'
        checks = [muxs_ingestor.log['DevAddr'] == devaddr for devaddr in self.devaddrs]
        found = muxs_ingestor.log['DevAddr'] > -1e15
        for check in checks:
            found |= check
        relevant = muxs_ingestor.log.loc[found]
        relevant.loc[:,new_col] = relevant['Event']
        self.timeline[new_col] = None
        self.timeline = pd.concat([self.timeline, relevant[['Timestamp', new_col]]], join='outer')
        self.timeline = self.timeline.sort_values(by='Timestamp', axis=0)
        self.timeline = self.timeline.reset_index(drop=True)

        print()

    def print(self):
        """Print the timeline to screen
        """
        pass

    def check_for_errors(self):
        """Analyze the sequence of events for operational errors
        """
        pass


def main(args):
    ingestors = []

    # Gateway ingestors
    gw0_logs = [f for f in args.logfolder.glob('*station-0.log')]
    if len(gw0_logs) > 1:
        raise FileExistsError(f'There should only be one *station-0.log file. Instead found: {gw0_logs}')
    gw0_ingestor = GatewayLogIngestor(gw0_logs[0])
    ingestors.append(gw0_ingestor)
    gw1_logs = [f for f in args.logfolder.glob('*station-1.log')]
    if len(gw1_logs) > 1:
        raise FileExistsError(f'There should only be one *station-1.log file. Instead found: {gw1_logs}')
    gw1_ingestor = GatewayLogIngestor(gw1_logs[0])
    ingestors.append(gw1_ingestor)

    # Joins ingestor
    joins_logs = [f for f in args.logfolder.glob('*joins.log')]
    if len(joins_logs) > 1:
        raise FileExistsError(f'There should only be one *joins.log file. Instead found: {joins_logs}')
    joins_ingestor = JoinsLogIngestor(joins_logs[0])
    ingestors.append(joins_ingestor)

    # Nwks ingestor
    nwks_logs = [f for f in args.logfolder.glob('*nwks.log')]
    if len(nwks_logs) > 1:
        raise FileExistsError(f'There should only be one *nwks.log file. Instead found: {nwks_logs}')
    nwks_ingestor = NwksLogIngestor(nwks_logs[0])
    ingestors.append(nwks_ingestor)

    # Door ingestor
    door_logs = [f for f in args.logfolder.glob('*door.log')]
    if len(nwks_logs) > 1:
        raise FileExistsError(f'There should only be one *door.log file. Instead found: {door_logs}')
    door_ingestor = DoorLogIngestor(door_logs[0])
    ingestors.append(door_ingestor)

    # Muxs ingestor
    muxs_logs = [f for f in args.logfolder.glob('*muxs.log')]
    if len(muxs_logs) > 1:
        raise FileExistsError(f'There should only be one *muxs.log file. Instead found: {muxs_logs}')
    muxs_ingestor = MuxsLogIngestor(muxs_logs[0])
    ingestors.append(muxs_ingestor)

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
    print()
    

if __name__ == "__main__":
    logger.info('asdf')
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