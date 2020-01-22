"""
Messing with the Data file
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split


class MessingWithData:

    def __init__(self, dir, filename):
        self.dir = dir  # directory
        self.file = filename  # filename

    def read_file(self):
        # do this to fix reading problem to read_csv - dtype={"user_id": int, "username": object}
        col_dtypes = {
            'MachineIdentifier': 'object',
            'ProductName': 'object',
            'EngineVersion': 'object',
            'AppVersion': 'object',
            'AvSigVersion': 'object',
            'IsBeta': 'int8',
            'RtpStateBitfield': 'float16',
            'IsSxsPassiveMode': 'int8',
            'DefaultBrowsersIdentifier': 'float32',  # was 'float16'
            'AVProductStatesIdentifier': 'float32',
            'AVProductsInstalled': 'float16',
            'AVProductsEnabled': 'float16',
            'HasTpm': 'int8',
            'CountryIdentifier': 'int16',
            'CityIdentifier': 'float32',
            'OrganizationIdentifier': 'float16',
            'GeoNameIdentifier': 'float16',
            'LocaleEnglishNameIdentifier': 'int16',  # was 'int8'
            'Platform': 'object',
            'Processor': 'object',
            'OsVer': 'object',
            'OsBuild': 'int16',
            'OsSuite': 'int16',
            'OsPlatformSubRelease': 'object',
            'OsBuildLab': 'object',
            'SkuEdition': 'object',
            'IsProtected': 'float16',
            'AutoSampleOptIn': 'int8',
            'PuaMode': 'object',
            'SMode': 'float16',
            'IeVerIdentifier': 'float16',
            'SmartScreen': 'object',
            'Firewall': 'float16',
            'UacLuaenable': 'float64',  # was 'float32'
            'Census_MDC2FormFactor': 'object',
            'Census_DeviceFamily': 'object',
            'Census_OEMNameIdentifier': 'float32',  # was 'float16'
            'Census_OEMModelIdentifier': 'float32',
            'Census_ProcessorCoreCount': 'float16',
            'Census_ProcessorManufacturerIdentifier': 'float16',
            'Census_ProcessorModelIdentifier': 'float32',  # was 'float16'
            'Census_ProcessorClass': 'object',
            'Census_PrimaryDiskTotalCapacity': 'float64',  # was 'float32'
            'Census_PrimaryDiskTypeName': 'object',
            'Census_SystemVolumeTotalCapacity': 'float64',  # was 'float32'
            'Census_HasOpticalDiskDrive': 'int8',
            'Census_TotalPhysicalRAM': 'float32',
            'Census_ChassisTypeName': 'object',
            'Census_InternalPrimaryDiagonalDisplaySizeInInches': 'float32',  # was 'float16'
            'Census_InternalPrimaryDisplayResolutionHorizontal': 'float32',  # was 'float16'
            'Census_InternalPrimaryDisplayResolutionVertical': 'float32',  # was 'float16'
            'Census_PowerPlatformRoleName': 'object',
            'Census_InternalBatteryType': 'object',
            'Census_InternalBatteryNumberOfCharges': 'float64',  # was 'float32'
            'Census_OSVersion': 'object',
            'Census_OSArchitecture': 'object',
            'Census_OSBranch': 'object',
            'Census_OSBuildNumber': 'int16',
            'Census_OSBuildRevision': 'int32',
            'Census_OSEdition': 'object',
            'Census_OSSkuName': 'object',
            'Census_OSInstallTypeName': 'object',
            'Census_OSInstallLanguageIdentifier': 'float16',
            'Census_OSUILocaleIdentifier': 'int16',
            'Census_OSWUAutoUpdateOptionsName': 'object',
            'Census_IsPortableOperatingSystem': 'int8',
            'Census_GenuineStateName': 'object',
            'Census_ActivationChannel': 'object',
            'Census_IsFlightingInternal': 'float16',
            'Census_IsFlightsDisabled': 'float16',
            'Census_FlightRing': 'object',
            'Census_ThresholdOptIn': 'float16',
            'Census_FirmwareManufacturerIdentifier': 'float16',
            'Census_FirmwareVersionIdentifier': 'float32',
            'Census_IsSecureBootEnabled': 'int8',
            'Census_IsWIMBootEnabled': 'float16',
            'Census_IsVirtualDevice': 'float16',
            'Census_IsTouchEnabled': 'int8',
            'Census_IsPenCapable': 'int8',
            'Census_IsAlwaysOnAlwaysConnectedCapable': 'float16',
            'Wdft_IsGamer': 'float16',
            'Wdft_RegionIdentifier': 'float16',
            'HasDetections': 'float32',
        }

        # drop unnecessary features - have too many missing values to be of any use
        drop_list = {'PuaMode', 'Census_ProcessorClass', 'Census_InternalBatteryType', 'Census_IsFlightingInternal',
                     'Census_ThresholdOptIn', 'Census_IsWIMBootEnabled', 'SmartScreen', 'DefaultBrowsersIdentifier'}

        test_df = pd.read_csv(os.path.join(self.dir, self.file), dtype=col_dtypes)
        test_df = test_df.drop(columns=drop_list)

        print(test_df.shape)
        print(test_df)
        print(test_df.describe())
        print(test_df.info())
        X_train, X_test, y_train, y_test = self.train_test_split(test_df)

    def train_test_split(self, dataframe):
        """
        Splits train, test given a data frame
        :param dataframe: dataframe to be split
        :return: X_train, X_test, y_train, y_test
        """
        return train_test_split(dataframe, test_size=0.20, random_state=42)

    def convert_categorical_to_int(self, dataframe):
        """
        Converts the categorical, Object features to integer
        :param dataframe: the datafile in csv format
        :return: dataframe with integer values
        """

        pass


def main():
    m1 = MessingWithData('/Users/k.n./Downloads/microsoft-malware-prediction', 'train.csv')
    m1.read_file()


if __name__ == '__main__':
    main()
