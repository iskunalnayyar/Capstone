"""
Messing with the Data file
"""

import os
from random import randint

import pandas as pd
from sklearn import preprocessing
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
        drop_list = ['EngineVersion', 'AppVersion', 'AvSigVersion', 'OsBuildLab', 'Census_OSVersion', 'PuaMode',
                     'Census_PowerPlatformRoleName',
                     'Census_ProcessorClass', 'DefaultBrowsersIdentifier', 'Census_IsFlightingInternal',
                     'Census_InternalBatteryType'
            , 'Census_ThresholdOptIn', 'Census_IsWIMBootEnabled', 'SmartScreen', 'OrganizationIdentifier',
                     'AutoSampleOptIn', 'SMode', 'Census_IsPortableOperatingSystem', 'Census_DeviceFamily',
                     'UacLuaenable',
                     'Census_IsVirtualDevice', 'SkuEdition', 'Census_PrimaryDiskTypeName', 'Census_FlightRing']
        chunk_list = []
        drop_these = ['IsBeta', 'IsSxsPassiveMode', 'AVProductsEnabled', 'HasTpm', 'IsProtected',
                      'OsPlatformSubRelease',
                      'Census_HasOpticalDiskDrive', 'Census_IsFlightsDisabled', 'Census_IsPenCapable', 'OsVer',
                      'Census_MDC2FormFactor',
                      'Census_IsAlwaysOnAlwaysConnectedCapable', 'Wdft_IsGamer', 'ProductName', 'Platform', 'Processor',
                      'Census_OSBranch']

        test_df = pd.read_csv(os.path.join(self.dir, self.file), dtype=col_dtypes, index_col='idNumber',
                              chunksize=1000000)
        for chunk in test_df:
            chunk_list.append(chunk)

        test_df_concat = pd.concat(chunk_list)
        print(test_df_concat.shape)
        print(test_df_concat.columns.to_list())

        print()
        # prints the missing values as per column as a percentage
        # print((test_df_concat.isnull().sum()/test_df_concat.shape[0]).sort_values(ascending=False))

        # test_df_concat = test_df_concat.drop(columns=drop_list, axis=1)
        cols_list = test_df_concat.columns.to_list()
        test_df_concat = test_df_concat.dropna()
        print(test_df_concat.shape)
        # test_df = test_df.select_dtypes(exclude=['object'])

        # ignore these - 'EngineVersion', 'AppVersion', 'AvSigVersion', 'OsBuildLab', 'Census_OSVersion'

        print(test_df_concat.columns.to_list())

        # precautionary measure
        test_df_concat = test_df_concat.dropna()
        # y = test_df_concat['HasDetections']
        y = test_df_concat['class']
        # scaled_df = test_df_concat.drop(columns='HasDetections')
        scaled_df = test_df_concat.drop(columns='class')
        # scaled_df = self.conver_categorical_to_int(scaled_df, col_dtypes)
        scaled_df = self.scaling_data(scaled_df)
        print(scaled_df.shape)

        X_train, X_test, y_train, y_test = self.split_train_test(scaled_df, y)
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        return X_train, X_test, y_train, y_test, cols_list

    def split_train_test(self, x, y):
        """
        Splits train, test given a data frame
        :param dataframe: dataframe to be split
        :return: X_train, X_test, y_train, y_test
        """
        return train_test_split(x, y, test_size=0.20, random_state=0)

    def conver_categorical_to_int(self, dataframe, col_types):
        """
        This is gonna map all unique values in column to keys
        :param dataframe: input dataframe
        :param col_types:
        :return:
        """
        mapping = dict()
        columns_list = dataframe.columns.to_list()
        print("Mapping...")
        for key, val in col_types.items():
            temp_dict = dict()
            if key in columns_list:
                if val == 'object':
                    temp_list = dataframe[key].unique()
                    print(key, len(temp_list))

                    for x in temp_list:
                        value = randint(0, 500)
                        if value not in mapping.values():
                            temp_dict[x] = value
                    dataframe[key].replace(temp_dict, inplace=True)
            mapping[key] = temp_dict

        print(dataframe.head())

        print()
        print(mapping.keys())
        dataframe.head()
        print("Mapping done!")
        return dataframe

    def scaling_data(self, dataframe):
        """
        Performing normalization
        :param dataframe:
        :return:
        """
        min_max_scaler = preprocessing.MinMaxScaler()
        scaled_df = min_max_scaler.fit_transform(dataframe)
        df = pd.DataFrame(scaled_df)
        return df


def main():
    md = MessingWithData('/Users/k.n./Downloads/', 'breast-cancer-wisconsin.csv')
    md.read_file()


if __name__ == '__main__':
    main()
