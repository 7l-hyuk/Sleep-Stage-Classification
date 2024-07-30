from glob import glob

from pyedflib import highlevel
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

from src.data.stats_utils import (
    band_pass_filter,
    esis,
    mmd,
    fourier_transformed_stats_values
    )


class BaseDataSet:
    '''
    Make eeg or sleep stage data by using sleep-edf dataset.

    Parameters
    ----------
    base_path : str, default './sleep-edf-database-expanded-1.0.0/'
        The path of sleep_edf dataset.

    Attributes
    ----------
    base_path : str
    psg_paths : list
    hypnogram_paths : list
    '''
    def __init__(
            self,
            base_path: str = 'C:/Users/7lhyu/Documents/sleep-edf/sleep_edf/data/sleep-edf-database-expanded-1.0.0/'
            ):
        self.base_path = base_path
        paths = glob(base_path + '**/*.edf')
        self.psg_paths = [path for path in paths if path.endswith('PSG.edf')]
        self.hypnogram_paths = [
            path for path in paths if path.endswith('Hypnogram.edf')
            ]

    def make_eeg_df(
            self,
            id: int,
            column_select: list[int] = None
            ) -> pd.DataFrame:
        '''
        Make eeg dataframe.

        Parameters
        ----------
        id : int
            Specific ID in the EEG data
        column_select : list[int], default None
            Specific EEG channel

        Returns
        -------
        pandas.DataFrame
            Dataframe that contains selected EEG dataset

        Examples
        --------
        >>> dataset = BaseDataSet()
        >>> eef_df = dataset.make_eeg_df(0, [0, 1])
        '''
        psg = highlevel.read_edf(self.psg_paths[id])
        eeg, info = psg[0], psg[1]
        eeg_df = pd.DataFrame()

        if column_select:
            for i in column_select:
                eeg_df = pd.concat(
                    [eeg_df, pd.DataFrame(eeg[i])],
                    axis=1,
                    ignore_index=True
                    )
            eeg_df.columns = [info[i]['label'] for i in column_select]
            return eeg_df

        for i in range(len(eeg)):
            eeg_df = pd.concat(
                [eeg_df, pd.DataFrame(eeg[i])],
                axis=1,
                ignore_index=True
                )
        eeg_df.columns = [info[i]['label'] for i in range(len(info))]
        return eeg_df

    def make_stage_df(self, id: int) -> pd.DataFrame:
        '''
        Make sleep stage data

        Parameters
        ----------
        id : int

        Returns
        -------
        pandas.DataFrame
            Dataframe that contains time, duration and sleep_stage
        '''
        hypnogram = highlevel.read_edf(self.hypnogram_paths[id])
        stages = hypnogram[2]['annotations']
        stage_df = pd.DataFrame(stages)
        stage_df.columns = ['time', 'duration', 'sleep_stage']
        return stage_df


class FeatureEngineering(BaseDataSet):
    def __init__(self):
        super().__init__()
        self.df: pd.DataFrame = pd.DataFrame()
        self.columns: list[str] = list()
        self.selected_id: int = None
        self.seleted_column: list = None
        self.stats_funcs = {
            'mean': np.mean,
            'median': np.median,
            'max': np.max,
            'min': np.min,
            'std': np.std,
            'skew': skew,
            'kurtosis': kurtosis
        }

    def make_filtered_df(
            self,
            id_select: int,
            column_select: list[int, list] = None
            ) -> pd.DataFrame:
        eeg_df = self.make_eeg_df(id_select, column_select)
        self.selected_id = id_select
        self.seleted_column = column_select

        if not len(self.columns):
            BANDS = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
            for c in eeg_df.columns:
                self.columns += [f'{c}_{band}' for band in BANDS]

        filtered = []
        for column in eeg_df.columns:
            filtered += band_pass_filter(eeg_df[column])

        filtered_eeg_df = pd.DataFrame(columns=self.columns)
        for i in range(len(filtered)):
            filtered_eeg_df[self.columns[i]] = filtered[i]
        self.df = pd.concat([self.df, filtered_eeg_df], axis=1)
        self.columns = self.df.columns
        return filtered_eeg_df

    def make_statistic_df(
            self,
            init_df: list[int | list[int]] = None,
            mean: bool = True,
            median: bool = True,
            max_: bool = True,
            min_: bool = True,
            std: bool = True,
            skewness: bool = True,
            kurtosis_: bool = True,
            mmd_: bool = True,
            esis_: bool = True,
            epoch: int = 3000,
            ) -> pd.DataFrame:
        if init_df:
            self.df = self.make_eeg_df(init_df[0], init_df[1])
            self.columns = self.df.columns
            self.selected_id = init_df[0]
            self.seleted_column = init_df[1]

        df = self.df.copy()
        columns = self.columns
        for column in columns:
            self.df[f'diff0 {column}'] = df[column].diff()
            self.df.loc[0, f'diff0 {column}'] = 0
            
        self.columns = self.df.columns
        stats_types = np.array(list(self.stats_funcs.keys()))[[
            mean, median, max_, min_, std, skewness, kurtosis_
            ]]
        extracted_df = pd.DataFrame()
        df = self.df
        columns = self.columns

        for column in columns:
            datas = np.zeros((rows := len(df)//epoch, epoch))

            for row in range(0, rows):
                datas[row] = df[column][row*epoch:(row+1)*epoch].values

            for stats_type in stats_types:
                extracted_df[f'{column}_{stats_type}'] = \
                    self.stats_funcs[stats_type](datas, axis=1)
            if mmd_:
                extracted_df[f'{column}_mmd'] = \
                    [mmd(data) for data in datas]
            if esis_:
                band_name = column[column.find('_')+1:]
                extracted_df[f'{column}_esis'] = \
                    [esis(data, band_name) for data in datas]

        self.df = extracted_df
        self.columns = extracted_df.columns
        return extracted_df

    def make_fourier_transformed_df(
        self,
        id_select: int,
        column_select: list[int],
        low_freq: list[float] = None,
        high_freq: list[float] = None,
        epoch: int = 3000
    ) -> pd.DataFrame:
        df = self.make_eeg_df(id_select, column_select)
        columns = df.columns

        extracted_df = pd.DataFrame()
        stats = ['mean', 'median', 'min', 'max', 'std', "skew", "kurtosis"]
        for column in columns:
            datas = np.zeros((rows := len(df)//epoch, epoch))

            for row in range(0, rows):
                datas[row] = df[column][row*epoch:(row+1)*epoch].values

            if (low_freq is not None) and (high_freq is not None):
                for i in range(len(low_freq)):
                    transformed_datas = np.array([
                        fourier_transformed_stats_values(
                            data,
                            low_freq=low_freq[i],
                            high_freq=high_freq[i]
                            )
                        for data in datas
                        ])
                    for j in range(len(stats)):
                        extracted_df[
                            f'{column}_{low_freq[i]}~{high_freq[i]}\
                                _fourier_transed_{stats[j]}'
                            ] = transformed_datas[:, j]
            else:
                transformed_datas = np.array([
                    fourier_transformed_stats_values(data)
                    for data in datas
                    ])
                for i in range(len(stats)):
                    extracted_df[
                        f'{column}_fourier_transed_{stats[i]}'
                        ] = transformed_datas[:, i]

            self.df = pd.concat([self.df, extracted_df], axis=1)
            self.columns = self.df.columns
        return extracted_df

    def make_previous_next_data(self, pre_next_rate: int = 5) -> pd.DataFrame:
        df = self.df
        return_df = df.copy()
        for r in range(1, pre_next_rate+1):
            zero_rows = pd.DataFrame(
                data=np.zeros((r, len(df.columns))),
                columns=df.columns
                )
            previous_df = df.iloc[:-r, :]
            previous_df = pd.concat(
                [zero_rows, previous_df],
                ignore_index=True
                )
            previous_df.columns = [
                f'{r}previous_{column}'
                for column in previous_df.columns
            ]
            next_df = df.iloc[r:, :]
            next_df = pd.concat(
                [next_df, zero_rows],
                ignore_index=True
                )
            next_df.columns = [
                f'{r}next_{column}'
                for column in next_df.columns
            ]
            return_df = pd.concat([previous_df, next_df, return_df], axis=1)
        self.df = return_df
        self.columns = return_df.columns
        return df

    def make_labels(self, epoch: int = 3000) -> pd.DataFrame:
        df = self.df
        stage_df = self.make_stage_df(self.selected_id)
        sleep_stage = []
        sleep_stage_bound = [
            (stage_df['time'][i] + stage_df['duration'][i])
            for i in range(len(stage_df))
            ]
        for time in range(len(df)):
            end = (time+1)*int(epoch//100)
            for t in range(len(sleep_stage_bound)):
                if end <= sleep_stage_bound[t]:
                    sleep_stage.append(stage_df['sleep_stage'][t])
                    break
        df['sleep_stage'] = sleep_stage
        self.df = df
        self.columns = df.columns
        return df


def make_train_df(
        id_select: list[int],
        column_select: list[int],
        band_filter: bool = True,
        fourier_transform: bool = True,
        mean: bool = True,
        median: bool = True,
        max_: bool = True,
        min_: bool = True,
        std: bool = True,
        skewness: bool = True,
        kurtosis_: bool = True,
        mmd_: bool = True,
        esis_: bool = True,
        pre_next_rate: int = 5,
        epoch: int = 3000
        ) -> pd.DataFrame:
    train_df = pd.DataFrame()
    for id in id_select:
        dataset = FeatureEngineering()
        initialized = False
        if band_filter:
            initialized = True
            dataset.make_filtered_df(id_select=id, column_select=column_select)
        if not initialized:
            dataset.make_statistic_df(
                init_df=[id, column_select],
                mean=mean,
                median=median,
                max_=max_,
                min_=min_,
                std=std,
                skewness=skewness,
                kurtosis_=kurtosis_,
                mmd_=mmd_,
                esis_=esis_,
                epoch=epoch
                )
        else:
            dataset.make_statistic_df(
                mean=mean,
                median=median,
                max_=max_,
                min_=min_,
                std=std,
                skewness=skewness,
                kurtosis_=kurtosis_,
                mmd_=mmd_,
                esis_=esis_,
                epoch=epoch
            )
        if fourier_transform:
            dataset.make_fourier_transformed_df(
                id_select=id,
                column_select=column_select,
                epoch=epoch
                )
            dataset.make_fourier_transformed_df(
                id_select=id,
                column_select=column_select,
                low_freq=[0.5, 4, 8, 12, 30],
                high_freq=[4, 8, 12, 30, 45],
                epoch=epoch
                )

        dataset.make_previous_next_data(pre_next_rate=pre_next_rate)
        dataset.make_labels(epoch=epoch)
        dataset.df.drop(
            index=dataset.df[
                (dataset.df['sleep_stage'] != 'Sleep stage W') &
                (dataset.df['sleep_stage'] != 'Sleep stage 1') &
                (dataset.df['sleep_stage'] != 'Sleep stage 2') &
                (dataset.df['sleep_stage'] != 'Sleep stage 3') &
                (dataset.df['sleep_stage'] != 'Sleep stage 4') &
                (dataset.df['sleep_stage'] != 'Sleep stage R')
                ].index,
            axis=0,
            inplace=True
            )
        dataset.df.loc[
            dataset.df[(dataset.df['sleep_stage'] == "Sleep stage 1")
                       | (dataset.df['sleep_stage'] == "Sleep stage 2")
                       ]['sleep_stage'].index, "sleep_stage"] = "LS"
        dataset.df.loc[
            dataset.df[(dataset.df['sleep_stage'] == "Sleep stage 3")
                       | (dataset.df['sleep_stage'] == "Sleep stage 4")
                       ]['sleep_stage'].index, "sleep_stage"] = "DS"
        train_df = pd.concat([train_df, dataset.df], ignore_index=True)
    return train_df
