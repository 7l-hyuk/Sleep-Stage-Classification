from glob import glob
from pyedflib import highlevel
import pandas as pd
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq


class BaseDataSet:
    '''
    Make base data like eeg or sleep stage by using sleep-edf dataset.

    Parameters
    ----------
    base_path : str, default './sleep-edf-database-expanded-1.0.0/'
        The path of sleep_edf dataset.
    '''
    def __init__(
            self,
            base_path: str = 'C:/Users/7lhyu/Documents/sleep-edf/sleep_edf/sleep-edf-database-expanded-1.0.0/',
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
            Select specific id in the eeg dataset
        column_select : list[int], default None
            Select specific features you want to import

        Returns
        -------
        pandas.DataFrame
            Dataframe that contain eeg data

        Examples
        --------
        >>> dataset = BaseDataSet()
        >>> eeg_df = dataset.make_eed_df(0)
        >>> eeg_df.head()

        >>> eef_df = dataset.make_eeg_df(0, [0, 1])
        >>> eeg_df.head()

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

        parameters
        ----------
        id : int

        Returns
        -------
        pandas.DataFrame
            Dataframe that contain time, duration and sleep_stage
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

    def band_pass_filter(self, segment: pd.Series) -> list[np.ndarray]:
        eeg_bands = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 12), 'Beta': (12, 30), 'Gamma': (30, 45)}
        filtered_features = []
        for band in eeg_bands:
            low, high = eeg_bands[band]
            band_pass_filter = signal.butter(
                3, [low, high],
                btype='bandpass',
                fs=100,
                output='sos'
                )
            filtered = signal.sosfilt(band_pass_filter, segment)
            filtered_features += [filtered]
        return filtered_features

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
            filtered += self.band_pass_filter(eeg_df[column])

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
            max: bool = True,
            min: bool = True,
            std: bool = True,
            var: bool = True,
            peak: bool = True,
            ):
        if init_df:
            self.df = self.make_eeg_df(init_df[0], init_df[1])
            self.columns = self.df.columns
            self.selected_id = init_df[0]
            self.seleted_column = init_df[1]

        extracted_df = pd.DataFrame()
        df = self.df
        columns = self.columns
        for column in columns:
            datas = np.zeros((rows := len(df)//3000, 3000))

            for row in range(0, rows):
                datas[row] = df[column][row*3000:(row+1)*3000].values

            # 왜도 첨도 형상계수
            if mean:
                extracted_df[f'{column}_mean'] = np.mean(datas, axis=1)
            if median:
                extracted_df[f'{column}_median'] = np.median(datas, axis=1)
            if max:
                extracted_df[f'{column}_max'] = np.max(datas, axis=1)
            if min:
                extracted_df[f'{column}_min'] = np.min(datas, axis=1)
            if std:
                extracted_df[f'{column}_std'] = np.std(datas, axis=1)
            if var:
                extracted_df[f'{column}_var'] = np.var(datas, axis=1)
            if peak:
                extracted_df[f'{column}_peak'] = \
                    np.abs(np.max(datas, axis=1))+np.abs(np.min(datas, axis=1))

        self.df = extracted_df
        self.columns = extracted_df.columns
        return extracted_df

    def extract_fourier_transformed_statistic_values(
            self,
            segment: np.ndarray,
            sample_rate: float = 100.0,
            low_freq: float = None,
            high_freq: float = None
            ) -> np.ndarray:
        if (low_freq is not None) and (high_freq is not None):
            n = len(segment)
            yf = fft(segment)
            xf = fftfreq(n, 1 / sample_rate)
            amplitude = np.abs(yf[:n // 2])
            freq_mask = (xf[:n // 2] >= low_freq) & (xf[:n // 2] <= high_freq)
            selected_amplitude = amplitude[freq_mask]
            return np.mean(selected_amplitude),np.median(selected_amplitude),np.min(selected_amplitude),np.max(selected_amplitude),np.std(selected_amplitude)
        n = len(segment)
        yf = fft(segment)
        # xf = fftfreq(n, 1 / sample_rate)
        amplitude = np.abs(yf[:n // 2])
        return np.mean(amplitude), np.median(amplitude), np.min(amplitude), np.max(amplitude), np.std(amplitude)

    def make_fourier_transformed_df(
        self,
        id_select: int,
        column_select: list[int],
        low_freq: float = None,
        high_freq: float = None
    ) -> pd.DataFrame:
        df = self.make_eeg_df(id_select, column_select)
        columns = df.columns

        extracted_df = pd.DataFrame()
        for column in columns:
            datas = np.zeros((rows := len(df)//3000, 3000))

            for row in range(0, rows):
                datas[row] = df[column][row*3000:(row+1)*3000].values
            transformed_datas = np.array([
                self.extract_fourier_transformed_statistic_values(
                    data,
                    low_freq=low_freq,
                    high_freq=high_freq
                    )
                for data in datas
                ])
            if (low_freq is not None) and (high_freq is not None):
                extracted_df[f'{column}_selected_fourier_transed_mean'] = \
                    transformed_datas[:, 0]
                extracted_df[f'{column}_selected_fourier_transed_median'] = \
                    transformed_datas[:, 1]
                extracted_df[f'{column}_selected_fourier_transed_min'] = \
                    transformed_datas[:, 2]
                extracted_df[f'{column}_selected_fourier_transed_max'] = \
                    transformed_datas[:, 3]
                extracted_df[f'{column}_selected_fourier_transed_std'] = \
                    transformed_datas[:, 4]
            else:
                extracted_df[f'{column}_fourier_transed_mean'] = \
                    transformed_datas[:, 0]
                extracted_df[f'{column}_fourier_transed_median'] = \
                    transformed_datas[:, 1]
                extracted_df[f'{column}_fourier_transed_min'] = \
                    transformed_datas[:, 2]
                extracted_df[f'{column}_fourier_transed_max'] = \
                    transformed_datas[:, 3]
                extracted_df[f'{column}_fourier_transed_std'] = \
                    transformed_datas[:, 4]

            self.df = pd.concat([self.df, extracted_df], axis=1)
            self.columns = self.df.columns
        return extracted_df

    def make_previous_data(self):
        df = self.df
        previous_df = df.iloc[:-1, :]
        zero_row = pd.DataFrame(
            data=[[0 for _ in range(len(previous_df.columns))]],
            columns=previous_df.columns
            )
        previous_df = pd.concat([zero_row, previous_df], ignore_index=True)
        previous_df.columns = [
            f'before_{column}' for column in previous_df.columns
            ]
        df = pd.concat([previous_df, df], axis=1)
        self.df = df
        self.columns = df.columns
        return df

    def make_labels(self):
        df = self.df
        stage_df = self.make_stage_df(self.selected_id)
        sleep_stage = []
        sleep_stage_bound = [
            (stage_df['time'][i] + stage_df['duration'][i])
            for i in range(len(stage_df))
            ]
        for time in range(len(df)):
            end = (time+1)*30
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
        max: bool = True,
        min: bool = True,
        std: bool = True,
        var: bool = True,
        peak: bool = True
        ):
    train_df = pd.DataFrame()
    for id in id_select:
        dataset = FeatureEngineering()
        init = 0
        if band_filter:
            init = 1
            dataset.make_filtered_df(id_select=id, column_select=column_select)
        if not init:
            dataset.make_statistic_df(
                init_df=[id, column_select],
                mean=mean,
                median=median,
                max=max,
                min=min,
                std=std,
                var=var,
                peak=peak
                )
        else:
            dataset.make_statistic_df(
                mean=mean,
                median=median,
                max=max,
                min=min,
                std=std,
                var=var,
                peak=peak
            )
        if fourier_transform:
            dataset.make_fourier_transformed_df(
                id_select=id,
                column_select=column_select
                )
            dataset.make_fourier_transformed_df(
                id_select=id,
                column_select=column_select,
                low_freq=5,
                high_freq=30
                )

        dataset.make_previous_data()
        dataset.make_labels()
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
        dataset.df.loc[dataset.df[(dataset.df['sleep_stage'] == "Sleep stage 1")|(dataset.df['sleep_stage'] == "Sleep stage 2")]['sleep_stage'].index, "sleep_stage"] = "LS"
        dataset.df.loc[dataset.df[(dataset.df['sleep_stage'] == "Sleep stage 3")|(dataset.df['sleep_stage'] == "Sleep stage 4")]['sleep_stage'].index, "sleep_stage"] = "DS"
        train_df = pd.concat([train_df, dataset.df], ignore_index=True)
    return train_df
