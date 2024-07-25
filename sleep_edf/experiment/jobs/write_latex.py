import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from src.model.model import Model
from src.data.dataframe import make_train_df
from src.model.train import X_y_split


def _make_acc_table(func):
    def wrapper(file: str, metrics: list, metric_name: str):
        file.write(
            '''
\\begin{table}[H]
\\centering
\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}cccccc}
\\toprule
\\textbf{ID} & \\textbf{Accuracy (\\%)} & \\textbf{ID} & \\textbf{Accuracy (\\%)} & \\textbf{ID} & \\textbf{Accuracy (\%)} \\\\
\\midrule
    ''')
        func(file, metrics)
        file.write(
            f'''
\\bottomrule    
\\end{{tabular*}}
\\caption{{Prediction {metric_name} for Patients}}
\\label{{tab:accuracy}}
\\end{{table}}
''')
    return wrapper


def _make_stats_table(func):
    def wrapper(file: str, metrics: list, metric_name: str):
        file.write(
            '''
\\begin{table}[H]
\\centering
\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lr}
\\toprule
\\textbf{Statistic} & \\textbf{Value (\\%)} \\\\
\\midrule
    ''')
        func(file, metrics)
        file.write(
            f'''
\\bottomrule
\\end{{tabular*}}
\\caption{{Statistics of {metric_name}}}
\\label{{tab:stats}}
\\end{{table}}
''')
    return wrapper


@_make_acc_table
def write_accuracy_line(file, metrics):
    for i in range(len(metrics)//3):
        file.write(
            f'{metrics[i*3][0]} & {metrics[i*3][1]*100:.2f} & {metrics[1+i*3][0]} & {metrics[1+i*3][1]*100:.2f} & {metrics[2+i*3][0]} & {metrics[2+i*3][1]*100:.2f} \\\\\n')
    if len(metrics) % 3 != 0:
        end_line = []
        for acc in metrics[-(len(metrics) % 3):]:
            end_line += [str(acc[0]), str(round(acc[1]*100, 2))]
        line = " & ".join(end_line) + " &" * (4//((len(metrics) % 3))) + " \\\\\n"
        file.write(line)


@_make_stats_table
def write_stats_table(file, metrics):
    metric = [metric[1] for metric in metrics]
    file.write(
        f'''
Mean & {np.mean(metric)*100:.2f} \\\\
Median & {np.median(metric)*100:.2f} \\\\
Standard Deviation & {np.std(metric)*100:.2f} \\\\
Minimum & {np.min(metric)*100:.2f} \\\\
Maximum & {np.max(metric)*100:.2f} \\\\
''')


def save_metrics_graph(
        metrics: dict[str, list],
        start: int,
        end: int,
        path: str
        ):
    for metric in metrics:
        plt.figure(figsize=(15, 5))
        plt.plot(
            range(start, end+1),
            [acc[1] for acc in metrics[metric]],
            color='black',
            marker="."
            )
        plt.xlabel('patient id', fontdict={'size': 16})
        plt.ylabel('accuracy', fontdict={'size': 16})
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.savefig(path + f'/{metric}.png')


def make_metrics(cfg, start, end):
    column_select = cfg.train_data_setting.column_select
    band_filter = cfg.train_data_setting.band_filter
    fourier_transform = cfg.train_data_setting.fourier_transform
    mean = cfg.train_data_setting.mean
    median = cfg.train_data_setting.median
    max_ = cfg.train_data_setting.max_
    min_ = cfg.train_data_setting.min_
    std = cfg.train_data_setting.std
    skewness = cfg.train_data_setting.skewness
    kurtosis_ = cfg.train_data_setting.kurtosis_
    mmd_ = cfg.train_data_setting.mmd_
    esis_ = cfg.train_data_setting.esis_
    pre_next_rate = cfg.train_data_setting.pre_next_rate
    epoch = cfg.train_data_setting.epoch
    model = Model()
    model.train_test(cfg)

    accuracys = []
    f1_weighteds = []
    f1_macros = []

    for id in range(start, end+1):
        try:
            test_df = make_train_df(
                [id],
                column_select,
                band_filter=band_filter,
                fourier_transform=fourier_transform,
                mean=mean,
                median=median,
                max_=max_,
                min_=min_,
                std=std,
                skewness=skewness,
                kurtosis_=kurtosis_,
                esis_=esis_,
                mmd_=mmd_,
                pre_next_rate=pre_next_rate,
                epoch=epoch
            )
            X, y, _ = X_y_split(test_df)
            y_pred = model.model.predict(X)
            acc = accuracy_score(y, y_pred)
            f1_weighted = f1_score(y, y_pred, average="weighted")
            f1_macro = f1_score(y, y_pred, average="macro")
            accuracys += [(id, acc)]
            f1_weighteds += [(id, f1_weighted)]
            f1_macros += [(id, f1_macro)]
            print(f'Pass the ID: {id}, acc: {acc*100} %')
        except Exception as e:
            print(e)
            print(f'ID: {id} has somthing problems!')
    return accuracys, f1_weighteds, f1_macros


def write_metrics_grpah(file):
    file.write(
            '''
\\begin{figure}[H]
\\centering
\\includegraphics[width=\\textwidth]{Accuracy.png}
\\caption{Accuracy for Each Patient}
\\label{tab:perfomance}
\\end{figure}

\\begin{figure}[H]
\\centering
\\includegraphics[width=\\textwidth]{Weighted F1-score.png}
\\caption{Weighted F1-score for Each Patient}
\\label{tab:perfomance}
\\end{figure}

\\begin{figure}[H]
\\centering
\\includegraphics[width=\\textwidth]{Macro F1-score.png}
\\caption{Macro F1-score for Each Patient}
\\label{tab:perfomance}
\\end{figure}
    ''')
