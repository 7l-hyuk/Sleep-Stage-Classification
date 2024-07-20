import os
import sys
from datetime import datetime
import hydra
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['HYDRA_FULL_ERROR'] = "1"
from src.model import Model
from src.data import make_train_df
from src.train import X_y_split


@hydra.main(
        version_base=None,
        config_path="../conf/Bpass_Ftrans",
        config_name="P20"
        )
def main(cfg):
    column_select = cfg.train_data_setting.column_select
    band_filter = cfg.train_data_setting.band_filter
    fourier_transform = cfg.train_data_setting.fourier_transform
    mean = cfg.train_data_setting.mean,
    median = cfg.train_data_setting.median,
    max = cfg.train_data_setting.max
    min = cfg.train_data_setting.min
    std = cfg.train_data_setting.std
    var = cfg.train_data_setting.var
    peak = cfg.train_data_setting.peak
    pre_next_rate = cfg.train_data_setting.pre_next_rate,
    epoch = cfg.train_data_setting.epoch

    train_num = cfg.train_data_setting.id_select
    end_id = 152
    model = Model()
    model.train_test(cfg)
    accuracys = []
    f1_weighteds = []
    f1_macros = []
    for id in range(train_num, end_id+1):
        try:
            test_df = make_train_df(
                [id],
                column_select,
                band_filter=band_filter,
                fourier_transform=fourier_transform,
                mean=mean,
                median=median,
                max=max,
                min=min,
                std=std,
                var=var,
                peak=peak,
                pre_next_rate=15,
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
        except:
            print(f'ID: {id} has somthing problems!')

    time = datetime.now()
    name = time.strftime("%Y%m%d_%H%M%S")
    BASE_PATH = f'./sleep_edf/experiment/Performance/{name}'
    os.mkdir(BASE_PATH)

    metric = {
        'accuracy': accuracys,
        'weighted_f1_score': f1_weighteds,
        'macro_f1_score': f1_macros
        }
    for m in metric:
        plt.figure(figsize=(15, 5))
        plt.plot(
            range(train_num, train_num+len(metric[m])),
            [acc[1] for acc in metric[m]],
            color='black',
            marker="."
            )
        plt.xlabel('patient id', fontdict={'size': 16})
        plt.ylabel('accuracy', fontdict={'size': 16})
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.savefig(BASE_PATH + f'/{m}.png')

    with open(
        BASE_PATH + f"/Performance_Evaluation({name}).tex", "w"
    ) as file:
        file.write(
            '''
\\documentclass{article}
\\usepackage{booktabs}
\\usepackage{siunitx}
\\usepackage{geometry}
\\usepackage{graphicx}
\\usepackage{float}
\\geometry{a4paper, margin=1in}
\\title{Performace Evaluations}
\\author{GiHyuk Kwon}

\\begin{document}

\\begin{table}[H]
\\centering
\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}cccccc}
\\toprule
\\textbf{ID} & \\textbf{Accuracy (\\%)} & \\textbf{ID} & \\textbf{Accuracy (\\%)} & \\textbf{ID} & \\textbf{Accuracy (\%)} \\\\
\\midrule
    ''')
        for i in range(len(accuracys)//3):
            file.write(f'{accuracys[i*3][0]} & {accuracys[i*3][1]*100:.2f} & {accuracys[1+i*3][0]} & {accuracys[1+i*3][1]*100:.2f} & {accuracys[2+i*3][0]} & {accuracys[2+i*3][1]*100:.2f} \\\\\n')
        if len(accuracys) % 3 != 0:
            end_line = []
            for acc in accuracys[-(len(accuracys) % 3):]:
                end_line += [str(acc[0]), str(round(acc[1]*100, 2))]
            line = " & ".join(end_line) + " &" * (4//((len(accuracys) % 3))) + " \\\\\n"
            file.write(line)
        file.write(
            '''
\\bottomrule    
\\end{tabular*}
\\caption{Prediction Accuracy for Patients}
\\label{tab:accuracy}
\\end{table}
''')

        file.write(
            '''
\\begin{table}[H]
\\centering
\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lr}
\\toprule
\\textbf{Statistic} & \\textbf{Value (\\%)} \\\\
\\midrule
    ''')

        metric = [acc[1] for acc in accuracys]
        file.write(
            f'''
Mean & {np.mean(metric)*100:.2f} \\\\
Median & {np.median(metric)*100:.2f} \\\\
Standard Deviation & {np.std(metric)*100:.2f} \\\\
Minimum & {np.min(metric)*100:.2f} \\\\
Maximum & {np.max(metric)*100:.2f} \\\\
''')
        file.write(
            '''
\\bottomrule
\\end{tabular*}
\\caption{Statistics of Accuracy}
\\label{tab:stats}
\\end{table}
''')

        file.write(
            '''
\\begin{table}[H]
\\centering
\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}cccccc}
\\toprule
\\textbf{ID} & \\textbf{Accuracy (\\%)} & \\textbf{ID} & \\textbf{Accuracy (\\%)} & \\textbf{ID} & \\textbf{Accuracy (\%)} \\\\
\\midrule
    ''')
        for i in range(len(f1_weighteds)//3):
            file.write(f'{f1_weighteds[i*3][0]} & {f1_weighteds[i*3][1]*100:.2f} & {f1_weighteds[1+i*3][0]} & {f1_weighteds[1+i*3][1]*100:.2f} & {f1_weighteds[2+i*3][0]} & {f1_weighteds[2+i*3][1]*100:.2f} \\\\\n')
        if len(f1_weighteds) % 3 != 0:
            end_line = []
            for acc in f1_weighteds[-(len(f1_weighteds) % 3):]:
                end_line += [str(acc[0]), str(round(acc[1]*100, 2))]
            line = " & ".join(end_line) + " &" * (4//((len(f1_weighteds) % 3))) + " \\\\\n"
            file.write(line)
        file.write(
            '''
\\bottomrule    
\\end{tabular*}
\\caption{Prediction Weighted F1-score for Patients}
\\label{tab:weighted}
\\end{table}
''')
        file.write(
            '''
\\begin{table}[H]
\\centering
\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lr}
\\toprule
\\textbf{Statistic} & \\textbf{Value (\\%)} \\\\
\\midrule
    ''')
        
        metric = [acc[1] for acc in f1_weighteds]
        file.write(
            f'''
Mean & {np.mean(metric)*100:.2f} \\\\
Median & {np.median(metric)*100:.2f} \\\\
Standard Deviation & {np.std(metric)*100:.2f} \\\\
Minimum & {np.min(metric)*100:.2f} \\\\
Maximum & {np.max(metric)*100:.2f} \\\\
''')
        file.write(
            '''
\\bottomrule
\\end{tabular*}
\\caption{Statistics of Weighted F1-score}
\\label{tab:stats}
\\end{table}
''')

        file.write(
            '''
\\begin{table}[H]
\\centering
\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}cccccc}
\\toprule
\\textbf{ID} & \\textbf{Accuracy (\\%)} & \\textbf{ID} & \\textbf{Accuracy (\\%)} & \\textbf{ID} & \\textbf{Accuracy (\%)} \\\\
\\midrule
    ''')
        
        for i in range(len(f1_macros)//3):
            file.write(f'{f1_macros[i*3][0]} & {f1_macros[i*3][1]*100:.2f} & {f1_macros[1+i*3][0]} & {f1_macros[1+i*3][1]*100:.2f} & {f1_macros[2+i*3][0]} & {f1_macros[2+i*3][1]*100:.2f} \\\\\n')
        if len(f1_macros) % 3 != 0:
            end_line = []
            for acc in f1_macros[-(len(f1_macros) % 3):]:
                end_line += [str(acc[0]), str(round(acc[1]*100, 2))]
            line = " & ".join(end_line) + " &" * (4//((len(f1_macros) % 3))) + " \\\\\n"
            file.write(line)

        file.write(
            '''
\\bottomrule    
\\end{tabular*}
\\caption{Prediction Macro F1-score for Patients}
\\label{tab:weighted}
\\end{table}
''')
        file.write(
            '''
\\begin{table}[H]
\\centering
\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lr}
\\toprule
\\textbf{Statistic} & \\textbf{Value (\\%)} \\\\
\\midrule
    ''')

        metric = [acc[1] for acc in f1_macros]
        file.write(
            f'''
Mean & {np.mean(metric)*100:.2f} \\\\
Median & {np.median(metric)*100:.2f} \\\\
Standard Deviation & {np.std(metric)*100:.2f} \\\\
Minimum & {np.min(metric)*100:.2f} \\\\
Maximum & {np.max(metric)*100:.2f} \\\\
''')
        file.write(
            '''
\\bottomrule
\\end{tabular*}
\\caption{Statistics of Macro F1-score}
\\label{tab:stats}
\\end{table}
''')
        file.write(
            '''
\\begin{figure}[H]
\\centering
\\includegraphics[width=\\textwidth]{accuracy.png}
\\caption{Accuracy for Each Patient}
\\label{tab:perfomance}
\\end{figure}

\\begin{figure}[H]
\\centering
\\includegraphics[width=\\textwidth]{weighted_f1_score.png}
\\caption{Weighted F1-score for Each Patient}
\\label{tab:perfomance}
\\end{figure}

\\begin{figure}[H]
\\centering
\\includegraphics[width=\\textwidth]{macro_f1_score.png}
\\caption{Macro F1-score for Each Patient}
\\label{tab:perfomance}
\\end{figure}

\\end{document}
    ''')


if __name__ == '__main__':
    main()
