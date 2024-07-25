import os

from datetime import datetime
import hydra

import setting
from jobs.write_latex import (
    make_metrics,
    save_metrics_graph,
    write_accuracy_line,
    write_stats_table,
    write_metrics_grpah
    )


@hydra.main(
        version_base=None,
        config_path="../conf/Bpass_Ftrans",
        config_name="P40"
        )
def main(cfg):
    start_id = cfg.train_data_setting.id_select
    END_ID = 152

    accuracys, f1_weighteds, f1_macros = \
        make_metrics(cfg, start_id, END_ID)
    metrics = {
        'Accuracy': accuracys,
        'Weighted F1-score': f1_weighteds,
        'Macro F1-score': f1_macros
        }

    time = datetime.now()
    name = time.strftime("%Y%m%d_%H%M%S")
    BASE_PATH = f'./Performance/{name}'
    os.mkdir(BASE_PATH)

    save_metrics_graph(metrics, start_id, END_ID, BASE_PATH)

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
''')
        for metric in metrics:
            write_accuracy_line(file, metrics[metric], metric)
            write_stats_table(file, metrics[metric], metric)
        write_metrics_grpah(file)

        file.write('\\end{document}')


if __name__ == '__main__':
    main()
