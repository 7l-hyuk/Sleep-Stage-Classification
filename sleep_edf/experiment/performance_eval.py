import os
import sys
from datetime import datetime
import hydra
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['HYDRA_FULL_ERROR'] = "1"
from Performance.jobs.write_latex import (
    make_metrics,
    save_metrics_graph,
    write_accuracy_line,
    write_stats_table,
    write_metrics_grpah
    )


@hydra.main(
        version_base=None,
        config_path="../conf/Bpass_Ftrans",
        config_name="P1"
        )
def main(cfg):
    start_id = cfg.train_data_setting.id_select
    end_id = 5

    accuracys, f1_weighteds, f1_macros = \
        make_metrics(cfg, start_id, end_id)
    metrics = {
        'accuracy': accuracys,
        'weighted_f1_score': f1_weighteds,
        'macro_f1_score': f1_macros
        }

    time = datetime.now()
    name = time.strftime("%Y%m%d_%H%M%S")
    BASE_PATH = f'./sleep_edf/experiment/Performance/{name}'
    os.mkdir(BASE_PATH)

    save_metrics_graph(metrics, start_id, end_id, BASE_PATH)

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
        for metrics in [accuracys, f1_weighteds, f1_macros]:
            write_accuracy_line(file, metrics)
            write_stats_table(file, metrics)
        write_metrics_grpah(file)

        file.write('\\end{document}')


if __name__ == '__main__':
    main()
