import hydra
import xgboost as xgb
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score

import setting
from src.data.dataframe import make_train_df
from src.model.train import train_valid_test_split, X_y_split


class Model:
    '''
    Attributes
    ----------
    model : xgboost.XGBClassifier
        trained model
    test_accuracy : float, default None
        test data accuracy
    test_f1_weight : float, default None
        weighted F1-score of test data
    test_f1_macro : float, default None
        macro F1-score of test data
    unknown_accuracy : float, default None
        accuracy of unknown person
    unknown_f1_weight : float, default None
        weighted F1-score of unknown person
    unknown_f1_macro : float, default None
        macro F1-score of unknown person
    '''
    def __init__(self):
        self.model = None
        self.test_accuracy = None
        self.test_f1_weight = None
        self.test_f1_macro = None
        self.unknown_accuracy = None
        self.unknown_f1_weight = None
        self.unknown_f1_macro = None

    def train_test(self, cfg: DictConfig):
        id_select = list(range(cfg.train_data_setting.id_select))
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
        unknown = cfg.train_data_setting.unknown
        oversampling = cfg.train_data_setting.oversampling,
        epoch = cfg.train_data_setting.epoch

        train_df = make_train_df(
            id_select,
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
        X_train, X_valid, X_test, y_train, y_valid, y_test, TARS = \
            train_valid_test_split(
                train_df=train_df,
                test_size=cfg.train_data_setting.test_size,
                valid_size=cfg.train_data_setting.valid_size,
                random_state=cfg.train_data_setting.random_state,
                oversampling=oversampling
            )

        xgb_clf = xgb.XGBClassifier(
            objective=cfg.model_setting.objective,
            num_class=len(TARS),
            learning_rate=cfg.model_setting.learning_rate,
            early_stopping_rounds=cfg.model_setting.early_stopping_rounds,
            n_estimators=cfg.model_setting.n_estimators
        )
        # class_weights = [1, 1, 10, 10]
        # weights = [class_weights[label] for label in y_train]
        xgb_clf.fit(X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    verbose=True,
                    # sample_weight=weights
                    )

        y_pred_test = xgb_clf.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_f1_weight = f1_score(y_test, y_pred_test, average="weighted")
        test_f1_macro = f1_score(y_test, y_pred_test, average="macro")

        unknown_df = make_train_df(
            unknown,
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
            mmd_=mmd_,
            esis_=esis_,
            pre_next_rate=pre_next_rate,
            epoch=epoch
            )
        X, y, _ = X_y_split(unknown_df)
        y_pred_test = xgb_clf.predict(X)
        unknown_accuracy = accuracy_score(y, y_pred_test)
        unknown_f1_weight = f1_score(y, y_pred_test, average="weighted")
        unknown_f1_macro = f1_score(y, y_pred_test, average="macro")

        self.model = xgb_clf
        self.test_accuracy = test_accuracy
        self.test_f1_weight = test_f1_weight
        self.test_f1_macro = test_f1_macro
        self.unknown_accuracy = unknown_accuracy
        self.unknown_f1_weight = unknown_f1_weight
        self.unknown_f1_macro = unknown_f1_macro


@hydra.main(
        version_base=None,
        config_path="../../conf/Bpass_Ftrans",
        config_name="P1"
        )
def main(cfg):
    model = Model()
    model.train_test(cfg)
    print(f'''
{type(model.model)}
Test Accuracy
{model.test_accuracy*100:.3f} %
{model.test_f1_weight*100:.3f} %
{model.test_f1_macro*100:.3f} %

Unknown Accuracy
{model.unknown_accuracy*100:.3f} %
{model.unknown_f1_weight*100:.3f} %
{model.unknown_f1_macro*100:.3f} %
''')


if __name__ == '__main__':
    main()
