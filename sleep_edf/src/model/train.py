from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import RandomOverSampler


def X_y_split(train_df):
    X, y = train_df[train_df.columns[:-1]], train_df[train_df.columns[-1]]
    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    TARS = {
        y.value_counts().index[i]: i
        for i in range(len(y.value_counts().index))
        }
    y = y.map(TARS)
    return X, y, TARS


def train_valid_test_split(
        train_df,
        test_size: float,
        valid_size: float,
        random_state: int,
        oversampling: bool = False
        ) -> tuple[list]:
    X, y = train_df[train_df.columns[:-1]], train_df[train_df.columns[-1]]
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
        )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=valid_size,
        random_state=random_state
        )

    scaler = RobustScaler()
    X_train, X_valid, X_test = (
        scaler.fit_transform(X_train),
        scaler.fit_transform(X_valid),
        scaler.fit_transform(X_test)
        )

    TARS = {
        y_train.value_counts().index[i]: i
        for i in range(len(y_train.value_counts().index))
        }
    y_train, y_valid, y_test = (
        y_train.map(TARS),
        y_valid.map(TARS),
        y_test.map(TARS)
        )
    if oversampling:
        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)
    return X_train, X_valid, X_test, y_train, y_valid, y_test, TARS
