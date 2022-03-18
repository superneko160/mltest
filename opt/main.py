import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

CROSS_VAL_NUM = 5  # 交差検証の回数

# 指定された属性の値を四捨五入し整数に変換
def round_number(df):
    ave_columns = ['AveRooms', 'AveBedrms', 'AveOccup']
    for col in ave_columns:
        df[col] = np.round(df[col])
    return df

# 標準偏差の２倍以上の値を除去
def std_exclude(df):
    columns = df[['MedInc', 'AveRooms', 'Population', 'AveOccup']].columns
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        boder = np.abs(df[col] - mean) / std
        df = df[(boder < 2)]
    return df

# 区域の人口は、少ない（few）か、普通（usually）か、多い（many）か
# 大体の区域では600人から3000人ということから、この範囲を指標とする
def category(df):
    if df < 600:
        return 'few'
    elif df > 3000:
        return 'many'
    else:
        return 'usually'

# データの前処理
def custom_conversion(dataframe):
    df = dataframe.copy()
    # 四捨五入して整数に変換
    df = round_number(df)
    # サンプルの調査ミスとして取り除く
    df = df[df['HouseAge'] < 52]
    df = df[df['Price'] < 5]
    df = std_exclude(df)
    # 平均部屋数に対して平均寝室数を比較
    df['Bedrms_per_Rooms'] = df['AveBedrms'] / df['AveRooms']
    df['Population_Feature'] = df['Population'].apply(category)  # category関数を属性Populationに適用
    # カテゴリ属性をダミー変数化
    feature_dummies = pd.get_dummies(df['Population_Feature'], drop_first=True)
    df = pd.concat([df, feature_dummies], axis=1)
    # X：説明変数、y：目的変数
    X = df.drop(['AveBedrms', 'Price', 'Population_Feature'], axis=1)
    y = df['Price'] 
    return X, y

# 誤差を返す(RMSE:Root Mean Squared Error)
def rmse(pred, y):
    mse = mean_squared_error(y, pred)
    return np.sqrt(mse)

# 通常の誤差、交差検定の評価、平均、標準偏差
def print_scores(normal, scores):
    print('Normal: ', normal)
    print('Score: ', scores)
    print('Mean: ', scores.mean())
    print('Standard deviation: ', scores.std())

# 線形回帰モデルを適用
def apply_linearregresion(X_s, y):
    lin_reg = LinearRegression()
    lin_reg.fit(X_s, y)
    # 予測結果
    pred = lin_reg.predict(X_s)
    # 交差検証（scornigは評価方法、ここでは平均二乗誤差、cvは検証回数）
    scores = cross_val_score(lin_reg, X_s, y, scoring='neg_mean_squared_error', cv=CROSS_VAL_NUM)
    # scikit-learnの交差検証では負数（マイナス）の平均二乗誤差が求められるので、平方根をとる際にマイナスを掛ける
    rmse_score = np.sqrt(-scores)
    # 通常の誤差、交差検証を行ったときの評価、平均、標準偏差
    print_scores(rmse(pred, y), rmse_score)

# SVMモデルを適用
def apply_svm(X_s, y):
    svm_reg = SVR()
    svm_reg.fit(X_s, y)
    # 予測結果
    pred = svm_reg.predict(X_s)
    # 交差検証（scornigは評価方法、ここでは平均二乗誤差、cvは検証回数）
    scores = cross_val_score(svm_reg, X_s, y, scoring='neg_mean_squared_error', cv=CROSS_VAL_NUM)
    # scikit-learnの交差検証では負数（マイナス）の平均二乗誤差が求められるので、平方根をとる際にマイナスを掛ける
    rmse_score = np.sqrt(-scores)
    # 通常の誤差、交差検証を行ったときの評価、平均、標準偏差
    print_scores(rmse(pred, y), rmse_score)

# ランダムフォレストモデルを適用
def apply_randomforest(X_s, y):
    forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    forest_reg.fit(X_s, y)
    # 予測結果
    pred = forest_reg.predict(X_s)
    # 交差検証（scornigは評価方法、ここでは平均二乗誤差、cvは検証回数）
    scores = cross_val_score(forest_reg, X_s, y, scoring='neg_mean_squared_error', cv=CROSS_VAL_NUM)
    # scikit-learnの交差検証では負数（マイナス）の平均二乗誤差が求められるので、平方根をとる際にマイナスを掛ける
    rmse_score = np.sqrt(-scores)
    # 通常の誤差、交差検証を行ったときの評価、平均、標準偏差
    print_scores(rmse(pred, y), rmse_score)

# 勾配ブースティング決定木モデルを適用
def apply_gradientboosting(X_s, y):
    gb_reg = GradientBoostingRegressor(random_state=42)
    gb_reg.fit(X_s, y)
    # 予測結果
    pred = gb_reg.predict(X_s)
    # 交差検証（scornigは評価方法、ここでは平均二乗誤差、cvは検証回数）
    scores = cross_val_score(gb_reg, X_s, y, scoring='neg_mean_squared_error', cv=CROSS_VAL_NUM)
    # scikit-learnの交差検証では負数（マイナス）の平均二乗誤差が求められるので、平方根をとる際にマイナスを掛ける
    rmse_score = np.sqrt(-scores)
    # 通常の誤差、交差検証を行ったときの評価、平均、標準偏差
    print_scores(rmse(pred, y), rmse_score)

# 多層パーセプトロンモデルを適用
def apply_mlpregressor(X_s, y):
    mlp_reg = MLPRegressor(max_iter=300, random_state=42)
    mlp_reg.fit(X_s, y)
    # 予測結果
    pred = mlp_reg.predict(X_s)
    # 交差検証（scornigは評価方法、ここでは平均二乗誤差、cvは検証回数）
    scores = cross_val_score(mlp_reg, X_s, y, scoring='neg_mean_squared_error', cv=CROSS_VAL_NUM)
    # scikit-learnの交差検証では負数（マイナス）の平均二乗誤差が求められるので、平方根をとる際にマイナスを掛ける
    rmse_score = np.sqrt(-scores)
    # 通常の誤差、交差検証を行ったときの評価、平均、標準偏差
    print_scores(rmse(pred, y), rmse_score)

# メイン処理
def main():
    housing = fetch_california_housing()  # データ取得

    X = pd.DataFrame(housing.data, columns=housing.feature_names)  # X：説明変数
    y = pd.DataFrame(housing.target, columns=['Price'])  # y：目的変数（正解ラベル）
    # 訓練セットとテストセットを8：2に分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_set = pd.concat([X_train, y_train], axis=1)  # 訓練セットをデータフレームに格納
    test_set = pd.concat([X_test, y_test], axis=1)  # テストセットをデータフレームに格納
    # 前処理された説明変数と目的変数を出力
    X, y = custom_conversion(train_set)
    # 各項目の平均値と50%の値が近くなっていれば大きな外れ値は除去できたと判断して良い
    # X.describe()

    # 説明変数をスケーリング
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # 処理に時間がかかるので1モデルずつ実行する
    # 線形回帰モデル
    apply_linearregresion(X_s, y)
    # SVMモデル
    # apply_svm(X_s, y)
    # ランダムフォレストモデル
    # apply_randomforest(X_s, y)
    # 勾配ブースティングモデル
    # apply_gradientboosting(X_s, y)
    # 多層パーセプトロンモデル
    # apply_mlpregressor(X_s, y)

if __name__ == '__main__':
    main()