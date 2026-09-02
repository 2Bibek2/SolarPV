# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# %%
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    # Use sqrt of MSE for RMSE for sklearn versions that don't support squared=False
    rmse = mean_squared_error(y_test, preds) ** 0.5
    return r2, mae, rmse


# %%
def main():
    # paths
    gen_path = 'Plant_2_Generation_Data.csv'
    wea_path = 'Plant_2_Weather_Sensor_Data.csv'
    print('Loading CSVs...')
    gen = pd.read_csv(gen_path)
    wea = pd.read_csv(wea_path)
    print('Rows gen:', len(gen), 'wea:', len(wea))

# %%
    # datetime
    gen['DATE_TIME'] = pd.to_datetime(gen['DATE_TIME'], errors='coerce')
    wea['DATE_TIME'] = pd.to_datetime(wea['DATE_TIME'], errors='coerce')

# %%
    # drop PLANT_ID if exists
    for df in (gen, wea):
        if 'PLANT_ID' in df.columns:
            df.drop(columns=['PLANT_ID'], inplace=True)

    if 'SOURCE_KEY' in wea.columns:
        wea = wea.drop(columns=['SOURCE_KEY'])

# %%
    df = pd.merge(gen, wea, on='DATE_TIME', how='inner')
    print('Merged rows:', len(df))

# %%
    # basic cleaning
    before = len(df)
    df = df[~df['DATE_TIME'].isna()]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if 'IRRADIATION' in df.columns:
        df = df[df['IRRADIATION'] >= 0]
    core = [c for c in ['IRRADIATION','AMBIENT_TEMPERATURE','MODULE_TEMPERATURE','AC_POWER'] if c in df.columns]
    df = df.dropna(subset=core)
    after = len(df)
    print('Rows removed:', before - after, 'remaining:', after)

# %%
    # features
    df['hour'] = df['DATE_TIME'].dt.hour
    df['minute'] = df['DATE_TIME'].dt.minute
    df['day'] = df['DATE_TIME'].dt.day
    df['month'] = df['DATE_TIME'].dt.month
    df['day_of_week'] = df['DATE_TIME'].dt.dayofweek
    df['week'] = df['DATE_TIME'].dt.isocalendar().week.astype(int)
    df['total_minutes'] = df['hour']*60 + df['minute']

# %%
    features_primary = [f for f in ['IRRADIATION','AMBIENT_TEMPERATURE','MODULE_TEMPERATURE','hour','minute','day','month','day_of_week'] if f in df.columns]
    if 'SOURCE_KEY' in df.columns:
        source_dummies = pd.get_dummies(df['SOURCE_KEY'], prefix='SRC', drop_first=True)
        X_primary = pd.concat([df[features_primary].reset_index(drop=True), source_dummies.reset_index(drop=True)], axis=1)
    else:
        X_primary = df[features_primary].copy()
    y = df['AC_POWER'].copy()

# %%
    # time-based chronological split based on unique DATE_TIME values
    df = df.sort_values('DATE_TIME').reset_index(drop=True)
    unique_times = pd.Series(df['DATE_TIME'].unique()).sort_values().reset_index(drop=True)
    n_times = len(unique_times)
    split_time_idx = int(n_times * 0.8)
    train_times = unique_times.iloc[:split_time_idx]
    test_times = unique_times.iloc[split_time_idx:]

    # ensure all observations with same DATE_TIME stay together
    train_mask = df['DATE_TIME'].isin(train_times)
    test_mask = df['DATE_TIME'].isin(test_times)

    X_primary_ch = X_primary.loc[df.index].reset_index(drop=True)
    y_ch = y.loc[df.index].reset_index(drop=True)

    X_train = X_primary_ch[train_mask.values].reset_index(drop=True)
    X_test = X_primary_ch[test_mask.values].reset_index(drop=True)
    y_train = y_ch[train_mask.values].reset_index(drop=True)
    y_test = y_ch[test_mask.values].reset_index(drop=True)

    # prints required by user
    train_start, train_end = (train_times.iloc[0], train_times.iloc[-1]) if len(train_times) > 0 else (None, None)
    test_start, test_end = (test_times.iloc[0], test_times.iloc[-1]) if len(test_times) > 0 else (None, None)
    print('training date range:', train_start, 'to', train_end)
    print('testing date range:', test_start, 'to', test_end)
    print('unique timestamps - train:', len(train_times), 'test:', len(test_times))
    print('rows - train:', X_train.shape[0], 'test:', X_test.shape[0])

    # models
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    dtr = DecisionTreeRegressor(random_state=42)
    dtr.fit(X_train, y_train)
    rfr = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rfr.fit(X_train, y_train)

    models = {'LinearRegression': lr, 'DecisionTree': dtr, 'RandomForest': rfr}
    results = []
    for name, model in models.items():
        r2, mae, rmse = evaluate_model(model, X_test, y_test)
        results.append({'model': name, 'r2': r2, 'mae': mae, 'rmse': rmse})
        print(f"{name}: R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

    # baseline mean predictor
    baseline_pred = np.full(len(y_test), y_train.mean())
    b_r2 = r2_score(y_test, baseline_pred)
    b_mae = mean_absolute_error(y_test, baseline_pred)
    b_rmse = mean_squared_error(y_test, baseline_pred) ** 0.5
    results.append({'model': 'BaselineMean', 'r2': b_r2, 'mae': b_mae, 'rmse': b_rmse})
    print(f"BaselineMean: R2={b_r2:.4f}, MAE={b_mae:.4f}, RMSE={b_rmse:.4f}")

    res_df = pd.DataFrame(results).sort_values(by='r2', ascending=False).reset_index(drop=True)
    print('\nSummary:\n', res_df)
    # save comparison
    res_df.to_csv('model_comparison.csv', index=False)
    print('Saved model comparison to model_comparison.csv')

if __name__ == '__main__':
    main()
