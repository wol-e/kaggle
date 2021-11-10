from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=10,
    max_depth=5,
    n_jobs=-1
)
