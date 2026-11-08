import pandas as pd
import numpy as np

raw = pd.read_csv('zindi data/Phase 3 Train.csv')
raw['ts'] = pd.to_datetime(raw.timestamp_ist.str[:19])
raw['v'] = raw.pm25.clip(upper=300)
p = raw.pivot_table(index='ts', columns='location_id', values='v', aggfunc='mean')
full = pd.date_range(p.index.min().floor('D'), '2026-11-03 00:00', freq='h')
p = p.reindex(full)

# Cross-validate across 4 post-monsoon 48-hour cutoffs
cutoffs = [
    pd.Timestamp('2026-10-15 00:00'),
    pd.Timestamp('2026-10-20 00:00'),
    pd.Timestamp('2026-10-25 00:00'),
    pd.Timestamp('2026-10-30 00:00')
]

configs = [
    {'name': 'User_719_style_NO_RAMP', 'w3': 0.0, 'w7': 0.5, 'w14': 0.25, 'wnet': 0.25, 'lookback': 180, 'alpha': 0.5, 'ramp': False},
    {'name': 'User_719_style_WITH_RAMP', 'w3': 0.0, 'w7': 0.5, 'w14': 0.25, 'wnet': 0.25, 'lookback': 180, 'alpha': 0.5, 'ramp': True},
    {'name': 'Postmonsoon_45d_NO_RAMP', 'w3': 0.4, 'w7': 0.4, 'w14': 0.0, 'wnet': 0.2, 'lookback': 45, 'alpha': 0.75, 'ramp': False},
    {'name': 'Postmonsoon_45d_WITH_RAMP', 'w3': 0.4, 'w7': 0.4, 'w14': 0.0, 'wnet': 0.2, 'lookback': 45, 'alpha': 0.75, 'ramp': True},
    {'name': 'Recent_Responsive_NO_RAMP', 'w3': 0.5, 'w7': 0.3, 'w14': 0.0, 'wnet': 0.2, 'lookback': 30, 'alpha': 0.70, 'ramp': False},
    {'name': 'Recent_Responsive_WITH_RAMP', 'w3': 0.5, 'w7': 0.3, 'w14': 0.0, 'wnet': 0.2, 'lookback': 30, 'alpha': 0.70, 'ramp': True},
]

for cfg in configs:
    fold_rmses = []
    for c in cutoffs:
        hist = p.loc[:c]
        actual = p.loc[c + pd.Timedelta(hours=1): c + pd.Timedelta(hours=48)]
        net7 = hist.tail(168).mean().mean()
        
        errs = []
        for s in p.columns:
            x = hist[s]
            level = cfg['w3'] * x.tail(72).mean() + cfg['w7'] * x.tail(168).mean() + cfg['w14'] * x.tail(336).mean() + cfg['wnet'] * net7
            xs = x.tail(cfg['lookback'] * 24)
            sh = xs.groupby((xs.index.hour - 1) % 24).mean() / xs.mean()
            
            y_true = actual[s].dropna()
            if len(y_true) < 10: continue
            hs = (y_true.index.hour - 1) % 24
            shape = 1.0 + cfg['alpha'] * (hs.map(sh).fillna(1.0).values - 1.0)
            pred = level * shape
            if cfg['ramp']:
                d = (y_true.index - c) / pd.Timedelta(days=1)
                r = np.where(d <= 9, 1 + 0.4 * d / 9, 1.4 + 0.025 * (d - 9))
                pred = pred * r
            err = np.mean((y_true.values - pred)**2)
            errs.append(err)
        if len(errs) > 0:
            fold_rmses.append(np.sqrt(np.mean(errs)))
    fold_str = ', '.join([f'{r:.2f}' for r in fold_rmses])
    print(f"{cfg['name']:<30}: Mean RMSE = {np.mean(fold_rmses):.4f}, Folds: [{fold_str}]")
