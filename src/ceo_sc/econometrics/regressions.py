"""Financial econometrics: cross-sectional and Fama-MacBeth regressions,
robustness/subsample/sensitivity analysis, and significance testing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ceo_sc.utils.logging_utils import get_logger

logger = get_logger(__name__)


def cross_sectional_regression(df: pd.DataFrame, y_col: str, x_cols: list[str],
                                date_col: str = "date") -> pd.DataFrame:
    """Run one OLS regression per date, return a DataFrame of coefficients per date."""
    records = []
    for dt, group in df.groupby(date_col):
        sub = group[[y_col] + x_cols].dropna()
        if len(sub) <= len(x_cols) + 1:
            continue
        X = sm.add_constant(sub[x_cols])
        model = sm.OLS(sub[y_col], X).fit()
        record = {"date": dt, "n_obs": len(sub)}
        record.update({f"coef_{c}": model.params.get(c, np.nan) for c in ["const"] + x_cols})
        record.update({f"tstat_{c}": model.tvalues.get(c, np.nan) for c in ["const"] + x_cols})
        record["r_squared"] = model.rsquared
        records.append(record)
    return pd.DataFrame(records)


def fama_macbeth(df: pd.DataFrame, y_col: str, x_cols: list[str],
                  date_col: str = "date", newey_west_lags: int = 3) -> pd.DataFrame:
    """Fama-MacBeth two-pass regression with Newey-West adjusted standard errors.

    Returns a summary DataFrame indexed by coefficient name with columns:
    mean, std_error, t_stat, p_value.
    """
    cs = cross_sectional_regression(df, y_col, x_cols, date_col)
    coef_cols = [c for c in cs.columns if c.startswith("coef_")]

    results = []
    for col in coef_cols:
        name = col.replace("coef_", "")
        series = cs[col].dropna()
        if series.empty:
            continue
        X = np.ones((len(series), 1))
        ols = sm.OLS(series.to_numpy(), X).fit(
            cov_type="HAC", cov_kwds={"maxlags": newey_west_lags}
        )
        results.append({
            "coefficient": name,
            "mean": ols.params[0],
            "std_error": ols.bse[0],
            "t_stat": ols.tvalues[0],
            "p_value": ols.pvalues[0],
            "n_periods": len(series),
        })
    return pd.DataFrame(results).set_index("coefficient")


def robustness_test(df: pd.DataFrame, y_col: str, x_cols: list[str],
                     control_specs: list[list[str]], date_col: str = "date") -> pd.DataFrame:
    """Re-run Fama-MacBeth across several sets of control variables.

    ``control_specs`` is a list of extra-control-variable lists to add on
    top of ``x_cols`` for each specification.
    """
    rows = []
    for i, controls in enumerate(control_specs):
        spec_cols = x_cols + controls
        fm = fama_macbeth(df, y_col, spec_cols, date_col)
        for coef in x_cols:
            if coef in fm.index:
                row = fm.loc[coef].to_dict()
                row["spec"] = i
                row["controls"] = ",".join(controls) if controls else "none"
                row["coefficient"] = coef
                rows.append(row)
    return pd.DataFrame(rows)


def subsample_analysis(df: pd.DataFrame, y_col: str, x_cols: list[str], split_col: str,
                        date_col: str = "date") -> pd.DataFrame:
    """Run Fama-MacBeth separately for each distinct value of ``split_col``."""
    rows = []
    for value, sub in df.groupby(split_col):
        fm = fama_macbeth(sub, y_col, x_cols, date_col)
        fm = fm.reset_index()
        fm["subsample"] = value
        rows.append(fm)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def sensitivity_analysis(df: pd.DataFrame, y_col: str, x_cols: list[str],
                          param_grid: dict[str, list], transform_fn, date_col: str = "date") -> pd.DataFrame:
    """Re-run Fama-MacBeth over a grid of feature-construction parameters.

    ``transform_fn(df, **params)`` should return a transformed copy of ``df``
    with the same ``x_cols`` recomputed under the given parameters.
    """
    from itertools import product

    keys = list(param_grid.keys())
    rows = []
    for combo in product(*param_grid.values()):
        params = dict(zip(keys, combo))
        transformed = transform_fn(df, **params)
        fm = fama_macbeth(transformed, y_col, x_cols, date_col).reset_index()
        for k, v in params.items():
            fm[k] = v
        rows.append(fm)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def out_of_sample_evaluation(df: pd.DataFrame, y_col: str, x_cols: list[str],
                              date_col: str = "date", train_frac: float = 0.7) -> dict:
    """Fit on the first ``train_frac`` of dates (chronologically), evaluate R^2 on the rest."""
    dates = sorted(df[date_col].unique())
    split_idx = int(len(dates) * train_frac)
    train_dates, test_dates = dates[:split_idx], dates[split_idx:]

    train = df[df[date_col].isin(train_dates)].dropna(subset=[y_col] + x_cols)
    test = df[df[date_col].isin(test_dates)].dropna(subset=[y_col] + x_cols)

    X_train = sm.add_constant(train[x_cols])
    model = sm.OLS(train[y_col], X_train).fit()

    X_test = sm.add_constant(test[x_cols], has_constant="add")
    preds = model.predict(X_test)
    resid = test[y_col].to_numpy() - preds.to_numpy()
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((test[y_col].to_numpy() - test[y_col].mean()) ** 2)

    return {
        "in_sample_r2": model.rsquared,
        "out_of_sample_r2": 1 - ss_res / ss_tot if ss_tot else np.nan,
        "n_train": len(train),
        "n_test": len(test),
    }


def significance_test(coef_mean: float, std_error: float, n_obs: int, alpha: float = 0.05) -> dict:
    """Two-sided t-test for a single estimated coefficient."""
    from scipy import stats

    t_stat = coef_mean / std_error if std_error else np.nan
    df_ = max(n_obs - 1, 1)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_)) if not np.isnan(t_stat) else np.nan
    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "significant": bool(p_value < alpha) if not np.isnan(p_value) else False,
    }
