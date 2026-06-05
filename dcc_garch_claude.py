# Libraries to use in the analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from arch import arch_model
from scipy.optimize import minimize

def csv_cleaning(df):
    """Format CSV files to have the same structure."""
    df = df.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)
    df = df.iloc[::-1].reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date']).astype(str)
    return df

def calculate_return(df):
    """Calculate log returns of a price DataFrame."""
    returns = df.copy().set_index('Date')
    returns = returns.pct_change().dropna()
    return returns

def fit_univariate_garch(return_series):
    """
    Fit GARCH(1,1) to a single return series.
    Returns: (result, conditional_volatility, standardized_residuals)
    """
    y = pd.to_numeric(return_series, errors="coerce") \
          .replace([np.inf, -np.inf], np.nan) \
          .dropna()
    y = y * 100  # scale to percentage returns

    am = arch_model(y, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
    res = am.fit(disp="off")

    cond_vol  = res.conditional_volatility
    std_resid = res.std_resid

    return res, cond_vol, std_resid

def _dcc_loglik(params, std_resids):
    """
    Negative log-likelihood for the DCC(1,1) second step.

    The Q process:
        Q_t = (1 - a - b) * Qbar  +  a * e_{t-1} e'_{t-1}  +  b * Q_{t-1}

    The conditional correlation matrix:
        R_t = diag(Q_t)^{-1/2}  *  Q_t  *  diag(Q_t)^{-1/2}

    Log-likelihood contribution at t:
        - (log|R_t|  +  e'_t R_t^{-1} e_t  -  e'_t e_t)

    Parameters
    ----------
    params : [alpha, beta]   scalars satisfying alpha, beta > 0 and alpha+beta < 1
    std_resids : (T, 2) ndarray of standardised residuals from step 1
    """
    alpha, beta = params

    if alpha <= 0 or beta <= 0 or alpha + beta >= 1:
        return 1e10  # infeasible region

    T = len(std_resids)
    e = std_resids  # shape (T, 2)

    # Unconditional correlation matrix (Qbar) – target of mean-reversion
    Qbar = e.T @ e / T  # (2, 2)

    # Initialise Q_0 = Qbar
    Q = Qbar.copy()

    llh = 0.0
    for t in range(1, T):
        e_lag = e[t - 1, :].reshape(-1, 1)           # (2,1)
        Q = (1 - alpha - beta) * Qbar \
            + alpha * (e_lag @ e_lag.T) \
            + beta  * Q                              # (2,2)  Q_t

        # Standardise Q to get the correlation matrix R_t
        q_diag_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(Q)))
        R = q_diag_inv_sqrt @ Q @ q_diag_inv_sqrt   # (2,2)  R_t

        e_t = e[t, :]                               # (2,)

        # Numerically stable log-det and inverse for a 2×2 matrix
        sign, log_det_R = np.linalg.slogdet(R)
        if sign <= 0:                               # non-positive-definite
            return 1e10
        R_inv = np.linalg.inv(R)

        quad_R   = e_t @ R_inv @ e_t               # e'_t R_t^{-1} e_t
        quad_eye = e_t @ e_t                        # e'_t I e_t  (same as sum of squares)

        llh += log_det_R + quad_R - quad_eye        # Engle (2002) eq. 13

    return llh  # we minimise, so no negative sign flip needed (already positive)


def fit_dcc(std_resids_matrix):
    """
    Fit DCC(1,1) given a (T, 2) matrix of standardised residuals.

    Returns
    -------
    alpha   : float   – ARCH effect in correlation dynamics
    beta    : float   – GARCH effect in correlation dynamics
    Q_path  : (T, 2, 2) ndarray – evolution of the Q matrix
    R_path  : (T, 2, 2) ndarray – evolution of the correlation matrix R_t
    rho_t   : (T,) ndarray  – the (0,1) off-diagonal element = dynamic correlation
    """
    e = std_resids_matrix.values if hasattr(std_resids_matrix, 'values') \
        else np.array(std_resids_matrix)

    T = len(e)
    Qbar = e.T @ e / T

    # ── Optimise DCC parameters ──────────────────────────────────────────────
    best_llh = np.inf
    best_res = None

    # Multiple starting points to avoid local minima
    for a0, b0 in [(0.05, 0.90), (0.10, 0.85), (0.02, 0.95)]:
        res = minimize(
            _dcc_loglik,
            x0=[a0, b0],
            args=(e,),
            method='L-BFGS-B',
            bounds=[(1e-6, 0.5), (1e-6, 0.9999)],
            options={'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res.fun < best_llh and res.success:
            best_llh = res.fun
            best_res = res

    if best_res is None:
        raise RuntimeError("DCC optimisation failed to converge.")

    alpha, beta = best_res.x
    print(f"\n  DCC parameters →  α (ARCH) = {alpha:.6f}   β (GARCH) = {beta:.6f}")
    print(f"  α + β = {alpha + beta:.6f}  (must be < 1 for stationarity)")
    print(f"  Log-likelihood = {-best_llh:.4f}")

    # ── Reconstruct the full path of Q_t and R_t ─────────────────────────────
    Q_path = np.zeros((T, 2, 2))
    R_path = np.zeros((T, 2, 2))
    rho_t  = np.zeros(T)

    Q = Qbar.copy()
    for t in range(T):
        if t > 0:
            e_lag = e[t - 1, :].reshape(-1, 1)
            Q = (1 - alpha - beta) * Qbar \
                + alpha * (e_lag @ e_lag.T) \
                + beta  * Q

        q_diag_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(Q)))
        R = q_diag_inv_sqrt @ Q @ q_diag_inv_sqrt

        Q_path[t] = Q
        R_path[t] = R
        rho_t[t]  = R[0, 1]   # off-diagonal = dynamic correlation

    return alpha, beta, Q_path, R_path, rho_t


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_dcc_results(dates, rho_t, candidate_name, portfolio_name="EDEN Portfolio"):
    """
    Two-panel figure:
      - Top:    dynamic conditional correlation ρ_t over time
      - Bottom: rolling 60-day average for smoothed trend
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(
        f"DCC-GARCH(1,1): Dynamic Correlation\n{candidate_name}  ↔  {portfolio_name}",
        fontsize=14, fontweight='bold'
    )

    dates_dt = pd.to_datetime(dates)

    # ── Top: raw dynamic correlation ─────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(dates_dt, rho_t, color='steelblue', linewidth=0.8, alpha=0.85, label='ρ_t')
    ax1.axhline(np.mean(rho_t), color='crimson', linestyle='--', linewidth=1.2,
                label=f'Mean ρ = {np.mean(rho_t):.3f}')
    ax1.axhline(0, color='grey', linestyle=':', linewidth=0.8)
    ax1.fill_between(dates_dt, rho_t, np.mean(rho_t),
                     where=(rho_t > np.mean(rho_t)), alpha=0.15, color='crimson')
    ax1.fill_between(dates_dt, rho_t, np.mean(rho_t),
                     where=(rho_t < np.mean(rho_t)), alpha=0.15, color='steelblue')
    ax1.set_ylabel("Conditional Correlation ρ_t")
    ax1.set_ylim(-1, 1)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Bottom: smoothed trend ────────────────────────────────────────────────
    ax2 = axes[1]
    rho_series = pd.Series(rho_t, index=dates_dt)
    rolling_mean = rho_series.rolling(window=60, min_periods=20).mean()
    ax2.plot(dates_dt, rolling_mean, color='darkorange', linewidth=1.5, label='60-day rolling avg')
    ax2.axhline(np.mean(rho_t), color='crimson', linestyle='--', linewidth=1.2,
                label=f'Full-sample mean = {np.mean(rho_t):.3f}')
    ax2.axhline(0, color='grey', linestyle=':', linewidth=0.8)
    ax2.set_ylabel("Smoothed ρ_t")
    ax2.set_ylim(-1, 1)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=30)

    plt.tight_layout()
    plt.savefig(f"dcc_{candidate_name.replace(' ', '_')}.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Chart saved → dcc_{candidate_name.replace(' ', '_')}.png")


def print_dcc_summary(candidate_name, rho_t):
    """Print a human-readable interpretation of the DCC results."""
    mean_rho   = np.mean(rho_t)
    max_rho    = np.max(rho_t)
    min_rho    = np.min(rho_t)
    range_rho  = max_rho - min_rho
    recent_rho = np.mean(rho_t[-60:])   # last ~3 months

    print(f"\n{'─'*55}")
    print(f"  DCC-GARCH Summary: {candidate_name}")
    print(f"{'─'*55}")
    print(f"  Full-sample mean ρ   :  {mean_rho:+.4f}")
    print(f"  Recent (last 60d) ρ  :  {recent_rho:+.4f}")
    print(f"  Min / Max ρ          :  {min_rho:+.4f}  /  {max_rho:+.4f}")
    print(f"  Range (max - min)    :  {range_rho:.4f}")

    # Diversification interpretation
    print(f"\n  Diversification read:")
    if abs(mean_rho) < 0.20:
        label = "LOW correlation → strong diversification candidate"
    elif abs(mean_rho) < 0.50:
        label = "MODERATE correlation → partial diversification"
    else:
        label = "HIGH correlation → limited diversification benefit"
    print(f"    {label}")

    # Trend interpretation
    trend = recent_rho - mean_rho
    direction = "RISING" if trend > 0.05 else "FALLING" if trend < -0.05 else "STABLE"
    print(f"    Recent trend: {direction}  (Δ = {trend:+.4f} vs full-sample mean)")
    print(f"{'─'*55}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    current_portfolio = [
        {'ticker': 'ASML',   'market value of shares': 5000,  'path': 'ASML_1d_data.csv'},
        {'ticker': 'NOVO-B', 'market value of shares': 5000,  'path': 'NOVO-B.CO_1d_data.csv'},
        {'ticker': 'PVH',    'market value of shares': 7500,  'path': 'PVH_1d_data.csv'},
        {'ticker': 'XLU',    'market value of shares': 7500,  'path': 'XLU_1d_data.csv'},
        {'ticker': 'GLD',    'market value of shares': 10000, 'path': 'GLD_1d_data.csv'},
    ]

    possible_stocks = [
        {'ticker': 'Orion Chips',  'path': 'orion_eod.csv'},
        {'ticker': 'Cleanergy',    'path': 'cleanergy_eod.csv'},
        {'ticker': 'NovaTerra AI', 'path': 'novaterra_eod.csv'},
    ]

    # ── Build price matrix & portfolio return series ──────────────────────────
    price_matrix = pd.DataFrame(columns=('Date',))
    portfolio_weights = []
    total_value = sum(a['market value of shares'] for a in current_portfolio)

    for asset in current_portfolio:
        portfolio_weights.append(asset['market value of shares'] / total_value)
        asset_df = csv_cleaning(pd.read_csv(asset['path']))
        asset_df = asset_df.rename(columns={'Close': asset['ticker']})
        price_matrix = price_matrix.merge(asset_df, on='Date', how='outer')

    price_matrix = price_matrix.sort_values('Date').reset_index(drop=True).dropna()
    return_df    = calculate_return(price_matrix)
    portfolio_weights = np.array(portfolio_weights)
    portfolio_return  = return_df @ portfolio_weights

    # ── Step 1: fit univariate GARCH on the portfolio ─────────────────────────
    print("Fitting GARCH(1,1) on portfolio return series…")
    port_res, port_cond_vol, port_std_resid = fit_univariate_garch(portfolio_return)

    # ── Step 1 + 2 for each candidate ────────────────────────────────────────
    dcc_results = {}  # store for comparison

    for candidate in possible_stocks:
        print(f"\n{'═'*55}")
        print(f"  Processing candidate: {candidate['ticker']}")
        print(f"{'═'*55}")

        # Load candidate, clean, compute returns
        cand_df = pd.read_csv(candidate['path'])
        cand_df['Date'] = pd.to_datetime(cand_df['Date']).astype(str)
        cand_return = calculate_return(cand_df).squeeze()

        # Step 1 – univariate GARCH on candidate
        print(f"  Fitting GARCH(1,1) on {candidate['ticker']}…")
        cand_res, cand_cond_vol, cand_std_resid = fit_univariate_garch(cand_return)

        # ── Align the two standardised residual series on common dates ────────
        port_std_df = port_std_resid.rename("portfolio")           # Series with Date index
        cand_std_df = cand_std_resid.rename(candidate['ticker'])   # Series with Date index

        # Inner join on dates present in both series
        std_matrix = pd.concat([port_std_df, cand_std_df], axis=1, join='inner').dropna()
        # std_matrix is now (T, 2), columns = ['portfolio', ticker]

        print(f"  Common observations for DCC fit: {len(std_matrix)}")

        # Step 2 – fit DCC on the joint standardised residuals
        print(f"  Fitting DCC(1,1)…")
        alpha, beta, Q_path, R_path, rho_t = fit_dcc(std_matrix)

        dcc_results[candidate['ticker']] = {
            'alpha':   alpha,
            'beta':    beta,
            'rho_t':   rho_t,
            'dates':   std_matrix.index,
            'mean_rho': np.mean(rho_t),
        }

        # Interpretation
        print_dcc_summary(candidate['ticker'], rho_t)

        # Plot
        plot_dcc_results(std_matrix.index, rho_t, candidate['ticker'])

    # ── Final comparison table ────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print("  DCC-GARCH Comparison Across Candidates")
    print(f"{'═'*55}")
    print(f"  {'Ticker':<18} {'Mean ρ':>8} {'Recent ρ':>10} {'Range':>8}  {'α':>7}  {'β':>7}")
    print(f"  {'─'*18} {'─'*8} {'─'*10} {'─'*8}  {'─'*7}  {'─'*7}")
    for name, r in dcc_results.items():
        recent = np.mean(r['rho_t'][-60:])
        rng    = np.max(r['rho_t']) - np.min(r['rho_t'])
        print(f"  {name:<18} {r['mean_rho']:>+8.4f} {recent:>+10.4f} {rng:>8.4f}  "
              f"{r['alpha']:>7.5f}  {r['beta']:>7.5f}")
    print(f"{'═'*55}")
    print("\n  Lower |mean ρ| = better diversifier.")
    print("  Higher β = more persistent correlation regimes (correlation 'sticks').")


if __name__ == "__main__":
    main()
