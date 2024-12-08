# OptionAnalyzerQ.py

import numpy as np
import pandas as pd
import logging
from scipy.optimize import minimize, brentq
from scipy.stats import norm
from scipy.integrate import quad
import multiprocessing
from functools import partial
from typing import Dict, Optional, Any

logger = logging.getLogger("OptionAnalysisLogger")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

def estimate_heston_parameters(variance_series, delta_t=1/252):
    try:
        variance_series = variance_series.dropna()
        variance_series = variance_series[variance_series > 0]
        n = len(variance_series)

        if n < 2:
            logger.error("Not enough data points to estimate Heston parameters.")
            return {'v0': np.nan, 'theta': np.nan, 'kappa': np.nan, 'sigma_v': np.nan, 'rho': 0.0}

        v = variance_series.values

        def neg_log_likelihood(params, v, delta_t):
            v0, kappa, theta, sigma_v = params
            if kappa <= 0 or sigma_v <= 0 or v0 <= 0 or theta <= 0:
                return np.inf
            neg_ll = 0.0
            v_prev = v0
            length = len(v)
            for t in range(length):
                if t > 0:
                    mu = v_prev + kappa*(theta - v_prev)*delta_t
                    var = sigma_v**2 * v_prev * delta_t
                    var = max(var, 1e-12)
                    neg_ll += 0.5*(np.log(2*np.pi*var) + ((v[t]-mu)**2)/var)
                    v_prev = v[t]
                else:
                    mu = v0
                    var = sigma_v**2 * v0 * delta_t
                    var = max(var, 1e-12)
                    neg_ll += 0.5*(np.log(2*np.pi*var) + ((v[t]-mu)**2)/var)
            return neg_ll

        initial_v0 = variance_series.iloc[0]
        initial_kappa = 1.0
        initial_theta = variance_series.mean()
        initial_sigma_v = 0.1

        initial_guess = [initial_v0, initial_kappa, initial_theta, initial_sigma_v]
        bounds = [(1e-8, None), (1e-8, None), (1e-8, None), (1e-8, None)]

        result = minimize(neg_log_likelihood, initial_guess, args=(v, delta_t), 
                          method='L-BFGS-B', bounds=bounds)

        if result.success:
            v0_est, kappa_est, theta_est, sigma_v_est = result.x
            logger.info(f"Estimated Heston parameters: v0={v0_est}, theta={theta_est}, kappa={kappa_est}, sigma_v={sigma_v_est}, rho=0.0")
            return {
                'v0': v0_est,
                'theta': theta_est,
                'kappa': kappa_est,
                'sigma_v': sigma_v_est,
                'rho': 0.0
            }
        else:
            logger.error("MLE optimization did not converge for Heston parameters.")
            return {'v0': np.nan, 'theta': np.nan, 'kappa': np.nan, 'sigma_v': np.nan, 'rho':0.0}

    except Exception as e:
        logger.error(f"Error estimating Heston parameters: {e}")
        return {'v0': np.nan, 'theta': np.nan, 'kappa': np.nan, 'sigma_v': np.nan, 'rho':0.0}


def heston_char_func(phi, S0, r, T, v0, theta, kappa, sigma_v, rho, j):
    i = complex(0,1)
    alpha = -0.5*(phi**2 + i*phi)
    beta = kappa - rho*sigma_v*i*phi*j
    gamma = 0.5*sigma_v**2
    d = np.sqrt(beta**2 -4*alpha*gamma)
    b = (beta - d)/(beta + d)
    b = np.where(np.abs(b)<1e-14, 1e-14, b)
    term1 = i*phi*(np.log(S0) + r*T)
    term2 = (theta*kappa)/(sigma_v**2)*((beta - d)*T - 2*np.log((1-b*np.exp(-d*T))/(1-b)))
    cf = np.exp(term1 + term2 + (v0/sigma_v**2)*(beta - d)*(1 - np.exp(-d*T))/(1-b*np.exp(-d*T))
                )
    return cf

def heston_pj(j, S0, K, r, T, v0, theta, kappa, sigma_v, rho):
    def integrand(phi):
        i = complex(0,1)
        cf = heston_char_func(phi, S0, r, T, v0, theta, kappa, sigma_v, rho, j)
        numerator = np.exp(-i*phi*np.log(K))*cf
        val = numerator/(i*phi)
        return val.real
    val, err = quad(integrand, 0.0, np.inf)
    return 0.5 + val/np.pi

def heston_call_price(S0, K, r, T, v0, theta, kappa, sigma_v, rho):
    if T<=0:
        return max(S0-K,0)
    P1 = heston_pj(1, S0, K, r, T, v0, theta, kappa, sigma_v, rho)
    P2 = heston_pj(2, S0, K, r, T, v0, theta, kappa, sigma_v, rho)
    call = S0*P1 - K*np.exp(-r*T)*P2
    return call

def black_scholes_call_price(S, K, r, T, sigma):
    if T<=0:
        return max(S-K,0)
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def implied_vol_from_price(S, K, r, T, target_price):
    if T<=0:
        return np.nan
    def f(sig):
        return black_scholes_call_price(S,K,r,T,sig)-target_price
    try:
        return brentq(f, 1e-8,5.0)
    except:
        return np.nan

class HestonVolSurface:
    def __init__(self, v0, theta, kappa, sigma_v, rho):
        self.v0 = v0
        self.theta = theta
        self.kappa = kappa
        self.sigma_v = sigma_v
        self.rho = rho

    def get_iv(self, S0, K, r, T, option_type='CALL'):
        call_price = heston_call_price(S0, K, r, T, self.v0, self.theta, self.kappa, self.sigma_v, self.rho)
        if option_type.upper()=='CALL':
            iv = implied_vol_from_price(S0, K, r, T, call_price)
            return iv
        else:
            put_price = call_price - S0 + K*np.exp(-r*T)
            iv = implied_vol_from_price(S0, K, r, T, call_price)
            return iv

class OptionAnalyzer:
    def __init__(self,
                 cleaned_data: pd.DataFrame,
                 historical_data: Dict[str, pd.DataFrame],
                 risk_free_rate: float = 0.01,
                 market_scenario: str = 'Normal Market'):
        self.cleaned_data = cleaned_data
        self.historical_data = historical_data
        self.risk_free_rate = risk_free_rate
        self.market_scenario = market_scenario

        # Ensure open_positions column is present
        if 'open_positions' not in self.cleaned_data.columns:
            logger.warning("'open_positions' column not found in cleaned_data. Creating with default 0.")
            self.cleaned_data['open_positions'] = 0

        self.simulation_results = {}
        self.pop_results = {}
        self.breakeven_points = {}
        self.sharpe_ratios = {}
        self.target_date_distribution = {}
        self.cash_flows = {}
        self.inferred_market_sentiment = 'neutral'
        self.market_views = {}
        self.var = {}
        self.cvar = {}
        self.payout_ratios = {}
        self.scenario_analysis_results = {}
        self.metrics_stats = {}

        variance_series = self.estimate_realized_variance()
        heston_params = estimate_heston_parameters(variance_series)

        if not np.isnan(heston_params['v0']):
            self.heston_model_params = heston_params
            self.vol_surface = HestonVolSurface(heston_params['v0'],
                                                heston_params['theta'],
                                                heston_params['kappa'],
                                                heston_params['sigma_v'],
                                                heston_params['rho'])
        else:
            logger.warning("Falling back to flat vol = 0.2 since Heston params invalid.")
            self.heston_model_params = None
            self.vol_surface = None
            self.flat_vol = 0.2

        self.calculate_moneyness()

    def estimate_realized_variance(self):
        if len(self.historical_data)==0:
            return pd.Series(dtype=float)
        first_key = next(iter(self.historical_data))
        df = self.historical_data[first_key].copy()
        df = df.sort_values('date')
        df['return'] = np.log(df['close']/df['close'].shift(1))
        rv = (df['return']**2).dropna()
        return rv

    def calculate_moneyness(self):
        logger.info("Calculating moneyness using forward price.")
        if 'last_spot_price' not in self.cleaned_data.columns:
            raise KeyError("'last_spot_price' column missing.")
        self.cleaned_data['T'] = self.cleaned_data['days']/365
        self.cleaned_data['forward_price'] = self.cleaned_data['last_spot_price']*np.exp(self.risk_free_rate*self.cleaned_data['T'])
        self.cleaned_data['moneyness'] = np.where(
            self.cleaned_data['option_type'].str.upper()=='CALL',
            self.cleaned_data['forward_price']/self.cleaned_data['strike_price'],
            self.cleaned_data['strike_price']/self.cleaned_data['forward_price']
        )

    def simulate_price_heston(self, S0, r, T, v0, theta, kappa, sigma_v, rho, num_simulations):
        dt = T
        Z1 = np.random.normal(0,1,num_simulations)
        Z2 = np.random.normal(0,1,num_simulations)
        Z2 = rho*Z1 + np.sqrt(1-rho**2)*Z2
        v_t = np.maximum(v0 + kappa*(theta - v0)*dt + sigma_v*np.sqrt(v0*dt)*Z2, 1e-12)
        S_T = S0*np.exp((r -0.5*v_t)*dt + np.sqrt(v_t*dt)*Z1)
        return S_T

    def simulate_price_risk_neutral(self, S0: float, K: float, r: float, T: float, num_simulations: int) -> np.ndarray:
        if self.heston_model_params:
            p = self.heston_model_params
            S_T = self.simulate_price_heston(S0, r, T, p['v0'], p['theta'], p['kappa'], p['sigma_v'], p['rho'], num_simulations)
        else:
            sigma = self.flat_vol
            Z = np.random.normal(0,1,num_simulations)
            S_T = S0*np.exp((r-0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
        return S_T

    def monte_carlo_simulation_worker(self, option_data: Dict[str, Any], num_simulations: int = 10000) -> Dict[str, Any]:
        try:
            S0 = option_data['last_spot_price']
            K = option_data['strike_price']
            option_type = option_data['option_type'].upper()
            T = option_data['days']/252
            r = self.risk_free_rate
            premium_long = option_data['ask_price']
            premium_short = option_data['bid_price']
            contract_size = option_data['contract_size']

            S_T = self.simulate_price_risk_neutral(S0, K, r, T, num_simulations)

            if option_type=='CALL':
                payoff = np.maximum(S_T - K,0)
            elif option_type=='PUT':
                payoff = np.maximum(K - S_T,0)
            else:
                payoff = np.zeros(num_simulations)

            pl_long = (payoff - premium_long)*contract_size
            pl_short = (premium_short - payoff)*contract_size
            return {'option_name': option_data['option_name'], 'pl_long': pl_long, 'pl_short': pl_short, 'S_T': S_T}
        except Exception as e:
            logger.error(f"Error in simulation for {option_data.get('option_name','Unknown')}: {e}")
            return {'option_name': option_data.get('option_name','Unknown'),'pl_long':None,'pl_short':None,'S_T':None}

    def monte_carlo_simulation(self, num_simulations: int = 10000):
        logger.info("Running Monte Carlo simulations.")
        # Instead of creating a pool here, we run a simple loop:
        options_data = self.cleaned_data.to_dict(orient='records')
        
        # Single-threaded loop:
        results = []
        for od in options_data:
            res = self.monte_carlo_simulation_worker(od, num_simulations=num_simulations)
            results.append(res)

        for result in results:
            option_name = result['option_name']
            pl_long = result['pl_long']
            pl_short = result['pl_short']
            S_T = result['S_T']
            self.simulation_results[option_name] = {'long': pl_long, 'short': pl_short}
            self.target_date_distribution[option_name] = S_T
            if isinstance(pl_long,np.ndarray):
                ask_p = self.cleaned_data.loc[self.cleaned_data['option_name']==option_name,'ask_price'].values[0]
                bid_p = self.cleaned_data.loc[self.cleaned_data['option_name']==option_name,'bid_price'].values[0]
                c_size = self.cleaned_data.loc[self.cleaned_data['option_name']==option_name,'contract_size'].values[0]
                self.cash_flows[option_name] = {
                    'long': {'initial': -ask_p*c_size, 'final': pl_long},
                    'short': {'initial': bid_p*c_size, 'final': pl_short}
                }

    def is_valid_array(self, arr: np.ndarray) -> bool:
        return isinstance(arr, np.ndarray) and arr.size>0 and not np.isnan(arr).all()

    def calculate_pop(self):
        logger.info("Calculating Probability of Profit (PoP).")
        for option, results in self.simulation_results.items():
            pl_long = results.get('long', np.nan)
            pl_short = results.get('short', np.nan)
            pop_long = (np.sum(pl_long>0)/len(pl_long))*100 if self.is_valid_array(pl_long) else np.nan
            pop_short = (np.sum(pl_short>0)/len(pl_short))*100 if self.is_valid_array(pl_short) else np.nan
            self.pop_results[option] = {'long':pop_long,'short':pop_short}

    def calculate_breakeven(self):
        logger.info("Calculating breakeven points and probabilities.")
        for idx, row in self.cleaned_data.iterrows():
            option_name = row['option_name']
            option_type = row['option_type'].upper()
            K = row['strike_price']
            premium_long = row['ask_price']
            premium_short = row['bid_price']
            r = self.risk_free_rate
            T = row['days']/365

            if option_type=='CALL':
                breakeven_long = K + premium_long
                breakeven_short = K + premium_short
            elif option_type=='PUT':
                breakeven_long = K - premium_long
                breakeven_short = K - premium_short
            else:
                breakeven_long = breakeven_short = np.nan

            adjusted_breakeven_long = breakeven_long*np.exp(r*T) if not np.isnan(breakeven_long) else np.nan
            adjusted_breakeven_short = breakeven_short*np.exp(r*T) if not np.isnan(breakeven_short) else np.nan

            S_T = self.target_date_distribution.get(option_name, np.array([]))
            if self.is_valid_array(S_T):
                if option_type=='CALL':
                    prob_exceed_long = np.mean(S_T>adjusted_breakeven_long)
                    prob_exceed_short = np.mean(S_T>adjusted_breakeven_short)
                    losses_long = (adjusted_breakeven_long - S_T)[S_T<adjusted_breakeven_long]
                    losses_short = (adjusted_breakeven_short - S_T)[S_T<adjusted_breakeven_short]
                else:
                    prob_exceed_long = np.mean(S_T<adjusted_breakeven_long)
                    prob_exceed_short = np.mean(S_T<adjusted_breakeven_short)
                    losses_long = (S_T - adjusted_breakeven_long)[S_T>adjusted_breakeven_long]
                    losses_short = (S_T - adjusted_breakeven_short)[S_T>adjusted_breakeven_short]

                cvar_long = np.mean(losses_long) if len(losses_long)>0 else 0
                cvar_short = np.mean(losses_short) if len(losses_short)>0 else 0
            else:
                prob_exceed_long = prob_exceed_short = cvar_long = cvar_short = np.nan

            self.breakeven_points[option_name] = {
                'long': adjusted_breakeven_long,
                'short': adjusted_breakeven_short,
                'prob_exceed_long': prob_exceed_long,
                'prob_exceed_short': prob_exceed_short,
                'cvar_at_breakeven_long': cvar_long,
                'cvar_at_breakeven_short': cvar_short
            }

    def calculate_sharpe_ratio(self, rf: float =0.01):
        logger.info("Calculating Sharpe Ratios.")
        for option, positions in self.simulation_results.items():
            pl_long = positions.get('long', np.nan)
            option_row = self.cleaned_data[self.cleaned_data['option_name']==option]
            if option_row.empty:
                self.sharpe_ratios[option] = {'long':np.nan,'short':np.nan}
                continue
            days = option_row.iloc[0]['days']
            if self.is_valid_array(pl_long):
                initial_investment_long = option_row.iloc[0]['ask_price']*option_row.iloc[0]['contract_size']
                returns_long = pl_long/initial_investment_long if initial_investment_long>0 else np.nan
                if self.is_valid_array(returns_long):
                    mean_r_long = np.mean(returns_long)*(365/days)
                    std_r_long = np.std(returns_long)*np.sqrt(365/days)
                    sharpe_long = (mean_r_long - rf)/std_r_long if std_r_long>0 else np.nan
                else:
                    sharpe_long = np.nan
            else:
                sharpe_long = np.nan

            pl_short = positions.get('short', np.nan)
            if self.is_valid_array(pl_short):
                initial_prem_short = option_row.iloc[0]['bid_price'] * option_row.iloc[0]['contract_size']
                returns_short = pl_short / initial_prem_short if initial_prem_short>0 else np.nan
                if self.is_valid_array(returns_short):
                    mean_r_short = np.mean(returns_short)*(365/days)
                    std_r_short = np.std(returns_short)*np.sqrt(365/days)
                    sharpe_short = (mean_r_short - rf)/std_r_short if std_r_short>0 else np.nan
                else:
                    sharpe_short = np.nan
            else:
                sharpe_short = np.nan

            self.sharpe_ratios[option] = {'long':sharpe_long,'short':sharpe_short}

    def calculate_var_cvar(self, confidence_level=0.95):
        logger.info(f"Calculating VaR and CVaR at {confidence_level*100}% confidence.")
        for option, positions in self.simulation_results.items():
            pl_long = positions.get('long', np.nan)
            pl_short = positions.get('short', np.nan)
            if self.is_valid_array(pl_long):
                var_long = np.percentile(pl_long,(1-confidence_level)*100)
                cvar_long = pl_long[pl_long<=var_long].mean() if np.sum(pl_long<=var_long)>0 else var_long
            else:
                var_long = cvar_long = np.nan

            if self.is_valid_array(pl_short):
                var_short = np.percentile(pl_short,(1-confidence_level)*100)
                cvar_short = pl_short[pl_short<=var_short].mean() if np.sum(pl_short<=var_short)>0 else var_short
            else:
                var_short = cvar_short = np.nan

            self.var[option] = {'long':var_long,'short':var_short}
            self.cvar[option] = {'long':cvar_long,'short':cvar_short}

    def calculate_payout_ratio(self):
        logger.info("Calculating payout ratios and premium efficiency.")
        for option, cflows in self.cash_flows.items():
            option_row = self.cleaned_data[self.cleaned_data['option_name']==option]
            if option_row.empty:
                self.payout_ratios[option] = {
                    'long':np.nan,'short':np.nan,
                    'premium_efficiency_long':np.nan,
                    'premium_efficiency_short':np.nan
                }
                continue
            row = option_row.iloc[0]
            contract_size = row['contract_size']
            premium_long = row['ask_price']
            premium_short = row['bid_price']
            pl_long = cflows['long']['final']
            pl_short = cflows['short']['final']
            if self.is_valid_array(pl_long):
                avg_pl_long = np.mean(pl_long)
                initial_investment_long = premium_long*contract_size
                payout_long = avg_pl_long/initial_investment_long if initial_investment_long>0 else np.nan
                premium_efficiency_long = 1.0/initial_investment_long if initial_investment_long>0 else 0
            else:
                payout_long = premium_efficiency_long = np.nan

            if self.is_valid_array(pl_short):
                avg_pl_short = np.mean(pl_short)
                initial_prem_short = premium_short*contract_size
                payout_short = avg_pl_short/initial_prem_short if initial_prem_short>0 else np.nan
                premium_efficiency_short = initial_prem_short
            else:
                payout_short = premium_efficiency_short = np.nan

            self.payout_ratios[option] = {
                'long': payout_long,
                'short': payout_short,
                'premium_efficiency_long': premium_efficiency_long,
                'premium_efficiency_short': premium_efficiency_short
            }

    def determine_market_views(self):
        logger.info("Determining market views from moneyness.")
        for _,row in self.cleaned_data.iterrows():
            option = row['option_name']
            option_type = row['option_type'].upper()
            S = row['last_spot_price']
            K = row['strike_price']
            if option_type=='CALL':
                self.market_views[option]='bullish' if S>K else 'bearish'
            elif option_type=='PUT':
                self.market_views[option]='bearish' if S<K else 'bullish'
            else:
                self.market_views[option]='neutral'

    def infer_market_sentiment(self):
        logger.info("Inferring market sentiment from open interest and skew.")
        # open_positions should now be guaranteed present
        total_call_open = self.cleaned_data[self.cleaned_data['option_type'].str.upper()=='CALL']['open_positions'].sum()
        total_put_open = self.cleaned_data[self.cleaned_data['option_type'].str.upper()=='PUT']['open_positions'].sum()
        if (total_call_open + total_put_open)==0:
            self.inferred_market_sentiment='neutral'
            return
        put_call_ratio = total_put_open/total_call_open if total_call_open>0 else np.inf
        avg_call_vol = 0.2
        avg_put_vol = 0.2
        vol_skew = avg_put_vol - avg_call_vol

        if put_call_ratio>1.0 and vol_skew>0:
            self.inferred_market_sentiment='bearish'
        elif put_call_ratio<1.0 and vol_skew<0:
            self.inferred_market_sentiment='bullish'
        else:
            self.inferred_market_sentiment='neutral'

    def get_recommendations(self, user_market_view: str):
        logger.info("Generating recommendations.")
        self.calculate_pop()
        self.calculate_sharpe_ratio()
        self.calculate_var_cvar()
        self.calculate_payout_ratio()
        self.calculate_breakeven()
        self.determine_market_views()
        self.infer_market_sentiment()

        recommendations=[]
        for option in self.cleaned_data['option_name']:
            pop_long = self.pop_results.get(option,{}).get('long',np.nan)
            pop_short = self.pop_results.get(option,{}).get('short',np.nan)
            final_score = np.nanmax([pop_long,pop_short])
            user_view = user_market_view.lower()
            if final_score>60:
                if user_view=='bullish':
                    recommendation='Buy' if pop_long>pop_short else 'Hold'
                elif user_view=='bearish':
                    recommendation='Sell' if pop_short>pop_long else 'Hold'
                else:
                    recommendation='Buy' if pop_long>pop_short else 'Sell'
            else:
                recommendation='Hold'
            recommendations.append({
                'OptionName':option,
                'Recommendation':recommendation,
                'PoP_Long':pop_long,
                'PoP_Short':pop_short,
                'InferredMarketSentiment':self.inferred_market_sentiment
            })

        self.recommendations_df = pd.DataFrame(recommendations)
        logger.info("Recommendations generated.")

    def perform_scenario_analysis(self, scenarios: Dict[str,dict], num_simulations=10000):
        logger.info("Performing scenario analysis.")
        base_params = self.heston_model_params.copy() if self.heston_model_params else None
        base_r = self.risk_free_rate
        scenario_results={}

        for scenario_name, adjustments in scenarios.items():
            logger.info(f"Applying scenario: {scenario_name}")
            if base_params:
                p = base_params.copy()
                if 'sigma_v_factor' in adjustments:
                    p['sigma_v'] *= adjustments['sigma_v_factor']
                if 'kappa_factor' in adjustments:
                    p['kappa'] *= adjustments['kappa_factor']
                if 'theta_factor' in adjustments:
                    p['theta'] *= adjustments['theta_factor']
                if 'r_shift' in adjustments:
                    self.risk_free_rate = base_r+adjustments['r_shift']

                self.heston_model_params = p
            else:
                if 'r_shift' in adjustments:
                    self.risk_free_rate = base_r+adjustments['r_shift']

            self.monte_carlo_simulation(num_simulations=num_simulations)
            self.calculate_pop()
            self.calculate_payout_ratio()

            sc_metrics={}
            for option in self.cleaned_data['option_name']:
                pop_long = self.pop_results.get(option,{}).get('long',np.nan)
                pop_short = self.pop_results.get(option,{}).get('short',np.nan)
                pay_long = self.payout_ratios.get(option,{}).get('long',np.nan)
                pay_short = self.payout_ratios.get(option,{}).get('short',np.nan)
                sc_metrics[option]={
                    'PoP_Long':pop_long,
                    'PoP_Short':pop_short,
                    'Payout_Long':pay_long,
                    'Payout_Short':pay_short
                }
            scenario_results[scenario_name]=sc_metrics

            if base_params:
                self.heston_model_params=base_params.copy()
            self.risk_free_rate=base_r

        self.scenario_analysis_results=scenario_results
        logger.info("Scenario analysis completed.")

    def compile_metrics_data(self) -> pd.DataFrame:
        logger.info("Compiling metrics data.")
        data = []
        for option in self.cleaned_data['option_name']:
            metrics = {
                'OptionName': option,
                'Sharpe_Long': self.sharpe_ratios.get(option,{}).get('long',np.nan),
                'Sharpe_Short': self.sharpe_ratios.get(option,{}).get('short',np.nan),
                'PoP_Long': self.pop_results.get(option,{}).get('long',np.nan),
                'PoP_Short': self.pop_results.get(option,{}).get('short',np.nan),
                'VaR_Long': self.var.get(option,{}).get('long',np.nan),
                'VaR_Short': self.var.get(option,{}).get('short',np.nan),
                'CVaR_Long': self.cvar.get(option,{}).get('long',np.nan),
                'CVaR_Short': self.cvar.get(option,{}).get('short',np.nan),
                'Payout_Long': self.payout_ratios.get(option,{}).get('long',np.nan),
                'Payout_Short': self.payout_ratios.get(option,{}).get('short',np.nan),
                'Premium_Efficiency_Long': self.payout_ratios.get(option,{}).get('premium_efficiency_long',np.nan),
                'Premium_Efficiency_Short': self.payout_ratios.get(option,{}).get('premium_efficiency_short',np.nan)
            }
            # Breakeven Points and Probability
            be = self.breakeven_points.get(option,{})
            metrics['Breakeven_Long'] = be.get('long',np.nan)
            metrics['Breakeven_Short'] = be.get('short',np.nan)
            # Compute breakeven percentages if possible
            spot_row = self.cleaned_data[self.cleaned_data['option_name']==option]
            if not spot_row.empty:
                S0 = spot_row.iloc[0]['last_spot_price']
                if np.isnan(be.get('long',np.nan)) or S0<=0:
                    metrics['Breakeven_Long_Pct'] = np.nan
                else:
                    metrics['Breakeven_Long_Pct'] = ((be.get('long',np.nan)/S0)-1)*100

                if np.isnan(be.get('short',np.nan)) or S0<=0:
                    metrics['Breakeven_Short_Pct'] = np.nan
                else:
                    metrics['Breakeven_Short_Pct'] = ((be.get('short',np.nan)/S0)-1)*100
            else:
                metrics['Breakeven_Long_Pct'] = np.nan
                metrics['Breakeven_Short_Pct'] = np.nan

            data.append(metrics)
        metrics_df = pd.DataFrame(data)
        return metrics_df
