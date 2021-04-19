"""Optimised version of strategy.py to calculate performance of
a leveraged investment strategy"""

import datetime as dt
import math
import time
from itertools import product

import numpy as np
import pandas as pd

import simulate
from debt import Debt

debt_available = {'SU': Debt(), 'Nordnet': Debt()}


def determine_investment(phase, pv_u, tv_u, s, td, pi_rf, dst, g, period):
    # returns cash, new_equity, new_debt

    if phase == 1:
        # Check if gearing cap has been reached
        equity = tv_u + s - td
        if td > (equity * g):
            new_debt = 0
        else:
            new_debt = nd(g, s, tv_u, td, dst, period)
        return 0, s, new_debt

    if phase == 2:
        stocks_sold = max(pv_u - dst, 0)
        debt_repayment = min(td, s + stocks_sold)
        repayment_left = debt_repayment

        if 'Nordnet' in debt_available.keys():
            repayment = min(debt_repayment, debt_available['Nordnet'].debt_amount)
            debt_available['Nordnet'].prepayment(repayment)
            repayment_left = debt_repayment - repayment

        if 'SU' in debt_available.keys():
            debt_available['SU'].prepayment(repayment_left)

        leftover_savings = max(s - debt_repayment - stocks_sold, 0)
        return 0, leftover_savings, -debt_repayment

    if phase == 3:
        return 0, s, 0

    if phase == 4:
        desired_cash = (1 - pi_rf) * (tv_u + s)
        desired_savings = (pi_rf) * (tv_u + s)
        change_in_stock = desired_savings - pv_u
        return desired_cash, change_in_stock, 0


# Function assumes monthly periods
def nd(g, s, tv_u, td, dst, period):
    equity = tv_u + s - td
    total_desired_debt = min(g / (g + 1) * dst, equity * g)
    remaining_debt_needed = max(0, total_desired_debt - td)

    SU_amount, Nordnet_amount = 0, 0

    if period <= 60 and 'SU' in debt_available.keys():
        # Has SU already been taken?
        SU_amount = min(3234, remaining_debt_needed)
        debt_available['SU'].add_debt(SU_amount)

        remaining_debt_needed -= SU_amount

    if 'Nordnet' in debt_available.keys():
        # Has Nordnet already been taken?
        Nordnet_amount = min(max(0, g * equity), remaining_debt_needed)
        debt_available['Nordnet'].add_debt(Nordnet_amount)

    return SU_amount + Nordnet_amount


def interest_all_debt():
    interest_bill = 0
    for debt in debt_available.values():
        interest_bill += debt.calculate_interest()

    return interest_bill


def phase_check(phase, pi_rf, pi_rm, pi_hat, td):
    if phase == 4:
        return 4

    if td > 0:
        # has target not been reached?
        if pi_hat < pi_rm and phase <= 1:
            return 1
        # if target has been reached once and debt remains, stay in phase 2
        return 2

    # if target has been reached and no debt remains
    # is the value still above the target?
    if pi_hat < pi_rf:
        return 3
    return 4


def calc_pi(gamma, sigma, mr, rate, cost=0):
    return (mr - cost - rate) / (gamma * sigma)


def calculate_return(savings_in, returns, gearing_cap, pi_rf, pi_rm, rf):
    # Running controls
    len_savings = len(savings_in)
    assert len_savings == len(returns), 'Investment plan should be same no of periods as market'

    # Setting up constants and dataframe for calculation
    ses_val = savings_in.sum()  # Possibly add more sophisticated discounting
    ist = pi_rm * ses_val
    columns = ['period', 'savings', 'cash', 'new_equity', 'new_debt', 'total_debt', 'nip', 'pv_p',
               'interest', 'market_returns', 'pv_u', 'tv_u', 'equity', 'dst', 'phase', 'pi_hat',
               'g_hat', 'SU_debt', 'Nordnet_debt']

    len_columns = len(columns)

    pp = np.zeros((len_savings, len_columns))

    period, savings, cash, new_equity, new_debt, total_debt, nip, pv_p, interest, market_returns, pv_u, tv_u, equity, dst, phase, pi_hat, g_hat, SU_debt, Nordnet_debt = range(
        len_columns)

    pp[:, period] = range(len_savings)
    pp[:, market_returns] = returns
    pp[:, savings] = savings_in
    pp[0, market_returns] = 0

    # Initializing debt objects
    if 'SU' in debt_available.keys():
        debt_available['SU'] = Debt(rate_structure=[[0, 0, 0.04]], rate_structure_type='relative', initial_debt=0)

    if 'Nordnet' in debt_available.keys():
        debt_available['Nordnet'] = Debt(rate_structure=[[0, .4, 0.02], [.4, .6, 0.03], [.6, 0, 0.07]],
                                         rate_structure_type='relative', initial_debt=0)

    # Period 0 primo
    pp[0, cash] = 0
    pp[0, new_equity] = pp[0, savings]
    pp[0, new_debt] = pp[0, new_equity] * gearing_cap
    pp[0, total_debt] = pp[0, new_debt]
    pp[0, SU_debt] = min(pp[0, new_debt], 3248)
    pp[0, Nordnet_debt] = max(0, pp[0, new_debt] - 3248)
    pp[0, nip] = pp[0, new_debt] + pp[0, new_equity]
    pp[0, pv_p] = pp[0, nip]
    pp[0, pi_hat] = pp[0, pv_p] / ses_val

    # Period 0 ultimo
    pp[0, interest] = pp[0, new_debt] * max(interest_all_debt(), 0)
    pp[0, pv_u] = pp[0, pv_p]
    pp[0, tv_u] = pp[0, pv_u] + pp[0, cash]
    pp[0, equity] = pp[0, tv_u] - pp[0, total_debt]
    pp[0, dst] = ist
    pp[0, phase] = 1

    # Looping over all reminaning periods
    for i in range(1, len_savings):

        # Period t > 0 primo
        if not (pp[i - 1, tv_u] <= 0 and (pp[i - 1, interest] > pp[i, savings])):

            pp[i, cash] = pp[i - 1, cash] * (1 + rf)
            pp[i, cash], pp[i, new_equity], pp[i, new_debt] = determine_investment(
                pp[i - 1, phase], pp[i - 1, pv_u],
                pp[i - 1, tv_u], pp[i, savings], pp[i - 1, total_debt],
                pi_rf, pp[i - 1, dst], gearing_cap, pp[i, period])

            if 'SU' in debt_available.keys():
                pp[i, SU_debt] = debt_available['SU'].debt_amount
            if 'Nordnet' in debt_available.keys():
                pp[i, Nordnet_debt] = debt_available['Nordnet'].debt_amount

            pp[i, total_debt] = pp[i - 1, total_debt] + pp[i, new_debt]
            pp[i, nip] = pp[i, new_equity] + max(0, pp[i, new_debt])
            pp[i, pv_p] = pp[i - 1, pv_u] + pp[i, nip]

            # Period t > 0 ultimo
            if pp[i, period] == 60 and 'SU' in debt_available.keys():
                debt_available['SU'].rate_structure = [[0, 0, 0.01]]

            pp[i, interest] = max(interest_all_debt(), 0)
            pp[i, pv_u] = pp[i, pv_p] * (1 + pp[i, market_returns]) - pp[i, interest]
            pp[i, tv_u] = pp[i, pv_u] + pp[i, cash]
            pp[i, equity] = pp[i, tv_u] - pp[i, total_debt]
            pp[i, pi_hat] = min(pp[i, pv_u] / ses_val, pp[i, pv_u] / pp[i, tv_u])
            pp[i, phase] = phase_check(pp[i - 1, phase], pi_rf, pi_rm, pp[i, pi_hat], pp[i, total_debt])
            target_pi = pi_rm if pp[i - 1, phase] < 3 else pi_rf
            pp[i, dst] = max(pp[i, tv_u] * target_pi, ist)  # Moving stock target
            # pp[i, dst] = max(pp[i-1, dst], max(pp[i, tv_u]*target_pi, ist))  # Locked stock target at highest previous position

        else:
            print('Warning: Fatal wipeout')
            pp[i:, [savings, cash, new_equity, new_debt, nip, pv_p,
                    interest, pv_u, tv_u, pi_hat, g_hat]] = 0

            cols = [total_debt, SU_debt, Nordnet_debt, equity, dst, phase]
            pp[i:, cols] = pp[i - 1, cols]

            break

    pp[:, g_hat] = pp[:, total_debt] / pp[:, equity]
    pp = pd.DataFrame(pp, columns=columns)

    return pp


def calculate100return(savings_in, returns):
    # Running controls
    len_savings = len(savings_in)
    assert len_savings == len(returns), 'Investment plan should be same no of periods as market'

    columns = ['period', 'savings', 'pv_p', 'market_returns', 'tv_u']

    pp = np.empty((len_savings, len(columns)))

    period, savings, pv_p, market_returns, tv_u = range(5)

    pp[:, period] = range(len_savings)
    pp[:, market_returns] = returns
    pp[:, savings] = savings_in
    pp[0, market_returns] = 0
    pp[0, pv_p] = pp[0, savings]
    pp[0, tv_u] = pp[0, savings]

    for i in range(1, len_savings):
        # Period t > 0 primo
        pp[i, pv_p] = pp[i - 1, tv_u] + pp[i, savings]

        # Period t > 0 ultimo
        pp[i, tv_u] = pp[i, pv_p] * (1 + pp[i, market_returns])

    pp = pd.DataFrame(pp, columns=columns)
    return pp


def calculate9050return(savings_in, returns, rf):
    # Strategy where 90% of value is initially invested in stocks, rest in risk free asset
    # Ratio of stocks falls linearly to 50% by age 65 and stays there

    # Running controls
    len_savings = len(savings_in)
    assert len_savings == len(returns), 'Investment plan should be same no of periods as market'

    columns = ['period', 'savings', 'cash', 'pv_p', 'market_returns', 'pv_u', 'tv_u', 'ratio']
    len_columns = len(columns)

    pp = np.empty((len_savings, len_columns))

    period, savings, cash, pv_p, market_returns, pv_u, tv_u, ratio = range(len_columns)

    pp[:, period] = range(len_savings)
    pp[:, market_returns] = returns
    pp[:, savings] = savings_in
    pp[0, market_returns] = 0
    pp[0, pv_p] = pp[0, savings] * 0.9
    pp[0, cash] = pp[0, savings] * 0.1
    pp[0, pv_u] = pp[0, pv_p]
    pp[0, tv_u] = pp[0, savings]
    pp[0, ratio] = 90

    for i in range(1, len_savings):
        ratio_val = max(90 - pp[i, period] / 12, 50)
        pp[i, ratio] = ratio_val

        # Period t > 0 primo
        pp[i, pv_p] = pp[i - 1, pv_u] + pp[i, savings] * (ratio_val / 100)
        pp[i, cash] = pp[i - 1, cash] * (1 + rf) + pp[i, savings] * (1 - ratio_val / 100)

        # Period t > 0 ultimo
        pp[i, pv_u] = pp[i, pv_p] * (1 + pp[i, market_returns])
        pp[i, tv_u] = pp[i, pv_u] + pp[i, cash]

    pp = pd.DataFrame(pp, columns=columns)

    return pp


def summary_stats(returns, values, h, annual_rf):
    mean = ((values[-1] / values[0]) ** (1 / h) - 1) * 3.46410
    std = returns.std() * 3.46410  # sqrt(12) = 3.46410
    return np.array([mean, std, (mean - annual_rf) / std, values[-1]]).transpose()


def main(investments_in, sim_type, random_state, gearing_cap, gamma, sigma, mr,
         yearly_rf, yearly_rm, cost):
    
    vars_for_name = (sim_type, random_state, gearing_cap, gamma, sigma, mr, yearly_rf, yearly_rm, cost)
    vars_for_name_len = len(vars_for_name)-1
    
    out_str = [str(x) + '_' if vars_for_name.index(x) != vars_for_name_len else str(x) for x in vars_for_name]
    
    try:
        pd.read_pickle('sims/' + sim_type + '/' + ''.join(out_str) + '.bz2')
        print('skipping...')

    except FileNotFoundError:
        returns = np.load('market_lookup/' + sim_type + '/' + str(random_state) + '.npy')

        rf = math.exp(yearly_rf / 12) - 1

        pi_rf = calc_pi(gamma, sigma, mr, yearly_rf, cost)
        pi_rm = calc_pi(gamma, sigma, mr, yearly_rm, cost)

        port = calculate_return(investments_in, returns, gearing_cap, pi_rf, pi_rm, rf)
        port100 = calculate100return(investments_in, returns)
        port9050 = calculate9050return(investments_in, returns, rf)

        # Joining normal strategies on to geared
        port['100'] = port100['tv_u']
        port['9050'] = port9050['tv_u']

        # Reducing size of port
        # Setting period as index
        port.set_index('period', drop=True, inplace=True)

        # Dropping non-essential columns
        port.drop(columns=['nip', 'pv_u', 'equity', 'pi_hat', 'g_hat'], inplace=True)

        # Convert selected float columns to integer values
        flt_cols = ['savings', 'cash', 'new_equity', 'new_debt', 'total_debt',
                    'pv_p', 'interest', 'tv_u', 'dst', 'phase', '100', '9050']

        port.loc[:, flt_cols] = port.loc[:, flt_cols].astype(int)
        for debt in ['SU_debt', 'Nordnet_debt']:
            try:
                port.loc[:, [debt]] = port.loc[:, [debt]].astype(int)
            except KeyError:
                pass
        
        # Using compressed pickle to store data efficiently
        port.to_pickle('sims/' + sim_type + '/' + ''.join(out_str) + '.bz2', compression="bz2")

        print('dumped sim')


if __name__ == "__main__":
    from multiprocessing.pool import Pool

    # Creating returns to simulate
    spx = pd.read_csv('^GSPC.csv', index_col=0)
    savings_year = pd.read_csv('investment_plan_year.csv', sep=';', index_col=0)
    savings_year.index = pd.to_datetime(savings_year.index, format='%Y')
    savings_month = (savings_year.resample('BMS').pad() / 12)['Earnings'].values

    YEARLY_RF = 0.015
    YEARLY_RM = 0.04  # Weighted average of margin rates

    # --- Fixed parameters ----
    investments = savings_month * 0.05

    START = dt.date(2020, 1, 1)
    END = dt.date(2070, 1, 31)
    Market = simulate.Market(spx.iloc[-7500:, -2], START, END)

    GAMMA = 2.5
    SIGMA = Market.yearly.pct_change().std() ** 2
    MR = (Market.yearly[-1] / Market.yearly[0]) ** (1 / len(Market.yearly)) - 1
    COST = 0
    # --- End fixed parameters ----

    # Creating list of arguments
    a = [[investments], ['garch', 'norm', 'draw', 't'], range(100), (1, 1.5),
         (1.8, 2.0, 2.2), [SIGMA], [MR], (0.01, 0.02, 0.03), (0.04, 0.05, 0.06, 0.07, 0.08), (0, 0.002)]

    #main(investments_in, sim_type, random_state, gearing_cap, gamma, sigma, mr,
    #     yearly_rf, yearly_rm, cost)

    comb_args = tuple(product(*a))

    arg_iter = (i for i in comb_args)
    num_sims = sum(1 for _ in arg_iter)
    print('number of simulations to run: ', num_sims)
    #import cProfile, pstats
    #profiler = cProfile.Profile()
    #profiler.enable()
    #main(investments, 'garch', 1, 1.5, 2.0, SIGMA, MR, 0.01, 0.05, 0)
    #profiler.disable()
    #stats = pstats.Stats(profiler)
    #stats.strip_dirs()
    #stats.sort_stats('cumtime')
    #stats.print_stats()

    with Pool() as p:
        tic = time.perf_counter()
        p.starmap(main, comb_args)
        toc = time.perf_counter()
        print(f"Script took {toc - tic:0.5f} seconds")
