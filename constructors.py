import numpy as np
import pdb
import python_utils.python_utils.basic as basic
import pandas as pd
import fxns as fxns
import monotonic.monotonic.rule as rule

def get_auc_fitter(supp, zmax, num_tries, lam, p, obj='pos', lbs=[]):
    phi = np.vectorize(lambda i: i**p)
    get_succ = fxns.serial_get_succ(max_num=None)
    get_priority = lambda xs, ys, rs: 1.
    get_lbs = []
    if obj == 'pos':
        if 'ideal' in lbs:
            get_lbs.append(lambda xs, ys, rs: fxns.get_ideal_lb_end(phi, lam, xs, ys, rs))
        get_obj_pre = lambda xs, ys, rs: fxns.get_obj_pre(phi, lam, xs, ys, rs)
        get_obj_end = lambda xs, ys, rs: fxns.get_obj_end(phi, lam, xs, ys, rs)
        get_max_auc = lambda ys: fxns.get_max_auc(phi, ys)
    elif obj == 'neg':
        if 'ideal' in lbs:
            get_lbs.append(lambda xs, ys, rs: fxns.get_ideal_lb_end_neg(phi, lam, xs, ys, rs))
        get_obj_pre = lambda xs, ys, rs: fxns.get_obj_pre_neg(phi, lam, xs, ys, rs)
        get_obj_end = lambda xs, ys, rs: fxns.get_obj_end_neg(phi, lam, xs, ys, rs)
        get_max_auc = lambda ys: fxns.get_max_auc_neg(phi, ys)
    else:
        assert False

    get_lb_end = lambda xs, ys, rs: fxns.get_max_lb_end(get_lbs, xs, ys, rs)
    get_length_penalty = lambda xs, ys, rs: fxns.get_length_penalty(phi, lam, xs, ys, rs)
    single_fitter_horse = lambda xs, ys, best_rs, best_obj, state: fxns.fit(get_succ, get_priority, get_lb_end, get_obj_pre, get_obj_end, get_max_auc, get_length_penalty, xs, ys, best_rs, best_obj, state)
    fitter_horse = lambda xs, ys: fxns.multiple_try_fit(num_tries, single_fitter_horse, xs, ys)
    fitter = fxns.rule_fitter(rule.rule_miner_f(supp=15, zmax=2), fxns.auc_fitter(fitter_horse, str({'p':p,'lam':lam,'num_tries':num_tries,'obj':obj})))
    return fitter

def get_loss(p, obj):
    phi = np.vectorize(lambda i: i**p)
    if obj == 'pos':
        return lambda ys, ys_hat: fxns.get_auc(phi, ys, ys_hat)
    elif obj == 'neg':
        return lambda ys, ys_hat: fxns.get_auc_neg(phi, ys, ys_hat)
    else:
        assert False
