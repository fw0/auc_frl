import numpy as np
import pdb
import python_utils.python_utils.basic as basic
import pandas as pd

verbose = False
eps = .001

def get_scores(xs, rs):
    # returns scores between 0 and len(rs), highest being best
    cs = xs[:,rs]
    pos = get_pos(cs)
    return len(rs) - pos

def get_pos(cs):
    # cs does not include default rule.  positions can be equal to cs.shape[1]
    if cs.shape[1] == 0:
        return np.zeros(len(cs), dtype=int)
    pos = np.argmax(cs, axis=1)
    pos[np.sum(cs, axis=1) == 0] = cs.shape[1]
    return pos.astype(int)

def get_supps(pos, L):
    # returns length N vector.  needs L to know length of list
    zs = np.zeros((len(pos), L+1), dtype=int)
    zs[np.arange(len(pos), dtype=int),pos] = 1
    return np.sum(zs, axis=0)
    
def get_obj_pre(phi, lam, xs, ys, rs):
    
    if rs == []:
        return 0.
    
    N = len(xs)
    L = len(rs)
    zero_cs = xs[(1.-ys).astype(bool)][:,rs]
    zero_pos = get_pos(zero_cs)
    zero_supps = get_supps(zero_pos, L)
    pessimistic_ranks = np.sum(1.-ys) - np.cumsum(zero_supps)
    phis = phi(pessimistic_ranks)

    one_cs = xs[ys.astype(bool)][:,rs]
    one_pos = get_pos(one_cs)
    one_supps = get_supps(one_pos, L)

    return np.sum(phis * one_supps)

def get_obj_pre_neg(phi, lam, xs, ys, rs):
    
    if rs == []:
        return 0.
    
    N = len(xs)
    L = len(rs)
    one_cs = xs[ys.astype(bool)][:,rs]
    one_pos = get_pos(one_cs)
    one_supps = get_supps(one_pos, L)
    pessimistic_ranks = np.cumsum(one_supps[::-1])[::-1]
    phis = phi(pessimistic_ranks)

    zero_cs = xs[(1.-ys).astype(bool)][:,rs]
    zero_pos = get_pos(zero_cs)
    zero_supps = get_supps(zero_pos, L)

    return -np.sum(phis * zero_supps)

def get_auc(phi, ys, ys_hat):
    # get "position" of each data sample
    idx = np.argsort(ys_hat)
    sorted_ys, sorted_ys_hat = ys[idx], ys_hat[idx]
    neq_prev = sorted_ys_hat[1:] != sorted_ys_hat[0:-1]
    neq_prev = np.insert(neq_prev, 0, [False])
    zeros_before = np.cumsum(1.-sorted_ys[:-1])
    zeros_before = np.insert(zeros_before, 0, [0])
    to_fill = zeros_before * neq_prev
    zero_ranks = phi(np.maximum.accumulate(to_fill))
    #unnormalized = np.sum((1.-ys) * zero_ranks)
    unnormalized = np.sum(ys * zero_ranks)
#    pdb.set_trace()
    return unnormalized / get_max_auc(phi, ys)

def get_auc_neg(phi, ys, ys_hat):
    # get "position" of each data sample
    idx = np.argsort(ys_hat)
    sorted_ys, sorted_ys_hat = ys[idx], ys_hat[idx]
    neq_next = sorted_ys_hat[0:-1] != sorted_ys_hat[1:]
    neq_next = np.insert(neq_next, len(neq_next), [1])
    ones_before = np.cumsum(sorted_ys)
#    zeros_before = np.insert(zeros_before, 0, [0])
    to_fill = (ones_before * neq_next).astype(float)
    to_fill[to_fill==0] = np.nan
    one_ranks = phi(np.fmin.accumulate(to_fill[::-1])[::-1])
    unnormalized = np.sum((1.-ys) * one_ranks)
#    pdb.set_trace()
    return unnormalized / get_max_auc_neg(phi, ys)

def get_obj_end(phi, lam, xs, ys, rs):
    N = len(xs)
    cs = xs[:,rs]
    is_default = np.sum(cs, axis=1) == 0
    return np.sum(ys[is_default]) * phi(0)

def get_obj_end_neg(phi, lam, xs, ys, rs):
    N = len(xs)
    cs = xs[:,rs]
    is_default = np.sum(cs, axis=1) == 0
    num_default = np.sum(is_default)
    return -np.sum((1.-ys)[is_default]) * phi(num_default - np.sum((1.-ys)[is_default]))

def get_max_auc(phi, ys):
    return np.sum(ys) * phi(len(ys) - np.sum(ys))

def get_max_auc_neg(phi, ys):
    return np.sum(1.-ys) * phi(len(ys) - np.sum(1.-ys))

def get_ideal_lb_end(phi, lam, xs, ys, rs):
    N = len(xs)
    cs = xs[:,rs]
    is_default = np.sum(cs, axis=1) == 0
    num_default = np.sum(is_default)
    return np.sum(ys[is_default]) * phi(num_default - np.sum(ys[is_default]))

def get_ideal_lb_end_neg(phi, lam, xs, ys, rs):
    return 0.

def get_max_lb_end(lb_ends, xs, ys, rs):
    return max([lb_end(xs, ys, rs) for lb_end in lb_ends])

def get_length_penalty(phi, lam, xs, ys, rs):
    return lam * len(rs)

def fit(get_succ, get_priority, get_lb_end, get_obj_pre, get_obj_end, get_max_auc, get_length_penalty, xs, ys, best_rs=None,  best_obj=0, state=None):
    # get_succ: antecedent -> list of child antedecents
    # get_priority: antecedent -> priority

    if not (state is None):
        get_succ.set_state(state, xs, ys)
    
    import queue
    q = queue.PriorityQueue(maxsize=0)
    q.put((get_priority(xs, ys, []),[]))

    max_auc = get_max_auc(ys)
#    pdb.set_trace()
    while not q.empty():
        rs_priority, rs = q.get()
        length_penalty, obj_pre, lb_end = get_length_penalty(xs, ys, rs), get_obj_pre(xs, ys, rs), get_lb_end(xs, ys, rs)
#        pdb.set_trace()
        #print (length_penalty - ((obj_pre + lb_end) / max_auc)), best_obj
        if best_obj is None or (length_penalty - ((obj_pre + lb_end) / max_auc)) <= best_obj:
            obj_end = get_obj_end(xs, ys, rs)
            obj = (length_penalty - ((obj_pre + obj_end) / max_auc))
            #print obj, 'obj'
            if verbose:
                print 'obj', obj, 'best', best_obj, 'rs', rs
            if best_obj is None or obj < best_obj:
                best_obj = obj
                best_rs = rs
            for child in get_succ(xs, ys, rs):
                if verbose:
                    print 'child', child
                q.put((get_priority(xs, ys, child), child))

    return best_rs, best_obj

def multiple_try_fit(num_tries, fit, xs, ys):
    best_obj = None
    best_rs = None
    for i in xrange(num_tries):
#        print i, best_obj
        rs, obj = fit(xs, ys, best_rs, best_obj, i)
        if best_obj is None or obj < best_obj:
            best_rs = rs
            best_obj = obj
#    print best_rs
    return best_rs

class serial_get_succ(object):

    def __init__(self, max_num=None):
        self.max_num = max_num
    
    def set_state(self, i, xs, ys):
        self.order = np.arange(xs.shape[1])
        state = np.random.get_state()
        np.random.seed(i)
        np.random.shuffle(self.order)
        np.random.set_state(state)
        self.pos = 0

    def __call__(self, xs, ys, rs):
        if (self.max_num is None and self.pos >= len(self.order)) or (not (self.max_num is None) and self.pos >= self.max_num):
            return []
        ans = self.order[0:(self.pos+1)]
        self.pos += 1
        return [ans]

class auc_predictor(object):

    def __init__(self, rs):
        self.rs = rs

    def predict_proba(self, xs):
        return get_scores(xs, self.rs)
    
class auc_fitter(object):

    def __init__(self, horse, _repr):
        self.horse, self._repr = horse, _repr

    def __repr__(self):
        return self._repr

    def fit(self, xs, ys):
        rs = self.horse(xs, ys)
        return auc_predictor(rs)

class rule_predictor(object):

    def __init__(self, rules, horse):
        self.rules, self.horse = rules, horse

    def predict_proba(self, xs):
        rule_xs = np.array([rule.batch_call(xs) for rule in self.rules]).astype(float).T
        return self.horse.predict_proba(rule_xs)

    def __repr__(self):
        return '_'.join(map(repr, np.array(self.rules)[self.horse.rs]))
#        return repr(self.rule_names[self.predictor.rs])
    
class rule_fitter(object):

    def __init__(self, rule_miner, fitter):
        self.rule_miner, self.fitter = rule_miner, fitter

    def fit(self, xs, ys, x_names=None):
        rules = self.rule_miner(xs, ys, x_names)
        rule_xs = np.array([rule.batch_call(xs) for rule in rules]).astype(float).T
        return rule_predictor(rules, self.fitter.fit(rule_xs, ys))

    def __repr__(self):
        return '%s_%s' % (repr(self.rule_miner), repr(self.fitter))


def kfold_cv(k, xs, ys, i):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=k)
    train_idxs, test_idxs = list(kf.split(xs, ys))[i]
    return xs[train_idxs], ys[train_idxs], xs[test_idxs], ys[test_idxs]

def sample_cv(sample_prop, xs, ys, i):
    state = np.random.get_state()
    np.random.seed(i)
    order = np.random.permutation(len(xs))
    np.random.set_state(state)
    shuffle_xs, shuffle_ys = xs[order,:], ys[order]
    num = int(sample_prop * len(xs))
    return shuffle_xs[0:num], shuffle_ys[0:num], shuffle_xs[num:], shuffle_ys[num:]

class cv_fitter(object):

    def __init__(self, fitters, cv, cv_iters, loss):
        self.fitters, self.cv, self.cv_iters, self.loss = fitters, cv, cv_iters, loss

    def fit(self, xs, ys, x_names=None):
        d = {}
        fitter_d = {}
        for fitter in self.fitters:
            fitter_key = repr(fitter)
            d[fitter_key] = []
            fitter_d[fitter_key] = fitter
            for i in xrange(self.cv_iters):
                xs_train, ys_train, xs_test, ys_test = self.cv(xs, ys, i)
                predictor = fitter.fit(xs, ys, x_names)
                ys_test_hat = predictor.predict_proba(xs_test)
                d[fitter_key].append(self.loss(ys_test, ys_test_hat))
        df = pd.DataFrame(d)
#        print df
        best_fitter = fitter_d[df.mean(axis=0).argmin()]
        best_predictor = best_fitter.fit(xs, ys, x_names)
        best_predictor.fit_info = df
        return best_predictor
        
