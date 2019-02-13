import numpy as np
from itertools import product


class ModelEvaluation:
    def __init__(self, ids, me_params):
        self.ids = ids
        self.me_method_label = me_params["method"]
        self.me_params = me_params[me_params["method"]]
        self.max_iters = 0
        self.me_method = self.get_me_method()

    def __getitem__(self, item):
        return self.me_method(item, self.ids, self.me_params)

    def set_new_ids(self, ids):
        self.ids = ids
        self.me_method = self.get_me_method()

    def get_me_method(self):
        if self.me_method_label == "window_based":
            self.max_iters = int(
                np.floor((len(self.ids) - self.me_params["warmup_period"]) / self.me_params["test_size"])) - 1
            return window_based_sets

        elif self.me_method_label == "user_defined":
            self.max_iters = np.unique(self.me_params["ud_steps"])
            return user_defined

        elif self.me_method_label == "holdout_end":
            self.max_iters = 1
            return holdout_end

        elif self.me_method_label == "k_fold_cv":
            self.max_iters = self.me_params["k"]
            return k_fold_cv

        elif self.me_method_label == "iid_bootstrap":
            self.max_iters = self.me_params["B"]
            return iid_bootstrap

        elif self.me_method_label == "stationary_bootstrap":
            self.max_iters = self.me_params["B"]
            return stationary_bootstrap

        elif self.me_method_label == "circular_bootstrap":
            self.max_iters = self.me_params["B"]
            return circular_bootstrap

        elif self.me_method_label == "block_bootstrap":
            self.max_iters = self.me_params["B"]
            return block_bootstrap

        elif self.me_method_label == "naive":
            self.max_iters = 1
            return naive

        elif self.me_method_label == "block_cv":
            self.max_iters = int(np.ceil(len(self.ids) / (self.me_params["test_size"])))
            return bcv_method

        elif self.me_method_label == "partitioned_cv":
            self.max_iters = self.me_params["subset_size"]
            return pcv_method

        elif self.me_method_label == "hvblock_cv":
            total_size = self.me_params["test_size"] + 2 * self.me_params["gap_size"]
            self.max_iters = int(np.ceil(len(self.ids) / total_size))
            return hvbcv_method

        elif self.me_method_label == "markov_cv":
            if self.me_params["gap_size"] % 3 == 0:
                m = int(2 * self.me_params["gap_size"] / 3)
            else:
                m = 2 * int(np.floor(self.me_params["gap_size"] / 3)) + 2
            self.max_iters = 2 * m
            return mkcv_method

        elif self.me_method_label == "combsym_cv":
            # create the cscv lists
            splits = self.me_params["splits"]
            cscv_ids = {}
            for x in range(splits // 2):
                cscv_ids[str(x)] = list(range(x, splits))
            cscv_ids = list(product(*[cscv_ids[k] for k in cscv_ids.keys()]))
            cscv_ids = [csid for csid in cscv_ids if np.prod(np.diff(csid) > 0)]
            cscv_ids = [list(cscv_ids[x]) + list(cscv_ids[-(x + 1)]) for x in range(len(cscv_ids))]

            # compute number of folds and pass cscv method
            self.max_iters = len(cscv_ids)
            return cscv_method


def k_fold_cv(i, ids, kfold_params):
    # some reproducibility stuff
    rng = np.random.get_state()
    np.random.seed(seed=kfold_params["init_stat"])

    # shuffle index
    np.random.shuffle(ids)
    fold_size = int(np.floor(len(ids) / kfold_params["k"]))

    # pick depending on i
    np.random.set_state(rng)
    if i == 0:
        return {"train_" + str(i): ids[fold_size:],
                "test_" + str(i): ids[:fold_size]
                }
    elif i == (kfold_params["k"] - 1):
        return {"train_" + str(i): ids[:i * fold_size],
                "test_" + str(i): ids[i * fold_size:]
                }
    else:
        return {"train_" + str(i): ids[:i * fold_size] + ids[(i + 1) * fold_size:],
                "test_" + str(i): ids[i * fold_size:(i + 1) * fold_size]
                }


def user_defined(i, ids, ud_params):
    # pre-calculation
    steps = ud_params["ud_steps"]
    train_steps = [i * ud_params["stride_size"], ud_params["warmup_period"] + i * ud_params["test_size"]]
    test_steps = [ud_params["warmup_period"] + i * ud_params["test_size"],
                  ud_params["warmup_period"] + (i + 1) * ud_params["test_size"]]

    train_ids = (np.array(steps) >= train_steps[0]) * (np.array(steps) < train_steps[1])
    test_ids = (np.array(steps) >= test_steps[0]) * (np.array(steps) < test_steps[1])

    # split in training and test folds
    me_sets = {"train_" + str(i): list(np.array(ids)[train_ids]),
               "test_" + str(i): list(np.array(ids)[test_ids])}

    return me_sets


def holdout_end(i, ids, hold_params):
    return {"train_" + str(i): ids[:-hold_params["holdout_size"]],
            "test_" + str(i): ids[-hold_params["holdout_size"]:]}


def naive(i, ids, naive_params=None):
    return {"train_" + str(i): ids,
            "test_" + str(i): ids}


def window_based_sets(i, ids, wb_params):
    # pre-calculation
    id_len = len(ids)
    n_iters = int(np.floor((id_len - wb_params["warmup_period"]) / wb_params["test_size"])) - 1

    # split in training and test folds
    me_sets = {}
    if i < n_iters:
        me_sets["train_" + str(i)] = ids[i * wb_params["stride_size"]:(wb_params["warmup_period"] + i * wb_params[
            "test_size"])]
        me_sets["test_" + str(i)] = ids[(wb_params["warmup_period"] + i * wb_params["test_size"]):(
            wb_params["warmup_period"] + (i + 1) * wb_params["test_size"])]
    elif i == n_iters:
        # final set
        me_sets["train_" + str(i)] = ids[i * wb_params["stride_size"]:(
            wb_params["warmup_period"] + i * wb_params["test_size"])]
        me_sets["test_" + str(i)] = ids[(wb_params["warmup_period"] + i * wb_params["test_size"]):]
    else:
        raise ValueError("i is too big to fetch any window training or test data")

    return me_sets


def iid_bootstrap(i, ids, boot_params):
    # some reproducibility stuff
    rng = np.random.get_state()
    np.random.seed(seed=boot_params["init_stat"] + i)

    # resampled indices
    rs_idx_list = np.random.choice(ids, size=len(ids), replace=True)

    # indices not considered
    out_idx_list = np.delete(ids, np.unique(rs_idx_list))
    np.random.set_state(rng)

    return {"train_" + str(i): np.array(ids)[rs_idx_list].tolist(),
            "test_" + str(i): np.array(ids)[out_idx_list].tolist()}


def block_bootstrap(i, ids, boot_params):
    # some reproducibility stuff
    rng = np.random.get_state()
    np.random.seed(seed=(boot_params["init_stat"] + i))

    # form blocks
    blocks = []
    block_size = boot_params["block_size"]
    for z in range(len(ids) - block_size + 1):
        blocks.append(ids[z:(z + block_size)])

    # resample blocks
    block_idx_list = np.random.choice(len(blocks), size=int(np.ceil(len(ids) / block_size)), replace=True)

    # concatenate indices
    rs_idx_list = blocks[block_idx_list[0]]
    for z in range(1, len(block_idx_list)):
        rs_idx_list = np.concatenate([rs_idx_list, blocks[block_idx_list[z]]])

    # indices not considered
    rs_idx_list = np.array(rs_idx_list[:len(ids)])
    out_idx_list = np.delete(ids, np.unique(rs_idx_list))
    np.random.set_state(rng)

    return {"train_" + str(i): np.array(ids)[rs_idx_list].tolist(),
            "test_" + str(i): np.array(ids)[out_idx_list].tolist()}


def circular_bootstrap(i, ids, boot_params):
    # some reproducibility stuff
    rng = np.random.get_state()
    np.random.seed(seed=boot_params["init_stat"] + i)

    # get starting indices
    block_size = boot_params["block_size"]
    circ_idx_list = np.random.choice(ids, size=int(np.ceil(len(ids) / block_size)), replace=True)

    # get id's in the block and concatenate them
    rs_idx_list = []
    for z in range(len(circ_idx_list)):
        # not going to surpass the last index available
        if (ids[circ_idx_list[z]] + block_size) < len(ids):
            rs_idx_list = np.concatenate([rs_idx_list, ids[circ_idx_list[z]:circ_idx_list[z] + block_size]])
        else:  # in case it surpass
            rem = (ids[circ_idx_list[z]] + block_size) - len(ids)
            rs_idx_list = np.concatenate([rs_idx_list, ids[circ_idx_list[z]:]])
            rs_idx_list = np.concatenate([rs_idx_list, ids[:rem]])

    # indices not considered
    rs_idx_list = np.array(rs_idx_list[:len(ids)]).astype(int)
    out_idx_list = np.delete(ids, np.unique(rs_idx_list))
    np.random.set_state(rng)

    return {"train_" + str(i): np.array(ids)[rs_idx_list].tolist(),
            "test_" + str(i): np.array(ids)[out_idx_list].tolist()}


def stationary_bootstrap(i, ids, boot_params):
    # some reproducibility stuff
    rng = np.random.get_state()
    np.random.seed(seed=boot_params["init_stat"] + i)

    # get starting indices
    block_size = boot_params["block_size"]
    circ_idx_list = np.random.choice(ids, size=int(5 * np.ceil(len(ids) / block_size)), replace=True)

    # get id's in the block and concatenate them
    rs_idx_list = []
    for z in range(len(circ_idx_list)):
        # get random block size
        random_size = np.random.geometric(1.0 / block_size)
        # not going to surpass the last index available
        if (ids[circ_idx_list[z]] + random_size) < len(ids):
            rs_idx_list = np.concatenate([rs_idx_list, ids[circ_idx_list[z]:(circ_idx_list[z]+random_size)]])
        else:  # in case it surpass
            rem = (ids[circ_idx_list[z]] + random_size) - len(ids)
            rs_idx_list = np.concatenate([rs_idx_list, ids[circ_idx_list[z]:]])
            rs_idx_list = np.concatenate([rs_idx_list, ids[:rem]])

    # indices not considered
    rs_idx_list = np.array(rs_idx_list[:len(ids)]).astype(int)
    out_idx_list = np.delete(ids, np.unique(rs_idx_list))
    np.random.set_state(rng)

    return {"train_" + str(i): np.array(ids)[rs_idx_list].tolist(),
            "test_" + str(i): np.array(ids)[out_idx_list].tolist()}


def bcv_method(i, ids, bcv_params):

    # remaining folders
    test_size = bcv_params["test_size"]

    # split training and testing
    test_ids = ids[i*test_size:(i+1)*test_size]
    train_ids = ids[:i*test_size] + ids[(i+1)*test_size:]

    return {"train_" + str(i): train_ids, "test_" + str(i): test_ids}


def pcv_method(i, ids, pcv_params):  # to be tested -- api style may be hard
    # some reproducibility stuff
    rng = np.random.get_state()
    np.random.seed(seed=pcv_params["init_stat"])

    # create lists
    sub_size = pcv_params["subset_size"]
    pcv_list = [ids[z::sub_size] for z in range(sub_size)]
    test_size = int(len(pcv_list[i]) * pcv_params["test_perc"]/100.0)

    # shuffle and split
    np.random.shuffle(pcv_list[i])
    test_ids = pcv_list[i][:test_size]
    train_ids = pcv_list[i][test_size:]

    np.random.set_state(rng)
    return {"train_" + str(i): train_ids, "test_" + str(i): test_ids}


def hvbcv_method(i, ids, hvbcv_params):

    # remaining folders
    test_size = hvbcv_params["test_size"]
    gap_size = hvbcv_params["gap_size"]
    total_size = test_size + 2 * gap_size

    # creating folders
    all_idx = list(range(total_size * i, total_size * (i+1)))
    test_idx = list(range(total_size * i, total_size * (i+1)))[gap_size:-gap_size]
    test_idx = list(set(test_idx).intersection(set(range(len(ids)))))
    test_ids = [ids[x] for x in test_idx]
    train_ids = list(set(ids) - set(all_idx))

    return {"train_" + str(i): train_ids, "test_" + str(i): test_ids}


def mkcv_method(z, ids, mkcv_params):  # to be tested -- api style may be hard

    # pre-allocation
    gap_size = mkcv_params["gap_size"] # autocorrelation component
    from itertools import compress
    i, j, r, d, mkcv_folders = 1, -1, np.random.uniform(), [], {"train_" + str(z): [], "test_" + str(z): []}
    if gap_size % 3 == 0:
        m = int(2 * gap_size / 3)
    else:
        m = 2 * int(np.floor(gap_size / 3)) + 2

    # computing d
    if r < 0.25:
        d.append(i)
        i += 1
        d.append(i)
        i += 1
    elif r < 0.50:
        d.append(i)
        i += 1
        d.append(j)
        j -= 1
    elif r < 0.75:
        d.append(j)
        j -= 1
        d.append(i)
        i += 1
    else:
        d.append(j)
        j -= 1
        d.append(j)
        j -= 1

    for t in range(2, len(ids)):
        if (d[t - 1] > 0) & (d[t - 2] > 0):
            d.append(j)
            j -= 1
        elif (d[t - 1] < 0) & (d[t - 2] < 0):
            d.append(i)
            i += 1
        else:
            if np.random.uniform() > 0.5:
                d.append(j)
                j -= 1
            else:
                d.append(i)
                i += 1

    # creating subsets
    Id = list(map(lambda x: x % m + 1 + (x > 0) * m, d))
    Su = []

    for sub in range(1, (2 * m + 1)):
        Su.append(list(compress(ids, list(map(lambda x: x == sub, Id)))))

    # separating in in-sample and out-sample folders
    se = list(compress(Su[z], list(map(lambda x: x % 2 == 1, Su[z]))))
    so = list(compress(Su[z], list(map(lambda x: x % 2 == 0, Su[z]))))

    if z % 2 == 0:
        mkcv_folders["train_" + str(z)].append([ids[x] for x in so])
        mkcv_folders["test_" + str(z)].append([ids[x] for x in se])
    elif z % 2 == 1:
        mkcv_folders["train_" + str(z)].append([ids[x] for x in se])
        mkcv_folders["test_" + str(z)].append([ids[x] for x in so])

    return mkcv_folders


def cscv_method(i, ids, cscv_params):

    # create the cscv lists
    splits = cscv_params["splits"]
    half_split = int(splits / 2)
    cscv_ids = {}
    for x in range(half_split):
        cscv_ids[str(x)] = list(range(x, splits))
    cscv_ids = list(product(*[cscv_ids[k] for k in cscv_ids.keys()]))
    cscv_ids = [csid for csid in cscv_ids if np.prod(np.diff(csid) > 0)]
    cscv_ids = [list(cscv_ids[x]) + list(cscv_ids[-(x+1)]) for x in range(len(cscv_ids))]

    # associate each index with a sublist
    id_size = int(np.floor(len(ids) / splits))
    id_list = [ids[(id_size * z):(id_size * (z + 1))] for z in range(splits - 1)] + [ids[(id_size * (splits - 1)):]]

    # assign train and test folds
    return {"train_" + str(i): np.concatenate([id_list[cscv_ids[i][x]] for x in range(half_split)]),
            "test_" + str(i): np.concatenate([id_list[cscv_ids[i][x]] for x in range(half_split, splits)])}

# tscv_params["tscv_method"] in ["mbased_ar"]:

#       # pre-allocation
#       boot_samples = tscv_params["tscv_method_params"][tscv_params["tscv_method"]]["boot_samples"]
#       tscv_df = {}

#        # train model
#        import statsmodels.tsa.ar_model as ar
#        lag = ar.AR(ts).select_order(tscv_params["tscv_method_params"][
# tscv_params["tscv_method"]]["max_order"], ic="aic")
#        ts_model = ar.AR(ts).fit(lag)

#        # get predicted and residuals
#        predicted, residuals = ts_model.fittedvalues, ts_model.resid

#        # take a bootstrap sample and get indexes
#        # main tscv folders
#        for b in range(boot_samples):
#            # draw a boot sample
#            in_idx, out_idx = blockboot_method(residuals.index, "boot_stationary",
#                                               tscv_params["tscv_method_params"][tscv_params["tscv_method"]]["block_size"])

#            # compute model-based bootstrap sample
#            boot_ts_data = pd.DataFrame(predicted.values + residuals.loc[in_idx].values,
#                                        index=predicted.index, columns=ts.columns)
#            df_ts = ts2df(boot_ts_data, n_lags)
#            tscv_df["in_" + str(b)] = df_ts.iloc[:-trad_horizon]
#            tscv_df["out_" + str(b)] = df_ts.iloc[-trad_horizon:]
