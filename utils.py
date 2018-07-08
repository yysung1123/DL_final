

import pickle


def parse(num):
    with open("tagged_data/{}.pkl".format(num), "rb") as f:
        p = pickle.load(f)
    ret = []
    if type(p) is not dict:
        return ret
    for s in p['sentences']:
        s_ret = []
        for t in s['tokens']:
            s_ret.append(t['word'])
        ret.append(s_ret)
    return ret
