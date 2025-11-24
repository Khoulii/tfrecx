import sys
import os
import optparse

import pandas as pd

import tfrecx.core

from functools import partial

log = partial(print, file=sys.stderr, flush=True)

def csv2tfrec():
    fn_in = sys.argv[1]
    fn_out = sys.argv[2]
    log(f"** II ** converting [{fn_in}] => [{fn_out}]: reading ...")
    df = pd.read_csv(fn_in)
    log(f"** II ** converting [{fn_in}] => [{fn_out}]: writing ...")
    tfrecx.core.pd_to_tfrec(df, fn_out)
    log(f"** II ** converting [{fn_in}] => [{fn_out}]: done.")
