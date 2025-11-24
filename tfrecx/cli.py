import sys
import os
import inspect


import pandas as pd

from inspect import signature
from typing import Sequence
from functools import partial

from . import core as tfrx_core

log = partial(print, file=sys.stderr, flush=True)

def cli_help(cmd_name, cmd, sig):
    log("\n\n"+("="*40)+"\n")
    log(" USAGE: ", getattr(cmd, "__doc__", cmd_name).strip())
    for k, v in sig.parameters.items():
        ann = v.annotation.__name__ if v.annotation is not inspect._empty else ""
        d = v.default if v.default is not inspect._empty else None
        log(f"** {k} {ann}: {d}")

def cast_n_bind(sig, *args, **kwargs):
    ba = sig.bind(*args, **kwargs)
    for k, v in ba.arguments.items():
        k_sig = sig.parameters[k]
        if k_sig.annotation is int or type(k_sig.default) is int:
            ba.arguments[k] = int(v)
        elif k_sig.annotation is float or type(k_sig.default) is float:
            ba.arguments[k] = float(v)
        elif k_sig.annotation is Sequence[int]:
            a = v.split(',')
            ba.arguments[k] = [int(i.strip()) for i in a]
    return ba

def cli(func):
    cmd_name = func.__name__
    sig = signature(func)
    n = len(sig.parameters)
    def wrapped():
        args = [i for i in sys.argv[1:] if "=" not in i]
        kwargs = dict([i.split('=', 1) for i in sys.argv[1:] if "=" in i])
        if args and args[0]=="help":
            cli_help(cmd_name, func, sig)
            return 0
        if n and not args and not kwargs and sig:
            cli_help(cmd_name, func, sig)
            return 0
        ba = cast_n_bind(sig, *args, **kwargs)
        return func(*ba.args, **ba.kwargs)
    return wrapped

@cli
def csv2tfrec(fn_in, fn_out=None, delimiter=","):
    """
    Converts a CSV file to a tfrec file.
    """
    if not fn_out:
        fn_out = fn_in+".tfrec"
    log(f"** II ** converting [{fn_in}] => [{fn_out}]: reading ...")
    df = pd.read_csv(fn_in, delimiter=delimiter)
    log(f"** II ** converting [{fn_in}] => [{fn_out}]: writing ...")
    tfrx_core.pd_to_tfrec(df, fn_out)
    log(f"** II ** converting [{fn_in}] => [{fn_out}]: done.")


@cli
def json2tfrec(fn_in, fn_out=None, lines: bool=True):
    """
    Converts a .json/.jsonl file to a tfrec file.
    """
    if not fn_out:
        fn_out = fn_in+".tfrec"
    log(f"** II ** converting [{fn_in}] => [{fn_out}]: reading ...")
    df = pd.read_json(fn_in, lines=lines)
    log(f"** II ** converting [{fn_in}] => [{fn_out}]: writing ...")
    tfrx_core.pd_to_tfrec(df, fn_out)
    log(f"** II ** converting [{fn_in}] => [{fn_out}]: done.")

@cli
def head(fn_in, n:int = 5):
    """
    Shows first N records in the tfrec file
    """
    for record in tfrx_core.head(fn_in, n):
        print(record)
