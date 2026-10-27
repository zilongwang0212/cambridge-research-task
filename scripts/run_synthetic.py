#!/usr/bin/env python3
from temporal_dte.experiment import run
from temporal_dte.plotting import make_figures

if __name__ == "__main__":
    _, _, _, summary = run()
    make_figures()
    print(summary)
