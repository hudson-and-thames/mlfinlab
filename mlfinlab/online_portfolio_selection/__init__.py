"""
Classes derived from Online Portfolio Selection module

STILL THINKING OF WAYS TO USE THIS
"""
from . import olps_utils
from mlfinlab.online_portfolio_selection.OLPS import OLPS
# Benchmarks
from mlfinlab.online_portfolio_selection.benchmarks.BAH import BAH
from mlfinlab.online_portfolio_selection.benchmarks.BESTSTOCK import BESTSTOCK
from mlfinlab.online_portfolio_selection.benchmarks.CRP import CRP
from mlfinlab.online_portfolio_selection.benchmarks.BCRP import BCRP
# Momentum
from mlfinlab.online_portfolio_selection.UP import UP
from mlfinlab.online_portfolio_selection.momentum.EG import EG
from mlfinlab.online_portfolio_selection.momentum.FTL import FTL
from mlfinlab.online_portfolio_selection.momentum.FTRL import FTRL
# Mean Reversion
from mlfinlab.online_portfolio_selection.pattern_matching.CORN import CORN
from mlfinlab.online_portfolio_selection.mean_reversion.PAMR import PAMR
from mlfinlab.online_portfolio_selection.mean_reversion.CWMR import CWMR
from mlfinlab.online_portfolio_selection.mean_reversion.OLMAR import OLMAR
# Pattern Matching
from mlfinlab.online_portfolio_selection.pattern_matching.CORN import CORN
from mlfinlab.online_portfolio_selection.pattern_matching.CORN_U import CORN_U
from mlfinlab.online_portfolio_selection.pattern_matching.CORN_K import CORN_K
