"""
Classes derived from Online Portfolio Selection module

"""
# Parent Method
from mlfinlab.online_portfolio_selection.online_portfolio_selection import OLPS
from mlfinlab.online_portfolio_selection.universal_portfolio import UniversalPortfolio

# Benchmarks
from mlfinlab.online_portfolio_selection.benchmarks.buy_and_hold import BuyAndHold
from mlfinlab.online_portfolio_selection.benchmarks.best_stock import BestStock
from mlfinlab.online_portfolio_selection.benchmarks.constant_rebalanced_portfolio import ConstantRebalancedPortfolio
from mlfinlab.online_portfolio_selection.benchmarks.best_constant_rebalanced_portfolio import BestConstantRebalancedPortfolio

# Momentum
from mlfinlab.online_portfolio_selection.momentum.exponential_gradient import ExponentialGradient
from mlfinlab.online_portfolio_selection.momentum.follow_the_leader import FollowTheLeader
from mlfinlab.online_portfolio_selection.momentum.follow_the_regularized_leader import FollowTheRegularizedLeader

# Mean Reversion
# from mlfinlab.online_portfolio_selection.pattern_matching.CORN import CORN
# from mlfinlab.online_portfolio_selection.mean_reversion.PAMR import PAMR
# from mlfinlab.online_portfolio_selection.mean_reversion.CWMR import CWMR
# from mlfinlab.online_portfolio_selection.mean_reversion.OLMAR import OLMAR

# Pattern Matching
# from mlfinlab.online_portfolio_selection.pattern_matching.CORN import CORN
# from mlfinlab.online_portfolio_selection.pattern_matching.CORN_U import CORN_U
# from mlfinlab.online_portfolio_selection.pattern_matching.CORN_K import CORN_K
# from mlfinlab.online_portfolio_selection.pattern_matching.SCORN import SCORN
# from mlfinlab.online_portfolio_selection.pattern_matching.SCORN_K import SCORN_K
# from mlfinlab.online_portfolio_selection.pattern_matching.FCORN import FCORN
# from mlfinlab.online_portfolio_selection.pattern_matching.FCORN_K import FCORN_K
