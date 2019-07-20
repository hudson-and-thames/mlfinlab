"""
Utility functions. In particular Chapter20 code on Multiprocessing and Vectorization
"""

from mlfinlab.util.fast_ewma import ewma
from mlfinlab.util.multiprocess import expand_call, lin_parts, mp_pandas_obj, nested_parts, process_jobs, process_jobs_, \
    report_progress
from mlfinlab.util.utils import get_daily_vol
