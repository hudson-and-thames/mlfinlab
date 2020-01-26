"""
Varoius codependence measure: mutual info, distance correlations, variation of information
"""

from mlfinlab.codependence.correlation import angular_distance, absolute_angular_distance, squared_angular_distance
from mlfinlab.codependence.information import mutual_info_score, get_optimal_number_of_bins, \
    variation_of_information_score