# flake8: noqa
"""
This library used heavy code from porespy, but we decided to not depend on it directly,
to avoid maintanance hell, and instead copy the necessary code (and simplify)
"""


from . import feature_extractors
from . import porosimetry
from . import basicmetrics
# from . import surface_area
# from . import permeability
# from . import permeability_from_lbm
# from . import curvature

from .feature_extractors import (make_feature_extractor,
                                 make_composite_feature_extractor,
                                 AVAILABLE_EXTRACTORS,
                                 EXTRACTORS_RETURN_KEYS_MAP)
from .dataset_variance import diagonal_variance