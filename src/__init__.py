import warnings
import numpy as np

# Configure global NumPy and Warning filters to prevent third-party library noise (CARL / ConfigSpace)
np.seterr(divide="ignore", invalid="ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered in.*divide.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Module .* not found.*")
