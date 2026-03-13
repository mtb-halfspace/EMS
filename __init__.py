"""
EMS (Emmerys Management System) Package
Centralized configuration and warning suppression for the project.
"""
import warnings
import os

# ============================================================================
# Warning Suppression Configuration
# ============================================================================
# This section suppresses non-essential warnings that clutter output without
# providing actionable information. Each filter is documented below.

# Suppress Pandas FutureWarning about mixed timezone datetime parsing
# This warning appears in load_dataset.py when parsing datetime columns
# The current behavior works correctly for our use case
warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    message='.*mixed time zones.*'
)

# Suppress all pandas FutureWarnings to avoid future deprecation noise
# Remove this if you need to see pandas deprecation warnings
warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    module='pandas.*'
)

# Suppress matplotlib font cache building messages
# These are informational only and don't affect functionality
warnings.filterwarnings(
    'ignore',
    message='.*Matplotlib is building.*'
)

# Configure matplotlib cache directory to avoid permission warnings
# This prevents warnings about .matplotlib directory not being writable
if 'MPLCONFIGDIR' not in os.environ:
    os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

# ============================================================================
# Note: To re-enable warnings for debugging, comment out the filters above
# or set the environment variable: PYTHONWARNINGS=default
# ============================================================================
