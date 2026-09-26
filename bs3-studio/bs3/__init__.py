"""BS Profiler 3.1: personality characterization (Big Five with MBTI notation), emotion, voice, face and speech analytics
from a video of Russian speech, by one model chosen for the analysis (OCEAN-AI or AMLAI 1.0)."""
__version__ = "3.1.0a1"
PRODUCT = "BS Profiler 3.1"
PRODUCT_SLUG = "BS_Profiler_3"
# the models the page offers, by their internal key, and the names the user sees (change request 3.1, section 2);
# exactly one of them is loaded and run per analysis
MODEL_TITLES = {"oceanai": "OCEAN-AI", "mm": "AMLAI 1.0"}
DEFAULT_MODEL = "oceanai"
LANG = "ru"          # the speech language of every analysis (3.1: Russian only, section 1 of the change request)
