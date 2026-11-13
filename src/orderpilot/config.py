"""Policy constants and runtime settings.

Business-rule constants live here so tools, tests and evals share one source of truth.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

load_dotenv()

# --- Policy ------------------------------------------------------------------
RETURN_WINDOW_DAYS: Final = 30
STUCK_DAYS: Final = 7
LOST_DAYS: Final = 14
MAX_AUTH_ATTEMPTS: Final = 3

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
DB_PATH: Final = Path(os.getenv("ORDERPILOT_DB_PATH", PROJECT_ROOT / "ecommerce.sqlite"))
