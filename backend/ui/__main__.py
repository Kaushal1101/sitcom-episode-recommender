import os
from pathlib import Path

from backend.ui.session_loop import run

db_path = Path(os.getenv("SITCOM_DB_PATH", "data/app.sqlite3"))
chroma_path = Path(os.getenv("SITCOM_CHROMA_PATH", "data/chroma"))
run(db_path=db_path, chroma_path=chroma_path)
