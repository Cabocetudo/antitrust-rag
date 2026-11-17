"""Download the official texts (or load the offline sample) and index them.

python scripts/ingest.py                       # download Lei 12.529/2011 + TFEU 101/102
python scripts/ingest.py --origin sample       # offline: index data/sample/
python scripts/ingest.py --download-only --save-raw data/raw
"""

import sys

from antitrust_rag.cli import ingest_main

if __name__ == "__main__":
    sys.exit(ingest_main())
