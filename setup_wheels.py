"""Compatibility entry point; the packaged implementation lives in initium.setup_wheels."""

import sys

from initium.setup_wheels import main

if __name__ == "__main__":
    sys.exit(main())
