#!/usr/bin/env python3
import sys

from recdiffusion.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "evaluate")
    main()
