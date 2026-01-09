"""Command-line interface for sigmutsel.

This module provides command-line access to sigmutsel utilities.

Usage:
    python -m sigmutsel setup        # Download reference data
    python -m sigmutsel check-data   # Check data status
"""

import sys


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable commands:")
        print("  setup         Download reference data files")
        print("  check-data    Check status of reference data files")
        return 1

    command = sys.argv[1]

    if command == "setup":
        # Remove command from argv so setup.main() can parse args
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from sigmutsel.setup import main as setup_main

        return setup_main()

    elif command == "check-data":
        from sigmutsel.locations import print_data_status

        print_data_status()
        return 0

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
