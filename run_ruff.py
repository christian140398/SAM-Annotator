#!/usr/bin/env python3
"""
Ruff formatting and linting script.
Runs both ruff format and ruff check --fix commands.
"""

import subprocess
import sys


def run_command(command, description):
    """Run a command and return success status."""
    print(f"\n🔄 {description}...")
    print(f"Running: {command}")

    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError:
        print(f"❌ {description} failed")
        return False


def main():
    """Run ruff format and check commands."""
    # Determine Python command based on argument
    python_cmd = "python"  # Default to py -3.11
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "1":
            python_cmd = "py -3.11"

        else:
            print(f"❌ Error: Invalid argument '{arg}'")
            sys.exit(1)

    print("🚀 Starting Ruff formatting and linting...")
    print(f"Using Python command: {python_cmd}")

    success = True

    # Run ruff format
    if not run_command(f"{python_cmd} -m ruff format .", "Ruff formatting"):
        success = False

    # Run ruff check --fix
    if not run_command(
        f"{python_cmd} -m ruff check . --fix", "Ruff linting and fixing"
    ):
        success = False

    if success:
        print("\n🎉 All Ruff commands completed successfully!")
        sys.exit(0)
    else:
        print("\n⚠️  Some Ruff commands failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
