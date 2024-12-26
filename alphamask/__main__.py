#!/usr/bin/env python3
import sys
from pathlib import Path

def print_help():
    """Print help message with available commands"""
    print("AlphaMask - A tool for protein structure prediction and analysis")
    print("\nAvailable commands:")
    print("  setup      Set up experiment directories")
    print("  run        Run experiments")
    print("  predict    Run individual predictions")
    print("  predict-job Run predictions in SLURM job")
    print("  check-jax  Check JAX/CUDA installation")
    print("\nUsage:")
    print("  python -m alphamask <command> [options]")
    print("\nFor help with a specific command:")
    print("  python -m alphamask <command> --help")

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ['-h', '--help']:
        print_help()
        sys.exit(0)

    command = sys.argv[1]
    # Remove the command from sys.argv so the subcommand parsers work correctly
    sys.argv.pop(1)

    try:
        if command == 'setup':
            from alphamask.scripts.setup_experiments import main as setup_main
            setup_main()
        elif command == 'run':
            from alphamask.scripts.run_experiments import main as run_main
            run_main()
        elif command in ['predict', 'predict-job']:
            from alphamask.cli.main import main as cli_main
            sys.argv.insert(1, command)  # Put the command back for the CLI parser
            cli_main()
        elif command == 'check-jax':
            from alphamask.scripts.check_jax import main as check_main
            check_main()
        else:
            print(f"Unknown command: {command}")
            print_help()
            sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 