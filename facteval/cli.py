"""
CLI – Command-line interface for FactEval.

Usage:
    facteval check input.json
    facteval check input.json --output output.json
    facteval check --answer "..." --context "ctx1" --context "ctx2"
"""

import argparse
import json
import sys
import logging


def main():
    """Entry point for the facteval CLI."""
    parser = argparse.ArgumentParser(
        prog="facteval",
        description="FactEval – Claim-level factuality evaluation with calibrated confidence.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── facteval check ───────────────────────────────────────────────────────
    check_parser = subparsers.add_parser(
        "check", help="Check an answer for factual accuracy against provided contexts."
    )
    check_parser.add_argument(
        "input_file", nargs="?", default=None,
        help='JSON file with "answer" and "contexts" keys.',
    )
    check_parser.add_argument(
        "--answer", "-a", type=str, default=None,
        help="The answer text to check (alternative to input file).",
    )
    check_parser.add_argument(
        "--context", "-c", action="append", default=None,
        help="Context passage (can be repeated). Alternative to input file.",
    )
    check_parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output file path. If not provided, prints to stdout.",
    )
    check_parser.add_argument(
        "--calibrator", type=str, default=None,
        help="Path to a pre-fitted calibrator pickle file.",
    )
    check_parser.add_argument(
        "--top-k", type=int, default=3,
        help="Number of evidence sentences to retrieve per claim (default: 3).",
    )
    check_parser.add_argument(
        "--max-claims", type=int, default=10,
        help="Maximum number of claims to extract (default: 10).",
    )
    check_parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "check":
        _run_check(args)


def _run_check(args):
    """Execute the check command."""
    # Configure logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(name)s | %(message)s")

    # Parse input
    answer, contexts = _parse_input(args)
    if answer is None:
        print("Error: Provide either an input JSON file or --answer + --context flags.", file=sys.stderr)
        sys.exit(1)

    # Import here to avoid slow import on --help
    from facteval.core import check

    # Run pipeline
    result = check(
        answer=answer,
        contexts=contexts,
        top_k=args.top_k,
        max_claims=args.max_claims,
        calibrator_path=args.calibrator,
    )

    # Output
    output_json = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"Results saved to {args.output}", file=sys.stderr)
    else:
        print(output_json)


def _parse_input(args) -> tuple[str | None, list[str]]:
    """Parse answer and contexts from file or CLI flags."""
    # Option 1: JSON file
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return data.get("answer"), data.get("contexts", [])

    # Option 2: CLI flags
    if args.answer:
        return args.answer, args.context or []

    # Option 3: stdin
    if not sys.stdin.isatty():
        data = json.load(sys.stdin)
        return data.get("answer"), data.get("contexts", [])

    return None, []


if __name__ == "__main__":
    main()
