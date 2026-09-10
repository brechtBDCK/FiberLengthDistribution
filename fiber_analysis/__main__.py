import argparse
from pathlib import Path

from .config import load
from .pipeline import analyze_image, images


def main():
    parser = argparse.ArgumentParser(prog="python -m fiber_analysis")
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--input", required=True)
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--methods", default="adaptive_skeleton,multiscale_ridge,line_segments")
    analyze.add_argument("--config")
    analyze.add_argument("--units-per-pixel", type=float)
    analyze.add_argument("--unit-name", default="px")
    analyze.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    cfg = load(args.config)
    found = images(Path(args.input))
    for index, path in enumerate(found, start=1):
        print(f"[{index}/{len(found)}] {path}", flush=True)
        analyze_image(path, Path(args.output), args.methods.split(","), cfg, args.units_per_pixel, args.unit_name, args.debug)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    print(f"Analyzed {len(found)} image(s); results: {args.output}")


if __name__ == "__main__":
    main()
