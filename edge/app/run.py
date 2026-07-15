"""Unified PipeSD Edge launcher for all supported modalities."""

import argparse
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="PipeSD unified Edge launcher")
    parser.add_argument("modality", choices=["text", "video"])
    if not argv or argv[0] in {"-h", "--help"}:
        parser.parse_args(argv)
    modality, remaining = argv[0], argv[1:]
    if modality not in {"text", "video"}:
        parser.error(f"invalid modality: {modality!r}")
    if modality == "text":
        from edge.app.run_edge import main as run_text
        return run_text(remaining)
    from edge.app.run_video_edge import main as run_video
    return run_video(remaining)


if __name__ == "__main__":
    main()
