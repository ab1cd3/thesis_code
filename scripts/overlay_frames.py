# scripts/overlay_frames.py
import argparse
from model_eval.vis.overlay import overlay_frames

def main():
    parser = argparse.ArgumentParser(
        description="Create GT+PRED overlay images for a series and multiple frames."
    )
    parser.add_argument("--series", required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    args = parser.parse_args()

    overlay_frames(video_series=args.series, frames=args.frames)

if __name__ == "__main__":
    main()