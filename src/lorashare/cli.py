"""CLI for lorashare: compress, inspect, and reconstruct LoRA adapters."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Configure logging for CLI
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="lorashare",
        description="Compress multiple PEFT LoRA adapters into a shared subspace",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── compress ──
    compress_p = subparsers.add_parser(
        "compress",
        help="Compress multiple LoRA adapters into SHARE format",
    )
    compress_p.add_argument(
        "adapters",
        nargs="+",
        help="Paths or HuggingFace Hub IDs of PEFT LoRA adapters",
    )
    compress_p.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory for SHARE checkpoint",
    )
    compress_p.add_argument(
        "-k", "--num-components",
        default="32",
        help="Number of shared components (integer or 'auto')",
    )
    compress_p.add_argument(
        "--variance",
        type=float,
        default=0.95,
        help="Explained variance threshold when using -k auto (default: 0.95)",
    )
    compress_p.add_argument(
        "--names",
        nargs="+",
        default=None,
        help="Custom names for each adapter (must match adapter count)",
    )
    compress_p.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device for computation (default: auto, uses GPU if available)",
    )
    compress_p.add_argument(
        "--layer-by-layer",
        action="store_true",
        help="Process one layer at a time (reduces memory usage)",
    )
    compress_p.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Process adapters in chunks (enables 100+ adapters)",
    )

    # ── info ──
    info_p = subparsers.add_parser(
        "info",
        help="Print compression statistics for a SHARE checkpoint",
    )
    info_p.add_argument("checkpoint", help="Path to SHARE checkpoint directory")

    # ── reconstruct ──
    recon_p = subparsers.add_parser(
        "reconstruct",
        help="Reconstruct adapter(s) from SHARE checkpoint to standard PEFT format",
    )
    recon_p.add_argument("checkpoint", help="Path to SHARE checkpoint directory")
    recon_p.add_argument("--adapter", help="Name of adapter to reconstruct")
    recon_p.add_argument(
        "--all", action="store_true", dest="reconstruct_all",
        help="Reconstruct all adapters",
    )
    recon_p.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory for reconstructed adapter(s)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "compress":
        _cmd_compress(args)
    elif args.command == "info":
        _cmd_info(args)
    elif args.command == "reconstruct":
        _cmd_reconstruct(args)


def _cmd_compress(args: argparse.Namespace) -> None:
    from lorashare import SHAREModel

    num_components: int | str
    if args.num_components == "auto":
        num_components = "auto"
    else:
        num_components = int(args.num_components)

    if args.names:
        if len(args.names) != len(args.adapters):
            print(
                f"Error: --names count ({len(args.names)}) "
                f"must match adapter count ({len(args.adapters)})",
                file=sys.stderr,
            )
            sys.exit(1)
        adapters: list[str] | dict[str, str] = dict(zip(args.names, args.adapters))
    else:
        adapters = args.adapters

    # Determine device
    device = None if args.device == "auto" else args.device

    # Show configuration
    print(f"Compressing {len(args.adapters)} adapters with k={num_components}...")
    if device:
        print(f"Using device: {device}")
    if args.layer_by_layer:
        print("Using layer-by-layer processing")
    if args.chunk_size:
        print(f"Using chunked processing (chunk_size={args.chunk_size})")

    share = SHAREModel.from_adapters(
        adapters,
        num_components=num_components,
        variance_threshold=args.variance,
        device=device,
        layer_by_layer=args.layer_by_layer,
        chunk_size=args.chunk_size,
    )
    share.save_pretrained(args.output)
    share.summary()
    print(f"Saved to {args.output}")


def _cmd_info(args: argparse.Namespace) -> None:
    from lorashare import SHAREModel

    share = SHAREModel.from_pretrained(args.checkpoint)
    share.summary()


def _cmd_reconstruct(args: argparse.Namespace) -> None:
    from lorashare import SHAREModel

    share = SHAREModel.from_pretrained(args.checkpoint)

    if args.reconstruct_all:
        names = share.adapter_names
    elif args.adapter:
        names = [args.adapter]
    else:
        print("Error: specify --adapter NAME or --all", file=sys.stderr)
        sys.exit(1)

    for name in names:
        out_dir = Path(args.output) / name if len(names) > 1 else Path(args.output)
        print(f"Reconstructing {name} -> {out_dir}")
        share.reconstruct(name, output_dir=out_dir)

    print("Done.")
