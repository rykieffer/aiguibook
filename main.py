#!/usr/bin/env python3
"""AIGUIBook - Main entry point for launching the Gradio GUI."""

from __future__ import annotations

import argparse
import logging
import os
import sys


def setup_logging(verbose: bool = False):
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="aiguibook-gui",
        description="AIGUIBook - AI-powered audiobook generator with character voices and emotion\n"
                    "AIGUIBook - Générateur de livres audio IA avec voix de personnages et émotions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Launch GUI on default port 7860
  python main.py --port 8080              # Launch GUI on port 8080
  python main.py --share                  # Create public shareable link
  python main.py --config config.yaml     # Use specific config file
  python main.py --verbose                # Enable debug logging
        """,
    )

    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for the Gradio web interface (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link via Gradio",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="0.0.0.0",
        help="Server hostname to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--no-queue",
        action="store_true",
        help="Disable Gradio queue (for testing)",
    )

    return parser.parse_args()


def main():
    """Main entry point: initialize config, build and launch GUI."""
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    # Initialize configuration
    from audiobook_ai.core.config import AudiobookConfig

    config = AudiobookConfig(config_path=args.config)
    config.load()

    # Validate config
    warnings = config.validate()
    for w in warnings:
        logger.warning(f"Config warning: {w}")

    # Build and launch the GUI
    from audiobook_ai.gui.app import AudiobookGUI

    gui = AudiobookGUI(config=config)
    gui.build()

    # Ensure the default projects root exists so Gradio's allowed_paths works
    os.makedirs(os.path.join(os.path.expanduser("~"), "audiobooks"), exist_ok=True)

    banner = """
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   AIGUIBook - AI Audiobook Generator                     ║
║   Générateur de Livres Audio IA                          ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """
    print(banner)

    print(f"Launching Gradio on http://{args.server_name}:{args.port}")
    if args.share:
        print("Creating public shareable link...")

    gui.launch(
        port=args.port,
        share=args.share,
        server_name=args.server_name,
    )


if __name__ == "__main__":
    main()
