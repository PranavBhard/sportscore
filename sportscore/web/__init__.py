"""Unified sportscore web application."""

import argparse


def run():
    """Entry point for ``sportscore-web`` console script."""
    parser = argparse.ArgumentParser(description="Sportscore unified web app")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    from sportscore.web.app import create_app

    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)
