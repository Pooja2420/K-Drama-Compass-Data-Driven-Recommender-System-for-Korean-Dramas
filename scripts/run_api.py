"""
Launch the K-Drama Compass FastAPI server with Uvicorn.

Usage
-----
    python scripts/run_api.py
    python scripts/run_api.py --host 0.0.0.0 --port 8080 --reload
    python scripts/run_api.py --workers 4          # multi-process (prod)

After starting, open:
    Swagger UI : http://localhost:8000/docs
    ReDoc      : http://localhost:8000/redoc
    Health     : http://localhost:8000/health

Demo credentials for POST /auth/token:
    username=demo  password=kdrama123
"""

import argparse

import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run K-Drama Compass API")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable hot-reload for development (disables multi-worker)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of Uvicorn worker processes (default: 1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Hot-reload is incompatible with multiple workers
    workers = 1 if args.reload else args.workers
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
