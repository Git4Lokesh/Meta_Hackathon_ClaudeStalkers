"""Root-level server/app.py — re-exports the FastAPI app from sre_env."""

from sre_env.server.app import app  # noqa: F401


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
