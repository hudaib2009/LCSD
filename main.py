from __future__ import annotations

import uvicorn

from backend.app.config import FASTAPI_HOST, FASTAPI_PORT


def main() -> None:
    uvicorn.run("backend.app.main:app", host=FASTAPI_HOST, port=FASTAPI_PORT, reload=False)


if __name__ == "__main__":
    main()
