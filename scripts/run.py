import argparse
import logging

import uvicorn

from visionflow.api.app import create_app
from visionflow.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cameras.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    uvicorn.run(create_app(cfg), host=cfg.api.host, port=cfg.api.port)


if __name__ == "__main__":
    main()
