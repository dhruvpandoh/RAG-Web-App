import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


def run_command(command: list[str]) -> None:
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def install_dependencies() -> None:
    run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def run_etl() -> None:
    run_command([sys.executable, "etl_pipeline.py"])


def run_featurization() -> None:
    run_command([sys.executable, "featurization_pipeline.py"])


def run_api() -> None:
    run_command([sys.executable, "main.py"])


def run_ui() -> None:
    run_command([sys.executable, "gradio_ui.py"])


def docker_compose_up() -> None:
    run_command(["docker", "compose", "up", "--build"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Deployment helper for ROS2 RAG Web App")
    parser.add_argument(
        "command",
        choices=["install", "etl", "featurize", "api", "ui", "docker"],
        help="Action to run",
    )

    args = parser.parse_args()

    if args.command == "install":
        install_dependencies()
    elif args.command == "etl":
        run_etl()
    elif args.command == "featurize":
        run_featurization()
    elif args.command == "api":
        run_api()
    elif args.command == "ui":
        run_ui()
    elif args.command == "docker":
        docker_compose_up()


if __name__ == "__main__":
    main()