import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import tempfile
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", default=str(Path(__file__).with_name("_gguf.py")))
    parser.add_argument("--mib", type=int, default=256)
    args = parser.parse_args()
    if args.mib < 4 or args.mib % 4:
        parser.error("--mib must be a positive multiple of four")
    spec = importlib.util.spec_from_file_location("pack", args.module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "input"
        root.mkdir()
        for i in range(4):
            with (root / f"{i}.bin").open("wb") as file:
                file.truncate(args.mib * 1024 * 1024 // 4)
        (root / "config.txt").write_text("chunk 50\n")
        output = Path(directory) / "model.gguf"
        start = time.perf_counter()
        module.pack(str(root), str(output), "act", "benchmark", {})
        elapsed = time.perf_counter() - start
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        digest = hashlib.sha256()
        with output.open("rb") as file:
            for chunk in iter(lambda: file.read(1 << 20), b""):
                digest.update(chunk)
        print(json.dumps({"seconds": elapsed, "peak_rss": peak_rss, "sha256": digest.hexdigest()}))


if __name__ == "__main__":
    main()
