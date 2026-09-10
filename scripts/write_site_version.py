"""Write the deployment marker consumed by version-sync.js."""
import json
from pathlib import Path
import sys


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: write_site_version.py SITE_DIR VERSION")

    site_dir = Path(sys.argv[1])
    version = sys.argv[2].strip()
    if not site_dir.is_dir() or not version:
        raise SystemExit("site directory and version are required")

    marker = site_dir / "site-version.json"
    marker.write_text(json.dumps({"version": version}) + "\n", encoding="utf-8")
    print(f"Wrote deployment marker: {version}")


if __name__ == "__main__":
    main()
