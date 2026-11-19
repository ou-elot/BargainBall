"""Download or verify the source files used by the expanded BargainBall dataset."""

import argparse
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile, is_zipfile

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "basketball_reference_latest.zip": {"url": "https://www.kaggle.com/api/v1/datasets/download/sumitrodatta/nba-aba-baa-stats", "members": ["Advanced.csv", "Player Per Game.csv", "Player Season Info.csv"], "directory": "basketball_reference"},
    "salaries/historical_salaries.csv": {"url": "https://raw.githubusercontent.com/logan-lauton/NBA-Player-HUD/main/NBA%20Salaries%281990-2023%29.csv", "required_columns": ["playerName", "seasonStartYear", "salary"]},
    "salaries/bbref_historical_salaries.csv": {"url": "https://raw.githubusercontent.com/ramvibhakar/basketball-analytics/master/data/player_salary.csv", "required_columns": ["season", "url", "salary", "team_url"]},
    "salaries/alternative_salaries.csv": {"url": "https://raw.githubusercontent.com/erikgregorywebb/datasets/master/nba-salaries.csv", "required_columns": ["name", "salary", "season"]},
    "salaries/hoopshype_snapshot.csv": {"url": "https://huggingface.co/datasets/Mr-Bridge/nba-salary-cap-contracts-2016-2026/resolve/main/player_salaries.csv", "required_columns": ["player", "salary", "season_year", "team_id"]}
}

for year in range(1997, 2027):
    filename = "sportsdataverse/player_season_stats_" + str(year) + ".parquet"
    SOURCES[filename] = {"url": "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/nba_stats_player_season_stats/player_season_stats_" + str(year) + ".parquet"}

for year in range(2010, 2027):
    filename = "bbref_salaries/bbref_salaries_" + str(year) + ".parquet"
    SOURCES[filename] = {"url": "https://basketball-data.billmill.org/nba/bbref/bbref_salaries/season=" + str(year) + "/data.parquet"}


def validate(blob, filename, source):
    members = {}

    if "members" in source:
        if not is_zipfile(io.BytesIO(blob)):
            raise ValueError(filename + ": response is not a ZIP archive")

        with ZipFile(io.BytesIO(blob)) as archive:
            for member in source["members"]:
                if member not in archive.namelist():
                    raise ValueError(filename + ": missing expected file " + member)

                members[member] = archive.read(member)

    elif filename.endswith(".parquet"):
        if blob[:4] != b"PAR1" or blob[-4:] != b"PAR1":
            raise ValueError(filename + ": response is not a Parquet file")

    else:
        reader = csv.DictReader(io.StringIO(blob.decode("utf-8-sig")))

        for column in source["required_columns"]:
            if reader.fieldnames is None or column not in reader.fieldnames:
                raise ValueError(filename + ": missing expected column " + column)

    return members


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []

    for filename, source in SOURCES.items():
        path = args.output / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        if args.refresh or not path.exists():
            print("Downloading " + filename, flush=True)

            for attempt in range(3):
                try:
                    request = Request(source["url"], headers={"User-Agent": "BargainBall-academic-project/1.0"})

                    with urlopen(request, timeout=120) as response:
                        blob = response.read()

                    members = validate(blob, filename, source)
                    temporary = path.with_suffix(path.suffix + ".part")
                    temporary.write_bytes(blob)
                    temporary.replace(path)
                    break
                except Exception:
                    if attempt == 2:
                        raise

        else:
            blob = path.read_bytes()
            members = validate(blob, filename, source)

        for member, data in members.items():
            destination = args.output / source["directory"]
            destination.mkdir(parents=True, exist_ok=True)
            (destination / member).write_bytes(data)

        records.append({"file": filename, "url": source["url"], "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest(), "verified_utc": datetime.now(timezone.utc).isoformat()})

    # This curated lookup is supplied in the project, not rebuilt from guesses.
    lookup = args.output / "salaries" / "BargainBall_listed_salaries.csv"

    if not lookup.exists():
        raise FileNotFoundError("Keep the bundled data/raw/salaries/BargainBall_listed_salaries.csv file.")

    blob = lookup.read_bytes()
    records.append({"file": str(lookup.relative_to(args.output)), "source": "previous project salary reconciliation", "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()})
    (args.output / "download_manifest.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print("Verified " + str(len(records)) + " source files. Run python prepare_data.py next.")


if __name__ == "__main__":
    main()
