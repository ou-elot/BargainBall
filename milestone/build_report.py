"""Build the milestone using installed TeX or a temporary portable Tectonic.

Run from CS230_Final: .venv/Scripts/python.exe milestone/build_report.py
"""

from pathlib import Path
import hashlib
import io
import json
import shutil
import subprocess
import urllib.request
import zipfile
from prepare_assets import main as prepare_assets

REPORT = Path(__file__).resolve().parent
TEMP = Path.home() / "AppData" / "Local" / "Temp" / "opencode"


def portable_tectonic():
    if(not TEMP.is_dir()):
        raise FileNotFoundError("An existing temporary build directory is required.")

    folder = TEMP / "cs230_tectonic_0_17_0"
    folder.mkdir(exist_ok=True)
    executable = folder / "tectonic.exe"

    if(not executable.exists()):
        request = urllib.request.Request("https://api.github.com/repos/tectonic-typesetting/tectonic/releases/tags/tectonic%400.17.0", headers={"User-Agent": "CS230-Milestone-Build"})
        release = json.load(urllib.request.urlopen(request, timeout=60))
        selected = None

        for asset in release["assets"]:
            if(asset["name"] == "tectonic-0.17.0-x86_64-pc-windows-msvc.zip"):
                selected = asset

        if(selected is None):
            raise ValueError("Official Windows compiler asset was not found.")

        archive = urllib.request.urlopen(selected["browser_download_url"], timeout=120).read()
        digest = selected.get("digest")

        if(digest and digest.startswith("sha256:") and hashlib.sha256(archive).hexdigest() != digest.split(":", 1)[1]):
            raise ValueError("Downloaded compiler hash does not match the official release.")

        with zipfile.ZipFile(io.BytesIO(archive)) as files:
            for name in files.namelist():
                if(Path(name).name == "tectonic.exe"):
                    executable.write_bytes(files.read(name))

        if(not executable.exists()):
            raise FileNotFoundError("The compiler archive did not contain tectonic.exe.")

    return executable


def main():
    prepare_assets()
    latexmk = shutil.which("latexmk")

    if(latexmk):
        command = [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    else:
        tectonic = shutil.which("tectonic")

        if(not tectonic):
            tectonic = str(portable_tectonic())

        command = [tectonic, "--keep-logs", "--keep-intermediates", "main.tex"]

    subprocess.run(command, cwd=REPORT, check=True)
    with zipfile.ZipFile(REPORT / "milestone_overleaf.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ["main.tex", "references.bib", "neurips_2026.sty", "figures/composite_losses.pdf", "README.md"]:
            archive.write(REPORT / name, arcname=name)

    print ("Built milestone PDF: " + str(REPORT / "main.pdf"))
    print ("Built Overleaf source package: " + str(REPORT / "milestone_overleaf.zip"))


if(__name__ == "__main__"):
    main()
