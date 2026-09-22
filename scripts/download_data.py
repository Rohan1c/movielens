"""Fetch and verify the MovieLens 100K archive.

Run this once before anything else::

    python scripts/download_data.py

The archive is downloaded to ``data/raw/ml-100k.zip``, checked twice - against the MD5 that
GroupLens publish alongside it, and against the MD5 recorded in ``configs/default.yaml`` -
and extracted to ``data/raw/ml-100k/``. Nothing under ``data/`` is committed, so this script
is the only way a fresh clone gets the dataset.

A checksum mismatch is treated as a hard failure. Silently training on a truncated or
substituted archive would invalidate every number in results/ with no visible symptom.

Certificate trust on Windows
----------------------------

Windows does not preload every trusted root certificate. Its own TLS stack fetches a
missing root on demand the first time it needs one, but Python reads the certificate store
as it stands and never triggers that fetch. GroupLens's current certificate chains to a
root that a fresh machine may not have cached yet, so on such a machine Python fails with
"unable to get local issuer certificate" while a browser loads the same URL fine.

If the system store cannot verify the chain, the download retries against certifi's copy
of Mozilla's root list. Verification is never switched off: the retry is a second, well
maintained trust list, and it makes the download behave the same on every machine
regardless of what that machine happens to have cached.
"""

import argparse
import hashlib
import shutil
import ssl
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# Running a file inside scripts/ puts scripts/ on the import path, not the repository
# root, so "import src" would fail on a fresh clone unless the package had been installed
# first. Putting the root on the path here means the documented commands work straight
# after a git clone, with pip install -e . left as an optional convenience.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.config import load_config

CHUNK_SIZE = 1024 * 256
EXPECTED_FILES = ["u.data", "u.item", "u.genre", "u.user"]


def open_url(url):
    """Open a URL with full certificate verification, trying two trust lists in turn."""
    try:
        return urllib.request.urlopen(url)
    except urllib.error.URLError as error:
        if not isinstance(error.reason, ssl.SSLCertVerificationError):
            raise
        try:
            import certifi
        except ImportError:
            raise SystemExit(
                "The system certificate store could not verify " + url + ".\n"
                "Install certifi (pip install -r requirements.txt) and retry; it supplies "
                "Mozilla's root list, which covers this certificate."
            )
        print("system store could not verify the chain; retrying with certifi's roots")
        context = ssl.create_default_context(cafile=certifi.where())
        return urllib.request.urlopen(url, context=context)


def download_archive(url, destination):
    print("downloading " + url)
    with open_url(url) as response:
        total = response.headers.get("Content-Length")
        if total is None:
            total_bytes = 0
        else:
            total_bytes = int(total)
        downloaded = 0
        with open(destination, "wb") as handle:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded = downloaded + len(chunk)
                report_progress(downloaded, total_bytes)
    print("")
    print("saved " + str(destination) + " (" + str(destination.stat().st_size) + " bytes)")


def report_progress(downloaded, total_bytes):
    if total_bytes <= 0:
        return
    percent = 100.0 * downloaded / total_bytes
    sys.stdout.write("\r  " + format(percent, ".1f") + "%")
    sys.stdout.flush()


def compute_md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def fetch_published_md5(url):
    """Read GroupLens's own published checksum for the archive.

    Their file is of the form "MD5 (ml-100k.zip) = <hash>", so the hash is the last token.
    """
    with open_url(url) as response:
        text = response.read().decode("utf-8", errors="replace")
    tokens = text.strip().split()
    if len(tokens) == 0:
        raise ValueError("published checksum file at " + url + " was empty")
    return tokens[-1].strip().lower()


def verify_checksum(path, expected, published=None):
    observed = compute_md5(path)
    if published is not None and published != expected:
        raise ValueError(
            "the checksum published by GroupLens (" + published + ") does not match the "
            "one pinned in configs/default.yaml (" + expected + "). Do not proceed until "
            "you know which is right and why they differ."
        )
    if observed == expected:
        print("checksum ok: " + observed)
        if published is not None:
            print("  confirmed against the checksum published by GroupLens")
        return
    message = [
        "checksum mismatch for " + str(path),
        "  expected: " + str(expected),
        "  observed: " + observed,
        "Delete the file and retry. If the observed hash is stable across retries the",
        "published archive has changed; update download.md5 in configs/default.yaml and",
        "say so in the report, because it means the dataset itself moved.",
    ]
    raise ValueError("\n".join(message))


def extract_archive(archive_path, raw_dir, extract_subdir):
    target = raw_dir / extract_subdir
    if target.exists():
        print("removing previous extraction at " + str(target))
        shutil.rmtree(target)
    print("extracting to " + str(raw_dir))
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(raw_dir)
    if not target.exists():
        raise FileNotFoundError("archive did not contain " + extract_subdir + "/")
    return target


def check_expected_files(extracted_dir):
    missing = []
    for name in EXPECTED_FILES:
        if not (extracted_dir / name).exists():
            missing.append(name)
    if len(missing) > 0:
        raise FileNotFoundError("extraction is missing: " + ", ".join(missing))
    print("found all expected files: " + ", ".join(EXPECTED_FILES))


def main():
    parser = argparse.ArgumentParser(description="Download and verify MovieLens 100K.")
    parser.add_argument("--config", default=None, help="path to a config file")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    config = load_config(args.config)
    settings = config.section("download")
    raw_dir = config.path("raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / settings["archive_name"]

    if archive_path.exists() and not args.force:
        print("archive already present at " + str(archive_path))
    else:
        download_archive(settings["url"], archive_path)

    published = None
    if "md5_url" in settings:
        published = fetch_published_md5(settings["md5_url"])

    verify_checksum(archive_path, settings["md5"], published)
    extracted_dir = extract_archive(archive_path, raw_dir, settings["extract_subdir"])
    check_expected_files(extracted_dir)
    print("done. next: python scripts/preprocess.py")


if __name__ == "__main__":
    main()
