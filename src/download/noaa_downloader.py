from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.config.settings import get_config


class NOAADownloader:
    """Fetch NOAA AIS daily files from the official NOAA archive when accessible."""

    def __init__(self, base_dir: str | Path | None = None):
        config = get_config()
        self.base_dir = Path(base_dir) if base_dir is not None else config.raw_dir
        self.base_url = config.noaa_base_url
        self.logger = logging.getLogger("noaa_downloader")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def candidate_urls(self, target_date: str) -> List[str]:
        date = datetime.strptime(target_date, "%Y-%m-%d")
        y = date.strftime("%Y")
        m = date.strftime("%m")
        d = date.strftime("%d")
        base = self.base_url.rstrip("/")
        return [f"{base}/csv2/csv{y}/ais-{y}-{m}-{d}.csv.zst"]

    def resolve_available_url(self, target_date: str) -> str | None:
        for candidate in self.candidate_urls(target_date):
            try:
                req = Request(
                    candidate, method="HEAD", headers={"User-Agent": "Mozilla/5.0"}
                )
                with urlopen(req, timeout=20) as response:
                    if response.status < 400:
                        return candidate
            except (HTTPError, URLError, TimeoutError, OSError):
                continue
        return None

    def download_file(
        self, target_date: str, output_path: str | Path | None = None
    ) -> dict:
        url = self.resolve_available_url(target_date)
        final_path = (
            Path(output_path)
            if output_path is not None
            else self.base_dir / f"{target_date}.csv.zst"
        )
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if url is None:
            return {
                "date": target_date,
                "status": "not_found",
                "path": str(final_path),
                "url": None,
            }
        if final_path.exists() and final_path.stat().st_size > 0:
            return {
                "date": target_date,
                "status": "already_downloaded",
                "path": str(final_path),
                "url": url,
            }

        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=60) as response, open(final_path, "wb") as fh:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
            return {
                "date": target_date,
                "status": "downloaded",
                "path": str(final_path),
                "url": url,
            }
        except Exception as exc:
            self.logger.error("Download failed for %s: %s", target_date, exc)
            return {
                "date": target_date,
                "status": "error",
                "path": str(final_path),
                "url": url,
                "error": str(exc),
            }

    def download_date_range(self, start: str, end: str) -> List[dict]:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        results: List[dict] = []
        current = start_dt
        while current <= end_dt:
            results.append(self.download_file(current.strftime("%Y-%m-%d")))
            current += timedelta(days=1)
        return results

    def download_dates(self, dates: Iterable[str]) -> List[dict]:
        return [self.download_file(date) for date in dates]

    def compute_sha256(self, path: str | Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
