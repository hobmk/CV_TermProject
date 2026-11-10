"""Build a diversified 200-ticker, 2010-2021 JKX I20/R5 dataset.

Run from the CV_TermProject repository root:
  python -m scripts.prepare_exp200

Reuses the EXACT original 100-ticker membership from data/exp100/processed/samples.csv,
then adds 100 historically listed companies from two *candidate* pools, diversified
across sectors. Pools are not verified market-cap buckets. Yahoo Finance provides
survivor-only historical data; see the generated manifest for limitations.

Resumable: raw CSVs are cached; images.npy is a single memory-mapped array;
completed per-ticker renders are journaled. Existing exp100 data are untouched.
"""
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

START, END = "2010-01-01", "2022-01-01"
WINDOW, HORIZON = 20, 5
COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]

# Pre-specified candidate pools: deliberately diversified and largely outside
# the original mega-cap universe. These are *not certified historical mid/small
# capitalization classes*. Selection labels report candidate intent only.
# A ticker's current market cap MUST NOT be used to claim 2010 market cap.
POOLS = {
 "mid_candidate": {
  "Technology": "SWKS TER VRSN FFIV JNPR MSI NTAP CRUS LSCC ON IPGP OLED MANH TYL WDC STX CACI MKSI KLIC VSH NXPI",
  "Financials": "HBAN RF KEY FITB CMA ZION MTB FNB EWBC CBSH OZK HWC BOH GBCI UMBF SNV BOKF IBOC CFFN WAFD WSFS",
  "Healthcare": "DXCM EXAS NBIX RGEN TECH BIO CRL WST QDEL NTRA HAE ICUI MASI UHS EHC ENSG SEM CHE LMAT MMSI",
  "Industrials": "WAB GATX WERN SAIA ODFL GPI ITRI KAI AOS LECO MSA LII TTEK FIX DY GVA MTZ STRL SSD OSK",
  "Consumer": "CBRL WEN PZZA CAKE TXRH JACK DPZ DRI JWN KSS COLM DECK VFC TOL MTH KBH SIG TPR CROX BKE",
  "Energy": "RRC MUR CNX SWN HP PTEN NBR RES GPRE CTRA SM NOG CVI HES OIS WTI SPH EPD ENB",
  "Utilities": "AWR SJW CWT AVA BKH NWN OTTR NWE UTL CPK HE IDA POR NJR SWX WTRG UGI VVC OGS",
  "RealEstate": "EXR CUBE NNN EPR ADC OHI LTC LXP RHP KRG AKR HIW BDN PEB SHO DEI RYN ELS SUI NHI SKT",
  "Materials": "SON SEE GPK OLN KALU KWR HUN HXL MTRN MTX CRS ATI CENX HL CDE SSRM ASH OI PKG",
  "Communication": "NYT GCI NXST GTN SSP CCOI IDT LUMN TDS SBGI UONE CHTR VZ DIS CMCSA EA TTWO NFLX",
 },
 "small_candidate": {
  "Technology": "DIOD POWI AEIS FORM COHU TTMI SANM UCTT PLAB OSIS PRGS BLKB CSGS VICR FEIM BELFB PENG SYNA AMBA AOSL",
  "Financials": "FULT BANF CATY COLB FIBK UCB PB CVBF FFBC AUB CBU FBP FHN HOMB IBCP UVSP BHB HBT PFS EFSC",
  "Healthcare": "CRVL CNMD IART OFIX USPH INSM BCRX EBS CYH HRTX VCEL SUPN PACB FOLD AMPH RMTI TGTX VSTM NVAX",
  "Industrials": "ALG FELE BMI GRC WTS PATK ROCK GFF LNN TNC ICFI CW AIT RRX FSS POWL ESE KEX WNC TISI",
  "Consumer": "AEO ANF GIII OXM LE SHOO WWW CAL GCO PLCE RRGB GIII HVT GES FOSL CATO BZH HOV GCO WINA",
  "Energy": "CRK GTE CLMT NBR GPRE GASS DLNG TELL NOG WTI RES GTE KOS CHRD VTLE TUSK REPX AMPY NGL",
  "Utilities": "MGEE NJR SWX ARTNA MSEX YORW RGCO CPK UTL OTTR CWT SJW AWR NWN NWE BKH AVA IDA POR HE",
  "RealEstate": "UHT LTC LXP SKT AKR PEB SHO BDN HIW EPR NHI ADC KRG CUBE RYN UMH GOOD SELF BRT INN",
  "Materials": "MERC SCL RYI CENX HL CDE MTRN KALU KWR HXL ATI MTX OLN HUN OI TROX ESI AXTA",
  "Communication": "GTN SSP IDT TDS SBGI UONE NYT CCOI GCI NXST LUMN IHRT TDAY CNSL CTL NCMI SIRI SALM",
 },
}


def candidate_order(group):
    """Round-robin by industry to avoid allocating all slots to one sector."""
    sectors = list(POOLS[group])
    streams = [POOLS[group][sec].split() for sec in sectors]
    for i in range(max(len(x) for x in streams)):
        for sec, stocks in zip(sectors, streams):
            if i < len(stocks):
                yield sec, stocks[i].upper()


def hash_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_csv(df):
    df = df.copy().reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="raise").dt.strftime("%Y-%m-%d")
    df = df.loc[df["Date"].between(START, "2021-12-31"), COLUMNS].copy()
    df = df.sort_values("Date").drop_duplicates("Date")
    for col in COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=COLUMNS[1:])
    df = df[(df[["Open", "High", "Low", "Close", "Adj Close"]] > 0).all(axis=1)]
    df = df[df["Volume"] >= 0]
    return df.reset_index(drop=True)


def get_prices(ticker, raw_dir, retries, delay):
    path = raw_dir / f"{ticker}.csv"
    if path.exists():
        try:
            data = canonical_csv(pd.read_csv(path))
            if len(data) > 200:
                return data
        except Exception as err:
            print(f"[INVALID CACHE] {ticker}: {err}", flush=True)
        # Do not silently change user-visible cached raw data.
        raise RuntimeError(f"Invalid cached raw file: {path}. Inspect/remove ONLY this file to redownload.")
    try:
        import yfinance as yf
    except ImportError as err:
        raise RuntimeError("Install yfinance in cv_termproject: pip install yfinance") from err
    for attempt in range(1, retries + 1):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*inplace keyword.*")
                history = yf.Ticker(ticker).history(
                    start=START, end=END, interval="1d",
                    auto_adjust=False, actions=False,
                )
            if history is None or history.empty:
                raise ValueError("empty Yahoo history")
            if "Adj Close" not in history.columns:
                raise ValueError("Adj Close missing; required for dividend-adjusted targets")
            df = history[COLUMNS[1:]].copy()
            dates = pd.DatetimeIndex(df.index)
            if dates.tz is not None:
                dates = dates.tz_localize(None)
            df.insert(0, "Date", dates.strftime("%Y-%m-%d"))
            df = canonical_csv(df)
            if len(df) < 200:
                raise ValueError("insufficient clean rows")
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False, float_format="%.12g")
            if delay > 0:
                time.sleep(delay)
            return df
        except Exception as err:
            if attempt == retries:
                print(f"[SKIP DOWNLOAD] {ticker}: {err}", flush=True)
                return None
            print(f"[RETRY] {ticker} ({attempt}/{retries}): {err}", flush=True)
            time.sleep(max(delay, 1.5 * attempt))
    return None


def eligibility(df, base=False):
    if df is None or len(df) < 1600:
        return False, "short history"
    if df.Date.iloc[-1] < "2021-12-15":
        return False, "no late-2021 coverage"
    late = df[df.Date.between("2015-01-01", "2021-12-31")]
    # Every selected stock must support the common 2015+ comparison.
    if len(late) < 1650 or late.Date.iloc[0] > "2015-02-01":
        return False, "insufficient 2015-2021 coverage"
    if not base:
        # Avoid late-IPO candidates where 'longer history' adds essentially nothing.
        earlier = df[df.Date < "2015-01-01"]
        if len(earlier) < 500:
            return False, "too little pre-2015 history"
    return True, "ok"


def price_info(df):
    historic = df[df.Date.between("2010-01-01", "2014-12-31")]
    median_dollar_volume = float((historic.Close * historic.Volume).median()) if len(historic) else np.nan
    return {
        "first_date": df.Date.iloc[0], "last_date": df.Date.iloc[-1],
        "n_price_days": int(len(df)),
        "n_days_pre2015": int(len(historic)),
        "median_dollar_volume_2010_2014": median_dollar_volume,
    }


def select_universe(args):
    args.root.mkdir(parents=True, exist_ok=True)
    raw_dir = args.root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.root / "universe200.csv"
    original_meta = pd.read_csv(args.legacy / "processed" / "samples.csv", usecols=["ticker"], dtype=str)
    legacy_tickers = sorted(original_meta.ticker.dropna().str.upper().unique().tolist())
    if len(legacy_tickers) != 100:
        raise ValueError(f"Expected 100 legacy tickers, found {len(legacy_tickers)}")
    selected = []
    used = set()
    if manifest_path.exists():
        prior = pd.read_csv(manifest_path, dtype={"ticker": str})
        if len(prior) != 200 or prior.ticker.nunique() != 200:
            raise ValueError("Existing universe manifest is not exactly 200 stocks")
        if sorted(prior.loc[prior.group == "legacy", "ticker"].tolist()) != legacy_tickers:
            raise ValueError("Existing universe manifest doesn't match legacy stocks")
        print(f"[MANIFEST] Reuse locked membership from {manifest_path}", flush=True)
        for entry in prior.itertuples(index=False):
            data = get_prices(entry.ticker, raw_dir, args.retries, args.delay)
            ok, why = eligibility(data, base=(entry.group == "legacy"))
            if not ok:
                raise ValueError(f"Previously accepted ticker {entry.ticker} now invalid: {why}")
        return prior

    for i, ticker in enumerate(legacy_tickers, 1):
        data = get_prices(ticker, raw_dir, args.retries, args.delay)
        ok, why = eligibility(data, base=True)
        if not ok:
            raise ValueError(f"Original stock {ticker}: {why}. Do not silently replace legacy stocks.")
        selected.append({"ticker": ticker, "group": "legacy", "sector": "original100", **price_info(data)})
        used.add(ticker)
        if i % 10 == 0:
            print(f"[LEGACY] {i}/100 downloaded and validated", flush=True)

    for group in ("mid_candidate", "small_candidate"):
        accepted = 0
        attempted = 0
        for sector, ticker in candidate_order(group):
            if accepted >= 50:
                break
            if ticker in used:
                continue
            attempted += 1
            data = get_prices(ticker, raw_dir, args.retries, args.delay)
            ok, reason = eligibility(data)
            if not ok:
                print(f"[REJECT] {ticker} {group} {reason}", flush=True)
                continue
            selected.append({"ticker": ticker, "group": group, "sector": sector, **price_info(data)})
            used.add(ticker)
            accepted += 1
            print(f"[ACCEPT] {group}: {accepted}/50 {ticker} ({sector})", flush=True)
        if accepted != 50:
            raise RuntimeError(
                f"Only selected {accepted}/50 from {group} after {attempted} downloads. "
                f"Expand POOLS['{group}'] with verified symbols and retry; existing CSV caches are reused."
            )
    manifest = pd.DataFrame(selected)
    if len(manifest) != 200 or manifest.ticker.nunique() != 200:
        raise AssertionError("Universe does not contain exactly 200 unique stocks")
    manifest.to_csv(manifest_path, index=False)
    print(f"[LOCKED] {manifest_path} (100 original + 50 mid-candidate + 50 small-candidate)", flush=True)
    return manifest


def adjusted(df):
    df = df.copy()
    adj_factor = df["Adj Close"].to_numpy(float) / df["Close"].to_numpy(float)
    for c in ("Open", "High", "Low", "Close"):
        df["A_" + c] = df[c].to_numpy(float) * adj_factor
    df["MA20"] = df.A_Close.rolling(20, min_periods=20).mean()
    return df


def build_metadata(manifest, raw_dir):
    rows = []
    for item in manifest.itertuples(index=False):
        d = adjusted(canonical_csv(pd.read_csv(raw_dir / f"{item.ticker}.csv")))
        dates = d.Date.to_numpy()
        closes = d.A_Close.to_numpy(float)
        # i >= 38 gives 20 chart days each with full past-only MA20.
        for i in range(38, len(d) - HORIZON):
            if dates[i + 5] >= END:
                continue
            ret = closes[i + 5] / closes[i] - 1.0
            rows.append((item.ticker, item.group, item.sector, dates[i - 19], dates[i], dates[i + 5], ret, int(ret > 0)))
    meta = pd.DataFrame(rows, columns=[
        "ticker", "group", "sector", "window_start", "decision_date", "target_date", "return_5d", "label"
    ])
    meta = meta.sort_values(["ticker", "decision_date"]).reset_index(drop=True)
    if meta.duplicated(["ticker", "decision_date"]).any():
        raise AssertionError("Duplicate ticker decision dates")
    if meta.empty or len(meta) < 300_000:
        raise ValueError(f"Unexpectedly small prepared data: {len(meta)}")
    return meta


def make_images(args, manifest):
    from scripts.jkx_i20_reproduce import render_i20
    image_root = args.images
    image_root.mkdir(parents=True, exist_ok=True)
    meta_path = image_root / "meta.csv"
    images_path = image_root / "images.npy"
    journal_path = image_root / "progress.json"
    flag_path = image_root / "READY.json"
    meta = build_metadata(manifest, args.root / "raw")
    if not meta_path.exists():
        meta.to_csv(meta_path, index=False, float_format="%.12g")
    else:
        existing = pd.read_csv(meta_path, dtype={"ticker": str, "decision_date": str})
        cols = ["ticker", "decision_date", "window_start", "target_date", "group", "label"]
        if len(existing) != len(meta) or not existing[cols].equals(meta[cols]):
            raise ValueError("Metadata differs from saved dataset; do not overwrite an existing experiment")
        if not np.allclose(existing.return_5d, meta.return_5d, rtol=0, atol=1e-9):
            raise ValueError("Adjusted returns differ from saved dataset")
    fingerprint = hash_file(meta_path)
    if flag_path.exists():
        ready = json.loads(flag_path.read_text(encoding="utf-8"))
        if ready.get("fingerprint") != fingerprint:
            raise ValueError("READY fingerprint mismatch")
        data = np.load(images_path, mmap_mode="r")
        if data.shape != (len(meta), 1, 64, 60):
            raise ValueError("READY image shape mismatch")
        print(f"[DATA READY] {len(meta):,} images at {images_path}", flush=True)
        return
    if images_path.exists():
        images = open_memmap(images_path, mode="r+")
        if images.shape != (len(meta), 1, 64, 60):
            raise ValueError("Existing array shape mismatch. Do not overwrite automatically.")
    else:
        if journal_path.exists():
            raise ValueError("Journal exists without array; remove incomplete experiment data explicitly")
        images = open_memmap(images_path, mode="w+", dtype="uint8", shape=(len(meta), 1, 64, 60))
    if journal_path.exists():
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        if journal.get("fingerprint") != fingerprint:
            raise ValueError("Image progress fingerprint mismatch")
        finished = set(journal.get("done", []))
    else:
        finished = set()
    count = 0
    for ticker, group in meta.groupby("ticker", sort=True):
        if ticker in finished:
            continue
        data = adjusted(canonical_csv(pd.read_csv(args.root / "raw" / f"{ticker}.csv")))
        indices = {date: i for i, date in enumerate(data.Date)}
        for row in group.itertuples():
            i = indices.get(row.decision_date)
            j = indices.get(row.target_date)
            if i is None or j != i + 5 or i < 38 or data.Date.iloc[i - 19] != row.window_start:
                raise ValueError(f"Index mismatch: {ticker} {row.decision_date}")
            image = render_i20(data.iloc[i - 19:i + 1])
            if image.shape != (64, 60):
                raise AssertionError("Unexpected rendered image size")
            images[row.Index, 0] = image
        images.flush()
        finished.add(ticker)
        journal_path.write_text(json.dumps({"fingerprint": fingerprint, "done": sorted(finished)}, indent=2), encoding="utf-8")
        count += 1
        if count % 10 == 0 or len(finished) == 200:
            print(f"[CHART] complete={len(finished)}/200 total_images={len(meta):,}", flush=True)
    if len(finished) != 200:
        raise AssertionError("Render incomplete")
    del images
    flag_path.write_text(json.dumps({"fingerprint": fingerprint, "count": len(meta), "stocks": 200}, indent=2), encoding="utf-8")
    journal_path.unlink(missing_ok=True)
    print(f"[READY] {len(meta):,} I20 charts, 100+100 stocks, 2010-2021", flush=True)
    print(f"[OUTPUT] {image_root}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Download and render 200-stock 2010-2021 JKX charts")
    p.add_argument("--legacy", type=Path, default=Path("data/exp100"))
    p.add_argument("--root", type=Path, default=Path("data/exp200"))
    p.add_argument("--images", type=Path, default=Path("data/exp200_jkx"))
    p.add_argument("--stage", choices=["all", "download", "render"], default="all")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--delay", type=float, default=0.3)
    a = p.parse_args()
    if a.retries < 1 or a.delay < 0:
        p.error("Invalid download settings")
    if a.stage == "render":
        universe_path = a.root / "universe200.csv"
        if not universe_path.exists():
            p.error("Download stage must complete first")
        universe = pd.read_csv(universe_path, dtype={"ticker": str})
        if len(universe) != 200:
            p.error("Manifest incomplete")
    else:
        universe = select_universe(a)
    if a.stage != "download":
        make_images(a, universe)
    print("[DONE] Preparation finished", flush=True)


if __name__ == "__main__":
    main()
