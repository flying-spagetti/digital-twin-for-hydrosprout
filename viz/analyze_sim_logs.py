#!/usr/bin/env python3
"""
analyze_sim_logs.py

Parse a sim_run log file produced by main.py sim_run and produce:
- parsed_sim.csv (time-indexed numeric time series)
- plots/ (PNG time-series + diagnostics)
- summary.txt (numeric summary)
- dashboard.html (interactive Plotly dashboard)

Usage:
    python viz/analyze_sim_logs.py /path/to/sim_run_20251122_123456.log

If no path provided, script will try to find the newest file in ./logs/
"""

import re
import sys
import json
from pathlib import Path
from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# -----------------------------
# Config
# -----------------------------
OUT_DIR = Path("viz_output")
PLOTS_DIR = OUT_DIR / "plots"
CSV_OUT = OUT_DIR / "parsed_sim.csv"
SUMMARY_OUT = OUT_DIR / "summary.txt"
DASH_OUT = OUT_DIR / "dashboard.html"

# keys regex -> canonical name used in dataframe
PATTERNS = {
    r"\bCanopy\b.*?:\s*([-+]?[0-9]*\.?[0-9]+)": "canopy",
    r"\bMoisture\b.*?:\s*([-+]?[0-9]*\.?[0-9]+)": "moisture",
    r"\bNutrient\b.*?:\s*([-+]?[0-9]*\.?[0-9]+)": "nutrient",
    r"\bMold probability\b.*?:\s*([-+]?[0-9]*\.?[0-9]+)": "mold_prob",
    r"\bTemperature \((?:scaled|scaled)\)\:\s*([-+]?[0-9]*\.?[0-9]+)": "temp_scaled",
    r"\bTemperature\:\s*([-+]?[0-9]*\.?[0-9]+)°C": "temp_c",
    r"\bTemp\b.*?:\s*([-+]?[0-9]*\.?[0-9]+)°?C?": "temp_c_alt",
    r"\bLux\b.*?:\s*([-+]?[0-9]*\.?[0-9]+)": "lux",
    r"\bShield position\b.*?:\s*([-+]?[0-9]*\.?[0-9]+)": "shield_pos",
    r"\bLAI\b.*?:\s*([-+]?[0-9]*\.?[0-9]+)": "LAI",
    r"\bTotal leaf biomass\b.*?:\s*([-+]?[0-9]*\.?[0-9]+)": "leaf_biomass_g",
    r"\bTotal root biomass\b.*?:\s*([-+]?[0-9]*\.?[0-9]+)": "root_biomass_g",
    r"\bTotal stem biomass\b.*?:\s*([-+]?[0-9]*\.?[0-9]+)": "stem_biomass_g",
    r"\bNumber of plants\b.*?:\s*([0-9]+)": "n_plants",
    r"\bEnergy consumed\b.*?:\s*([-+]?[0-9]*\.?[0-9]+)": "energy",
    r"\bEnergy:\s*([-+]?[0-9]*\.?[0-9]+)": "energy_alt",
    r"\bDelivered water\b.*?:\s*([-+]?[0-9]*\.?[0-9]+)": "delivered_water",
    r"\bHeater power\b.*?:\s*([-+]?[0-9]*\.?[0-9]+)": "heater_power",
    r"\bTranspiration\b.*?:\s*([-+]?[0-9]*\.?[0-9]+)": "transp_total_liters",
    r"\bVPD\b.*?:\s*([-+]?[0-9]*\.?[0-9]+)": "vpd_norm",
    r"\bReward\b.*?:\s*([-+]?[0-9]*\.?[0-9]+)": "reward",
    r"\bCumulative reward\b.*?:\s*([-+]?[0-9]*\.?[0-9]+)": "cum_reward",
    r"\bHour\s+([0-9]{1,2})\b": "hour",
    r"STEP\s+([0-9]+)\/([0-9]+)": "step_progress",  # captures two groups
    r"\bTerminated\b.*?:\s*(True|False)": "terminated",
    r"\bTruncated\b.*?:\s*(True|False)": "truncated",
}

# There are lines that directly print dicts like "Plant State After Step:" followed by dict
DICT_LINE_RE = re.compile(r"^\s*[{[]")  # starts with { or [

# -----------------------------
# Helper functions
# -----------------------------
def find_log_file_candidate(arg_path=None):
    if arg_path:
        p = Path(arg_path)
        if not p.exists():
            raise FileNotFoundError(f"Log file not found: {p}")
        return p
    # else find newest file in logs/
    logs_dir = Path("logs")
    if not logs_dir.exists():
        raise FileNotFoundError("No logs/ directory found and no path supplied.")
    files = sorted(logs_dir.glob("**/sim_run_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError("No sim_run_*.log files found in logs/")
    return files[0]

def try_parse_number(s):
    try:
        return float(s)
    except:
        try:
            return float(s.replace(',', ''))
        except:
            return None

# attempt to parse python-style dict string into dict (best-effort)
def safe_eval_dict(s):
    try:
        # tidy up common python repr differences to JSON
        js = s.replace("'", '"')
        # remove trailing commas before closing braces
        js = re.sub(r",\s*}", "}", js)
        js = re.sub(r",\s*]", "]", js)
        return json.loads(js)
    except Exception:
        return None

# -----------------------------
# Main parser
# -----------------------------
def parse_log_file(path):
    dfrows = []
    current = defaultdict(lambda: None)
    step_index = -1
    last_step_logged = -1

    with open(path, 'r', errors='ignore') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            # detect new STEP block lines
            mstep = re.search(r"STEP\s+([0-9]+)\/([0-9]+)", line)
            if mstep:
                # move to new step index (1-based readjusted to 0-based)
                try:
                    step_index = int(mstep.group(1)) - 1
                except:
                    step_index += 1
                # store previous current as a row if it had some step info
                if any(v is not None for v in current.values()):
                    # store with index = last_step_logged + 1 or step_index
                    idx = step_index if step_index >= 0 else last_step_logged + 1
                    row = dict(current)
                    row["_step"] = idx
                    dfrows.append(row)
                    current = defaultdict(lambda: None)
                    last_step_logged = idx
                # continue scanning rest of line too (it may include Hour or key values)
            # try matching each pattern
            for pat, cname in PATTERNS.items():
                try:
                    m = re.search(pat, line)
                except re.error:
                    continue
                if m:
                    # depending on capture groups, choose first numeric group
                    groups = m.groups()
                    if not groups:
                        continue
                    # prefer first numeric-like group
                    val = None
                    for g in groups[::-1]:  # check last group first (like terminated)
                        if g is None:
                            continue
                        if g in ("True", "False"):
                            val = True if g == "True" else False
                            break
                        num = try_parse_number(g)
                        if num is not None:
                            val = num
                            break
                        # else keep string
                        val = g
                    current[cname] = val
            # try to parse JSON-like dict lines (plant info / env info / hw info)
            # sometimes logger prints "Plant info: {'C':..., ...}"
            dict_match = re.search(r"Plant info:\s*(\{.*\})", line)
            if dict_match:
                dstr = dict_match.group(1)
                d = safe_eval_dict(dstr)
                if d:
                    # flatten known keys into current row
                    for k, v in d.items():
                        key = str(k).lower()
                        # canonicalize some keys:
                        if key in ("c", "canopy"):
                            current["canopy"] = try_parse_number(v)
                        elif key in ("m", "moisture"):
                            current["moisture"] = try_parse_number(v)
                        elif key in ("n", "nutrient"):
                            current["nutrient"] = try_parse_number(v)
                        elif key in ("p_mold","pmold","p_mold"):
                            current["mold_prob"] = try_parse_number(v)
                        else:
                            # store raw if numeric
                            num = try_parse_number(v)
                            if num is not None:
                                current[key] = num
                            else:
                                current[key] = v
            # also parse trailing dicts printed on separate lines (line starts with '{')
            if DICT_LINE_RE.match(line):
                # accumulate the bracketed text until closing bracket
                buf = line
                if not (line.endswith("}") or line.endswith("]")):
                    # read ahead (this is best-effort; logs are line-by-line so often not needed)
                    pass
                d = safe_eval_dict(buf)
                if d:
                    for k, v in d.items():
                        k0 = str(k).lower()
                        num = try_parse_number(v)
                        if num is not None:
                            current[k0] = num
                        else:
                            current[k0] = v

    # append last current if any
    if any(v is not None for v in current.values()):
        idx = last_step_logged + 1
        row = dict(current)
        row["_step"] = idx
        dfrows.append(row)

    # Build dataframe sorted by _step
    if not dfrows:
        raise RuntimeError("No numeric data parsed from log.")
    df = pd.DataFrame(dfrows)
    df = df.sort_values(by=["_step"]).reset_index(drop=True)
    # If no explicit hour column, create it: hour = step % 24
    if "hour" not in df.columns:
        df["hour"] = df["_step"] % 24
    # unify temp value: prefer temp_c, else use temp_scaled*40
    if "temp_c" not in df.columns and "temp_scaled" in df.columns:
        df["temp_c"] = df["temp_scaled"].astype(float) * 40.0
    # ensure numeric columns are floats
    for col in df.columns:
        if col.startswith("_"):
            continue
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            pass
    return df

# -----------------------------
# Plotting helpers
# -----------------------------
def save_basic_plots(df, outdir):
    outdir.mkdir(parents=True, exist_ok=True)

    # 1: canopy, moisture, nutrient
    plt.figure(figsize=(10,4))
    if "canopy" in df.columns:
        plt.plot(df["_step"], df["canopy"], label="canopy")
    if "moisture" in df.columns:
        plt.plot(df["_step"], df["moisture"], label="moisture")
    if "nutrient" in df.columns:
        plt.plot(df["_step"], df["nutrient"], label="nutrient")
    plt.xlabel("step")
    plt.legend()
    plt.title("Canopy / Moisture / Nutrient")
    plt.grid(True)
    plt.savefig(outdir / "canopy_moist_nutrient.png", dpi=150)
    plt.close()

    # 2: temp and lux (dual axis)
    if "temp_c" in df.columns or "lux" in df.columns:
        fig, ax1 = plt.subplots(figsize=(10,4))
        if "temp_c" in df.columns:
            ax1.plot(df["_step"], df["temp_c"], label="temp (°C)", color="tab:red")
            ax1.set_ylabel("Temp (°C)")
        ax2 = ax1.twinx()
        if "lux" in df.columns:
            ax2.plot(df["_step"], df["lux"], label="lux", color="tab:orange", alpha=0.8)
            ax2.set_ylabel("Lux")
        ax1.set_xlabel("step")
        ax1.set_title("Temp and Lux")
        fig.tight_layout()
        fig.savefig(outdir / "temp_lux.png", dpi=150)
        plt.close(fig)

    # 3: energy & water usage
    plt.figure(figsize=(10,4))
    if "energy" in df.columns:
        plt.plot(df["_step"], df["energy"], label="energy")
    if "delivered_water" in df.columns:
        plt.plot(df["_step"], df["delivered_water"].cumsum(), label="cumulative_water")
    plt.xlabel("step")
    plt.legend()
    plt.title("Energy & Water usage")
    plt.grid(True)
    plt.savefig(outdir / "energy_water.png", dpi=150)
    plt.close()

    # 4: biomass stacked if available
    if {"leaf_biomass_g","root_biomass_g","stem_biomass_g"}.issubset(set(df.columns)):
        plt.figure(figsize=(10,4))
        plt.stackplot(df["_step"], df["leaf_biomass_g"].fillna(0),
                      df["stem_biomass_g"].fillna(0), df["root_biomass_g"].fillna(0),
                      labels=["leaf_g","stem_g","root_g"])
        plt.legend(loc="upper left")
        plt.xlabel("step")
        plt.title("Total biomass per organ (g)")
        plt.savefig(outdir / "biomass_stacked.png", dpi=150)
        plt.close()

    # 5: mold probability
    if "mold_prob" in df.columns:
        plt.figure(figsize=(10,3))
        plt.plot(df["_step"], df["mold_prob"], label="mold_prob")
        plt.xlabel("step")
        plt.title("Mold probability")
        plt.grid(True)
        plt.savefig(outdir / "mold_prob.png", dpi=150)
        plt.close()

    # 6: reward curve
    if "reward" in df.columns:
        plt.figure(figsize=(10,3))
        plt.plot(df["_step"], df["reward"].fillna(0), label="reward")
        plt.plot(df["_step"], df["cum_reward"].ffill(), label="cum_reward")
        plt.xlabel("step")
        plt.legend()
        plt.title("Reward per step")
        plt.savefig(outdir / "reward_curve.png", dpi=150)
        plt.close()

    # 7: correlation heatmap
    try:
        numeric = df.select_dtypes(include=[np.number]).drop(columns=["_step"]).fillna(0)
        if numeric.shape[1] > 1:
            corr = numeric.corr()
            plt.figure(figsize=(8,6))
            plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            plt.colorbar()
            xticks = list(numeric.columns)
            plt.xticks(range(len(xticks)), xticks, rotation=45, ha="right")
            plt.yticks(range(len(xticks)), xticks)
            plt.title("Correlation matrix")
            plt.tight_layout()
            plt.savefig(outdir / "correlation.png", dpi=150)
            plt.close()
    except Exception as e:
        print("Failed to plot correlation:", e)

def write_summary(df, outpath):
    with open(outpath, "w") as f:
        f.write(f"Simulation analysis summary\nGenerated: {datetime.datetime.now()}\n\n")
        f.write(f"Rows parsed: {len(df)}\n")
        f.write("\nKey metrics (min / mean / max):\n")
        cols = ["canopy","moisture","nutrient","mold_prob","temp_c","lux","shield_pos","energy"]
        for c in cols:
            if c in df.columns:
                s = df[c].dropna()
                f.write(f"  {c}: {s.min():.4f} / {s.mean():.4f} / {s.max():.4f}\n")
        # find peaks
        if "canopy" in df.columns:
            imax = df["canopy"].idxmax()
            f.write(f"\nPeak canopy at step {int(df.loc[imax,'_step'])} value={df.loc[imax,'canopy']:.4f}\n")
        f.write("\nEnd of summary\n")

# -----------------------------
# Interactive dashboard (Plotly)
# -----------------------------
def build_dashboard(df, out_html):
    figs = []
    # canopy
    if "canopy" in df.columns:
        fig = px.line(df, x="_step", y="canopy", title="Canopy over time")
        figs.append(fig)
    # moisture + nutrient
    ycols = [c for c in ("moisture","nutrient","mold_prob") if c in df.columns]
    if ycols:
        fig = go.Figure()
        for c in ycols:
            fig.add_trace(go.Scatter(x=df["_step"], y=df[c], name=c))
        fig.update_layout(title="Moisture / Nutrient / Mold")
        figs.append(fig)
    # temp & lux with secondary axis
    if "temp_c" in df.columns or "lux" in df.columns:
        fig = make_temp_lux_plot(df)
        figs.append(fig)
    # energy & water
    if "energy" in df.columns or "delivered_water" in df.columns:
        fig = go.Figure()
        if "energy" in df.columns:
            fig.add_trace(go.Scatter(x=df["_step"], y=df["energy"], name="energy"))
        if "delivered_water" in df.columns:
            fig.add_trace(go.Scatter(x=df["_step"], y=df["delivered_water"].cumsum(), name="cumulative_water"))
        fig.update_layout(title="Energy and Cumulative Water")
        figs.append(fig)

    # Save into single HTML
    with open(out_html, "w") as f:
        f.write("<html><head><meta charset='utf-8' /></head><body>\n")
        f.write(f"<h1>Simulation Dashboard</h1>\n")
        f.write(f"<p>Generated: {datetime.datetime.now()}</p>\n")
        # dump each fig as div
        import plotly.io as pio
        for fig in figs:
            f.write(pio.to_html(fig, full_html=False, include_plotlyjs='cdn'))
            f.write("<hr/>\n")
        # include raw table top rows
        f.write("<h2>Top rows (parsed)</h2>\n")
        f.write(df.head(20).to_html(index=False, float_format='{:.4f}'.format))
        f.write("</body></html>\n")
    print(f"[dashboard] Saved interactive dashboard to {out_html}")

def make_temp_lux_plot(df):
    fig = make_subplots(rows=1, cols=1)
    if "temp_c" in df.columns:
        fig.add_trace(go.Scatter(x=df["_step"], y=df["temp_c"], name="temp_c (°C)"))
    if "lux" in df.columns:
        fig.add_trace(go.Scatter(x=df["_step"], y=df["lux"], name="lux"))
    fig.update_layout(title="Temp and Lux")
    return fig

# -----------------------------
# Main
# -----------------------------
def main(log_path=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        log_file = find_log_file_candidate(log_path)
    except Exception as e:
        print("Error locating log file:", e)
        return
    print("Parsing log file:", log_file)
    df = parse_log_file(log_file)
    print("Parsed rows:", len(df))
    # write CSV
    df.to_csv(CSV_OUT, index=False)
    print("Wrote parsed CSV to", CSV_OUT)
    # plots
    save_basic_plots(df, PLOTS_DIR)
    print("Saved plots to", PLOTS_DIR)
    # summary
    write_summary(df, SUMMARY_OUT)
    print("Wrote summary to", SUMMARY_OUT)
    # dashboard
    build_dashboard(df, DASH_OUT)
    print("Wrote dashboard to", DASH_OUT)
    print("Done.")

if __name__ == "__main__":
    logpath = sys.argv[1] if len(sys.argv) > 1 else None
    main(logpath)
