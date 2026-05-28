from pathlib import Path
import re
import zipfile
import warnings

import numpy as np
import pandas as pd


# ============================================================
# 0. 基础路径设置
# ============================================================

DATA_ROOT = Path(r"F:\桌面\Virtual Reality\客观数据\皮影实验数据\实验输出")

TRIAL_LOG_DIR = DATA_ROOT / "04_trial_logs"
CONTACT_EVENT_DIR = DATA_ROOT / "05_contact_event_logs"
TRIAL_SUMMARY_DIR = DATA_ROOT / "06_trial_summary"

PARAMETER_FILE = DATA_ROOT / "parameter_settings"
THRESHOLD_FILE = DATA_ROOT / "threshold_settings"
MANIFEST_FILE = DATA_ROOT / "trial_manifest"

OUT_DIR = Path(r"F:\桌面\Virtual Reality\客观数据\data_result")
OUT_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 1. 通用工具函数
# ============================================================

def find_existing_file(path: Path) -> Path:
    """
    兼容两种情况：
    1. 传入 parameter_settings.csv
    2. 传入 parameter_settings，但实际文件是 parameter_settings.csv
    """
    candidates = [
        path,
        path.with_suffix(".csv"),
        path.with_suffix(".CSV"),
        path.with_suffix(".xlsx"),
        path.with_suffix(".XLSX"),
    ]

    for p in candidates:
        if p.exists() and p.is_file():
            return p

    raise FileNotFoundError(f"找不到文件：{path}")


def find_summary_file(summary_dir: Path) -> Path:
    """
    在 06_trial_summary 文件夹中自动查找 Trial_Summary 文件。
    """
    preferred = [
        summary_dir / "Trial_Summary.csv",
        summary_dir / "trial_summary.csv",
        summary_dir / "TrialSummary.csv",
        summary_dir / "Trial_Summary.xlsx",
        summary_dir / "trial_summary.xlsx",
    ]

    for p in preferred:
        if p.exists() and p.is_file():
            return p

    csv_files = sorted(summary_dir.glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]

    xlsx_files = sorted(summary_dir.glob("*.xlsx"))
    if len(xlsx_files) == 1:
        return xlsx_files[0]

    raise FileNotFoundError(
        f"无法在 {summary_dir} 中唯一确定 Trial_Summary 文件。"
        f"找到 CSV: {len(csv_files)} 个，XLSX: {len(xlsx_files)} 个。"
    )


def read_table_smart(path: Path) -> pd.DataFrame:
    """
    自动读取 csv / xlsx，并兼容 utf-8-sig、utf-8、gbk。
    空文件返回空 DataFrame。
    """
    if path.suffix.lower() in [".xlsx", ".xls"]:
        try:
            return pd.read_excel(path)
        except Exception as e:
            print(f"[WARN] 读取 Excel 失败：{path} | {e}")
            return pd.DataFrame()

    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"[WARN] 读取 CSV 失败：{path} | {e}")
            return pd.DataFrame()

    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as e:
        print(f"[WARN] 读取 CSV 最终失败：{path} | {e}")
        return pd.DataFrame()


def normalize_col_name(c: str) -> str:
    """
    统一列名格式：去空格、小写、空格变下划线。
    """
    c = str(c).strip()
    c = re.sub(r"\s+", "_", c)
    c = c.replace("-", "_")
    return c.lower()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_col_name(c) for c in df.columns]
    return df


def bool_to_int(s: pd.Series) -> pd.Series:
    """
    把 True/False、true/false、1/0、yes/no 转为 0/1。
    """
    if s.dtype == bool:
        return s.astype(int)

    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
            "yes": 1,
            "no": 0,
            "y": 1,
            "n": 0,
            "nan": 0,
            "none": 0,
            "null": 0,
            "": 0,
        })
        .fillna(0)
        .astype(int)
    )


def first_existing_col(df: pd.DataFrame, candidates):
    """
    在候选列名中找第一个存在的列。
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_numeric_col(df: pd.DataFrame, col: str):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def safe_sheet_name(name: str) -> str:
    """
    Excel sheet name 最大 31 字符，且不能包含特殊符号。
    """
    name = str(name)
    name = re.sub(r"[\[\]\:\*\?\/\\]", "_", name)
    return name[:31]


def infer_trial_id_from_filename(file_name: str) -> str:
    """
    如果日志里没有 trial_id，就从文件名推断。
    例如：
    FrameLog_TC01_Ours_R01.csv -> TC01_Ours_R01
    ContactEvent_TEST_Ours.csv -> TEST_Ours
    """
    stem = Path(file_name).stem
    stem = re.sub(r"^FrameLog[_\-]?", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"^ContactEvent[_\-]?", "", stem, flags=re.IGNORECASE)
    return stem


# ============================================================
# 2. 读取所有原始数据
# ============================================================

def load_raw_data():
    print("\n========== 读取原始数据 ==========")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"TRIAL_LOG_DIR: {TRIAL_LOG_DIR}")
    print(f"CONTACT_EVENT_DIR: {CONTACT_EVENT_DIR}")
    print(f"TRIAL_SUMMARY_DIR: {TRIAL_SUMMARY_DIR}")
    print(f"OUT_DIR: {OUT_DIR}")

    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"数据根目录不存在：{DATA_ROOT}")

    if not TRIAL_LOG_DIR.exists():
        raise FileNotFoundError(f"缺少文件夹：{TRIAL_LOG_DIR}")

    if not CONTACT_EVENT_DIR.exists():
        raise FileNotFoundError(f"缺少文件夹：{CONTACT_EVENT_DIR}")

    if not TRIAL_SUMMARY_DIR.exists():
        raise FileNotFoundError(f"缺少文件夹：{TRIAL_SUMMARY_DIR}")

    parameter_file = find_existing_file(PARAMETER_FILE)
    threshold_file = find_existing_file(THRESHOLD_FILE)
    manifest_file = find_existing_file(MANIFEST_FILE)
    summary_file = find_summary_file(TRIAL_SUMMARY_DIR)

    print(f"parameter_settings: {parameter_file}")
    print(f"threshold_settings: {threshold_file}")
    print(f"trial_manifest: {manifest_file}")
    print(f"trial_summary: {summary_file}")

    parameter_settings = normalize_columns(read_table_smart(parameter_file))
    threshold_settings = normalize_columns(read_table_smart(threshold_file))
    trial_manifest = normalize_columns(read_table_smart(manifest_file))
    trial_summary = normalize_columns(read_table_smart(summary_file))

    frame_files = sorted(TRIAL_LOG_DIR.glob("*.csv")) + sorted(TRIAL_LOG_DIR.glob("*.xlsx"))
    contact_files = sorted(CONTACT_EVENT_DIR.glob("*.csv")) + sorted(CONTACT_EVENT_DIR.glob("*.xlsx"))

    frame_logs = []
    for f in frame_files:
        df = read_table_smart(f)
        if df.empty:
            empty = pd.DataFrame({"source_file": [f.name], "is_empty_file": [True]})
            frame_logs.append(empty)
            continue

        df = normalize_columns(df)
        df["source_file"] = f.name
        df["is_empty_file"] = False
        frame_logs.append(df)

    contact_logs = []
    for f in contact_files:
        df = read_table_smart(f)
        if df.empty:
            empty = pd.DataFrame({"source_file": [f.name], "is_empty_file": [True]})
            contact_logs.append(empty)
            continue

        df = normalize_columns(df)
        df["source_file"] = f.name
        df["is_empty_file"] = False
        contact_logs.append(df)

    frame_all = pd.concat(frame_logs, ignore_index=True, sort=False) if frame_logs else pd.DataFrame()
    contact_all = pd.concat(contact_logs, ignore_index=True, sort=False) if contact_logs else pd.DataFrame()

    if "trial_id" not in frame_all.columns and "source_file" in frame_all.columns:
        frame_all["trial_id"] = frame_all["source_file"].apply(infer_trial_id_from_filename)

    if "trial_id" not in contact_all.columns and "source_file" in contact_all.columns:
        contact_all["trial_id"] = contact_all["source_file"].apply(infer_trial_id_from_filename)

    print(f"FrameLog 文件数: {len(frame_files)}")
    print(f"ContactEvent 文件数: {len(contact_files)}")
    print(f"Trial_Summary 行数: {len(trial_summary)}")
    print(f"Trial_Manifest 行数: {len(trial_manifest)}")

    return {
        "parameter_settings": parameter_settings,
        "threshold_settings": threshold_settings,
        "trial_manifest": trial_manifest,
        "trial_summary": trial_summary,
        "frame_all": frame_all,
        "contact_all": contact_all,
        "frame_files": frame_files,
        "contact_files": contact_files,
        "parameter_file": parameter_file,
        "threshold_file": threshold_file,
        "manifest_file": manifest_file,
        "summary_file": summary_file,
    }


# ============================================================
# 3. 标记分析窗口 5-25s 或 summary 中的 analyzed_start/end
# ============================================================

def add_analysis_window(frame_all: pd.DataFrame, trial_summary: pd.DataFrame) -> pd.DataFrame:
    frame_all = frame_all.copy()

    if frame_all.empty:
        return frame_all

    if "timestamp" not in frame_all.columns:
        frame_all["in_analysis_window"] = True
        return frame_all

    if "trial_id" not in frame_all.columns or "trial_id" not in trial_summary.columns:
        frame_all["in_analysis_window"] = True
        return frame_all

    start_col = first_existing_col(
        trial_summary,
        ["analyzed_start", "analysis_start", "trim_start", "analysis_window_start"]
    )
    end_col = first_existing_col(
        trial_summary,
        ["analyzed_end", "analysis_end", "trim_end", "analysis_window_end"]
    )

    if start_col is None or end_col is None:
        # 若 summary 里没有窗口字段，则默认 5-25s。
        frame_all["_analysis_start"] = 5.0
        frame_all["_analysis_end"] = 25.0
    else:
        window = trial_summary[["trial_id", start_col, end_col]].copy()
        window[start_col] = pd.to_numeric(window[start_col], errors="coerce")
        window[end_col] = pd.to_numeric(window[end_col], errors="coerce")

        frame_all = frame_all.merge(window, on="trial_id", how="left")
        frame_all["_analysis_start"] = frame_all[start_col].fillna(5.0)
        frame_all["_analysis_end"] = frame_all[end_col].fillna(25.0)

    frame_all["timestamp"] = pd.to_numeric(frame_all["timestamp"], errors="coerce")
    frame_all["in_analysis_window"] = (
        (frame_all["timestamp"] >= frame_all["_analysis_start"]) &
        (frame_all["timestamp"] <= frame_all["_analysis_end"])
    )

    return frame_all


# ============================================================
# 4. 从 FrameLog 重算 trial-level 指标
# ============================================================

def recompute_from_frame_logs(frame_all: pd.DataFrame) -> pd.DataFrame:
    if frame_all.empty or "trial_id" not in frame_all.columns:
        return pd.DataFrame()

    df = frame_all.copy()

    if "in_analysis_window" in df.columns:
        df = df[df["in_analysis_window"] == True].copy()

    # 去掉空文件占位行
    if "is_empty_file" in df.columns:
        df = df[df["is_empty_file"] != True].copy()

    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby("trial_id", dropna=False)
    out = pd.DataFrame(index=grouped.size().index)
    out["frame_rows_from_frame"] = grouped.size()

    # 帧级布尔指标
    bool_metrics = {
        "external_contact_count_from_frame": [
            "is_external_contact",
            "external_contact",
            "external_contact_flag"
        ],
        "contact_attempt_count_from_frame": [
            "contact_attempt",
            "is_contact_attempt",
            "contact_attempt_flag"
        ],
        "penetration_frame_count_from_frame": [
            "is_penetrating",
            "penetrating",
            "penetration_flag"
        ],
        "audio_candidate_count_from_frame": [
            "audio_candidate",
            "audio_event",
            "audio_trigger"
        ],
        "pass_through_frame_count_from_frame": [
            "pass_through",
            "is_pass_through",
            "pass_through_flag"
        ],
        "visual_instability_count_from_frame": [
            "visual_instability",
            "is_visual_instability",
            "visual_instability_flag"
        ],
    }

    for new_col, candidates in bool_metrics.items():
        col = first_existing_col(df, candidates)
        if col is not None:
            df[col] = bool_to_int(df[col])
            out[new_col] = df.groupby("trial_id", dropna=False)[col].sum()

            if new_col == "penetration_frame_count_from_frame":
                out["penetration_frame_ratio_from_frame"] = (
                    df.groupby("trial_id", dropna=False)[col].mean()
                )

    # 空间深度指标
    depth_col = first_existing_col(
        df,
        ["raw_penetration_depth", "penetration_depth", "max_penetration_depth"]
    )
    if depth_col is not None:
        df[depth_col] = pd.to_numeric(df[depth_col], errors="coerce")
        out["mean_penetration_depth_from_frame"] = (
            df.groupby("trial_id", dropna=False)[depth_col].mean()
        )
        out["max_penetration_depth_from_frame"] = (
            df.groupby("trial_id", dropna=False)[depth_col].max()
        )
        out["p95_penetration_depth_from_frame"] = (
            df.groupby("trial_id", dropna=False)[depth_col]
            .quantile(0.95)
        )

    overlap_col = first_existing_col(
        df,
        ["raw_overlap_distance", "raw_overlap_depth", "overlap_distance", "overlap_depth"]
    )
    if overlap_col is not None:
        df[overlap_col] = pd.to_numeric(df[overlap_col], errors="coerce")
        out["mean_overlap_from_frame"] = (
            df.groupby("trial_id", dropna=False)[overlap_col].mean()
        )
        out["max_overlap_from_frame"] = (
            df.groupby("trial_id", dropna=False)[overlap_col].max()
        )
        out["p95_overlap_from_frame"] = (
            df.groupby("trial_id", dropna=False)[overlap_col]
            .quantile(0.95)
        )

    # 标准化 visual instability：events / 1000 frame rows
    if "visual_instability_count_from_frame" in out.columns:
        out["visual_instability_per_1000_frame_rows_from_frame"] = (
            out["visual_instability_count_from_frame"] /
            out["frame_rows_from_frame"].replace(0, np.nan) *
            1000
        )

    return out.reset_index()


# ============================================================
# 5. 从 ContactEvent 重算 event-level 指标
# ============================================================

def recompute_from_contact_events(contact_all: pd.DataFrame) -> pd.DataFrame:
    if contact_all.empty or "trial_id" not in contact_all.columns:
        return pd.DataFrame()

    df = contact_all.copy()

    # 去掉空文件占位行
    if "is_empty_file" in df.columns:
        df = df[df["is_empty_file"] != True].copy()

    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby("trial_id", dropna=False)
    out = pd.DataFrame(index=grouped.size().index)
    out["contact_event_count_from_event"] = grouped.size()

    # blocking success
    col = first_existing_col(df, ["blocking_success", "blocked", "is_blocked"])
    if col is not None:
        df[col] = bool_to_int(df[col])
        out["blocking_success_rate_from_event"] = (
            df.groupby("trial_id", dropna=False)[col].mean()
        )
        out["blocking_success_event_count_from_event"] = (
            df.groupby("trial_id", dropna=False)[col].sum()
        )

    # pass-through event
    col = first_existing_col(df, ["pass_through", "is_pass_through", "pass_through_event"])
    if col is not None:
        df[col] = bool_to_int(df[col])
        out["pass_through_event_count_from_event"] = (
            df.groupby("trial_id", dropna=False)[col].sum()
        )
        out["pass_through_event_rate_from_event"] = (
            df.groupby("trial_id", dropna=False)[col].mean()
        )

    # recovery time
    rec_time_col = first_existing_col(df, ["recovery_time", "recovery_duration"])
    if rec_time_col is not None:
        df[rec_time_col] = pd.to_numeric(df[rec_time_col], errors="coerce")
        valid_rec = df[df[rec_time_col] > 0].copy()
        if not valid_rec.empty:
            out["mean_recovery_time_from_event"] = (
                valid_rec.groupby("trial_id", dropna=False)[rec_time_col].mean()
            )
            out["max_recovery_time_from_event"] = (
                valid_rec.groupby("trial_id", dropna=False)[rec_time_col].max()
            )
            out["valid_recovery_event_count_from_event"] = (
                valid_rec.groupby("trial_id", dropna=False)[rec_time_col].count()
            )

    # recovery success
    recovered_col = first_existing_col(df, ["was_recovered", "recovered", "recovery_success"])
    timeout_col = first_existing_col(df, ["was_timeout", "timeout", "timeout_finalize", "is_timeout"])

    if recovered_col is not None:
        df[recovered_col] = bool_to_int(df[recovered_col])
        recovered_flag = df[recovered_col].copy()
    else:
        # 若没有 was_recovered，则用 recovered_time > contact_end_time 推断
        if "recovered_time" in df.columns and "contact_end_time" in df.columns:
            df["recovered_time"] = pd.to_numeric(df["recovered_time"], errors="coerce")
            df["contact_end_time"] = pd.to_numeric(df["contact_end_time"], errors="coerce")
            recovered_flag = (
                df["recovered_time"].notna()
                & df["contact_end_time"].notna()
                & (df["recovered_time"] > df["contact_end_time"])
            ).astype(int)
        else:
            recovered_flag = None

    if recovered_flag is not None:
        if timeout_col is not None:
            df[timeout_col] = bool_to_int(df[timeout_col])
            recovered_flag = recovered_flag.where(df[timeout_col] == 0, 0)

        df["_recovered_flag_for_analysis"] = recovered_flag
        out["recovered_event_count_from_event"] = (
            df.groupby("trial_id", dropna=False)["_recovered_flag_for_analysis"].sum()
        )
        out["recovery_success_rate_from_event"] = (
            df.groupby("trial_id", dropna=False)["_recovered_flag_for_analysis"].mean()
        )

    # penetration event max
    pen_col = first_existing_col(df, ["max_penetration_depth", "maximum_penetration_depth"])
    if pen_col is not None:
        df[pen_col] = pd.to_numeric(df[pen_col], errors="coerce")
        out["max_penetration_depth_from_event"] = (
            df.groupby("trial_id", dropna=False)[pen_col].max()
        )
        out["mean_event_max_penetration_depth_from_event"] = (
            df.groupby("trial_id", dropna=False)[pen_col].mean()
        )

    # response latency
    latency_col = first_existing_col(df, ["response_latency", "collision_response_latency"])
    if latency_col is not None:
        df[latency_col] = pd.to_numeric(df[latency_col], errors="coerce")
        out["mean_response_latency_from_event"] = (
            df.groupby("trial_id", dropna=False)[latency_col].mean()
        )
        out["max_response_latency_from_event"] = (
            df.groupby("trial_id", dropna=False)[latency_col].max()
        )

    # contact-state target deviation
    deviation_col = first_existing_col(
        df,
        ["contact_target_deviation", "target_deviation", "mean_target_deviation"]
    )
    if deviation_col is not None:
        df[deviation_col] = pd.to_numeric(df[deviation_col], errors="coerce")
        out["mean_contact_target_deviation_from_event"] = (
            df.groupby("trial_id", dropna=False)[deviation_col].mean()
        )

    return out.reset_index()


# ============================================================
# 6. 合并 consistency check
# ============================================================

def build_consistency_check(trial_summary, frame_recomputed, contact_recomputed):
    check = trial_summary.copy()

    if not frame_recomputed.empty:
        check = check.merge(frame_recomputed, on="trial_id", how="left")

    if not contact_recomputed.empty:
        check = check.merge(contact_recomputed, on="trial_id", how="left")

    diff_pairs = [
        ("external_contact_count", "external_contact_count_from_frame"),
        ("contact_attempt_count", "contact_attempt_count_from_frame"),
        ("penetration_frame_ratio", "penetration_frame_ratio_from_frame"),
        ("audio_candidate_count", "audio_candidate_count_from_frame"),
        ("audio_trigger_count", "audio_candidate_count_from_frame"),
        ("pass_through_count", "pass_through_frame_count_from_frame"),
        ("blocking_success_rate", "blocking_success_rate_from_event"),
        ("mean_response_latency", "mean_response_latency_from_event"),
        ("mean_recovery_time", "mean_recovery_time_from_event"),
        ("recovery_success_rate", "recovery_success_rate_from_event"),
        ("max_penetration_depth", "max_penetration_depth_from_frame"),
    ]

    for summary_col, recalc_col in diff_pairs:
        if summary_col in check.columns and recalc_col in check.columns:
            check[summary_col] = pd.to_numeric(check[summary_col], errors="coerce")
            check[recalc_col] = pd.to_numeric(check[recalc_col], errors="coerce")
            check[f"diff__{summary_col}__vs__{recalc_col}"] = (
                check[summary_col] - check[recalc_col]
            )

    return check


# ============================================================
# 7. 生成 task × condition 汇总表
# ============================================================

def build_aggregate_tables(trial_summary):
    summary = trial_summary.copy()

    candidate_metrics = [
        "mean_fps",
        "dropped_frame_rate",
        "convergence_ratio",
        "mean_end_effector_error",
        "max_end_effector_error",
        "mean_jitter",
        "visual_instability_count",
        "visual_instability_rate",
        "external_contact_count",
        "contact_attempt_count",
        "penetration_frame_ratio",
        "mean_penetration_depth",
        "max_penetration_depth",
        "blocking_success_rate",
        "mean_response_latency",
        "mean_recovery_time",
        "recovery_success_rate",
        "contact_target_deviation",
        "audio_candidate_count",
        "audio_trigger_count",
        "pass_through_count",
    ]

    metrics = [m for m in candidate_metrics if m in summary.columns]

    for m in metrics:
        summary[m] = pd.to_numeric(summary[m], errors="coerce")

    tables = {}

    if "task" in summary.columns and "condition" in summary.columns and metrics:
        aggregate = (
            summary
            .groupby(["task", "condition"], dropna=False)[metrics]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        aggregate.columns = [
            "_".join([str(x) for x in col if x != ""])
            for col in aggregate.columns
        ]
        tables["aggregate_by_task_condition"] = aggregate

        for task_name, part in summary.groupby("task", dropna=False):
            task_agg = (
                part
                .groupby("condition", dropna=False)[metrics]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            task_agg.columns = [
                "_".join([str(x) for x in col if x != ""])
                for col in task_agg.columns
            ]
            tables[f"aggregate_{safe_sheet_name(task_name)}"] = task_agg

    if "condition" in summary.columns and metrics:
        condition_agg = (
            summary
            .groupby("condition", dropna=False)[metrics]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        condition_agg.columns = [
            "_".join([str(x) for x in col if x != ""])
            for col in condition_agg.columns
        ]
        tables["aggregate_by_condition_all_tasks"] = condition_agg

    return tables


# ============================================================
# 8. 问题 trial 检查
# ============================================================

def build_problem_trials(trial_summary, consistency_check):
    problems = []

    # recovery_time > 0 但 recovery_success_rate = 0
    if "mean_recovery_time" in trial_summary.columns and "recovery_success_rate" in trial_summary.columns:
        tmp = trial_summary.copy()
        tmp["mean_recovery_time"] = pd.to_numeric(tmp["mean_recovery_time"], errors="coerce")
        tmp["recovery_success_rate"] = pd.to_numeric(tmp["recovery_success_rate"], errors="coerce")
        bad = tmp[(tmp["mean_recovery_time"] > 0) & (tmp["recovery_success_rate"] == 0)]

        for _, row in bad.iterrows():
            problems.append({
                "trial_id": row.get("trial_id", ""),
                "task": row.get("task", ""),
                "condition": row.get("condition", ""),
                "issue_type": "recovery_consistency",
                "issue": "mean_recovery_time > 0 but recovery_success_rate = 0",
            })

    # consistency diff 非零
    diff_cols = [c for c in consistency_check.columns if c.startswith("diff__")]
    for c in diff_cols:
        vals = pd.to_numeric(consistency_check[c], errors="coerce")
        bad = consistency_check[vals.abs() > 1e-6]

        for _, row in bad.iterrows():
            problems.append({
                "trial_id": row.get("trial_id", ""),
                "task": row.get("task", ""),
                "condition": row.get("condition", ""),
                "issue_type": "summary_recompute_difference",
                "issue": c,
                "diff_value": row.get(c, np.nan),
            })

    # Trial_Summary 重复 trial_id
    if "trial_id" in trial_summary.columns:
        dup_counts = trial_summary["trial_id"].value_counts()
        dup_ids = dup_counts[dup_counts > 1].index.tolist()

        for tid in dup_ids:
            rows = trial_summary[trial_summary["trial_id"] == tid]
            for _, row in rows.iterrows():
                problems.append({
                    "trial_id": row.get("trial_id", ""),
                    "task": row.get("task", ""),
                    "condition": row.get("condition", ""),
                    "issue_type": "duplicated_trial_id",
                    "issue": f"duplicated trial_id in Trial_Summary: {tid}",
                })

    return pd.DataFrame(problems)


# ============================================================
# 9. 生成 QC 文本报告
# ============================================================

def unique_trial_set(df):
    if df is not None and not df.empty and "trial_id" in df.columns:
        return set(df["trial_id"].dropna().astype(str))
    return set()


def build_qc_report(data, frame_recomputed, contact_recomputed, consistency_check, problem_trials, aggregate_tables):
    trial_summary = data["trial_summary"]
    trial_manifest = data["trial_manifest"]
    frame_all = data["frame_all"]
    contact_all = data["contact_all"]
    frame_files = data["frame_files"]
    contact_files = data["contact_files"]

    report = []
    report.append("VR Objective Validation QC Report")
    report.append("=" * 80)
    report.append(f"DATA_ROOT: {DATA_ROOT}")
    report.append(f"OUT_DIR: {OUT_DIR}")
    report.append("")
    report.append("File counts")
    report.append("-" * 80)
    report.append(f"FrameLog files: {len(frame_files)}")
    report.append(f"ContactEvent files: {len(contact_files)}")
    report.append(f"Trial_Summary rows: {len(trial_summary)}")
    report.append(f"Trial_Manifest rows: {len(trial_manifest)}")
    report.append("")

    summary_trials = unique_trial_set(trial_summary)
    manifest_trials = unique_trial_set(trial_manifest)
    frame_trials = unique_trial_set(frame_all)
    contact_trials = unique_trial_set(contact_all)

    report.append("Trial ID coverage")
    report.append("-" * 80)
    report.append(f"Manifest unique trials: {len(manifest_trials)}")
    report.append(f"Summary unique trials: {len(summary_trials)}")
    report.append(f"Frame unique trials: {len(frame_trials)}")
    report.append(f"Contact unique trials: {len(contact_trials)}")
    report.append("")

    def add_missing(title, values):
        report.append(f"{title}: {len(values)}")
        if len(values) > 0:
            report.append(", ".join(sorted(list(values))[:50]))
        report.append("")

    add_missing("Trials in summary but not frame", summary_trials - frame_trials)
    add_missing("Trials in frame but not summary", frame_trials - summary_trials)
    add_missing("Trials in summary but not contact", summary_trials - contact_trials)
    add_missing("Trials in contact but not summary", contact_trials - summary_trials)
    add_missing("Trials in manifest but not summary", manifest_trials - summary_trials)
    add_missing("Trials in summary but not manifest", summary_trials - manifest_trials)

    report.append("Column semantic check")
    report.append("-" * 80)
    if "audio_candidate" in frame_all.columns:
        report.append("[OK] FrameLog contains audio_candidate.")
    else:
        report.append("[WARN] FrameLog does not contain audio_candidate.")

    if "audio_event" in frame_all.columns:
        report.append("[WARN] FrameLog still contains audio_event.")

    if "audio_candidate_count" in trial_summary.columns:
        report.append("[OK] Trial_Summary contains audio_candidate_count.")
    else:
        report.append("[WARN] Trial_Summary does not contain audio_candidate_count.")

    if "audio_trigger_count" in trial_summary.columns:
        report.append("[WARN] Trial_Summary still contains audio_trigger_count.")
    report.append("")

    report.append("Recovery consistency check")
    report.append("-" * 80)
    if "mean_recovery_time" in trial_summary.columns and "recovery_success_rate" in trial_summary.columns:
        tmp = trial_summary.copy()
        tmp["mean_recovery_time"] = pd.to_numeric(tmp["mean_recovery_time"], errors="coerce")
        tmp["recovery_success_rate"] = pd.to_numeric(tmp["recovery_success_rate"], errors="coerce")
        bad = tmp[(tmp["mean_recovery_time"] > 0) & (tmp["recovery_success_rate"] == 0)]
        report.append(f"Rows with recovery_time > 0 but success_rate = 0: {len(bad)}")
        if len(bad) > 0:
            cols = [c for c in ["trial_id", "task", "condition", "mean_recovery_time", "recovery_success_rate"] if c in bad.columns]
            report.append(bad[cols].head(30).to_string(index=False))
    else:
        report.append("Recovery columns incomplete.")
    report.append("")

    report.append("Summary vs recomputed metric difference check")
    report.append("-" * 80)
    diff_cols = [c for c in consistency_check.columns if c.startswith("diff__")]
    if diff_cols:
        for c in diff_cols:
            vals = pd.to_numeric(consistency_check[c], errors="coerce")
            nonzero = int((vals.abs() > 1e-6).sum())
            report.append(f"{c}: nonzero rows = {nonzero}")
    else:
        report.append("No diff columns generated.")
    report.append("")

    report.append("Task × Condition trial counts")
    report.append("-" * 80)
    if "task" in trial_summary.columns and "condition" in trial_summary.columns:
        count_table = (
            trial_summary
            .groupby(["task", "condition"], dropna=False)
            .size()
            .reset_index(name="n_trials")
        )
        report.append(count_table.to_string(index=False))
    else:
        report.append("No task / condition columns in Trial_Summary.")
    report.append("")

    report.append("Problem trials")
    report.append("-" * 80)
    report.append(f"Problem rows: {len(problem_trials)}")
    if not problem_trials.empty:
        report.append(problem_trials.head(50).to_string(index=False))
    report.append("")

    report.append("Generated aggregate tables")
    report.append("-" * 80)
    for name, table in aggregate_tables.items():
        report.append(f"{name}: {table.shape[0]} rows × {table.shape[1]} columns")

    return "\n".join(report)


# ============================================================
# 10. 保存所有输出
# ============================================================

def save_outputs(data, frame_recomputed, contact_recomputed, consistency_check, aggregate_tables, problem_trials, qc_report):
    trial_summary = data["trial_summary"]
    parameter_settings = data["parameter_settings"]
    threshold_settings = data["threshold_settings"]
    trial_manifest = data["trial_manifest"]

    # CSV 输出
    (OUT_DIR / "00_QC_report.txt").write_text(qc_report, encoding="utf-8-sig")

    trial_summary.to_csv(OUT_DIR / "01_trial_summary_clean.csv", index=False, encoding="utf-8-sig")
    frame_recomputed.to_csv(OUT_DIR / "02_recomputed_from_frame_logs.csv", index=False, encoding="utf-8-sig")
    contact_recomputed.to_csv(OUT_DIR / "03_recomputed_from_contact_events.csv", index=False, encoding="utf-8-sig")
    consistency_check.to_csv(OUT_DIR / "04_trial_summary_consistency_check.csv", index=False, encoding="utf-8-sig")
    problem_trials.to_csv(OUT_DIR / "06_problem_trials.csv", index=False, encoding="utf-8-sig")

    for name, table in aggregate_tables.items():
        file_name = re.sub(r"[^0-9a-zA-Z_\-\u4e00-\u9fa5]+", "_", name)
        table.to_csv(OUT_DIR / f"05_{file_name}.csv", index=False, encoding="utf-8-sig")

    # Excel 总表
    excel_path = OUT_DIR / "VR_objective_validation_analysis.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        trial_summary.to_excel(writer, sheet_name="trial_summary", index=False)
        consistency_check.to_excel(writer, sheet_name="consistency_check", index=False)
        frame_recomputed.to_excel(writer, sheet_name="recomputed_frame", index=False)
        contact_recomputed.to_excel(writer, sheet_name="recomputed_contact", index=False)
        problem_trials.to_excel(writer, sheet_name="problem_trials", index=False)

        for name, table in aggregate_tables.items():
            table.to_excel(writer, sheet_name=safe_sheet_name(name), index=False)

        parameter_settings.to_excel(writer, sheet_name="parameter_settings", index=False)
        threshold_settings.to_excel(writer, sheet_name="threshold_settings", index=False)
        trial_manifest.to_excel(writer, sheet_name="trial_manifest", index=False)

    # 小 zip，后续可以发给 ChatGPT
    zip_path = OUT_DIR / "analysis_outputs_for_chatgpt.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in OUT_DIR.glob("*"):
            if f.name == zip_path.name:
                continue
            if f.is_file():
                z.write(f, arcname=f.name)

    print("\n========== 分析完成 ==========")
    print(f"输出文件夹：{OUT_DIR}")
    print(f"Excel 总表：{excel_path}")
    print(f"可上传给 ChatGPT 的小压缩包：{zip_path}")


# ============================================================
# 11. 主程序
# ============================================================

def main():
    data = load_raw_data()

    # 标准化 Trial_Summary 再进入后续流程
    data["trial_summary"] = normalize_columns(data["trial_summary"])
    data["trial_manifest"] = normalize_columns(data["trial_manifest"])
    data["frame_all"] = normalize_columns(data["frame_all"])
    data["contact_all"] = normalize_columns(data["contact_all"])

    # 标记分析窗口
    data["frame_all"] = add_analysis_window(data["frame_all"], data["trial_summary"])

    # 重算指标
    frame_recomputed = recompute_from_frame_logs(data["frame_all"])
    contact_recomputed = recompute_from_contact_events(data["contact_all"])

    # 一致性检查
    consistency_check = build_consistency_check(
        data["trial_summary"],
        frame_recomputed,
        contact_recomputed
    )

    # 汇总表
    aggregate_tables = build_aggregate_tables(data["trial_summary"])

    # 问题 trial
    problem_trials = build_problem_trials(data["trial_summary"], consistency_check)

    # QC 报告
    qc_report = build_qc_report(
        data,
        frame_recomputed,
        contact_recomputed,
        consistency_check,
        problem_trials,
        aggregate_tables
    )

    print("\n" + qc_report)

    # 保存
    save_outputs(
        data,
        frame_recomputed,
        contact_recomputed,
        consistency_check,
        aggregate_tables,
        problem_trials,
        qc_report
    )


if __name__ == "__main__":
    main()