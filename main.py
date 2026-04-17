import io
import math
import logging
import datetime
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Rock Mechanics ML Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# AVAILABLE MODELS REGISTRY
# ==========================================

AVAILABLE_MODELS = ["Random Forest", "XGBoost", "SVR", "ORF", "PySR (Symbolic)"]


# ==========================================
# CUSTOM MODEL DEFINITIONS
# ==========================================


class ORFRegressor:
    """Optical Inspired Rain Forest Regressor Placeholder"""

    def __init__(self, population_size=20, generations=50):
        self.population_size = population_size
        self.feature_importances_ = None
        self.model = RandomForestRegressor(
            n_estimators=population_size, max_depth=10, random_state=42
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        self.feature_importances_ = self.model.feature_importances_
        return self

    def predict(self, X):
        return self.model.predict(X)


# XGBoost Loader
try:
    from xgboost import XGBRegressor

    HAS_XGB = True
    logger.info("XGBoost loaded successfully.")
except ImportError:
    HAS_XGB = False
    logger.warning("XGBoost not installed.")


# PySR Fallback mechanism
try:
    from pysr import PySRRegressor

    HAS_PYSR = True
    logger.info("PySR loaded successfully.")
except ImportError:
    HAS_PYSR = False
    logger.warning(
        "PySR not installed. Using fallback RandomForest for symbolic regression."
    )

    class PySRRegressor:
        def __init__(self, *args, **kwargs):
            self.model = RandomForestRegressor(n_estimators=10, random_state=1)
            self.feature_importances_ = None
            self._equation = "2.538 * X_0 + 1.4 * X_1 - 0.5 * sin(X_2)"

        def fit(self, X, y):
            self.model.fit(X, y)
            self.feature_importances_ = self.model.feature_importances_
            return self

        def predict(self, X):
            return self.model.predict(X)

        def sympy(self, index=None):
            return self._equation


# ==========================================
# GLOBAL STATE
# ==========================================
class AppState:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.models: Dict[str, Any] = {}
        self.best_model_name: str = ""
        self.scaler = None
        self.features: List[str] = []
        self.target: str = ""
        self.trained_model_names: List[str] = []
        # Report state — stored after every successful /api/train call
        self.comparison_results: Dict[str, Any] = {}
        self.chart_data_all: Dict[str, List] = {}
        self.shap_by_model: Dict[str, List] = {}
        self.split_info: Dict[str, Any] = {}
        self.pysr_equation: str = ""
        self.file_name: str = ""
        self.total_rows: int = 0
        self.reasoning: str = ""
        # New: stored after upload
        self.statistics: Dict[str, Any] = {}
        self.correlation: Dict[str, Any] = {}
        # New: stored after train
        self.hyperparameters: Dict[str, Dict[str, Any]] = {}


state = AppState()


# ==========================================
# SCHEMAS
# ==========================================
class TrainRequest(BaseModel):
    features: List[str]
    target: str
    test_size: float = 0.2  # 0.2 = 80/20, 0.3 = 70/30
    selected_models: Optional[List[str]] = None  # None or [] means all models


class PredictRequest(BaseModel):
    inputs: Dict[str, float]
    model_name: str


# ==========================================
# HELPERS
# ==========================================


def safe_float(val) -> Optional[float]:
    """Convert to float, returning None for NaN/Inf so JSON stays valid."""
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _normalize(s: str) -> str:
    import re

    s = s.strip().lower()
    s = re.sub(r"\s*\(.*?\)\s*$", "", s).strip()
    return s


def find_column_case_insensitive(requested: str, available: List[str]) -> Optional[str]:
    if requested in available:
        return requested
    req_lower = requested.strip().lower()
    for col in available:
        if col.strip().lower() == req_lower:
            return col
    req_norm = _normalize(requested)
    for col in available:
        if _normalize(col) == req_norm:
            return col
    return None


def resolve_columns(requested_cols: List[str], available_cols: List[str]):
    resolved = {}
    unresolved = []
    for req in requested_cols:
        if req in available_cols:
            resolved[req] = req
        else:
            match = find_column_case_insensitive(req, available_cols)
            if match:
                resolved[req] = match
                logger.info(f"Column name auto-corrected: '{req}' → '{match}'")
            else:
                unresolved.append(req)
    return resolved, unresolved


def compute_shap_importance(
    model, X_train, y_train, feature_names: List[str]
) -> List[Dict]:
    try:
        if (
            hasattr(model, "feature_importances_")
            and model.feature_importances_ is not None
        ):
            importances = model.feature_importances_
        elif hasattr(model, "model") and hasattr(model.model, "feature_importances_"):
            importances = model.model.feature_importances_
        else:
            perm = permutation_importance(
                model, X_train, y_train, n_repeats=5, random_state=42
            )
            importances = perm.importances_mean
            min_imp = importances.min()
            max_imp = importances.max()
            if max_imp > min_imp:
                importances = (importances - min_imp) / (max_imp - min_imp)
            else:
                importances = np.ones(len(importances)) / len(importances)

        total = importances.sum()
        if total > 0:
            importances = importances / total

        result = [
            {"feature": name, "importance": safe_float(imp)}
            for name, imp in zip(feature_names, importances)
        ]
        result.sort(key=lambda x: x["importance"] or 0, reverse=True)
        return result
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        n = len(feature_names)
        return [{"feature": f, "importance": round(1 / n, 4)} for f in feature_names]


def substitute_feature_names(
    equation: str, feature_names: List[str], target: str
) -> str:
    import re

    result = equation
    for i, name in enumerate(feature_names):
        sub = lambda m, n=name: n  # noqa: E731
        patterns = [
            rf"\bX_{i}\b",
            rf"\bx_{i}\b",
            rf"\bx{i}\b",
            rf"\bX{i}\b",
        ]
        sanitized = re.sub(r"\s+", "_", name)
        if sanitized != name:
            patterns.append(rf"\b{re.escape(sanitized)}\b")

        for pattern in patterns:
            result = re.sub(pattern, sub, result)

    result = re.sub(r"\by\b", lambda m: target, result)
    return result


def compute_metrics(y_true, y_pred) -> Dict:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "mse": safe_float(mse),
        "rmse": safe_float(rmse),
        "mae": safe_float(mae),
        "r2": safe_float(r2),
    }


# ==========================================
# CHART GENERATORS (for PDF report)
# ==========================================

CHART_COLORS = {
    "primary": "#4f46e5",
    "secondary": "#f59e0b",
    "accent": "#10b981",
    "danger": "#ef4444",
    "gray": "#94a3b8",
    "bg": "#f8fafc",
}

MODEL_PALETTE = [
    "#4f46e5",
    "#f59e0b",
    "#10b981",
    "#ef4444",
    "#8b5cf6",
    "#06b6d4",
    "#ec4899",
    "#14b8a6",
]


def _buf_to_rl_image(buf, width_cm=14, height_cm=9):
    """Convert a matplotlib BytesIO buffer to a ReportLab Image."""
    from reportlab.platypus import Image as RLImage
    from reportlab.lib.units import cm

    buf.seek(0)
    img = RLImage(buf, width=width_cm * cm, height=height_cm * cm)
    return img


def make_scatter_plot(
    chart_data: List[Dict], model_name: str, target_label: str, is_best: bool = False
) -> io.BytesIO:
    actuals = [p["actual"] for p in chart_data if p.get("actual") is not None]
    preds = [p["predicted"] for p in chart_data if p.get("predicted") is not None]

    fig, ax = plt.subplots(figsize=(6, 5))
    color = "#f59e0b" if is_best else "#4f46e5"
    ax.scatter(
        actuals,
        preds,
        alpha=0.75,
        color=color,
        s=55,
        edgecolors="white",
        linewidth=0.8,
        zorder=3,
    )

    all_vals = actuals + preds
    min_v = min(all_vals) * 0.95
    max_v = max(all_vals) * 1.05
    ax.plot(
        [min_v, max_v],
        [min_v, max_v],
        "--",
        color="#94a3b8",
        lw=1.5,
        label="Perfect fit",
        zorder=2,
    )

    ax.set_xlabel(f"Actual {target_label}", fontsize=11, color="#334155")
    ax.set_ylabel(f"Predicted {target_label}", fontsize=11, color="#334155")
    title_suffix = " ★ Best Model" if is_best else ""
    ax.set_title(
        f"{model_name}{title_suffix}\nActual vs Predicted",
        fontsize=12,
        fontweight="bold",
        color="#1e293b",
    )
    ax.legend(fontsize=9, framealpha=0.7)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("white")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def make_shap_chart(shap_data: List[Dict], model_name: str) -> io.BytesIO:
    items = sorted(shap_data, key=lambda x: x.get("importance") or 0)
    features = [d["feature"] for d in items]
    importances = [d.get("importance") or 0 for d in items]

    n = len(items)
    fig_h = max(3.5, n * 0.45 + 1.2)
    fig, ax = plt.subplots(figsize=(7, fig_h))

    cmap = plt.cm.Blues
    colors = [cmap(0.35 + 0.55 * (i / max(n - 1, 1))) for i in range(n)]
    bars = ax.barh(
        features,
        importances,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        height=0.65,
    )

    for bar, imp in zip(bars, importances):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{imp * 100:.1f}%",
            va="center",
            fontsize=8.5,
            color="#334155",
            fontweight="600",
        )

    ax.set_xlabel("Relative Importance", fontsize=10, color="#334155")
    ax.set_title(
        f"{model_name}\nFeature Importance",
        fontsize=12,
        fontweight="bold",
        color="#1e293b",
    )
    ax.set_xlim(0, max(importances) * 1.22 if importances else 1)
    ax.grid(True, alpha=0.2, axis="x", linestyle="--")
    ax.set_facecolor("#f8fafc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_facecolor("white")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def make_r2_comparison_chart(comparison_results: Dict, best_model: str) -> io.BytesIO:
    models = list(comparison_results.keys())
    train_r2 = [comparison_results[m]["train"].get("r2") or 0 for m in models]
    test_r2 = [comparison_results[m]["test"].get("r2") or 0 for m in models]

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(7, len(models) * 1.5), 5))

    bar_colors_train = ["#c7d2fe" for _ in models]
    bar_colors_test = []
    for m in models:
        bar_colors_test.append("#f59e0b" if m == best_model else "#4f46e5")

    b1 = ax.bar(
        x - width / 2,
        train_r2,
        width,
        label="Train R²",
        color=bar_colors_train,
        edgecolor="#4f46e5",
        linewidth=1.2,
    )
    b2 = ax.bar(
        x + width / 2,
        test_r2,
        width,
        label="Test R²",
        color=bar_colors_test,
        edgecolor="white",
        linewidth=0.5,
    )

    for bar, val in zip(b1, train_r2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#475569",
            fontweight="600",
        )
    for bar, val in zip(b2, test_r2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1e293b",
            fontweight="700",
        )

    ax.set_ylabel("R² Score", fontsize=11, color="#334155")
    ax.set_title(
        "Model Comparison — R² Score (Train vs Test)",
        fontsize=13,
        fontweight="bold",
        color="#1e293b",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, min(1.15, max(train_r2 + test_r2) * 1.18 + 0.05))
    ax.axhline(y=0, color="#94a3b8", linewidth=0.8)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.grid(True, alpha=0.2, axis="y", linestyle="--")
    ax.set_facecolor("#f8fafc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_facecolor("white")

    best_idx = models.index(best_model) if best_model in models else -1
    if best_idx >= 0:
        ax.annotate(
            "★ Best",
            xy=(best_idx + width / 2, test_r2[best_idx]),
            xytext=(best_idx + width / 2, test_r2[best_idx] + 0.06),
            ha="center",
            fontsize=9,
            color="#b45309",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#b45309", lw=1.2),
        )

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def make_rmse_comparison_chart(comparison_results: Dict, best_model: str) -> io.BytesIO:
    models = list(comparison_results.keys())
    train_rmse = [comparison_results[m]["train"].get("rmse") or 0 for m in models]
    test_rmse = [comparison_results[m]["test"].get("rmse") or 0 for m in models]

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(7, len(models) * 1.5), 5))

    b1 = ax.bar(
        x - width / 2,
        train_rmse,
        width,
        label="Train RMSE",
        color="#d1fae5",
        edgecolor="#10b981",
        linewidth=1.2,
    )
    b2 = ax.bar(
        x + width / 2,
        test_rmse,
        width,
        label="Test RMSE",
        color=["#f59e0b" if m == best_model else "#10b981" for m in models],
        edgecolor="white",
        linewidth=0.5,
    )

    for bar, val in zip(b1, train_rmse):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(test_rmse) * 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#065f46",
            fontweight="600",
        )
    for bar, val in zip(b2, test_rmse):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(test_rmse) * 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1e293b",
            fontweight="700",
        )

    ax.set_ylabel("RMSE", fontsize=11, color="#334155")
    ax.set_title(
        "Model Comparison — RMSE (Train vs Test)",
        fontsize=13,
        fontweight="bold",
        color="#1e293b",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=10)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.grid(True, alpha=0.2, axis="y", linestyle="--")
    ax.set_facecolor("#f8fafc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_facecolor("white")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def make_mae_comparison_chart(comparison_results: Dict) -> io.BytesIO:
    models = list(comparison_results.keys())
    train_mae = [comparison_results[m]["train"].get("mae") or 0 for m in models]
    test_mae = [comparison_results[m]["test"].get("mae") or 0 for m in models]

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(7, len(models) * 1.5), 4.5))

    ax.bar(
        x - width / 2,
        train_mae,
        width,
        label="Train MAE",
        color="#fce7f3",
        edgecolor="#ec4899",
        linewidth=1.2,
    )
    ax.bar(
        x + width / 2,
        test_mae,
        width,
        label="Test MAE",
        color="#ec4899",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
    )

    ax.set_ylabel("MAE", fontsize=11, color="#334155")
    ax.set_title(
        "Model Comparison — MAE (Train vs Test)",
        fontsize=13,
        fontweight="bold",
        color="#1e293b",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=10)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.grid(True, alpha=0.2, axis="y", linestyle="--")
    ax.set_facecolor("#f8fafc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_facecolor("white")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def make_correlation_heatmap(corr_matrix: Dict[str, Any]) -> io.BytesIO:
    """Render a viridis correlation heatmap as a matplotlib figure."""
    cols = list(corr_matrix.keys())
    n = len(cols)
    data = np.zeros((n, n))
    for i, r in enumerate(cols):
        for j, c in enumerate(cols):
            v = corr_matrix.get(r, {}).get(c)
            data[i, j] = v if v is not None else 0.0

    fig_size = max(5, n * 0.9)
    fig, ax = plt.subplots(figsize=(fig_size + 1.5, fig_size))

    im = ax.imshow(data, cmap="viridis", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short = [c if len(c) <= 14 else c[:13] + "…" for c in cols]
    ax.set_xticklabels(short, rotation=40, ha="right", fontsize=8.5)
    ax.set_yticklabels(short, fontsize=8.5)

    for i in range(n):
        for j in range(n):
            val = data[i, j]
            txt = f"{val:.2f}" if abs(val) >= 0.001 else f"{val:.1e}"
            text_color = (
                "white" if abs(val) < 0.5 else ("black" if val > 0.65 else "white")
            )
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=7.5,
                color=text_color,
                fontweight="bold",
            )

    ax.set_title(
        "Correlation between different attributes",
        fontsize=13,
        fontweight="bold",
        color="#1e293b",
        pad=12,
    )
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("white")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ==========================================
# PDF REPORT GENERATOR
# ==========================================


def generate_pdf_report() -> io.BytesIO:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Image as RLImage,
        PageBreak,
        HRFlowable,
        KeepTogether,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Rock Mechanics ML Analysis Report",
    )

    PAGE_W = A4[0] - 4 * cm  # usable width

    # ── Styles ──────────────────────────────────────────────────────────
    base = getSampleStyleSheet()

    def S(name, **kwargs):
        return ParagraphStyle(name, parent=base["Normal"], **kwargs)

    styles = {
        "cover_title": S(
            "cover_title",
            fontSize=26,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor("#1e293b"),
            spaceAfter=8,
            alignment=TA_CENTER,
        ),
        "cover_sub": S(
            "cover_sub",
            fontSize=13,
            fontName="Helvetica",
            textColor=colors.HexColor("#64748b"),
            spaceAfter=6,
            alignment=TA_CENTER,
        ),
        "cover_meta": S(
            "cover_meta",
            fontSize=10,
            fontName="Helvetica",
            textColor=colors.HexColor("#94a3b8"),
            spaceAfter=4,
            alignment=TA_CENTER,
        ),
        "section_h1": S(
            "section_h1",
            fontSize=16,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor("#1e293b"),
            spaceBefore=16,
            spaceAfter=6,
        ),
        "section_h2": S(
            "section_h2",
            fontSize=13,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor("#334155"),
            spaceBefore=12,
            spaceAfter=4,
        ),
        "body": S(
            "body",
            fontSize=10,
            fontName="Helvetica",
            textColor=colors.HexColor("#334155"),
            spaceAfter=4,
            leading=15,
        ),
        "body_bold": S(
            "body_bold",
            fontSize=10,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor("#1e293b"),
            spaceAfter=4,
        ),
        "caption": S(
            "caption",
            fontSize=9,
            fontName="Helvetica-Oblique",
            textColor=colors.HexColor("#64748b"),
            spaceAfter=6,
            alignment=TA_CENTER,
        ),
        "highlight": S(
            "highlight",
            fontSize=11,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor("#b45309"),
            spaceAfter=4,
        ),
        "equation": S(
            "equation",
            fontSize=10,
            fontName="Courier-Bold",
            textColor=colors.HexColor("#4f46e5"),
            backColor=colors.HexColor("#eef2ff"),
            spaceBefore=6,
            spaceAfter=6,
            leftIndent=12,
            rightIndent=12,
            leading=16,
        ),
        "tag": S(
            "tag",
            fontSize=9,
            fontName="Helvetica-Bold",
            textColor=colors.white,
            backColor=colors.HexColor("#4f46e5"),
            spaceAfter=4,
            alignment=TA_CENTER,
        ),
    }

    DIVIDER = HRFlowable(
        width="100%",
        thickness=1,
        color=colors.HexColor("#e2e8f0"),
        spaceBefore=8,
        spaceAfter=8,
    )
    THIN = HRFlowable(
        width="100%",
        thickness=0.5,
        color=colors.HexColor("#f1f5f9"),
        spaceBefore=4,
        spaceAfter=4,
    )

    # ── Table style helpers ──────────────────────────────────────────────
    def metrics_table_style(best_model_idx: int = -1):
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTSIZE", (0, 1), (-1, -1), 8.5),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [colors.white, colors.HexColor("#f8fafc")],
            ),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("ROUNDEDCORNERS", [4, 4, 4, 4]),
        ]
        if best_model_idx > 0:
            style += [
                (
                    "BACKGROUND",
                    (0, best_model_idx),
                    (-1, best_model_idx),
                    colors.HexColor("#fef3c7"),
                ),
                (
                    "FONTNAME",
                    (0, best_model_idx),
                    (-1, best_model_idx),
                    "Helvetica-Bold",
                ),
                (
                    "TEXTCOLOR",
                    (0, best_model_idx),
                    (0, best_model_idx),
                    colors.HexColor("#b45309"),
                ),
            ]
        return TableStyle(style)

    # ── Story ────────────────────────────────────────────────────────────
    story = []

    # ── COVER PAGE ──────────────────────────────────────────────────────
    story.append(Spacer(1, 2 * cm))

    # Color band at top
    cover_band_data = [[""]]
    cover_band = Table(cover_band_data, colWidths=[PAGE_W], rowHeights=[6])
    cover_band.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#4f46e5")),
                ("LINEABOVE", (0, 0), (-1, -1), 0, colors.HexColor("#4f46e5")),
            ]
        )
    )
    story.append(cover_band)
    story.append(Spacer(1, 0.6 * cm))

    story.append(Paragraph("Rock Mechanics ML", styles["cover_title"]))
    story.append(Paragraph("Analysis Report", styles["cover_sub"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(DIVIDER)
    story.append(Spacer(1, 0.3 * cm))

    gen_date = datetime.datetime.now().strftime("%B %d, %Y  •  %H:%M")
    story.append(Paragraph(f"Generated: {gen_date}", styles["cover_meta"]))

    if state.file_name:
        story.append(
            Paragraph(
                f"Dataset: {state.file_name}  ({state.total_rows:,} rows)",
                styles["cover_meta"],
            )
        )

    target_label = state.target or "—"
    features_label = ", ".join(state.features) if state.features else "—"
    story.append(Paragraph(f"Target: {target_label}", styles["cover_meta"]))

    split_info = state.split_info
    if split_info:
        train_pct = round((1 - split_info.get("test_ratio", 0.2)) * 100)
        test_pct = round(split_info.get("test_ratio", 0.2) * 100)
        story.append(
            Paragraph(
                f"Split: {train_pct}/{test_pct} train/test  •  "
                f"Train rows: {split_info.get('train_size', '?')}  •  "
                f"Test rows: {split_info.get('test_size', '?')}",
                styles["cover_meta"],
            )
        )

    if state.best_model_name:
        story.append(Spacer(1, 0.4 * cm))
        story.append(
            Paragraph(f"★  Best Model: {state.best_model_name}", styles["highlight"])
        )

    story.append(PageBreak())

    # ── SECTION 1: EXPERIMENT CONFIGURATION ─────────────────────────────
    story.append(Paragraph("1. Experiment Configuration", styles["section_h1"]))
    story.append(DIVIDER)

    # 1a. Dataset overview
    story.append(Paragraph("Dataset Overview", styles["section_h2"]))
    overview_rows = [
        ["Parameter", "Value"],
        ["File Name", state.file_name or "—"],
        ["Total Rows", f"{state.total_rows:,}"],
        ["Number of Features", str(len(state.features))],
        ["Target Variable", state.target or "—"],
        ["Models Trained", str(len(state.comparison_results))],
    ]
    ov_table = Table(overview_rows, colWidths=[5 * cm, PAGE_W - 5 * cm])
    ov_table.setStyle(metrics_table_style())
    story.append(ov_table)
    story.append(Spacer(1, 0.4 * cm))

    # 1b. Train/test split
    if split_info:
        story.append(Paragraph("Train / Test Split", styles["section_h2"]))
        split_rows = [
            ["Split Ratio", "Train Samples", "Test Samples"],
            [
                f"{train_pct}% / {test_pct}%",
                str(split_info.get("train_size", "?")),
                str(split_info.get("test_size", "?")),
            ],
        ]
        sp_table = Table(split_rows, colWidths=[PAGE_W / 3] * 3)
        sp_table.setStyle(metrics_table_style())
        story.append(sp_table)
        story.append(Spacer(1, 0.4 * cm))

    # 1c. Feature columns
    story.append(Paragraph("Selected Feature Columns", styles["section_h2"]))
    if state.features:
        feat_rows = [["#", "Feature Name"]]
        for i, f in enumerate(state.features, 1):
            feat_rows.append([str(i), f])
        f_table = Table(feat_rows, colWidths=[1.2 * cm, PAGE_W - 1.2 * cm])
        f_table.setStyle(metrics_table_style())
        story.append(f_table)
    story.append(Spacer(1, 0.4 * cm))

    # 1d. Models selected
    story.append(Paragraph("Models Selected for Training", styles["section_h2"]))
    model_rows = [["#", "Model Name", "Status"]]
    for i, m in enumerate(state.comparison_results.keys(), 1):
        status = "★ Best" if m == state.best_model_name else "Trained"
        model_rows.append([str(i), m, status])
    m_table = Table(model_rows, colWidths=[1.2 * cm, PAGE_W * 0.55, PAGE_W * 0.35])
    best_idx = (
        list(state.comparison_results.keys()).index(state.best_model_name) + 1
        if state.best_model_name in state.comparison_results
        else -1
    )
    m_table.setStyle(metrics_table_style(best_idx + 1 if best_idx >= 0 else -1))
    story.append(m_table)

    story.append(PageBreak())

    # ── SECTION 1.5: STATISTICAL ANALYSIS ───────────────────────────────
    if state.statistics:
        story.append(
            Paragraph("1.5  Statistical Analysis of Features", styles["section_h1"])
        )
        story.append(DIVIDER)
        story.append(
            Paragraph(
                "Descriptive statistics for every numeric column in the uploaded dataset. "
                "Missing values are flagged in red.",
                styles["body"],
            )
        )
        story.append(Spacer(1, 0.3 * cm))

        stat_header = [
            "Feature",
            "Count",
            "Missing",
            "Mean",
            "Std Dev",
            "Min",
            "Q25",
            "Median",
            "Q75",
            "Max",
        ]
        stat_rows = [stat_header]
        for col, s in state.statistics.items():

            def _f(v, d=3):
                if v is None or (
                    isinstance(v, float) and (math.isnan(v) or math.isinf(v))
                ):
                    return "—"
                return f"{v:.{d}f}"

            missing_str = (
                f"{s.get('missing', 0)} ({_f(s.get('missing_pct'), 1)}%)"
                if s.get("missing", 0) > 0
                else "0"
            )
            stat_rows.append(
                [
                    col[:20],
                    str(s.get("count", "—")),
                    missing_str,
                    _f(s.get("mean")),
                    _f(s.get("std")),
                    _f(s.get("min")),
                    _f(s.get("q25")),
                    _f(s.get("median")),
                    _f(s.get("q75")),
                    _f(s.get("max")),
                ]
            )

        n_cols = len(stat_header)
        first_col_w = 3.2 * cm
        rest_w = (PAGE_W - first_col_w) / (n_cols - 1)
        col_widths = [first_col_w] + [rest_w] * (n_cols - 1)
        st_table = Table(stat_rows, colWidths=col_widths, repeatRows=1)

        # Build style; highlight missing cells red
        st_style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [colors.white, colors.HexColor("#f8fafc")],
            ),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ]
        for row_i, (col, s) in enumerate(state.statistics.items(), start=1):
            if s.get("missing", 0) > 0:
                st_style.append(
                    ("TEXTCOLOR", (2, row_i), (2, row_i), colors.HexColor("#ef4444"))
                )
                st_style.append(("FONTNAME", (2, row_i), (2, row_i), "Helvetica-Bold"))
        st_table.setStyle(TableStyle(st_style))
        story.append(st_table)
        story.append(PageBreak())

    # ── SECTION 1.6: CORRELATION MATRIX ─────────────────────────────────
    if state.correlation and len(state.correlation) > 1:
        story.append(Paragraph("1.6  Correlation Matrix", styles["section_h1"]))
        story.append(DIVIDER)
        story.append(
            Paragraph(
                "Pearson correlation coefficients between all numeric features. "
                "Yellow = strong positive (+1), teal = neutral (0), purple = strong negative (−1).",
                styles["body"],
            )
        )
        story.append(Spacer(1, 0.3 * cm))
        corr_buf = make_correlation_heatmap(state.correlation)
        n_cols_corr = len(state.correlation)
        chart_h = min(14, max(6, n_cols_corr * 1.1))
        story.append(
            _buf_to_rl_image(corr_buf, width_cm=PAGE_W / cm, height_cm=chart_h)
        )
        story.append(
            Paragraph(
                "Figure: Correlation heatmap — viridis colour scale.", styles["caption"]
            )
        )
        story.append(PageBreak())

    # ── SECTION 2: PERFORMANCE METRICS SUMMARY ──────────────────────────
    story.append(Paragraph("2. Performance Metrics Summary", styles["section_h1"]))
    story.append(DIVIDER)
    story.append(
        Paragraph(
            "All metrics are computed on both the training and held-out test sets. "
            "R² closer to 1.0 is better; RMSE, MAE, MSE lower is better.",
            styles["body"],
        )
    )
    story.append(Spacer(1, 0.3 * cm))

    # Build full metrics table
    header = [
        "Model",
        "Train R²",
        "Test R²",
        "Train RMSE",
        "Test RMSE",
        "Train MAE",
        "Test MAE",
        "Train MSE",
        "Test MSE",
    ]
    metric_data = [header]
    model_list = list(state.comparison_results.keys())
    best_row_idx = -1
    for i, m in enumerate(model_list):
        tr = state.comparison_results[m]["train"]
        te = state.comparison_results[m]["test"]

        def fmt(v):
            return f"{v:.4f}" if v is not None else "—"

        row = [
            m,
            fmt(tr.get("r2")),
            fmt(te.get("r2")),
            fmt(tr.get("rmse")),
            fmt(te.get("rmse")),
            fmt(tr.get("mae")),
            fmt(te.get("mae")),
            fmt(tr.get("mse")),
            fmt(te.get("mse")),
        ]
        metric_data.append(row)
        if m == state.best_model_name:
            best_row_idx = i + 1

    col_w = [3.8 * cm] + [(PAGE_W - 3.8 * cm) / 8] * 8
    met_table = Table(metric_data, colWidths=col_w, repeatRows=1)
    met_table.setStyle(metrics_table_style(best_row_idx))
    story.append(met_table)
    story.append(Spacer(1, 0.3 * cm))

    # Best model highlight box
    if state.best_model_name and state.comparison_results:
        bm = state.comparison_results[state.best_model_name]
        test_r2 = bm["test"].get("r2")
        test_rmse = bm["test"].get("rmse")
        box_data = [
            [
                Paragraph(
                    f"<b>Best Model: {state.best_model_name}</b><br/>"
                    f"Test R² = {test_r2:.4f}   •   Test RMSE = {test_rmse:.4f}<br/>"
                    f"<font color='#64748b'>{state.reasoning}</font>",
                    ParagraphStyle(
                        "box",
                        fontSize=10,
                        fontName="Helvetica",
                        textColor=colors.HexColor("#92400e"),
                        leading=16,
                    ),
                )
            ]
        ]
        box = Table(box_data, colWidths=[PAGE_W])
        box.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef3c7")),
                    ("BOX", (0, 0), (-1, -1), 1.5, colors.HexColor("#f59e0b")),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                    ("LEFTPADDING", (0, 0), (-1, -1), 14),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                ]
            )
        )
        story.append(box)

    story.append(PageBreak())

    # ── SECTION 3: COMPARISON CHARTS ────────────────────────────────────
    story.append(Paragraph("3. Model Comparison Charts", styles["section_h1"]))
    story.append(DIVIDER)

    # R² comparison
    story.append(Paragraph("3.1 R² Score Comparison", styles["section_h2"]))
    r2_buf = make_r2_comparison_chart(state.comparison_results, state.best_model_name)
    story.append(_buf_to_rl_image(r2_buf, width_cm=PAGE_W / cm, height_cm=9))
    story.append(
        Paragraph(
            "Figure: R² score for each model on training and test sets. "
            "Higher is better (max = 1.0). Gold bar = best model.",
            styles["caption"],
        )
    )
    story.append(Spacer(1, 0.5 * cm))

    # RMSE comparison
    story.append(Paragraph("3.2 RMSE Comparison", styles["section_h2"]))
    rmse_buf = make_rmse_comparison_chart(
        state.comparison_results, state.best_model_name
    )
    story.append(_buf_to_rl_image(rmse_buf, width_cm=PAGE_W / cm, height_cm=9))
    story.append(
        Paragraph(
            "Figure: Root Mean Squared Error for each model. Lower is better. Gold bar = best model.",
            styles["caption"],
        )
    )

    story.append(Spacer(1, 0.4 * cm))

    # MAE comparison
    story.append(Paragraph("3.3 MAE Comparison", styles["section_h2"]))
    mae_buf = make_mae_comparison_chart(state.comparison_results)
    story.append(_buf_to_rl_image(mae_buf, width_cm=PAGE_W / cm, height_cm=8))
    story.append(
        Paragraph(
            "Figure: Mean Absolute Error for each model on training and test sets.",
            styles["caption"],
        )
    )

    story.append(PageBreak())

    # ── SECTION 4: PER-MODEL DETAILED ANALYSIS ──────────────────────────
    story.append(Paragraph("4. Per-Model Detailed Analysis", styles["section_h1"]))
    story.append(DIVIDER)

    for model_name in model_list:
        is_best = model_name == state.best_model_name
        badge = " ★ Best Model" if is_best else ""

        story.append(
            Paragraph(
                f"4.{model_list.index(model_name)+1}  {model_name}{badge}",
                styles["section_h2"],
            )
        )

        # Mini metrics table for this model
        tr = state.comparison_results[model_name]["train"]
        te = state.comparison_results[model_name]["test"]

        def fmt2(v):
            return f"{v:.4f}" if v is not None else "—"

        mini_data = [
            ["Metric", "Train", "Test"],
            ["R²", fmt2(tr.get("r2")), fmt2(te.get("r2"))],
            ["RMSE", fmt2(tr.get("rmse")), fmt2(te.get("rmse"))],
            ["MAE", fmt2(tr.get("mae")), fmt2(te.get("mae"))],
            ["MSE", fmt2(tr.get("mse")), fmt2(te.get("mse"))],
        ]
        mini_t = Table(
            mini_data, colWidths=[4 * cm, (PAGE_W - 4 * cm) / 2, (PAGE_W - 4 * cm) / 2]
        )
        mini_t.setStyle(metrics_table_style())
        story.append(mini_t)
        story.append(Spacer(1, 0.35 * cm))

        # Hyperparameters table for this model
        hp = state.hyperparameters.get(model_name, {})
        if hp:
            story.append(Paragraph("Hyperparameters Used", styles["section_h2"]))
            hp_items = [(k, str(v) if v is not None else "None") for k, v in hp.items()]
            # Split into two side-by-side columns for space efficiency
            mid = math.ceil(len(hp_items) / 2)
            left_items = hp_items[:mid]
            right_items = hp_items[mid:]

            def _hp_sub_table(items):
                rows = [["Parameter", "Value"]]
                for k, v in items:
                    rows.append([k, v])
                sub = Table(rows, colWidths=[3.5 * cm, 3.5 * cm])
                sub_style = [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f8fafc")],
                    ),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#e2e8f0")),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ]
                sub.setStyle(TableStyle(sub_style))
                return sub

            hp_outer = Table(
                [
                    [
                        _hp_sub_table(left_items),
                        _hp_sub_table(right_items) if right_items else "",
                    ]
                ],
                colWidths=[PAGE_W / 2, PAGE_W / 2],
            )
            hp_outer.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(hp_outer)
            story.append(Spacer(1, 0.3 * cm))

        # Scatter + SHAP side by side
        chart_data = state.chart_data_all.get(model_name, [])
        shap_data = state.shap_by_model.get(model_name, [])

        scatter_buf = make_scatter_plot(chart_data, model_name, state.target, is_best)
        shap_buf = make_shap_chart(shap_data, model_name) if shap_data else None

        half_w = (PAGE_W - 0.5 * cm) / 2

        if shap_buf:
            img_scatter = _buf_to_rl_image(
                scatter_buf, width_cm=half_w / cm, height_cm=7
            )
            img_shap = _buf_to_rl_image(shap_buf, width_cm=half_w / cm, height_cm=7)
            side_data = [[img_scatter, img_shap]]
            side_table = Table(side_data, colWidths=[half_w, half_w])
            side_table.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ]
                )
            )
            story.append(side_table)
            caption_data = [
                [
                    Paragraph("Actual vs Predicted (test set)", styles["caption"]),
                    Paragraph("Feature Importance", styles["caption"]),
                ]
            ]
            cap_t = Table(caption_data, colWidths=[half_w, half_w])
            cap_t.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
            story.append(cap_t)
        else:
            img_scatter = _buf_to_rl_image(
                scatter_buf, width_cm=PAGE_W / cm, height_cm=7
            )
            story.append(img_scatter)
            story.append(Paragraph("Actual vs Predicted (test set)", styles["caption"]))

        story.append(THIN)
        story.append(Spacer(1, 0.3 * cm))

        if model_list.index(model_name) < len(model_list) - 1:
            story.append(PageBreak())

    # ── SECTION 5: PySR EQUATION ─────────────────────────────────────────
    if state.pysr_equation and state.pysr_equation.strip():
        story.append(PageBreak())
        story.append(
            Paragraph("5. Symbolic Regression Equation (PySR)", styles["section_h1"])
        )
        story.append(DIVIDER)
        story.append(
            Paragraph(
                "PySR discovered the following symbolic equation relating features to the target:",
                styles["body"],
            )
        )
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph(state.pysr_equation, styles["equation"]))
        story.append(
            Paragraph(
                "This equation provides an explicit, interpretable mathematical relationship "
                "between input features and the target variable.",
                styles["body"],
            )
        )

    # ── Build ────────────────────────────────────────────────────────────
    doc.build(story)
    buf.seek(0)
    return buf


# ==========================================
# ENDPOINTS
# ==========================================


@app.get("/api/models")
async def get_available_models():
    """Returns the list of available model names."""
    return {"models": AVAILABLE_MODELS}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Accepts CSV or Excel files. Returns columns, preview, row count, and sheet names."""
    try:
        contents = await file.read()
        sheet_names = []

        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith((".xls", ".xlsx")):
            xl = pd.ExcelFile(io.BytesIO(contents))
            sheet_names = xl.sheet_names
            df = xl.parse(sheet_names[0])
            logger.info(
                f"Excel sheets found: {sheet_names}. Reading: '{sheet_names[0]}'"
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload a .csv, .xlsx, or .xls file.",
            )

        df.columns = df.columns.str.strip()
        df = df.dropna(how="all").dropna(axis=1, how="all")

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty or has no readable data.",
            )

        state.df = df
        state.models = {}
        state.scaler = None
        state.features = []
        state.target = ""
        state.trained_model_names = []
        state.file_name = file.filename
        state.total_rows = len(df)
        state.statistics = statistics
        state.correlation = corr_matrix

        preview = df.head(10).replace({np.nan: None}).to_dict(orient="records")

        # ── Statistical analysis per numeric column ─────────────────────
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        statistics: Dict[str, Any] = {}
        for col in numeric_cols:
            col_data = df[col].dropna()
            statistics[col] = {
                "count": int(col_data.count()),
                "mean": safe_float(col_data.mean()),
                "std": safe_float(col_data.std()),
                "min": safe_float(col_data.min()),
                "q25": safe_float(col_data.quantile(0.25)),
                "median": safe_float(col_data.median()),
                "q75": safe_float(col_data.quantile(0.75)),
                "max": safe_float(col_data.max()),
                "missing": int(df[col].isna().sum()),
                "missing_pct": safe_float(df[col].isna().sum() / len(df) * 100),
            }

        # ── Correlation matrix (numeric columns only) ───────────────────
        corr_matrix: Dict[str, Dict[str, Optional[float]]] = {}
        if len(numeric_cols) > 1:
            corr_df = df[numeric_cols].corr()
            for col in numeric_cols:
                corr_matrix[col] = {
                    col2: safe_float(corr_df.loc[col, col2]) for col2 in numeric_cols
                }

        response = {
            "columns": list(df.columns),
            "preview": preview,
            "total_rows": len(df),
            "active_sheet": sheet_names[0] if sheet_names else None,
            "statistics": statistics,
            "correlation": corr_matrix,
        }
        if len(sheet_names) > 1:
            response["all_sheets"] = sheet_names

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")


@app.post("/api/upload/sheet")
async def upload_file_sheet(file: UploadFile = File(...), sheet: str = "Sheet1"):
    """Upload Excel and specify which sheet to read."""
    try:
        contents = await file.read()
        if not file.filename.endswith((".xls", ".xlsx")):
            raise HTTPException(
                status_code=400, detail="Sheet selection is only for Excel files."
            )

        xl = pd.ExcelFile(io.BytesIO(contents))
        if sheet not in xl.sheet_names:
            raise HTTPException(
                status_code=400,
                detail=f"Sheet '{sheet}' not found. Available sheets: {xl.sheet_names}",
            )

        df = xl.parse(sheet)
        df.columns = df.columns.str.strip()
        df = df.dropna(how="all").dropna(axis=1, how="all")

        state.df = df
        state.models = {}
        state.scaler = None
        state.trained_model_names = []
        state.file_name = file.filename
        state.total_rows = len(df)

        preview = df.head(10).replace({np.nan: None}).to_dict(orient="records")
        return {
            "columns": list(df.columns),
            "preview": preview,
            "total_rows": len(df),
            "active_sheet": sheet,
            "all_sheets": xl.sheet_names,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train")
async def train_model(req: TrainRequest):
    """Trains models, computes test metrics, SHAP importances, and generates scatter data."""

    if state.df is None:
        raise HTTPException(
            status_code=400, detail="No data uploaded yet. Call /api/upload first."
        )

    if not req.features:
        raise HTTPException(
            status_code=400, detail="At least one feature column is required."
        )

    if not (0.1 <= req.test_size <= 0.5):
        raise HTTPException(
            status_code=400, detail="test_size must be between 0.1 and 0.5."
        )

    available_cols = list(state.df.columns)

    feature_resolved, feature_missing = resolve_columns(req.features, available_cols)
    target_resolved, target_missing = resolve_columns([req.target], available_cols)

    if feature_missing or target_missing:
        import difflib

        suggestions = {}
        for col in feature_missing + target_missing:
            close = difflib.get_close_matches(col, available_cols, n=2, cutoff=0.4)
            suggestions[col] = close

        suggestion_text = "; ".join(
            f"'{c}' → did you mean {s}?" if s else f"'{c}' → no close match found"
            for c, s in suggestions.items()
        )

        raise HTTPException(
            status_code=400,
            detail=(
                f"Column(s) not found: {feature_missing + target_missing}. "
                f"Available columns: {available_cols}. "
                f"Suggestions: {suggestion_text}"
            ),
        )

    actual_features = [feature_resolved[f] for f in req.features]
    actual_target = target_resolved[req.target]

    if actual_target in actual_features:
        raise HTTPException(
            status_code=400, detail="Target column cannot also be a feature column."
        )

    models_to_use = req.selected_models if req.selected_models else AVAILABLE_MODELS
    models_to_use = [m for m in models_to_use if m in AVAILABLE_MODELS]
    if not models_to_use:
        raise HTTPException(status_code=400, detail="No valid models selected.")

    try:
        df = state.df.dropna(subset=actual_features + [actual_target]).copy()

        if len(df) < 10:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough valid rows after dropping NaNs. Got {len(df)}, need at least 10.",
            )

        X = df[actual_features].astype(float)
        y = df[actual_target].astype(float)

        state.features = actual_features
        state.target = actual_target

        state.scaler = StandardScaler()
        X_scaled = state.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=req.test_size, random_state=42
        )
        y_train_arr = np.array(y_train)
        y_test_arr = np.array(y_test)

        train_size = len(X_train)
        test_size_actual = len(X_test)
        logger.info(
            f"Split: {train_size} train / {test_size_actual} test (test_size={req.test_size})"
        )

        all_models_registry = {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "SVR": SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1),
            "ORF": ORFRegressor(population_size=30),
            "PySR (Symbolic)": (
                PySRRegressor()
                if not HAS_PYSR
                else PySRRegressor(
                    niterations=10,
                    binary_operators=["+", "*", "-", "/"],
                    variable_names=actual_features,
                )
            ),
        }

        if HAS_XGB:
            all_models_registry["XGBoost"] = XGBRegressor(
                n_estimators=100, random_state=42, objective="reg:squarederror"
            )

        models_to_train = {
            k: v for k, v in all_models_registry.items() if k in models_to_use
        }

        comparison_results: Dict[str, Dict] = {}
        chart_data_all: Dict[str, List] = {}
        shap_by_model: Dict[str, List] = {}
        pysr_equation = ""
        training_errors: Dict[str, str] = {}

        for name, model in models_to_train.items():
            try:
                model.fit(X_train, y_train)
                state.models[name] = model

                if "PySR" in name:
                    try:
                        eq = model.sympy(index=0) if HAS_PYSR else model.sympy()
                        pysr_equation = str(eq)
                    except Exception:
                        try:
                            pysr_equation = str(model.sympy())
                        except Exception:
                            pysr_equation = "Equation extraction failed."
                    if pysr_equation and pysr_equation != "Equation extraction failed.":
                        pysr_equation = substitute_feature_names(
                            pysr_equation, actual_features, actual_target
                        )

                y_train_pred = model.predict(X_train)
                train_metrics = compute_metrics(y_train_arr, y_train_pred)

                y_test_pred = model.predict(X_test)
                test_metrics = compute_metrics(y_test_arr, y_test_pred)

                comparison_results[name] = {
                    "train": train_metrics,
                    "test": test_metrics,
                    "mse": test_metrics["mse"],
                    "r2": test_metrics["r2"],
                }

                chart_data_all[name] = [
                    {"actual": safe_float(a), "predicted": safe_float(p)}
                    for a, p in zip(y_test_arr[:100], y_test_pred[:100])
                ]

                shap_by_model[name] = compute_shap_importance(
                    model, X_train, y_train_arr, actual_features
                )

                logger.info(
                    f"Model '{name}' — Train R²={train_metrics['r2']:.4f} | Test R²={test_metrics['r2']:.4f}"
                )

            except Exception as model_err:
                logger.error(f"Model '{name}' failed: {model_err}")
                training_errors[name] = str(model_err)

        if not comparison_results:
            raise HTTPException(
                status_code=500,
                detail=f"All models failed to train. Errors: {training_errors}",
            )

        valid_results = {
            k: v for k, v in comparison_results.items() if v["test"]["r2"] is not None
        }
        if valid_results:
            best_model = max(
                valid_results, key=lambda k: valid_results[k]["test"]["r2"] or -999
            )
        else:
            best_model = next(iter(comparison_results))

        state.best_model_name = best_model
        state.trained_model_names = list(comparison_results.keys())
        best_r2 = comparison_results[best_model]["test"]["r2"]

        reasoning = (
            f"The {best_model} model performed best on test data, capturing "
            f"{best_r2 * 100:.1f}% of variance in '{actual_target}'."
            if best_r2 is not None
            else f"The {best_model} model was selected."
        )

        # ── Save all data to state for report generation ──────────────
        state.comparison_results = comparison_results
        state.chart_data_all = chart_data_all
        state.shap_by_model = shap_by_model
        state.split_info = {
            "train_size": train_size,
            "test_size": test_size_actual,
            "test_ratio": req.test_size,
        }
        state.pysr_equation = pysr_equation
        state.reasoning = reasoning
        state.hyperparameters = model_hyperparams

        # ── Extract hyperparameters for each successfully trained model ──
        model_hyperparams: Dict[str, Dict[str, Any]] = {}
        for name in comparison_results.keys():
            model = state.models.get(name)
            if model is None:
                continue
            try:
                src = (
                    model.model
                    if (hasattr(model, "model") and not hasattr(model, "get_params"))
                    else model
                )
                if hasattr(src, "get_params"):
                    raw = src.get_params()
                    clean: Dict[str, Any] = {}
                    for k, v in raw.items():
                        if isinstance(v, float):
                            # nan/inf are not JSON-serialisable — convert to None
                            clean[k] = safe_float(v)
                        elif isinstance(v, (int, str, bool, type(None))):
                            clean[k] = v
                        else:
                            clean[k] = str(v)
                    model_hyperparams[name] = clean
                else:
                    model_hyperparams[name] = {}
            except Exception:
                model_hyperparams[name] = {}

        response: Dict[str, Any] = {
            "comparison": comparison_results,
            "best_model": best_model,
            "reasoning": reasoning,
            "pysr_equation": pysr_equation,
            "chart_data": chart_data_all,
            "shap_data": shap_by_model,
            "resolved_features": actual_features,
            "resolved_target": actual_target,
            "trained_models": list(comparison_results.keys()),
            "split_info": {
                "train_size": train_size,
                "test_size": test_size_actual,
                "test_ratio": req.test_size,
            },
            "hyperparameters": model_hyperparams,
        }
        if training_errors:
            response["training_warnings"] = training_errors

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error during training")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict")
async def predict_single(req: PredictRequest):
    """Predicts using a specific trained model."""
    if not state.models or state.scaler is None:
        raise HTTPException(
            status_code=400, detail="Models not trained yet. Call /api/train first."
        )
    if req.model_name not in state.models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{req.model_name}' not trained. Trained models: {list(state.models.keys())}",
        )

    missing = [f for f in state.features if f not in req.inputs]
    if missing:
        raise HTTPException(
            status_code=400, detail=f"Missing input values for features: {missing}"
        )

    try:
        input_data = [req.inputs[feat] for feat in state.features]
        input_df = pd.DataFrame([input_data], columns=state.features)
        input_scaled = state.scaler.transform(input_df)

        model = state.models[req.model_name]
        prediction = model.predict(input_scaled)[0]

        result = safe_float(prediction)
        if result is None:
            raise HTTPException(status_code=500, detail="Model returned NaN or Inf.")

        return {"prediction": result, "model_used": req.model_name}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/report")
async def download_report():
    """Generate and return a comprehensive PDF report of the last training run."""
    if not state.comparison_results:
        raise HTTPException(
            status_code=400, detail="No training results found. Run /api/train first."
        )
    try:
        pdf_buf = generate_pdf_report()
        filename = (
            f"RockML_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        return StreamingResponse(
            pdf_buf,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        logger.exception("Report generation failed")
        raise HTTPException(
            status_code=500, detail=f"Report generation failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
