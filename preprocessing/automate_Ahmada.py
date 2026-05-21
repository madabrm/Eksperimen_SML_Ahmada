"""
automate_Ahmada.py
==================
Automated preprocessing pipeline untuk dataset Customer Churn.

Fungsi utama:
    - preprocess(): Melakukan seluruh tahapan preprocessing dan mengembalikan
      data yang siap dilatih (X_train, X_val, y_train, y_val, X_test, y_test)
    - save_preprocessed_data(): Menyimpan hasil preprocessing ke folder output

Tahapan preprocessing (sama dengan Eksperimen_Ahmada.ipynb, struktur berbeda):
    1. Drop kolom yang tidak diperlukan
    2. Deteksi outlier (IQR method)
    3. Encoding fitur kategorikal (Label + One-Hot)
    4. Handling missing values
    5. Feature scaling (StandardScaler)
    6. Train-test split (stratified)
"""

import os
import sys
import json
import pickle
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# ============================================================
# Logging Configuration
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# Konstanta
# ============================================================
COLUMNS_TO_DROP = [
    "customer_id",
    "split",
    "has_international_plan",
    "has_voice_mail_plan",
    "total_minutes",
    "total_calls",
    "total_charges",
    "avg_charge_per_minute",
    "support_call_rate",
]

BINARY_COLS = ["international_plan", "voice_mail_plan"]

MULTI_CAT_COLS = [
    "state",
    "usage_intensity",
    "customer_value_segment",
    "rule_based_churn_risk_level",
    "area_code",
]

SCALE_COLS = [
    "account_length",
    "area_code",
    "number_vmail_messages",
    "total_day_minutes",
    "total_day_calls",
    "total_day_charge",
    "total_eve_minutes",
    "total_eve_calls",
    "total_eve_charge",
    "total_night_minutes",
    "total_night_calls",
    "total_night_charge",
    "total_intl_minutes",
    "total_intl_calls",
    "total_intl_charge",
    "customer_service_calls",
    "high_service_calls",
    "international_plan",
    "voice_mail_plan",
    "rule_based_churn_risk_score",
]

TARGET_COL = "churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2


# ============================================================
# Custom Transformer (sklearn-compatible)
# ============================================================
class ChurnPreprocessor(BaseEstimator, TransformerMixin):
    """
    Transformer yang menggabungkan seluruh tahapan preprocessing
    agar bisa di-fit dan di-transform secara konsisten pada data
    train dan test.
    """

    def __init__(
        self,
        cols_to_drop: list = None,
        binary_cols: list = None,
        multi_cat_cols: list = None,
        scale_cols: list = None,
    ):
        self.cols_to_drop = cols_to_drop or COLUMNS_TO_DROP
        self.binary_cols = binary_cols or BINARY_COLS
        self.multi_cat_cols = multi_cat_cols or MULTI_CAT_COLS
        self.scale_cols = scale_cols or SCALE_COLS

        # Atribut yang di-fit
        self.scaler_ = None
        self.feature_names_out_ = None
        self.outlier_report_ = None

    def fit(self, X: pd.DataFrame, y=None):
        """Fit scaler pada data training."""
        df = self._drop_columns(X)
        df = self._encode_binary(df)
        df = self._encode_categorical(df)
        df = self._handle_missing(df)

        # Fit scaler
        actual_scale_cols = [c for c in self.scale_cols if c in df.columns]
        self.scaler_ = StandardScaler()
        self.scaler_.fit(df[actual_scale_cols])

        self.feature_names_out_ = df.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data menggunakan parameter yang sudah di-fit."""
        df = self._drop_columns(X)
        df = self._encode_binary(df)
        df = self._encode_categorical(df)
        df = self._handle_missing(df)

        # Transform dengan scaler yang sudah di-fit
        actual_scale_cols = [c for c in self.scale_cols if c in df.columns]
        df[actual_scale_cols] = self.scaler_.transform(df[actual_scale_cols])

        # Pastikan kolom sama dengan saat fit
        df = df.reindex(columns=self.feature_names_out_, fill_value=0)
        return df

    # --- Internal methods ---

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.cols_to_drop if c in df.columns]
        if cols:
            logger.info(f"Drop {len(cols)} kolom: {cols}")
        return df.drop(columns=cols, errors="ignore")

    def _encode_binary(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.binary_cols:
            if col in df.columns:
                df[col] = df[col].map({"Yes": 1, "No": 0})
        return df

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        if "area_code" in df.columns:
            df["area_code"] = df["area_code"].astype(str)
        df = pd.get_dummies(df, columns=self.multi_cat_cols, drop_first=False, dtype=int)
        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in ["int64", "float64"]:
                    df[col] = df[col].fillna(df[col].median())
                    logger.info(f"Imputasi {col} dengan median")
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
                    logger.info(f"Imputasi {col} dengan modus")
        return df

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deteksi outlier menggunakan IQR method (hanya report, tidak hapus)."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        report = []
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            if n_outliers > 0:
                report.append({
                    "feature": col,
                    "outliers": int(n_outliers),
                    "percentage": round(n_outliers / len(df) * 100, 2),
                })
        self.outlier_report_ = pd.DataFrame(report)
        return self.outlier_report_


# ============================================================
# Fungsi Utama: preprocess()
# ============================================================
def preprocess(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, Optional[pd.Series], pd.DataFrame]:
    """
    Melakukan seluruh tahapan preprocessing dan mengembalikan data yang siap dilatih.

    Parameters
    ----------
    train_df : pd.DataFrame
        DataFrame training mentah (belum dipreprocessing).
    test_df : pd.DataFrame
        DataFrame test mentah (belum dipreprocessing).
    test_size : float
        Proporsi data untuk validation set.
    random_state : int
        Seed untuk reproducibility.

    Returns
    -------
    X_train : pd.DataFrame
        Fitur training yang sudah dipreprocessing.
    X_val : pd.DataFrame
        Fitur validation yang sudah dipreprocessing.
    y_train : pd.Series
        Target training.
    y_val : pd.Series
        Target validation.
    X_test : pd.DataFrame
        Fitur test yang sudah dipreprocessing.
    y_test : pd.Series atau None
        Target test (jika ada).
    preprocessor : ChurnPreprocessor
        Objek preprocessor yang sudah di-fit (bisa disimpan untuk inference).
    """
    logger.info("=" * 60)
    logger.info("MULAI PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    # --- Step 0: Pisahkan fitur dan target ---
    logger.info("Step 0: Memisahkan fitur dan target")
    X_train_raw = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    X_test_raw = test_df.drop(columns=[TARGET_COL]) if TARGET_COL in test_df.columns else test_df.copy()
    y_test = test_df[TARGET_COL] if TARGET_COL in test_df.columns else None

    # --- Step 1: Deteksi outlier ---
    logger.info("Step 1: Deteksi outlier (IQR method)")
    preprocessor = ChurnPreprocessor()
    outlier_report = preprocessor.detect_outliers(X_train_raw)
    if outlier_report is not None and not outlier_report.empty:
        logger.info(f"Outlier terdeteksi pada {len(outlier_report)} fitur:")
        for _, row in outlier_report.iterrows():
            logger.info(f"  - {row['feature']}: {row['outliers']} outlier ({row['percentage']}%)")
    else:
        logger.info("Tidak ada outlier terdeteksi")

    # --- Step 2: Fit preprocessor pada train data ---
    logger.info("Step 2: Fit preprocessor pada training data")
    preprocessor.fit(X_train_raw)
    logger.info(f"Jumlah fitur setelah preprocessing: {len(preprocessor.feature_names_out_)}")

    # --- Step 3: Transform train data ---
    logger.info("Step 3: Transform training data")
    X_train_processed = preprocessor.transform(X_train_raw)

    # --- Step 4: Transform test data ---
    logger.info("Step 4: Transform test data")
    X_test_processed = preprocessor.transform(X_test_raw)

    # --- Step 5: Train-validation split ---
    logger.info(f"Step 5: Split data (test_size={test_size}, stratified)")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_processed,
        y_train,
        test_size=test_size,
        random_state=random_state,
        stratify=y_train,
    )
    logger.info(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.info(f"  X_val:   {X_val.shape}, y_val: {y_val.shape}")
    logger.info(f"  X_test:  {X_test_processed.shape}")

    logger.info("=" * 60)
    logger.info("PREPROCESSING SELESAI")
    logger.info("=" * 60)

    return X_train, X_val, y_train, y_val, X_test_processed, y_test, preprocessor


# ============================================================
# Fungsi: load_raw_data()
# ============================================================
def load_raw_data(raw_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Memuat data mentah dari folder raw.

    Parameters
    ----------
    raw_dir : str
        Path ke folder yang berisi file CSV mentah.

    Returns
    -------
    train_df : pd.DataFrame
    test_df : pd.DataFrame
    """
    logger.info(f"Memuat data mentah dari: {raw_dir}")

    train_df = pd.read_csv(os.path.join(raw_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(raw_dir, "test.csv"))

    logger.info(f"  train.csv: {train_df.shape}")
    logger.info(f"  test.csv:  {test_df.shape}")

    return train_df, test_df


# ============================================================
# Fungsi: save_preprocessed_data()
# ============================================================
def save_preprocessed_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: Optional[pd.Series],
    preprocessor: ChurnPreprocessor,
    output_dir: str,
) -> Dict[str, str]:
    """
    Menyimpan hasil preprocessing ke folder output dalam format CSV dan pickle.

    Parameters
    ----------
    output_dir : str
        Path ke folder output.

    Returns
    -------
    dict
        Mapping nama file ke path absolut.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = {}

    # Simpan sebagai CSV
    for name, df in [
        ("X_train", X_train),
        ("X_val", X_val),
        ("y_train", y_train),
        ("y_val", y_val),
        ("X_test", X_test),
    ]:
        path = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        saved_files[f"{name}.csv"] = path
        logger.info(f"Saved: {path}")

    if y_test is not None:
        path = os.path.join(output_dir, "y_test.csv")
        y_test.to_csv(path, index=False)
        saved_files["y_test.csv"] = path
        logger.info(f"Saved: {path}")

    # Simpan sebagai pickle (untuk modelling)
    train_data = {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "feature_names": X_train.columns.tolist(),
        "target_name": TARGET_COL,
    }
    train_pkl = os.path.join(output_dir, "train_data.pkl")
    with open(train_pkl, "wb") as f:
        pickle.dump(train_data, f)
    saved_files["train_data.pkl"] = train_pkl
    logger.info(f"Saved: {train_pkl}")

    test_data = {"X_test": X_test, "y_test": y_test}
    test_pkl = os.path.join(output_dir, "test_data.pkl")
    with open(test_pkl, "wb") as f:
        pickle.dump(test_data, f)
    saved_files["test_data.pkl"] = test_pkl
    logger.info(f"Saved: {test_pkl}")

    # Simpan preprocessor (scaler + metadata)
    meta_pkl = os.path.join(output_dir, "preprocessing_metadata.pkl")
    with open(meta_pkl, "wb") as f:
        pickle.dump(preprocessor, f)
    saved_files["preprocessing_metadata.pkl"] = meta_pkl
    logger.info(f"Saved: {meta_pkl}")

    return saved_files


# ============================================================
# Fungsi: run_pipeline() — Entry point utama
# ============================================================
def run_pipeline(
    raw_dir: str = None,
    output_dir: str = None,
) -> Dict[str, Any]:
    """
    Menjalankan seluruh pipeline preprocessing dari awal hingga penyimpanan.

    Parameters
    ----------
    raw_dir : str, optional
        Path ke folder raw data. Default: ../customer_churn_raw relatif terhadap script.
    output_dir : str, optional
        Path ke folder output. Default: ./customer_churn_preprocessing relatif terhadap script.

    Returns
    -------
    dict
        Informasi hasil pipeline (shapes, saved files, dll).
    """
    # Resolve path relatif terhadap lokasi file ini
    script_dir = Path(__file__).resolve().parent

    if raw_dir is None:
        raw_dir = str(script_dir.parent / "customer_churn_raw")
    if output_dir is None:
        output_dir = str(script_dir / "customer_churn_preprocessing")

    logger.info(f"Raw data directory:  {raw_dir}")
    logger.info(f"Output directory:    {output_dir}")

    # Load
    train_df, test_df = load_raw_data(raw_dir)

    # Preprocess
    X_train, X_val, y_train, y_val, X_test, y_test, preprocessor = preprocess(
        train_df, test_df
    )

    # Save
    saved_files = save_preprocessed_data(
        X_train, X_val, y_train, y_val, X_test, y_test, preprocessor, output_dir
    )

    # Summary
    result = {
        "train_shape": X_train.shape,
        "val_shape": X_val.shape,
        "test_shape": X_test.shape,
        "n_features": X_train.shape[1],
        "saved_files": saved_files,
    }

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Train: {result['train_shape']}")
    logger.info(f"Val:   {result['val_shape']}")
    logger.info(f"Test:  {result['test_shape']}")
    logger.info(f"Features: {result['n_features']}")
    logger.info(f"Files saved: {len(result['saved_files'])}")
    for fname, fpath in result["saved_files"].items():
        size_kb = os.path.getsize(fpath) / 1024
        logger.info(f"  - {fname} ({size_kb:.1f} KB)")

    return result


# ============================================================
# CLI Entry Point
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Automated preprocessing pipeline untuk Customer Churn")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Path ke folder raw data (default: ../customer_churn_raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path ke folder output (default: ./customer_churn_preprocessing)",
    )
    args = parser.parse_args()

    result = run_pipeline(raw_dir=args.raw_dir, output_dir=args.output_dir)

    print("\n✅ Preprocessing pipeline selesai!")
    print(f"   Train: {result['train_shape']}")
    print(f"   Val:   {result['val_shape']}")
    print(f"   Test:  {result['test_shape']}")
    print(f"   Features: {result['n_features']}")
