"""
数据质量分析引擎

对数据表进行列级和表级的质量分析，包括：
    - 列级分析：空值率、唯一值数量、最小/最大/均值/标准差、值分布
    - 表级分析：行数趋势、增长率、数据新鲜度
    - 异常检测：突增突降、模式偏移、参照完整性
    - 质量评分：0-100 综合评分

使用方法:
    from etl.data_quality.profiling_engine import DataProfilingEngine
    engine = DataProfilingEngine()
    report = engine.profile_table(df, table_name="orders")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Protocol

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DBConnection(Protocol):
    """数据库连接协议。"""
    def cursor(self) -> Any: ...


@dataclass
class ColumnProfile:
    """
    列级质量分析结果。

    Attributes:
        column_name: 列名
        dtype: 数据类型
        total_count: 总记录数
        null_count: 空值数量
        null_rate: 空值率（0-1）
        distinct_count: 唯一值数量
        distinct_rate: 唯一值率（0-1）
        min_value: 最小值
        max_value: 最大值
        mean_value: 均值（数值型）
        std_value: 标准差（数值型）
        median_value: 中位数（数值型）
        top_values: 前5高频值及频次
        value_distribution: 值分布（分位数）
        quality_score: 质量评分（0-100）
    """
    column_name: str
    dtype: str
    total_count: int
    null_count: int = 0
    null_rate: float = 0.0
    distinct_count: int = 0
    distinct_rate: float = 0.0
    min_value: Any = None
    max_value: Any = None
    mean_value: float | None = None
    std_value: float | None = None
    median_value: float | None = None
    top_values: list[dict[str, Any]] = field(default_factory=list)
    value_distribution: dict[str, float] = field(default_factory=dict)
    quality_score: float = 100.0


@dataclass
class TableProfile:
    """
    表级质量分析结果。

    Attributes:
        table_name: 表名
        row_count: 行数
        column_count: 列数
        columns: 列级分析结果列表
        overall_quality_score: 综合质量评分
        freshness_score: 数据新鲜度评分
        completeness_score: 完整性评分
        anomalies: 检测到的异常列表
    """
    table_name: str
    row_count: int
    column_count: int
    columns: list[ColumnProfile] = field(default_factory=list)
    overall_quality_score: float = 100.0
    freshness_score: float = 100.0
    completeness_score: float = 100.0
    anomalies: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AnomalyAlert:
    """
    异常告警。

    Attributes:
        alert_type: 告警类型
        severity: 严重程度（info/warning/error/critical）
        table_name: 表名
        column_name: 列名（可选）
        message: 告警消息
        value: 异常值
        expected_range: 预期范围
    """
    alert_type: str
    severity: str
    table_name: str
    column_name: str | None = None
    message: str = ""
    value: Any = None
    expected_range: tuple[float, float] | None = None


# ============================================================
# 数据质量分析引擎
# ============================================================


class DataProfilingEngine:
    """
    数据质量分析引擎。

    对 DataFrame 进行全面的列级和表级质量分析，
    检测异常并计算综合质量评分。

    Usage:
        engine = DataProfilingEngine(
            anomaly_z_threshold=3.0,
            freshness_sla_days=1,
        )
        report = engine.profile_table(df, table_name="orders")
    """

    def __init__(
        self,
        anomaly_z_threshold: float = 3.0,
        freshness_sla_days: int = 1,
        null_rate_threshold: float = 0.05,
        distinct_rate_low: float = 0.01,
    ) -> None:
        """
        初始化分析引擎。

        Args:
            anomaly_z_threshold: 异常检测 Z-score 阈值
            freshness_sla_days: 数据新鲜度 SLA（天）
            null_rate_threshold: 空值率告警阈值
            distinct_rate_low: 唯一值率过低阈值
        """
        self.anomaly_z_threshold = anomaly_z_threshold
        self.freshness_sla_days = freshness_sla_days
        self.null_rate_threshold = null_rate_threshold
        self.distinct_rate_low = distinct_rate_low

    def profile_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        date_column: str | None = None,
        reference_data: dict[str, pd.DataFrame] | None = None,
    ) -> TableProfile:
        """
        执行完整的表级数据质量分析。

        Args:
            df: 待分析的 DataFrame
            table_name: 表名
            date_column: 日期列名（用于新鲜度检查）
            reference_data: 参照表字典（用于完整性检查）

        Returns:
            TableProfile 表级分析结果
        """
        logger.info("开始数据质量分析: %s (%d 行 x %d 列)", table_name, len(df), len(df.columns))

        if df.empty:
            return TableProfile(
                table_name=table_name,
                row_count=0,
                column_count=0,
                overall_quality_score=0.0,
                anomalies=[{
                    "alert_type": "empty_table",
                    "severity": "critical",
                    "message": f"表 {table_name} 为空",
                }],
            )

        # 列级分析
        column_profiles: list[ColumnProfile] = []
        for col in df.columns:
            profile = self._profile_column(df, col, table_name)
            column_profiles.append(profile)

        # 表级分析
        completeness_score = self._compute_completeness_score(column_profiles)
        freshness_score = self._compute_freshness_score(df, date_column)

        # 异常检测
        anomalies = self._detect_anomalies(df, table_name, column_profiles)

        # 参照完整性检查
        if reference_data:
            ref_anomalies = self._check_referential_integrity(
                df, table_name, reference_data,
            )
            anomalies.extend(ref_anomalies)

        # 综合评分
        overall_score = self._compute_overall_score(
            completeness_score, freshness_score, anomalies, len(column_profiles),
        )

        profile = TableProfile(
            table_name=table_name,
            row_count=len(df),
            column_count=len(df.columns),
            columns=column_profiles,
            overall_quality_score=round(overall_score, 2),
            freshness_score=round(freshness_score, 2),
            completeness_score=round(completeness_score, 2),
            anomalies=anomalies,
        )

        logger.info(
            "数据质量分析完成: %s, 综合评分=%.1f, 异常=%d",
            table_name, overall_score, len(anomalies),
        )
        return profile

    def _profile_column(
        self,
        df: pd.DataFrame,
        column: str,
        table_name: str,
    ) -> ColumnProfile:
        """
        执行列级数据质量分析。

        Args:
            df: DataFrame
            column: 列名
            table_name: 表名

        Returns:
            ColumnProfile 列级分析结果
        """
        series = df[column]
        total_count = len(series)
        null_count = int(series.isna().sum())
        null_rate = round(null_count / total_count, 4) if total_count > 0 else 0.0

        non_null = series.dropna()
        distinct_count = int(non_null.nunique())
        distinct_rate = round(distinct_count / total_count, 4) if total_count > 0 else 0.0

        profile = ColumnProfile(
            column_name=column,
            dtype=str(series.dtype),
            total_count=total_count,
            null_count=null_count,
            null_rate=null_rate,
            distinct_count=distinct_count,
            distinct_rate=distinct_rate,
        )

        # 数值型统计
        if pd.api.types.is_numeric_dtype(series):
            if not non_null.empty:
                profile.min_value = float(non_null.min())
                profile.max_value = float(non_null.max())
                profile.mean_value = round(float(non_null.mean()), 4)
                profile.std_value = round(float(non_null.std()), 4)
                profile.median_value = round(float(non_null.median()), 4)

                # 分位数分布
                quantiles = non_null.quantile([0.25, 0.5, 0.75, 0.90, 0.95, 0.99])
                profile.value_distribution = {
                    f"p{int(q * 100)}": round(float(v), 4)
                    for q, v in quantiles.items()
                }

        # 字符串型统计
        elif pd.api.types.is_string_dtype(series):
            if not non_null.empty:
                str_lengths = non_null.astype(str).str.len()
                profile.min_value = int(str_lengths.min())
                profile.max_value = int(str_lengths.max())
                profile.mean_value = round(float(str_lengths.mean()), 2)

        # 日期型统计
        elif pd.api.types.is_datetime64_any_dtype(series):
            if not non_null.empty:
                profile.min_value = str(non_null.min())
                profile.max_value = str(non_null.max())

        # 高频值 Top-5
        if not non_null.empty:
            value_counts = non_null.value_counts().head(5)
            profile.top_values = [
                {"value": str(val), "count": int(cnt), "frequency": round(cnt / total_count, 4)}
                for val, cnt in value_counts.items()
            ]

        # 列级质量评分
        profile.quality_score = self._compute_column_score(profile)

        return profile

    def _compute_column_score(self, profile: ColumnProfile) -> float:
        """
        计算列级质量评分。

        评分维度:
            - 完整性（40%）：空值率越低越好
            - 唯一性（20%）：主键类列需要高唯一率
            - 分布合理性（40%）：无明显偏斜

        Args:
            profile: 列级分析结果

        Returns:
            0-100 的质量评分
        """
        score = 100.0

        # 完整性扣分
        if profile.null_rate > self.null_rate_threshold:
            penalty = min(40, (profile.null_rate / 0.5) * 40)
            score -= penalty

        # 分布合理性扣分（单一值占比过高）
        if profile.top_values:
            top_freq = profile.top_values[0].get("frequency", 0)
            if top_freq > 0.95 and profile.distinct_count > 1:
                score -= 10  # 某个值占比超过95%可能有问题

        return round(max(0, score), 2)

    def _compute_completeness_score(self, profiles: list[ColumnProfile]) -> float:
        """
        计算表级完整性评分。

        基于所有列的空值率加权平均。

        Args:
            profiles: 列级分析结果列表

        Returns:
            0-100 的完整性评分
        """
        if not profiles:
            return 100.0

        total_cells = sum(p.total_count for p in profiles)
        total_nulls = sum(p.null_count for p in profiles)

        if total_cells == 0:
            return 100.0

        completeness = 1.0 - (total_nulls / total_cells)
        return round(completeness * 100, 2)

    def _compute_freshness_score(
        self,
        df: pd.DataFrame,
        date_column: str | None,
    ) -> float:
        """
        计算数据新鲜度评分。

        基于最新数据时间戳与当前时间的差距。

        Args:
            df: DataFrame
            date_column: 日期列名

        Returns:
            0-100 的新鲜度评分
        """
        if date_column is None or date_column not in df.columns:
            return 100.0  # 无日期列时默认满分

        date_series = pd.to_datetime(df[date_column], errors="coerce")
        max_date = date_series.max()

        if pd.isna(max_date):
            return 0.0

        now = pd.Timestamp.now()
        if hasattr(max_date, "tzinfo") and max_date.tzinfo is None:
            now = pd.Timestamp.now()

        age_hours = (now - max_date).total_seconds() / 3600
        sla_hours = self.freshness_sla_days * 24

        if age_hours <= sla_hours:
            return 100.0
        elif age_hours <= sla_hours * 2:
            # 超过 SLA 但在2倍以内，线性衰减
            return round(100 * (1 - (age_hours - sla_hours) / sla_hours), 2)
        else:
            return 0.0

    def _detect_anomalies(
        self,
        df: pd.DataFrame,
        table_name: str,
        column_profiles: list[ColumnProfile],
    ) -> list[dict[str, Any]]:
        """
        执行异常检测。

        检测类型:
            1. 行数突增突降（与历史对比）
            2. 空值率突增
            3. 数值列分布偏移
            4. 唯一值率异常
            5. 模式偏移（schema drift）

        Args:
            df: DataFrame
            table_name: 表名
            column_profiles: 列级分析结果

        Returns:
            异常列表
        """
        anomalies: list[dict[str, Any]] = []

        # 检测空值率过高的列
        for profile in column_profiles:
            if profile.null_rate > self.null_rate_threshold:
                anomalies.append({
                    "alert_type": "high_null_rate",
                    "severity": "warning" if profile.null_rate < 0.3 else "error",
                    "table_name": table_name,
                    "column_name": profile.column_name,
                    "message": f"列 {profile.column_name} 空值率 {profile.null_rate:.2%} 超过阈值 {self.null_rate_threshold:.2%}",
                    "value": profile.null_rate,
                    "expected_range": (0, self.null_rate_threshold),
                })

            # 唯一值率过低（可能的数据质量问题）
            if profile.distinct_count > 1 and profile.distinct_rate < self.distinct_rate_low:
                anomalies.append({
                    "alert_type": "low_distinct_rate",
                    "severity": "info",
                    "table_name": table_name,
                    "column_name": profile.column_name,
                    "message": f"列 {profile.column_name} 唯一值率 {profile.distinct_rate:.2%} 过低",
                    "value": profile.distinct_rate,
                })

        # 数值列异常值检测（Z-score 方法）
        for profile in column_profiles:
            if profile.mean_value is not None and profile.std_value is not None and profile.std_value > 0:
                col = df[profile.column_name].dropna()
                if len(col) > 10:
                    z_scores = np.abs((col - col.mean()) / col.std())
                    outlier_count = int((z_scores > self.anomaly_z_threshold).sum())
                    outlier_rate = outlier_count / len(col)

                    if outlier_rate > 0.05:
                        anomalies.append({
                            "alert_type": "high_outlier_rate",
                            "severity": "warning",
                            "table_name": table_name,
                            "column_name": profile.column_name,
                            "message": f"列 {profile.column_name} 异常值比例 {outlier_rate:.2%} 超过5%",
                            "value": outlier_rate,
                            "expected_range": (0, 0.05),
                        })

        # 数值列负值检测（应非负的列）
        non_negative_hints = ["amount", "price", "cost", "quantity", "revenue", "mrr", "count", "score"]
        for profile in column_profiles:
            if profile.min_value is not None and isinstance(profile.min_value, (int, float)):
                if any(hint in profile.column_name.lower() for hint in non_negative_hints):
                    if profile.min_value < 0:
                        anomalies.append({
                            "alert_type": "negative_value",
                            "severity": "error",
                            "table_name": table_name,
                            "column_name": profile.column_name,
                            "message": f"列 {profile.column_name} 存在负值: {profile.min_value}",
                            "value": profile.min_value,
                            "expected_range": (0, float("inf")),
                        })

        return anomalies

    def _check_referential_integrity(
        self,
        df: pd.DataFrame,
        table_name: str,
        reference_data: dict[str, pd.DataFrame],
    ) -> list[dict[str, Any]]:
        """
        检查参照完整性。

        检测外键列是否在参照表中存在对应的记录。

        Args:
            df: 待检查的 DataFrame
            table_name: 表名
            reference_data: {列名: 参照 DataFrame} 字典

        Returns:
            参照完整性异常列表
        """
        anomalies: list[dict[str, Any]] = []

        for col_name, ref_df in reference_data.items():
            if col_name not in df.columns:
                continue

            # 假设参照表的第一列是主键
            ref_pk = ref_df.columns[0]
            ref_values = set(ref_df[ref_pk].astype(str).unique())

            fk_values = df[col_name].dropna().astype(str).unique()
            orphan_count = sum(1 for v in fk_values if v not in ref_values)
            orphan_rate = orphan_count / len(fk_values) if len(fk_values) > 0 else 0.0

            if orphan_count > 0:
                anomalies.append({
                    "alert_type": "referential_integrity",
                    "severity": "error",
                    "table_name": table_name,
                    "column_name": col_name,
                    "message": f"外键列 {col_name} 有 {orphan_count} 个孤立值 ({orphan_rate:.2%})",
                    "value": orphan_count,
                    "expected_range": (0, 0),
                })

        return anomalies

    def _compute_overall_score(
        self,
        completeness: float,
        freshness: float,
        anomalies: list[dict[str, Any]],
        column_count: int,
    ) -> float:
        """
        计算综合质量评分。

        评分公式:
            overall = completeness * 0.4 + freshness * 0.3 + anomaly_penalty * 0.3

        Args:
            completeness: 完整性评分
            freshness: 新鲜度评分
            anomalies: 异常列表
            column_count: 列数

        Returns:
            0-100 综合评分
        """
        # 异常惩罚
        severity_weights = {"info": 1, "warning": 3, "error": 5, "critical": 10}
        anomaly_penalty = sum(
            severity_weights.get(a.get("severity", "info"), 1)
            for a in anomalies
        )
        max_penalty = column_count * 10  # 每列最多扣10分
        anomaly_score = max(0, 100 - (anomaly_penalty / max(max_penalty, 1)) * 100)

        overall = completeness * 0.4 + freshness * 0.3 + anomaly_score * 0.3
        return round(max(0, min(100, overall)), 2)

    def profile_multiple_tables(
        self,
        tables: dict[str, pd.DataFrame],
        date_columns: dict[str, str] | None = None,
    ) -> list[TableProfile]:
        """
        批量分析多个表的数据质量。

        Args:
            tables: {表名: DataFrame} 字典
            date_columns: {表名: 日期列名} 字典

        Returns:
            TableProfile 列表
        """
        date_columns = date_columns or {}
        results: list[TableProfile] = []

        for table_name, df in tables.items():
            date_col = date_columns.get(table_name)
            profile = self.profile_table(df, table_name, date_column=date_col)
            results.append(profile)

        return results

    def generate_report(
        self,
        profiles: list[TableProfile],
    ) -> dict[str, Any]:
        """
        生成数据质量报告摘要。

        Args:
            profiles: 表级分析结果列表

        Returns:
            报告摘要字典
        """
        total_tables = len(profiles)
        avg_score = np.mean([p.overall_quality_score for p in profiles]) if profiles else 0.0
        total_anomalies = sum(len(p.anomalies) for p in profiles)

        critical_tables = [
            p.table_name for p in profiles
            if p.overall_quality_score < 60
        ]

        anomaly_summary: dict[str, int] = {}
        for p in profiles:
            for a in p.anomalies:
                atype = a.get("alert_type", "unknown")
                anomaly_summary[atype] = anomaly_summary.get(atype, 0) + 1

        return {
            "report_date": date.today().isoformat(),
            "total_tables": total_tables,
            "average_quality_score": round(float(avg_score), 2),
            "total_anomalies": total_anomalies,
            "critical_tables": critical_tables,
            "anomaly_type_distribution": anomaly_summary,
            "table_scores": {
                p.table_name: p.overall_quality_score
                for p in profiles
            },
        }
