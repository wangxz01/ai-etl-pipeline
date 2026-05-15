"""
DAU/MAU 计算引擎

计算日活跃用户（DAU）、周活跃用户（WAU）、月活跃用户（MAU）以及用户粘性系数。
提供 SQL 和 pandas 两种实现，支持交叉验证结果一致性。

SQL 实现用于数据仓库批处理，pandas 实现用于实时计算和离线分析。

使用方法:
    from etl.user_analytics.dau_mau_etl import DAUMAUCalculator
    calculator = DAUMAUCalculator(db_connection=conn)
    result = calculator.run(target_date=date(2025, 1, 15))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Protocol

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================
# 类型定义
# ============================================================


class DBConnection(Protocol):
    """数据库连接协议，支持 psycopg2 / SQLAlchemy 等接口。"""

    def cursor(self) -> Any: ...


@dataclass
class DAUMAUResult:
    """
    DAU/MAU 计算结果。

    Attributes:
        metric_date: 指标日期
        dau: 日活跃用户数
        wau: 周活跃用户数（含当天往前7天）
        mau: 月活跃用户数（含当天往前30天）
        sticky_factor: 粘性系数（DAU/MAU）
        segments: 按维度拆分的详细数据
    """
    metric_date: date
    dau: int = 0
    wau: int = 0
    mau: int = 0
    sticky_factor: float = 0.0
    segments: list[dict[str, Any]] = field(default_factory=list)


# ============================================================
# SQL 查询模板
# ============================================================

# 测试账号过滤条件 —— 排除内部测试和机器人账号
TEST_ACCOUNT_FILTER = """
    AND u.email NOT LIKE 'test_%'
    AND u.email NOT LIKE 'bot_%'
    AND u.email NOT LIKE 'demo_%'
    AND u.email NOT LIKE 'qa_%'
    AND u.email NOT LIKE '%@test-company.com'
"""

# DAU 计算 SQL —— 按日期统计活跃用户数
SQL_DAU = """
WITH daily_active AS (
    SELECT
        e.event_date,
        e.user_id,
        u.region,
        u.plan_type,
        e.device
    FROM user_events e
    INNER JOIN users u ON e.user_id = u.user_id
    WHERE e.event_date = %(target_date)s
      AND u.status IN ('active', 'trial')
      {test_filter}
),
-- 按维度聚合
dau_by_segment AS (
    SELECT
        event_date AS metric_date,
        'dau' AS metric_name,
        COUNT(DISTINCT user_id) AS metric_value,
        COALESCE(region, 'ALL') AS region,
        COALESCE(plan_type, 'ALL') AS plan_type,
        COALESCE(device, 'ALL') AS device
    FROM daily_active
    GROUP BY ROLLUP (region, plan_type, device)
)
SELECT * FROM dau_by_segment
ORDER BY region, plan_type, device;
"""

# WAU 计算 SQL —— 滚动7天窗口
SQL_WAU = """
WITH weekly_active AS (
    SELECT
        %(target_date)s AS metric_date,
        e.user_id,
        u.region,
        u.plan_type,
        e.device
    FROM user_events e
    INNER JOIN users u ON e.user_id = u.user_id
    WHERE e.event_date BETWEEN %(start_week)s AND %(target_date)s
      AND u.status IN ('active', 'trial')
      {test_filter}
),
wau_by_segment AS (
    SELECT
        metric_date,
        'wau' AS metric_name,
        COUNT(DISTINCT user_id) AS metric_value,
        COALESCE(region, 'ALL') AS region,
        COALESCE(plan_type, 'ALL') AS plan_type,
        COALESCE(device, 'ALL') AS device
    FROM weekly_active
    GROUP BY ROLLUP (region, plan_type, device)
)
SELECT * FROM wau_by_segment
ORDER BY region, plan_type, device;
"""

# MAU 计算 SQL —— 滚动30天窗口
SQL_MAU = """
WITH monthly_active AS (
    SELECT
        %(target_date)s AS metric_date,
        e.user_id,
        u.region,
        u.plan_type,
        e.device
    FROM user_events e
    INNER JOIN users u ON e.user_id = u.user_id
    WHERE e.event_date BETWEEN %(start_month)s AND %(target_date)s
      AND u.status IN ('active', 'trial')
      {test_filter}
),
mau_by_segment AS (
    SELECT
        metric_date,
        'mau' AS metric_name,
        COUNT(DISTINCT user_id) AS metric_value,
        COALESCE(region, 'ALL') AS region,
        COALESCE(plan_type, 'ALL') AS plan_type,
        COALESCE(device, 'ALL') AS device
    FROM monthly_active
    GROUP BY ROLLUP (region, plan_type, device)
)
SELECT * FROM mau_by_segment
ORDER BY region, plan_type, device;
"""

# 7天和30天趋势 SQL
SQL_DAU_TREND = """
WITH daily_counts AS (
    SELECT
        event_date,
        COUNT(DISTINCT user_id) AS dau
    FROM user_events e
    INNER JOIN users u ON e.user_id = u.user_id
    WHERE e.event_date BETWEEN %(start_date)s AND %(end_date)s
      AND u.status IN ('active', 'trial')
      {test_filter}
    GROUP BY event_date
)
SELECT
    event_date,
    dau,
    AVG(dau) OVER (ORDER BY event_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS dau_7d_avg,
    AVG(dau) OVER (ORDER BY event_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS dau_30d_avg
FROM daily_counts
ORDER BY event_date;
"""

# 粘性系数 SQL
SQL_STICKY_FACTOR = """
WITH dau_count AS (
    SELECT COUNT(DISTINCT user_id) AS dau
    FROM user_events e
    INNER JOIN users u ON e.user_id = u.user_id
    WHERE e.event_date = %(target_date)s
      AND u.status IN ('active', 'trial')
      {test_filter}
),
mau_count AS (
    SELECT COUNT(DISTINCT user_id) AS mau
    FROM user_events e
    INNER JOIN users u ON e.user_id = u.user_id
    WHERE e.event_date BETWEEN %(start_month)s AND %(target_date)s
      AND u.status IN ('active', 'trial')
      {test_filter}
)
SELECT
    dau.dau,
    mau.mau,
    CASE WHEN mau.mau > 0 THEN ROUND(dau.dau::NUMERIC / mau.mau, 4) ELSE 0 END AS sticky_factor
FROM dau_count, mau_count;
"""


# ============================================================
# pandas 实现
# ============================================================


class DAUMAUCalculator:
    """
    DAU/MAU 计算器。

    提供基于 pandas 的活跃用户计算实现，可与 SQL 结果交叉验证。
    支持按地区、套餐、设备维度拆分，自动过滤测试账号。

    Usage:
        calculator = DAUMAUCalculator(db_connection=conn)
        result = calculator.run(target_date=date(2025, 1, 15))
    """

    # 测试账号邮箱模式
    TEST_EMAIL_PATTERNS = [
        "test_", "bot_", "demo_", "qa_",
        "@test-company.com",
    ]

    def __init__(
        self,
        db_connection: DBConnection | None = None,
        timezone: str = "Asia/Shanghai",
        exclude_test_accounts: bool = True,
    ) -> None:
        """
        初始化计算器。

        Args:
            db_connection: 数据库连接（可选，用于 SQL 模式）
            timezone: 目标时区，用于日期对齐
            exclude_test_accounts: 是否排除测试账号
        """
        self.db_connection = db_connection
        self.timezone = timezone
        self.exclude_test_accounts = exclude_test_accounts

    def run(
        self,
        events_df: pd.DataFrame,
        users_df: pd.DataFrame,
        target_date: date,
    ) -> DAUMAUResult:
        """
        执行完整的 DAU/MAU 计算流程。

        Args:
            events_df: 用户事件 DataFrame，需包含 user_id, event_date, event_type, device
            users_df: 用户 DataFrame，需包含 user_id, email, region, plan_type, status
            target_date: 目标计算日期

        Returns:
            DAUMAUResult 包含所有指标和分维度数据
        """
        logger.info("开始 DAU/MAU 计算，目标日期: %s", target_date)

        # 数据预处理
        events = self._preprocess_events(events_df)
        users = self._preprocess_users(users_df)

        # 过滤测试账号
        if self.exclude_test_accounts:
            test_mask = users["email"].apply(self._is_test_account)
            valid_users = users[~test_mask]
            events = events[events["user_id"].isin(valid_users["user_id"])]
        else:
            valid_users = users

        # 合并用户属性到事件
        merged = events.merge(
            valid_users[["user_id", "region", "plan_type"]],
            on="user_id",
            how="inner",
        )

        # 计算 DAU
        dau_df = merged[merged["event_date"] == target_date]
        dau = dau_df["user_id"].nunique()
        dau_segments = self._aggregate_by_segments(dau_df, "dau")

        # 计算 WAU（含当天往前7天）
        start_week = target_date - timedelta(days=6)
        wau_df = merged[
            (merged["event_date"] >= start_week) & (merged["event_date"] <= target_date)
        ]
        wau = wau_df["user_id"].nunique()
        wau_segments = self._aggregate_by_segments(wau_df, "wau")

        # 计算 MAU（含当天往前30天）
        start_month = target_date - timedelta(days=29)
        mau_df = merged[
            (merged["event_date"] >= start_month) & (merged["event_date"] <= target_date)
        ]
        mau = mau_df["user_id"].nunique()
        mau_segments = self._aggregate_by_segments(mau_df, "mau")

        # 粘性系数
        sticky_factor = round(dau / mau, 4) if mau > 0 else 0.0

        result = DAUMAUResult(
            metric_date=target_date,
            dau=dau,
            wau=wau,
            mau=mau,
            sticky_factor=sticky_factor,
            segments=dau_segments + wau_segments + mau_segments,
        )

        logger.info(
            "DAU=%d, WAU=%d, MAU=%d, Sticky=%.4f",
            dau, wau, mau, sticky_factor,
        )
        return result

    def run_sql(
        self,
        target_date: date,
    ) -> list[dict[str, Any]]:
        """
        使用 SQL 执行 DAU/MAU 计算（需要数据库连接）。

        Args:
            target_date: 目标计算日期

        Returns:
            查询结果列表

        Raises:
            RuntimeError: 未配置数据库连接时抛出
        """
        if self.db_connection is None:
            raise RuntimeError("SQL 模式需要数据库连接")

        results: list[dict[str, Any]] = []
        params = {
            "target_date": target_date,
            "start_week": target_date - timedelta(days=6),
            "start_month": target_date - timedelta(days=29),
        }
        test_filter = TEST_ACCOUNT_FILTER if self.exclude_test_accounts else ""

        cursor = self.db_connection.cursor()
        try:
            for sql_template, metric_name in [
                (SQL_DAU, "dau"),
                (SQL_WAU, "wau"),
                (SQL_MAU, "mau"),
            ]:
                sql = sql_template.format(test_filter=test_filter)
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                for row in rows:
                    results.append({
                        "metric_date": row[0],
                        "metric_name": row[1],
                        "metric_value": row[2],
                        "region": row[3] or "ALL",
                        "plan_type": row[4] or "ALL",
                        "device": row[5] or "ALL",
                    })
                logger.info("SQL %s 计算完成，%d 条记录", metric_name, len(rows))
        except Exception as e:
            logger.error("SQL DAU/MAU 计算失败: %s", e)
            raise
        finally:
            cursor.close()

        return results

    def run_trend(
        self,
        events_df: pd.DataFrame,
        users_df: pd.DataFrame,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        计算 DAU 趋势数据，包含7日和30日移动平均。

        Args:
            events_df: 事件数据
            users_df: 用户数据
            start_date: 趋势起始日期
            end_date: 趋势结束日期

        Returns:
            包含 dau, dau_7d_avg, dau_30d_avg 列的 DataFrame
        """
        events = self._preprocess_events(events_df)
        users = self._preprocess_users(users_df)

        if self.exclude_test_accounts:
            test_mask = users["email"].apply(self._is_test_account)
            valid_users = users[~test_mask]
            events = events[events["user_id"].isin(valid_users["user_id"])]
        else:
            valid_users = users

        # 只保留活跃用户的事件
        active_users = valid_users[valid_users["status"].isin(["active", "trial"])]
        events = events[events["user_id"].isin(active_users["user_id"])]

        # 按日期统计 DAU
        date_range = pd.date_range(start_date, end_date, freq="D")
        daily_counts: list[dict[str, Any]] = []
        for d in date_range:
            day_events = events[events["event_date"] == d.date()]
            daily_counts.append({
                "event_date": d.date(),
                "dau": day_events["user_id"].nunique(),
            })

        trend_df = pd.DataFrame(daily_counts)

        if trend_df.empty:
            return trend_df

        # 计算移动平均
        trend_df["dau_7d_avg"] = trend_df["dau"].rolling(window=7, min_periods=1).mean()
        trend_df["dau_30d_avg"] = trend_df["dau"].rolling(window=30, min_periods=1).mean()

        logger.info("DAU 趋势计算完成，%d 天数据", len(trend_df))
        return trend_df

    def compute_sticky_factor_series(
        self,
        events_df: pd.DataFrame,
        users_df: pd.DataFrame,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        计算每日粘性系数时间序列（DAU/MAU）。

        Args:
            events_df: 事件数据
            users_df: 用户数据
            start_date: 起始日期
            end_date: 结束日期

        Returns:
            包含 event_date, dau, mau, sticky_factor 列的 DataFrame
        """
        events = self._preprocess_events(events_df)
        users = self._preprocess_users(users_df)

        if self.exclude_test_accounts:
            test_mask = users["email"].apply(self._is_test_account)
            valid_users = users[~test_mask]
            events = events[events["user_id"].isin(valid_users["user_id"])]
        else:
            valid_users = users

        active_users = valid_users[valid_users["status"].isin(["active", "trial"])]
        events = events[events["user_id"].isin(active_users["user_id"])]

        results: list[dict[str, Any]] = []
        date_range = pd.date_range(start_date, end_date, freq="D")

        for d in date_range:
            current = d.date()
            # DAU
            dau_users = set(events[events["event_date"] == current]["user_id"].unique())
            dau = len(dau_users)

            # MAU（前30天）
            month_start = current - timedelta(days=29)
            mau_users = set(
                events[
                    (events["event_date"] >= month_start) & (events["event_date"] <= current)
                ]["user_id"].unique()
            )
            mau = len(mau_users)

            sticky = round(dau / mau, 4) if mau > 0 else 0.0

            results.append({
                "event_date": current,
                "dau": dau,
                "mau": mau,
                "sticky_factor": sticky,
            })

        return pd.DataFrame(results)

    def cross_validate(
        self,
        events_df: pd.DataFrame,
        users_df: pd.DataFrame,
        target_date: date,
        tolerance: int = 2,
    ) -> dict[str, bool]:
        """
        交叉验证 pandas 和 SQL 计算结果的一致性。

        Args:
            events_df: 事件数据
            users_df: 用户数据
            target_date: 目标日期
            tolerance: 允许的误差范围（绝对值）

        Returns:
            各指标验证结果（True=一致）

        Raises:
            RuntimeError: 未配置数据库连接
        """
        if self.db_connection is None:
            raise RuntimeError("交叉验证需要数据库连接")

        # pandas 计算
        pandas_result = self.run(events_df, users_df, target_date)

        # SQL 计算
        sql_results = self.run_sql(target_date)

        # 比较结果
        sql_metrics: dict[str, int] = {}
        for row in sql_results:
            key = row["metric_name"]
            if row["region"] == "ALL" and row["plan_type"] == "ALL" and row["device"] == "ALL":
                sql_metrics[key] = row["metric_value"]

        validation: dict[str, bool] = {}
        for metric, pandas_value in [
            ("dau", pandas_result.dau),
            ("wau", pandas_result.wau),
            ("mau", pandas_result.mau),
        ]:
            sql_value = sql_metrics.get(metric, 0)
            diff = abs(pandas_value - sql_value)
            validation[metric] = diff <= tolerance
            if not validation[metric]:
                logger.warning(
                    "%s 不一致: pandas=%d, sql=%d, diff=%d",
                    metric, pandas_value, sql_value, diff,
                )

        logger.info("交叉验证结果: %s", validation)
        return validation

    # ============================================================
    # 内部方法
    # ============================================================

    def _preprocess_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理事件数据，确保类型正确。"""
        result = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(result["event_date"]):
            result["event_date"] = pd.to_datetime(result["event_date"]).dt.date
        if "event_date" in result.columns:
            result["event_date"] = result["event_date"].apply(
                lambda x: x if isinstance(x, date) else pd.Timestamp(x).date()
            )
        result["user_id"] = result["user_id"].astype(str)
        return result

    def _preprocess_users(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理用户数据，确保类型正确。"""
        result = df.copy()
        result["user_id"] = result["user_id"].astype(str)
        return result

    def _is_test_account(self, email: str) -> bool:
        """判断是否为测试账号。"""
        if not isinstance(email, str):
            return False
        email_lower = email.lower()
        return any(pattern.lower() in email_lower for pattern in self.TEST_EMAIL_PATTERNS)

    def _aggregate_by_segments(
        self,
        df: pd.DataFrame,
        metric_name: str,
    ) -> list[dict[str, Any]]:
        """
        按维度聚合活跃用户数。

        生成全维度（ALL）、地区、套餐、以及地区×套餐的聚合结果。

        Args:
            df: 已过滤的事件数据（已合并用户属性）
            metric_name: 指标名称（dau/wau/mau）

        Returns:
            分维度指标列表
        """
        if df.empty:
            return [{
                "metric_name": metric_name,
                "metric_value": 0,
                "region": "ALL",
                "plan_type": "ALL",
                "device": "ALL",
            }]

        segments: list[dict[str, Any]] = []

        # 全局总计
        total = df["user_id"].nunique()
        segments.append({
            "metric_name": metric_name,
            "metric_value": total,
            "region": "ALL",
            "plan_type": "ALL",
            "device": "ALL",
        })

        # 按地区拆分
        for region in df["region"].unique():
            count = df[df["region"] == region]["user_id"].nunique()
            segments.append({
                "metric_name": metric_name,
                "metric_value": count,
                "region": region,
                "plan_type": "ALL",
                "device": "ALL",
            })

        # 按套餐拆分
        for plan in df["plan_type"].unique():
            count = df[df["plan_type"] == plan]["user_id"].nunique()
            segments.append({
                "metric_name": metric_name,
                "metric_value": count,
                "region": "ALL",
                "plan_type": plan,
                "device": "ALL",
            })

        # 按设备拆分
        if "device" in df.columns:
            for device in df["device"].unique():
                count = df[df["device"] == device]["user_id"].nunique()
                segments.append({
                    "metric_name": metric_name,
                    "metric_value": count,
                    "region": "ALL",
                    "plan_type": "ALL",
                    "device": device,
                })

        # 地区 × 套餐交叉维度
        for region in df["region"].unique():
            for plan in df["plan_type"].unique():
                mask = (df["region"] == region) & (df["plan_type"] == plan)
                count = df[mask]["user_id"].nunique()
                if count > 0:
                    segments.append({
                        "metric_name": metric_name,
                        "metric_value": count,
                        "region": region,
                        "plan_type": plan,
                        "device": "ALL",
                    })

        return segments
