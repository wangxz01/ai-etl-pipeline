"""
用户留存分析引擎

基于群组（Cohort）方法计算用户留存率，支持周留存和月留存。
提供 SQL（含 WITH RECURSIVE）和 pandas 两种实现。

留存分析核心逻辑：
    1. 确定每个用户的注册周/月作为群组
    2. 追踪该群组在第 N 个周期是否仍有活跃行为
    3. 计算留存率 = 留存用户数 / 群组初始用户数

使用方法:
    from etl.user_analytics.retention_etl import RetentionAnalyzer
    analyzer = RetentionAnalyzer()
    result = analyzer.run(events_df, users_df, period_type='week')
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Protocol

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DBConnection(Protocol):
    """数据库连接协议。"""
    def cursor(self) -> Any: ...


@dataclass
class CohortData:
    """
    群组留存数据。

    Attributes:
        period_type: 周期类型（week/month）
        cohort_date: 群组起始日期
        cohort_size: 群组初始用户数
        retention_curve: {period_number: retention_rate} 留存曲线
        segments: 分维度留存数据
    """
    period_type: str
    cohort_date: date
    cohort_size: int
    retention_curve: dict[int, float] = field(default_factory=dict)
    segments: list[dict[str, Any]] = field(default_factory=list)


# ============================================================
# SQL 查询模板
# ============================================================

# 使用 WITH RECURSIVE 构建群组和周期
SQL_WEEKLY_RETENTION = """
WITH RECURSIVE cohort_users AS (
    -- 第一层：确定每个用户的注册周（群组）
    SELECT
        u.user_id,
        u.region,
        u.plan_type,
        DATE_TRUNC('week', u.signup_date)::DATE AS cohort_week
    FROM users u
    WHERE u.signup_date BETWEEN %(cohort_start)s AND %(cohort_end)s
      AND u.email NOT LIKE 'test_%%'
      AND u.email NOT LIKE 'bot_%%'
),
-- 生成 0-12 周的周期序列
weeks AS (
    SELECT 0 AS week_number
    UNION ALL
    SELECT week_number + 1 FROM weeks WHERE week_number < 12
),
-- 用户每周活跃标记
user_weekly_active AS (
    SELECT DISTINCT
        cu.user_id,
        cu.cohort_week,
        cu.region,
        cu.plan_type,
        DATE_TRUNC('week', e.event_date)::DATE AS active_week
    FROM cohort_users cu
    INNER JOIN user_events e ON cu.user_id = e.user_id
    WHERE e.event_date BETWEEN cu.cohort_week AND cu.cohort_week + INTERVAL '84 days'
),
-- 计算每个群组每周的留存用户数
cohort_retention AS (
    SELECT
        cu.cohort_week,
        w.week_number,
        cu.region,
        cu.plan_type,
        COUNT(DISTINCT cu.user_id) AS cohort_size_base,
        COUNT(DISTINCT uwa.user_id) AS retained_users
    FROM cohort_users cu
    CROSS JOIN weeks w
    LEFT JOIN user_weekly_active uwa
        ON cu.user_id = uwa.user_id
        AND uwa.active_week = cu.cohort_week + (w.week_number * INTERVAL '7 days')
    GROUP BY cu.cohort_week, w.week_number, cu.region, cu.plan_type
)
SELECT
    cohort_week,
    week_number,
    region,
    plan_type,
    cohort_size_base,
    retained_users,
    CASE
        WHEN cohort_size_base > 0
        THEN ROUND(retained_users::NUMERIC / cohort_size_base, 4)
        ELSE 0
    END AS retention_rate
FROM cohort_retention
ORDER BY cohort_week, week_number;
"""

# 月留存 SQL
SQL_MONTHLY_RETENTION = """
WITH RECURSIVE cohort_users AS (
    SELECT
        u.user_id,
        u.region,
        u.plan_type,
        DATE_TRUNC('month', u.signup_date)::DATE AS cohort_month
    FROM users u
    WHERE u.signup_date BETWEEN %(cohort_start)s AND %(cohort_end)s
      AND u.email NOT LIKE 'test_%%'
      AND u.email NOT LIKE 'bot_%%'
),
months AS (
    SELECT 0 AS month_number
    UNION ALL
    SELECT month_number + 1 FROM months WHERE month_number < 12
),
user_monthly_active AS (
    SELECT DISTINCT
        cu.user_id,
        cu.cohort_month,
        cu.region,
        cu.plan_type,
        DATE_TRUNC('month', e.event_date)::DATE AS active_month
    FROM cohort_users cu
    INNER JOIN user_events e ON cu.user_id = e.user_id
    WHERE e.event_date >= cu.cohort_month
      AND e.event_date < cu.cohort_month + INTERVAL '13 months'
),
cohort_retention AS (
    SELECT
        cu.cohort_month,
        m.month_number,
        cu.region,
        cu.plan_type,
        COUNT(DISTINCT cu.user_id) AS cohort_size_base,
        COUNT(DISTINCT uwa.user_id) AS retained_users
    FROM cohort_users cu
    CROSS JOIN months m
    LEFT JOIN user_monthly_active uwa
        ON cu.user_id = uwa.user_id
        AND uwa.active_month = cu.cohort_month + (m.month_number * INTERVAL '1 month')
    GROUP BY cu.cohort_month, m.month_number, cu.region, cu.plan_type
)
SELECT
    cohort_month,
    month_number,
    region,
    plan_type,
    cohort_size_base,
    retained_users,
    CASE
        WHEN cohort_size_base > 0
        THEN ROUND(retained_users::NUMERIC / cohort_size_base, 4)
        ELSE 0
    END AS retention_rate
FROM cohort_retention
ORDER BY cohort_month, month_number;
"""

# 留存率环比变化
SQL_RETENTION_PERIOD_OVER_PERIOD = """
WITH current_period AS (
    SELECT
        cohort_week,
        week_number,
        retained_users,
        retention_rate,
        LAG(retention_rate) OVER (PARTITION BY week_number ORDER BY cohort_week) AS prev_retention
    FROM metrics.retention
    WHERE period_type = 'week'
      AND region = 'ALL'
      AND plan_type = 'ALL'
)
SELECT
    cohort_week,
    week_number,
    retained_users,
    retention_rate,
    prev_retention,
    CASE
        WHEN prev_retention > 0
        THEN ROUND((retention_rate - prev_retention) / prev_retention, 4)
        ELSE 0
    END AS retention_change_pct
FROM current_period
ORDER BY cohort_week, week_number;
"""


# ============================================================
# pandas 实现
# ============================================================


class RetentionAnalyzer:
    """
    群组留存分析器。

    使用 pandas 实现周/月留存计算，生成留存矩阵和留存曲线。
    支持按地区、套餐维度拆分。

    Usage:
        analyzer = RetentionAnalyzer()
        result = analyzer.run(events_df, users_df, period_type='week')
        matrix = analyzer.build_retention_matrix(result)
    """

    def __init__(
        self,
        max_periods: int = 12,
        exclude_test_accounts: bool = True,
    ) -> None:
        """
        初始化留存分析器。

        Args:
            max_periods: 最大追踪周期数（默认12周/12月）
            exclude_test_accounts: 是否排除测试账号
        """
        self.max_periods = max_periods
        self.exclude_test_accounts = exclude_test_accounts
        self.test_patterns = ["test_", "bot_", "demo_", "qa_", "@test-company.com"]

    def run(
        self,
        events_df: pd.DataFrame,
        users_df: pd.DataFrame,
        period_type: str = "week",
        cohort_start: date | None = None,
        cohort_end: date | None = None,
    ) -> list[CohortData]:
        """
        执行留存分析。

        Args:
            events_df: 事件数据（user_id, event_date）
            users_df: 用户数据（user_id, email, signup_date, region, plan_type）
            period_type: "week" 或 "month"
            cohort_start: 群组起始日期（可选）
            cohort_end: 群组结束日期（可选）

        Returns:
            CohortData 列表，每个群组一个元素
        """
        logger.info("开始留存分析，周期类型: %s", period_type)

        events = self._preprocess_events(events_df)
        users = self._preprocess_users(users_df)

        # 过滤测试账号
        if self.exclude_test_accounts:
            test_mask = users["email"].apply(self._is_test_account)
            users = users[~test_mask]
            events = events[events["user_id"].isin(users["user_id"])]

        # 确定群组字段
        if period_type == "week":
            users["cohort_date"] = users["signup_date"].apply(self._get_week_start)
            events["period_date"] = events["event_date"].apply(self._get_week_start)
        else:
            users["cohort_date"] = users["signup_date"].apply(
                lambda d: d.replace(day=1)
            )
            events["period_date"] = events["event_date"].apply(
                lambda d: d.replace(day=1)
            )

        # 过滤群组日期范围
        if cohort_start:
            users = users[users["cohort_date"] >= cohort_start]
        if cohort_end:
            users = users[users["cohort_date"] <= cohort_end]

        # 按群组聚合
        results: list[CohortData] = []
        cohort_groups = users.groupby("cohort_date")

        for cohort_date, cohort_users in cohort_groups:
            cohort_user_ids = set(cohort_users["user_id"])
            cohort_size = len(cohort_user_ids)

            if cohort_size < 5:
                continue  # 跳过太小的群组

            # 获取该群组用户的所有事件
            cohort_events = events[events["user_id"].isin(cohort_user_ids)]

            # 计算每个周期的留存
            retention_curve: dict[int, float] = {}
            for period_num in range(self.max_periods + 1):
                if period_type == "week":
                    target_period = cohort_date + timedelta(weeks=period_num)
                else:
                    month = cohort_date.month + period_num
                    year = cohort_date.year + (month - 1) // 12
                    month = (month - 1) % 12 + 1
                    target_period = cohort_date.replace(year=year, month=month)

                # 在该周期内有事件的用户
                active_in_period = cohort_events[
                    cohort_events["period_date"] == target_period
                ]["user_id"].nunique()

                retention_curve[period_num] = round(
                    active_in_period / cohort_size, 4
                ) if cohort_size > 0 else 0.0

            results.append(CohortData(
                period_type=period_type,
                cohort_date=cohort_date,
                cohort_size=cohort_size,
                retention_curve=retention_curve,
            ))

        logger.info(
            "留存分析完成，%d 个群组，%s 周期",
            len(results), period_type,
        )
        return results

    def run_with_segments(
        self,
        events_df: pd.DataFrame,
        users_df: pd.DataFrame,
        period_type: str = "week",
        segment_columns: list[str] | None = None,
    ) -> dict[str, list[CohortData]]:
        """
        分维度执行留存分析。

        Args:
            events_df: 事件数据
            users_df: 用户数据
            period_type: 周期类型
            segment_columns: 拆分维度列名（如 ["region", "plan_type"]）

        Returns:
            按 "维度=值" 分组的 CohortData 列表字典
        """
        if segment_columns is None:
            segment_columns = ["region", "plan_type"]

        users = self._preprocess_users(users_df)
        events = self._preprocess_events(events_df)

        if self.exclude_test_accounts:
            test_mask = users["email"].apply(self._is_test_account)
            users = users[~test_mask]
            events = events[events["user_id"].isin(users["user_id"])]

        results: dict[str, list[CohortData]] = {}

        for col in segment_columns:
            if col not in users.columns:
                logger.warning("维度列 %s 不存在，跳过", col)
                continue

            for value in users[col].unique():
                segment_users = users[users[col] == value]
                segment_events = events[events["user_id"].isin(segment_users["user_id"])]

                key = f"{col}={value}"
                segment_results = self.run(
                    segment_events, segment_users, period_type,
                )
                results[key] = segment_results

                logger.info("维度 %s: %d 个群组", key, len(segment_results))

        return results

    def run_sql(
        self,
        db_connection: DBConnection,
        period_type: str = "week",
        cohort_start: date | None = None,
        cohort_end: date | None = None,
    ) -> pd.DataFrame:
        """
        使用 SQL 执行留存分析。

        Args:
            db_connection: 数据库连接
            period_type: 周期类型
            cohort_start: 群组起始日期
            cohort_end: 群组结束日期

        Returns:
            留存结果 DataFrame
        """
        if cohort_start is None:
            cohort_start = date(2024, 1, 1)
        if cohort_end is None:
            cohort_end = date(2024, 12, 31)

        sql = SQL_WEEKLY_RETENTION if period_type == "week" else SQL_MONTHLY_RETENTION
        period_col = "week_number" if period_type == "week" else "month_number"
        cohort_col = "cohort_week" if period_type == "week" else "cohort_month"

        cursor = db_connection.cursor()
        try:
            cursor.execute(sql, {
                "cohort_start": cohort_start,
                "cohort_end": cohort_end,
            })
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            result_df = pd.DataFrame(rows, columns=columns)
            logger.info("SQL 留存分析完成，%d 行", len(result_df))
            return result_df
        except Exception as e:
            logger.error("SQL 留存分析失败: %s", e)
            raise
        finally:
            cursor.close()

    def build_retention_matrix(
        self,
        cohort_results: list[CohortData],
    ) -> pd.DataFrame:
        """
        将群组留存数据构建为留存矩阵。

        矩阵行=群组日期，列=周期编号（0, 1, 2...），值为留存率。

        Args:
            cohort_results: 群组留存数据列表

        Returns:
            留存率矩阵 DataFrame
        """
        if not cohort_results:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for cohort in cohort_results:
            row: dict[str, Any] = {
                "cohort_date": cohort.cohort_date,
                "cohort_size": cohort.cohort_size,
            }
            for period, rate in cohort.retention_curve.items():
                row[f"period_{period}"] = rate
            rows.append(row)

        matrix_df = pd.DataFrame(rows)
        matrix_df = matrix_df.sort_values("cohort_date").reset_index(drop=True)

        logger.info("留存矩阵构建完成，%d 个群组", len(matrix_df))
        return matrix_df

    def compute_average_retention_curve(
        self,
        cohort_results: list[CohortData],
    ) -> dict[int, float]:
        """
        计算所有群组的平均留存曲线。

        Args:
            cohort_results: 群组留存数据列表

        Returns:
            {period_number: average_retention_rate} 平均留存曲线
        """
        if not cohort_results:
            return {}

        period_rates: dict[int, list[float]] = {}
        for cohort in cohort_results:
            for period, rate in cohort.retention_curve.items():
                if period not in period_rates:
                    period_rates[period] = []
                period_rates[period].append(rate)

        avg_curve: dict[int, float] = {}
        for period, rates in sorted(period_rates.items()):
            avg_curve[period] = round(float(np.mean(rates)), 4)

        logger.info("平均留存曲线计算完成，%d 个周期", len(avg_curve))
        return avg_curve

    def detect_retention_anomalies(
        self,
        cohort_results: list[CohortData],
        z_threshold: float = 2.0,
    ) -> list[dict[str, Any]]:
        """
        检测留存异常（显著偏离平均值的群组）。

        Args:
            cohort_results: 群组留存数据列表
            z_threshold: Z-score 阈值，超过此值视为异常

        Returns:
            异常信息列表
        """
        avg_curve = self.compute_average_retention_curve(cohort_results)
        if not avg_curve:
            return []

        # 计算每个周期留存率的标准差
        period_rates: dict[int, list[float]] = {}
        for cohort in cohort_results:
            for period, rate in cohort.retention_curve.items():
                if period not in period_rates:
                    period_rates[period] = []
                period_rates[period].append(rate)

        period_stats: dict[int, tuple[float, float]] = {}
        for period, rates in period_rates.items():
            period_stats[period] = (float(np.mean(rates)), float(np.std(rates)))

        anomalies: list[dict[str, Any]] = []
        for cohort in cohort_results:
            for period, rate in cohort.retention_curve.items():
                mean, std = period_stats.get(period, (0.0, 0.0))
                if std > 0:
                    z_score = (rate - mean) / std
                    if abs(z_score) > z_threshold:
                        anomalies.append({
                            "cohort_date": cohort.cohort_date,
                            "period": period,
                            "retention_rate": rate,
                            "average_rate": round(mean, 4),
                            "z_score": round(z_score, 2),
                            "direction": "above" if z_score > 0 else "below",
                        })

        logger.info("检测到 %d 个留存异常", len(anomalies))
        return anomalies

    # ============================================================
    # 内部方法
    # ============================================================

    def _preprocess_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理事件数据。"""
        result = df.copy()
        result["user_id"] = result["user_id"].astype(str)
        if "event_date" in result.columns:
            result["event_date"] = pd.to_datetime(result["event_date"]).apply(
                lambda x: x.date() if hasattr(x, "date") else x
            )
        return result

    def _preprocess_users(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理用户数据。"""
        result = df.copy()
        result["user_id"] = result["user_id"].astype(str)
        if "signup_date" in result.columns:
            result["signup_date"] = pd.to_datetime(result["signup_date"]).apply(
                lambda x: x.date() if hasattr(x, "date") else x
            )
        return result

    def _is_test_account(self, email: str) -> bool:
        """判断是否为测试账号。"""
        if not isinstance(email, str):
            return False
        return any(p in email.lower() for p in self.test_patterns)

    @staticmethod
    def _get_week_start(d: date) -> date:
        """获取日期所在周的周一。"""
        return d - timedelta(days=d.weekday())
