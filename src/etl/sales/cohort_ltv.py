"""
客户生命周期价值（LTV）分析引擎

基于月度群组计算实际和预测的客户 LTV。
支持 ARPU/ARPPU 分析和历史群组曲线投影。

核心逻辑:
    1. 按注册月构建群组
    2. 追踪每个群组每月的收入贡献
    3. 计算累计 LTV 曲线
    4. 基于历史群组衰减率预测未来 LTV

使用方法:
    from etl.sales.cohort_ltv import CohortLTVAnalyzer
    analyzer = CohortLTVAnalyzer()
    result = analyzer.run(orders_df, subscriptions_df, users_df)
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
class CohortLTVResult:
    """
    群组 LTV 结果。

    Attributes:
        cohort_month: 群组月份
        cohort_size: 群组用户数
        months_data: 每月详细数据（收入、活跃用户等）
        actual_ltv: 实际累计 LTV
        projected_ltv: 预测 LTV（基于衰减率投影）
    """
    cohort_month: date
    cohort_size: int
    months_data: list[dict[str, Any]] = field(default_factory=list)
    actual_ltv: float = 0.0
    projected_ltv: float = 0.0


# ============================================================
# SQL 查询模板
# ============================================================

SQL_COHORT_REVENUE = """
WITH user_cohorts AS (
    SELECT
        user_id,
        DATE_TRUNC('month', signup_date)::DATE AS cohort_month
    FROM users
    WHERE email NOT LIKE 'test_%%'
      AND email NOT LIKE 'bot_%%'
),
monthly_revenue AS (
    SELECT
        uc.cohort_month,
        DATE_TRUNC('month', o.order_date)::DATE AS revenue_month,
        COUNT(DISTINCT o.user_id) AS active_users,
        SUM(o.total_amount) AS total_revenue
    FROM orders o
    INNER JOIN user_cohorts uc ON o.user_id = uc.user_id
    WHERE o.status IN ('paid', 'delivered')
    GROUP BY uc.cohort_month, DATE_TRUNC('month', o.order_date)
)
SELECT
    cohort_month,
    revenue_month,
    EXTRACT(YEAR FROM revenue_month) * 12 + EXTRACT(MONTH FROM revenue_month)
        - EXTRACT(YEAR FROM cohort_month) * 12 - EXTRACT(MONTH FROM cohort_month) AS months_since,
    active_users,
    total_revenue
FROM monthly_revenue
ORDER BY cohort_month, revenue_month;
"""

SQL_ARPU_BY_COHORT = """
WITH user_cohorts AS (
    SELECT user_id, DATE_TRUNC('month', signup_date)::DATE AS cohort_month
    FROM users
    WHERE email NOT LIKE 'test_%%'
),
cohort_sizes AS (
    SELECT cohort_month, COUNT(*) AS cohort_size
    FROM user_cohorts
    GROUP BY cohort_month
),
monthly_revenue AS (
    SELECT
        uc.cohort_month,
        DATE_TRUNC('month', o.order_date)::DATE AS revenue_month,
        SUM(o.total_amount) AS total_revenue
    FROM orders o
    INNER JOIN user_cohorts uc ON o.user_id = uc.user_id
    WHERE o.status IN ('paid', 'delivered')
    GROUP BY uc.cohort_month, DATE_TRUNC('month', o.order_date)
)
SELECT
    cs.cohort_month,
    mr.total_revenue / cs.cohort_size AS arpu,
    mr.total_revenue / NULLIF(mr.active_users, 0) AS arppu,
    cs.cohort_size,
    SUM(mr.total_revenue) OVER (PARTITION BY cs.cohort_month ORDER BY mr.revenue_month) / cs.cohort_size AS cum_ltv
FROM cohort_sizes cs
INNER JOIN monthly_revenue mr ON cs.cohort_month = mr.cohort_month
ORDER BY cs.cohort_month, mr.revenue_month;
"""


# ============================================================
# pandas 实现
# ============================================================


class CohortLTVAnalyzer:
    """
    客户生命周期价值分析器。

    基于月度群组方法计算客户 LTV，包括实际累计和预测值。
    使用历史群组衰减率投影未来 LTV。

    Usage:
        analyzer = CohortLTVAnalyzer(projection_months=24)
        results = analyzer.run(orders_df, subscriptions_df, users_df)
    """

    def __init__(
        self,
        projection_months: int = 24,
        decay_model: str = "exponential",
        min_cohort_size: int = 10,
    ) -> None:
        """
        初始化 LTV 分析器。

        Args:
            projection_months: LTV 预测的最大月数
            decay_model: 衰减模型（"exponential" 或 "linear"）
            min_cohort_size: 最小群组大小（低于此值跳过）
        """
        self.projection_months = projection_months
        self.decay_model = decay_model
        self.min_cohort_size = min_cohort_size

    def run(
        self,
        orders_df: pd.DataFrame,
        subscriptions_df: pd.DataFrame | None = None,
        users_df: pd.DataFrame | None = None,
    ) -> list[CohortLTVResult]:
        """
        执行完整的群组 LTV 分析。

        Args:
            orders_df: 订单数据
            subscriptions_df: 订阅数据（可选，用于补充收入）
            users_df: 用户数据（可选，用于群组构建）

        Returns:
            CohortLTVResult 列表
        """
        logger.info("开始群组 LTV 分析，预测 %d 个月", self.projection_months)

        orders = self._preprocess_orders(orders_df)
        users = self._preprocess_users(users_df) if users_df is not None else None

        # 构建群组
        cohort_data = self._build_cohorts(orders, users)

        # 计算每月指标
        results: list[CohortLTVResult] = []
        for cohort_month, data in sorted(cohort_data.items()):
            if data["cohort_size"] < self.min_cohort_size:
                continue

            months = self._compute_monthly_metrics(data)
            actual_ltv = self._compute_actual_ltv(months)
            projected_ltv = self._project_ltv(months, data["cohort_size"])

            results.append(CohortLTVResult(
                cohort_month=cohort_month,
                cohort_size=data["cohort_size"],
                months_data=months,
                actual_ltv=round(actual_ltv, 2),
                projected_ltv=round(projected_ltv, 2),
            ))

        logger.info("LTV 分析完成，%d 个群组", len(results))
        return results

    def _build_cohorts(
        self,
        orders: pd.DataFrame,
        users: pd.DataFrame | None,
    ) -> dict[date, dict[str, Any]]:
        """
        构建月度群组数据。

        如果有用户数据，按注册月分组；否则按首单月份分组。

        Args:
            orders: 订单数据
            users: 用户数据（可选）

        Returns:
            {cohort_month: {cohort_size, orders_df}} 字典
        """
        cohorts: dict[date, dict[str, Any]] = {}

        if users is not None and "signup_date" in users.columns:
            # 按注册月分组
            merged = orders.merge(users[["user_id", "signup_date"]], on="user_id", how="left")
            merged["cohort_month"] = pd.to_datetime(merged["signup_date"]).apply(
                lambda x: x.replace(day=1).date() if hasattr(x, "date") else x
            )

            for cohort_month in merged["cohort_month"].unique():
                if pd.isna(cohort_month):
                    continue
                cm = cohort_month if isinstance(cohort_month, date) else pd.Timestamp(cohort_month).date()
                cohort_users = set(merged[merged["cohort_month"] == cohort_month]["user_id"].unique())
                cohort_orders = merged[merged["user_id"].isin(cohort_users)]

                cohorts[cm] = {
                    "cohort_size": len(cohort_users),
                    "orders": cohort_orders,
                    "user_ids": cohort_users,
                }
        else:
            # 按首单月份分组
            first_orders = orders.groupby("user_id")["order_date"].min().reset_index()
            first_orders["cohort_month"] = first_orders["order_date"].apply(
                lambda d: d.replace(day=1) if isinstance(d, date) else d
            )

            for _, row in first_orders.iterrows():
                cm = row["cohort_month"]
                if cm not in cohorts:
                    cohorts[cm] = {"cohort_size": 0, "orders": pd.DataFrame(), "user_ids": set()}
                cohorts[cm]["user_ids"].add(row["user_id"])
                cohorts[cm]["cohort_size"] = len(cohorts[cm]["user_ids"])

            for cm in cohorts:
                uids = cohorts[cm]["user_ids"]
                cohorts[cm]["orders"] = orders[orders["user_id"].isin(uids)]

        return cohorts

    def _compute_monthly_metrics(
        self,
        cohort_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        计算群组每月的详细指标。

        Args:
            cohort_data: 群组数据（包含 orders 和 cohort_size）

        Returns:
            每月指标列表
        """
        cohort_orders = cohort_data["orders"]
        cohort_size = cohort_data["cohort_size"]
        user_ids = cohort_data["user_ids"]

        if cohort_orders.empty:
            return []

        # 确定群组起始月份
        first_date = pd.to_datetime(cohort_orders["order_date"]).min()
        if hasattr(first_date, "date"):
            cohort_start = first_date.date().replace(day=1)
        else:
            cohort_start = first_date.replace(day=1)

        # 按月聚合
        monthly_data: list[dict[str, Any]] = []
        for month_offset in range(self.projection_months):
            year = cohort_start.year + (cohort_start.month + month_offset - 1) // 12
            month = (cohort_start.month + month_offset - 1) % 12 + 1
            target_month = date(year, month, 1)

            month_orders = cohort_orders[
                pd.to_datetime(cohort_orders["order_date"]).apply(
                    lambda d: d.date().replace(day=1) if hasattr(d, "date") else d.replace(day=1)
                ) == target_month
            ]

            active_users = month_orders["user_id"].nunique()
            revenue = float(month_orders["amount"].sum()) if not month_orders.empty else 0.0
            arpu = revenue / cohort_size if cohort_size > 0 else 0.0

            monthly_data.append({
                "months_since": month_offset,
                "month_date": target_month,
                "active_users": active_users,
                "total_revenue": round(revenue, 2),
                "arpu": round(arpu, 2),
            })

        # 计算累计值
        cum_revenue = 0.0
        for m in monthly_data:
            cum_revenue += m["total_revenue"]
            m["cum_revenue"] = round(cum_revenue, 2)
            m["cum_ltv"] = round(cum_revenue / cohort_size, 2)

        return monthly_data

    def _compute_actual_ltv(self, months_data: list[dict[str, Any]]) -> float:
        """
        计算实际累计 LTV。

        Args:
            months_data: 月度数据列表

        Returns:
            累计 LTV 值
        """
        if not months_data:
            return 0.0
        return months_data[-1].get("cum_ltv", 0.0)

    def _project_ltv(
        self,
        months_data: list[dict[str, Any]],
        cohort_size: int,
    ) -> float:
        """
        基于历史衰减率预测完整 LTV。

        预测方法:
            1. 计算已观测月数的月均收入衰减率
            2. 指数衰减：revenue(t) = revenue(0) * e^(-lambda*t)
            3. 积分得到预测总 LTV = sum(revenue(t)) for t=0..inf

        Args:
            months_data: 已观测的月度数据
            cohort_size: 群组大小

        Returns:
            预测 LTV 值
        """
        if len(months_data) < 3:
            return months_data[-1].get("cum_ltv", 0.0) if months_data else 0.0

        # 提取已观测的收入序列
        revenues = [m["total_revenue"] for m in months_data if m["total_revenue"] > 0]
        if len(revenues) < 2:
            return months_data[-1].get("cum_ltv", 0.0)

        if self.decay_model == "exponential":
            # 拟合指数衰减率
            log_revenues = [np.log(max(r, 0.01)) for r in revenues]
            time_points = list(range(len(log_revenues)))

            if len(time_points) >= 2:
                # 线性回归拟合 log(revenue) = a - lambda * t
                coeffs = np.polyfit(time_points, log_revenues, 1)
                decay_rate = -coeffs[0]  # lambda
                initial_revenue = np.exp(coeffs[1])  # revenue(0)

                # 预测未来月份
                actual_cum = months_data[-1]["cum_revenue"]
                remaining_ltv = 0.0

                for t in range(len(months_data), self.projection_months):
                    projected_revenue = initial_revenue * np.exp(-decay_rate * t)
                    remaining_ltv += projected_revenue

                projected_total = actual_cum + remaining_ltv
                return projected_total / cohort_size if cohort_size > 0 else 0.0

        # 线性衰减备选
        if len(revenues) >= 2:
            avg_decline = (revenues[0] - revenues[-1]) / len(revenues)
            actual_cum = months_data[-1]["cum_revenue"]
            remaining = 0.0
            last_rev = revenues[-1]
            for _ in range(len(months_data), self.projection_months):
                last_rev = max(0, last_rev - avg_decline)
                remaining += last_rev
                if last_rev <= 0:
                    break
            return (actual_cum + remaining) / cohort_size if cohort_size > 0 else 0.0

        return months_data[-1].get("cum_ltv", 0.0)

    def compute_summary(
        self,
        results: list[CohortLTVResult],
    ) -> pd.DataFrame:
        """
        生成 LTV 摘要表。

        Args:
            results: 群组 LTV 结果列表

        Returns:
            摘要 DataFrame
        """
        rows: list[dict[str, Any]] = []
        for r in results:
            rows.append({
                "cohort_month": r.cohort_month,
                "cohort_size": r.cohort_size,
                "actual_ltv": r.actual_ltv,
                "projected_ltv": r.projected_ltv,
                "ltv_months_observed": len(r.months_data),
            })

        return pd.DataFrame(rows).sort_values("cohort_month")

    def run_sql(
        self,
        db_connection: DBConnection,
    ) -> pd.DataFrame:
        """
        使用 SQL 执行群组 LTV 计算。

        Args:
            db_connection: 数据库连接

        Returns:
            LTV 结果 DataFrame
        """
        cursor = db_connection.cursor()
        try:
            cursor.execute(SQL_COHORT_REVENUE)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            result_df = pd.DataFrame(rows, columns=columns)
            logger.info("SQL LTV 分析完成，%d 行", len(result_df))
            return result_df
        except Exception as e:
            logger.error("SQL LTV 分析失败: %s", e)
            raise
        finally:
            cursor.close()

    # ============================================================
    # 内部方法
    # ============================================================

    def _preprocess_orders(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理订单数据。"""
        result = df.copy()
        result["user_id"] = result["user_id"].astype(str)
        if "order_date" in result.columns:
            result["order_date"] = pd.to_datetime(result["order_date"]).apply(
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
