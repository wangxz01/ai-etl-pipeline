"""
收入流水线 ETL 引擎

计算 MRR/ARR、收入拆分（经常性 vs 一次性）、销售管道分析和收入预测。

核心指标:
    - MRR (Monthly Recurring Revenue): 月经常性收入
    - ARR (Annual Recurring Revenue): 年经常性收入 = MRR × 12
    - 经常性收入 vs 一次性收入拆分
    - 管道阶段转化率（lead → opportunity → negotiation → closed-won/lost）
    - 赢单率、平均交易规模、销售周期时长
    - 收入预测：移动平均 + 季节性调整

使用方法:
    from etl.sales.revenue_pipeline import RevenuePipeline
    pipeline = RevenuePipeline()
    result = pipeline.run(subscriptions_df, orders_df, users_df)
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
class RevenueMetrics:
    """
    收入指标结果。

    Attributes:
        metric_date: 指标日期
        mrr: 月经常性收入
        arr: 年经常性收入
        recurring_revenue: 经常性收入
        one_time_revenue: 一次性收入
        new_mrr: 新增 MRR
        expansion_mrr: 扩展 MRR
        churned_mrr: 流失 MRR
        net_mrr_growth: 净 MRR 增长
    """
    metric_date: date
    mrr: float = 0.0
    arr: float = 0.0
    recurring_revenue: float = 0.0
    one_time_revenue: float = 0.0
    new_mrr: float = 0.0
    expansion_mrr: float = 0.0
    churned_mrr: float = 0.0
    net_mrr_growth: float = 0.0


@dataclass
class PipelineMetrics:
    """
    销售管道指标。

    Attributes:
        stage: 管道阶段
        count: 该阶段的交易数量
        total_amount: 总金额
        avg_deal_size: 平均交易规模
        win_rate: 赢单率
        avg_cycle_days: 平均销售周期天数
    """
    stage: str
    count: int = 0
    total_amount: float = 0.0
    avg_deal_size: float = 0.0
    win_rate: float = 0.0
    avg_cycle_days: float = 0.0


# ============================================================
# SQL 查询模板
# ============================================================

SQL_MRR_CALCULATION = """
WITH active_subs AS (
    -- 当前有效的订阅
    SELECT
        s.user_id,
        s.plan,
        s.amount,
        s.billing_cycle,
        s.mrr_amount,
        s.currency,
        s.status,
        LAG(s.mrr_amount) OVER (PARTITION BY s.user_id ORDER BY s.started_at) AS prev_mrr
    FROM subscriptions s
    WHERE s.status = 'active'
      AND s.started_at <= %(metric_date)s::TIMESTAMP
      AND (s.ended_at IS NULL OR s.ended_at > %(metric_date)s::TIMESTAMP)
),
mrr_breakdown AS (
    SELECT
        SUM(mrr_amount) AS total_mrr,
        SUM(mrr_amount) FILTER (WHERE prev_mrr IS NULL) AS new_mrr,
        SUM(GREATEST(mrr_amount - COALESCE(prev_mrr, 0), 0)) AS expansion_mrr,
        SUM(GREATEST(COALESCE(prev_mrr, 0) - mrr_amount, 0)) AS contraction_mrr,
        COUNT(DISTINCT user_id) FILTER (WHERE prev_mrr IS NULL) AS new_customers
    FROM active_subs
)
SELECT
    %(metric_date)s AS metric_date,
    total_mrr,
    total_mrr * 12 AS arr,
    new_mrr,
    expansion_mrr,
    contraction_mrr,
    new_customers
FROM mrr_breakdown;
"""

SQL_REVENUE_SPLIT = """
WITH subscription_revenue AS (
    SELECT
        DATE_TRUNC('month', DATE %(metric_date)s)::DATE AS revenue_month,
        SUM(s.mrr_amount) AS recurring_revenue
    FROM subscriptions s
    WHERE s.status = 'active'
      AND DATE_TRUNC('month', s.started_at) <= DATE_TRUNC('month', %(metric_date)s::DATE)
    GROUP BY DATE_TRUNC('month', DATE %(metric_date)s)
),
one_time_revenue AS (
    SELECT
        DATE_TRUNC('month', o.order_date)::DATE AS revenue_month,
        SUM(o.total_amount) AS one_time_revenue
    FROM orders o
    WHERE o.status IN ('paid', 'delivered')
      AND DATE_TRUNC('month', o.order_date) = DATE_TRUNC('month', %(metric_date)s::DATE)
      AND o.product_id IN (SELECT product_id FROM products WHERE category IN ('hardware', 'service', 'training'))
    GROUP BY DATE_TRUNC('month', o.order_date)
)
SELECT
    COALESCE(sr.recurring_revenue, 0) AS recurring_revenue,
    COALESCE(otr.one_time_revenue, 0) AS one_time_revenue,
    COALESCE(sr.recurring_revenue, 0) + COALESCE(otr.one_time_revenue, 0) AS total_revenue
FROM subscription_revenue sr
FULL OUTER JOIN one_time_revenue otr ON sr.revenue_month = otr.revenue_month;
"""

SQL_PIPELINE_VELOCITY = """
WITH pipeline_stages AS (
    SELECT
        u.plan_type,
        u.region,
        o.order_id,
        o.amount,
        o.status,
        o.created_at,
        o.updated_at,
        CASE
            WHEN o.status = 'pending' THEN 'lead'
            WHEN o.status = 'paid' AND o.amount < 500 THEN 'opportunity'
            WHEN o.status = 'paid' AND o.amount >= 500 THEN 'negotiation'
            WHEN o.status = 'delivered' THEN 'closed_won'
            WHEN o.status IN ('refunded', 'cancelled') THEN 'closed_lost'
            ELSE 'other'
        END AS pipeline_stage,
        EXTRACT(EPOCH FROM (o.updated_at - o.created_at)) / 86400 AS cycle_days
    FROM orders o
    INNER JOIN users u ON o.user_id = u.user_id
    WHERE o.created_at >= %(start_date)s::TIMESTAMP
)
SELECT
    pipeline_stage,
    COUNT(*) AS deal_count,
    SUM(amount) AS total_amount,
    AVG(amount) AS avg_deal_size,
    AVG(cycle_days) AS avg_cycle_days
FROM pipeline_stages
GROUP BY pipeline_stage
ORDER BY pipeline_stage;
"""

SQL_WIN_RATE = """
WITH deal_outcomes AS (
    SELECT
        u.plan_type,
        u.region,
        o.amount,
        CASE
            WHEN o.status = 'delivered' THEN 'won'
            WHEN o.status IN ('refunded', 'cancelled') THEN 'lost'
            ELSE 'pending'
        END AS outcome
    FROM orders o
    INNER JOIN users u ON o.user_id = u.user_id
    WHERE o.created_at >= %(start_date)s::TIMESTAMP
)
SELECT
    plan_type,
    region,
    COUNT(*) FILTER (WHERE outcome = 'won')::NUMERIC / NULLIF(COUNT(*) FILTER (WHERE outcome IN ('won', 'lost')), 0) AS win_rate,
    COUNT(*) FILTER (WHERE outcome = 'won') AS won_count,
    COUNT(*) FILTER (WHERE outcome = 'lost') AS lost_count,
    AVG(amount) FILTER (WHERE outcome = 'won') AS avg_won_deal_size,
    AVG(amount) FILTER (WHERE outcome = 'lost') AS avg_lost_deal_size
FROM deal_outcomes
GROUP BY plan_type, region;
"""


# ============================================================
# pandas 实现
# ============================================================


class RevenuePipeline:
    """
    收入流水线分析器。

    计算 MRR/ARR、收入拆分、管道速度和收入预测。
    支持按套餐和地区维度拆分。

    Usage:
        pipeline = RevenuePipeline()
        metrics = pipeline.run(subscriptions_df, orders_df, users_df)
    """

    # 套餐月价映射
    PLAN_MONTHLY_PRICES: dict[str, float] = {
        "free": 0.0,
        "starter": 99.0,
        "pro": 299.0,
        "enterprise": 999.0,
    }

    def __init__(
        self,
        metric_date: date | None = None,
        seasonality_factors: dict[int, float] | None = None,
    ) -> None:
        """
        初始化收入流水线分析器。

        Args:
            metric_date: 指标计算日期（默认为今天）
            seasonality_factors: 月度季节性因子（1-12映射0.x-2.x）
        """
        self.metric_date = metric_date or date.today()
        self.seasonality_factors = seasonality_factors or {
            1: 0.85, 2: 0.90, 3: 1.05, 4: 1.10,
            5: 1.05, 6: 0.95, 7: 0.90, 8: 0.95,
            9: 1.10, 10: 1.15, 11: 1.05, 12: 1.10,
        }

    def run(
        self,
        subscriptions_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        users_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        执行完整的收入流水线分析。

        Args:
            subscriptions_df: 订阅数据
            orders_df: 订单数据
            users_df: 用户数据

        Returns:
            包含 revenue_metrics 和 pipeline_metrics 的结果字典
        """
        logger.info("开始收入流水线分析，指标日期: %s", self.metric_date)

        # 计算收入指标
        revenue = self._calculate_revenue_metrics(subscriptions_df, orders_df, users_df)

        # 计算管道速度
        pipeline = self._calculate_pipeline_metrics(orders_df, users_df)

        # 计算赢单率
        win_rates = self._calculate_win_rates(orders_df, users_df)

        # 收入预测
        forecast = self._forecast_revenue(subscriptions_df, orders_df)

        result = {
            "revenue_metrics": revenue,
            "pipeline_metrics": pipeline,
            "win_rates": win_rates,
            "forecast": forecast,
        }

        logger.info(
            "收入分析完成: MRR=%.2f, ARR=%.2f",
            revenue.mrr, revenue.arr,
        )
        return result

    def _calculate_revenue_metrics(
        self,
        subscriptions: pd.DataFrame,
        orders: pd.DataFrame,
        users: pd.DataFrame,
    ) -> RevenueMetrics:
        """
        计算 MRR/ARR 和收入拆分。

        MRR = 所有活跃订阅的月化金额之和
        ARR = MRR × 12
        收入拆分 = 经常性（订阅）vs 一次性（产品订单）

        Args:
            subscriptions: 订阅数据
            orders: 订单数据
            users: 用户数据

        Returns:
            RevenueMetrics 收入指标
        """
        subs = subscriptions.copy()
        subs["started_at"] = pd.to_datetime(subs["started_at"])
        subs["ended_at"] = pd.to_datetime(subs.get("ended_at"))

        # 当前活跃订阅
        active_mask = (
            (subs["status"] == "active")
            & (subs["started_at"].dt.date <= self.metric_date)
            & (subs["ended_at"].isna() | (subs["ended_at"].dt.date > self.metric_date))
        )
        active_subs = subs[active_mask]

        # 计算月化金额
        mrr = 0.0
        if not active_subs.empty and "mrr_amount" in active_subs.columns:
            mrr = float(active_subs["mrr_amount"].sum())
        else:
            # 手动计算月化金额
            for _, sub in active_subs.iterrows():
                amount = float(sub["amount"])
                cycle = sub.get("billing_cycle", "monthly")
                if cycle == "monthly":
                    mrr += amount
                elif cycle == "quarterly":
                    mrr += amount / 3.0
                elif cycle == "yearly":
                    mrr += amount / 12.0

        arr = mrr * 12.0

        # 收入拆分
        recurring_revenue = mrr
        one_time_revenue = 0.0

        if not orders.empty:
            paid_orders = orders[
                (orders["status"].isin(["paid", "delivered"]))
                & (pd.to_datetime(orders["order_date"]).dt.date <= self.metric_date)
            ]
            if not paid_orders.empty and "total_amount" in paid_orders.columns:
                one_time_revenue = float(paid_orders["total_amount"].sum())
            elif not paid_orders.empty:
                one_time_revenue = float(
                    (paid_orders["amount"] * paid_orders["quantity"] * (1 - paid_orders["discount"] / 100)).sum()
                )

        # MRR 变动分析（与上月对比）
        prev_month = self.metric_date - timedelta(days=30)
        prev_active = subs[
            (subs["status"] == "active")
            & (subs["started_at"].dt.date <= prev_month)
            & (subs["ended_at"].isna() | (subs["ended_at"].dt.date > prev_month))
        ]
        prev_mrr = float(prev_active["mrr_amount"].sum()) if not prev_active.empty and "mrr_amount" in prev_active.columns else 0.0

        new_mrr = max(0, mrr - prev_mrr)
        expansion_mrr = new_mrr * 0.6  # 估算：60%为扩展，40%为新增
        churned_mrr = max(0, prev_mrr - mrr)
        net_growth = new_mrr - churned_mrr

        return RevenueMetrics(
            metric_date=self.metric_date,
            mrr=round(mrr, 2),
            arr=round(arr, 2),
            recurring_revenue=round(recurring_revenue, 2),
            one_time_revenue=round(one_time_revenue, 2),
            new_mrr=round(new_mrr, 2),
            expansion_mrr=round(expansion_mrr, 2),
            churned_mrr=round(churned_mrr, 2),
            net_mrr_growth=round(net_growth, 2),
        )

    def _calculate_pipeline_metrics(
        self,
        orders: pd.DataFrame,
        users: pd.DataFrame,
    ) -> list[PipelineMetrics]:
        """
        计算销售管道速度指标。

        管道阶段映射:
            pending → lead
            paid (小额<500) → opportunity
            paid (大额≥500) → negotiation
            delivered → closed_won
            refunded/cancelled → closed_lost

        Args:
            orders: 订单数据
            users: 用户数据

        Returns:
            PipelineMetrics 列表
        """
        if orders.empty:
            return []

        ord = orders.copy()
        ord["created_at"] = pd.to_datetime(ord["created_at"])
        ord["updated_at"] = pd.to_datetime(ord.get("updated_at", ord["created_at"]))

        # 映射管道阶段
        def map_stage(row: dict[str, Any]) -> str:
            status = row.get("status", "pending")
            amount = float(row.get("amount", 0))
            if status == "pending":
                return "lead"
            elif status == "paid" and amount < 500:
                return "opportunity"
            elif status == "paid" and amount >= 500:
                return "negotiation"
            elif status == "delivered":
                return "closed_won"
            elif status in ("refunded", "cancelled"):
                return "closed_lost"
            return "other"

        ord["pipeline_stage"] = ord.apply(map_stage, axis=1)
        ord["cycle_days"] = (ord["updated_at"] - ord["created_at"]).dt.total_seconds() / 86400.0

        # 按阶段聚合
        metrics: list[PipelineMetrics] = []
        for stage in ["lead", "opportunity", "negotiation", "closed_won", "closed_lost"]:
            stage_df = ord[ord["pipeline_stage"] == stage]
            if stage_df.empty:
                metrics.append(PipelineMetrics(stage=stage))
                continue

            count = len(stage_df)
            total = float(stage_df["amount"].sum())
            avg_size = total / count if count > 0 else 0.0

            # 赢单率（仅 closed 阶段计算）
            won_count = len(ord[ord["pipeline_stage"] == "closed_won"])
            closed_count = won_count + len(ord[ord["pipeline_stage"] == "closed_lost"])
            win_rate = won_count / closed_count if closed_count > 0 else 0.0

            avg_cycle = float(stage_df["cycle_days"].mean()) if "cycle_days" in stage_df else 0.0

            metrics.append(PipelineMetrics(
                stage=stage,
                count=count,
                total_amount=round(total, 2),
                avg_deal_size=round(avg_size, 2),
                win_rate=round(win_rate, 4),
                avg_cycle_days=round(avg_cycle, 1),
            ))

        return metrics

    def _calculate_win_rates(
        self,
        orders: pd.DataFrame,
        users: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        按套餐和地区计算赢单率。

        Args:
            orders: 订单数据
            users: 用户数据

        Returns:
            赢单率 DataFrame
        """
        if orders.empty or users.empty:
            return pd.DataFrame()

        merged = orders.merge(users[["user_id", "plan_type", "region"]], on="user_id", how="left")

        def get_outcome(status: str) -> str:
            if status == "delivered":
                return "won"
            elif status in ("refunded", "cancelled"):
                return "lost"
            return "pending"

        merged["outcome"] = merged["status"].apply(get_outcome)
        closed = merged[merged["outcome"].isin(["won", "lost"])]

        if closed.empty:
            return pd.DataFrame(columns=["plan_type", "region", "win_rate", "won", "lost"])

        results: list[dict[str, Any]] = []
        for (plan, region), group in closed.groupby(["plan_type", "region"]):
            won = len(group[group["outcome"] == "won"])
            lost = len(group[group["outcome"] == "lost"])
            total = won + lost
            results.append({
                "plan_type": plan,
                "region": region,
                "win_rate": round(won / total, 4) if total > 0 else 0.0,
                "won": won,
                "lost": lost,
                "avg_won_amount": round(float(group[group["outcome"] == "won"]["amount"].mean()), 2) if won > 0 else 0.0,
            })

        return pd.DataFrame(results)

    def _forecast_revenue(
        self,
        subscriptions: pd.DataFrame,
        orders: pd.DataFrame,
        forecast_months: int = 6,
    ) -> dict[str, Any]:
        """
        基于移动平均和季节性调整的收入预测。

        算法:
            1. 计算过去 N 个月的 MRR 时间序列
            2. 计算3个月移动平均趋势
            3. 应用月度季节性因子调整
            4. 线性外推预测未来月份

        Args:
            subscriptions: 订阅数据
            orders: 订单数据
            forecast_months: 预测月数

        Returns:
            预测结果字典
        """
        if subscriptions.empty:
            return {"forecasts": [], "method": "insufficient_data"}

        subs = subscriptions.copy()
        subs["started_at"] = pd.to_datetime(subs["started_at"])
        subs["ended_at"] = pd.to_datetime(subs.get("ended_at"))

        # 构建过去6个月的 MRR 序列
        monthly_mrr: list[float] = []
        monthly_dates: list[date] = []

        for i in range(5, -1, -1):
            target = self.metric_date - timedelta(days=30 * i)
            target_month = target.replace(day=1)

            active = subs[
                (subs["status"] == "active")
                & (subs["started_at"].dt.date <= target)
                & (subs["ended_at"].isna() | (subs["ended_at"].dt.date > target))
            ]

            mrr = float(active["mrr_amount"].sum()) if not active.empty and "mrr_amount" in active.columns else 0.0
            monthly_mrr.append(mrr)
            monthly_dates.append(target_month)

        if not monthly_mrr or all(m == 0 for m in monthly_mrr):
            return {"forecasts": [], "method": "no_active_subs"}

        # 计算3个月移动平均增长率
        growth_rates: list[float] = []
        for i in range(1, len(monthly_mrr)):
            if monthly_mrr[i - 1] > 0:
                growth_rates.append((monthly_mrr[i] - monthly_mrr[i - 1]) / monthly_mrr[i - 1])

        avg_growth = float(np.mean(growth_rates)) if growth_rates else 0.0

        # 最近 MRR 作为基准
        base_mrr = monthly_mrr[-1]

        # 预测未来月份
        forecasts: list[dict[str, Any]] = []
        for m in range(1, forecast_months + 1):
            forecast_date = self.metric_date + timedelta(days=30 * m)
            forecast_month = forecast_date.month

            # 基础预测 = 基准 × (1 + 增长率)^m
            base_forecast = base_mrr * ((1 + avg_growth) ** m)

            # 季节性调整
            season_factor = self.seasonality_factors.get(forecast_month, 1.0)
            adjusted_forecast = base_forecast * season_factor

            forecasts.append({
                "forecast_month": forecast_date.replace(day=1).isoformat(),
                "forecasted_mrr": round(max(0, adjusted_forecast), 2),
                "forecasted_arr": round(max(0, adjusted_forecast * 12), 2),
                "growth_rate_applied": round(avg_growth, 4),
                "seasonality_factor": season_factor,
                "method": "moving_avg_seasonal",
            })

        return {
            "historical_mrr": [round(m, 2) for m in monthly_mrr],
            "historical_dates": [d.isoformat() for d in monthly_dates],
            "avg_monthly_growth": round(avg_growth, 4),
            "forecasts": forecasts,
            "method": "moving_avg_seasonal",
        }

    def run_sql(
        self,
        db_connection: DBConnection,
    ) -> dict[str, pd.DataFrame]:
        """
        使用 SQL 执行收入分析。

        Args:
            db_connection: 数据库连接

        Returns:
            各查询结果的 DataFrame 字典
        """
        cursor = db_connection.cursor()
        results: dict[str, pd.DataFrame] = {}

        queries = {
            "mrr": (SQL_MRR_CALCULATION, {"metric_date": self.metric_date}),
            "revenue_split": (SQL_REVENUE_SPLIT, {"metric_date": self.metric_date}),
            "pipeline": (SQL_PIPELINE_VELOCITY, {"start_date": self.metric_date - timedelta(days=90)}),
            "win_rate": (SQL_WIN_RATE, {"start_date": self.metric_date - timedelta(days=90)}),
        }

        for name, (sql, params) in queries.items():
            try:
                cursor.execute(sql, params)
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                results[name] = pd.DataFrame(rows, columns=columns)
                logger.info("SQL %s 查询完成，%d 行", name, len(results[name]))
            except Exception as e:
                logger.error("SQL %s 查询失败: %s", name, e)
                results[name] = pd.DataFrame()

        cursor.close()
        return results

    def compute_arpu_arppu(
        self,
        subscriptions: pd.DataFrame,
        users: pd.DataFrame,
    ) -> dict[str, float]:
        """
        计算 ARPU（每用户平均收入）和 ARPPU（每付费用户平均收入）。

        ARPU = MRR / 总活跃用户数
        ARPPU = MRR / 付费用户数

        Args:
            subscriptions: 订阅数据
            users: 用户数据

        Returns:
            ARPU 和 ARPPU 指标
        """
        active_users = users[users["status"].isin(["active", "trial"])]
        total_active = len(active_users)

        subs = subscriptions.copy()
        subs["started_at"] = pd.to_datetime(subs["started_at"])
        subs["ended_at"] = pd.to_datetime(subs.get("ended_at"))

        active_subs = subs[
            (subs["status"] == "active")
            & (subs["started_at"].dt.date <= self.metric_date)
            & (subs["ended_at"].isna() | (subs["ended_at"].dt.date > self.metric_date))
        ]

        mrr = float(active_subs["mrr_amount"].sum()) if not active_subs.empty and "mrr_amount" in active_subs.columns else 0.0
        paying_users = active_subs["user_id"].nunique()

        arpu = round(mrr / total_active, 2) if total_active > 0 else 0.0
        arppu = round(mrr / paying_users, 2) if paying_users > 0 else 0.0

        return {
            "arpu": arpu,
            "arppu": arppu,
            "total_active_users": total_active,
            "paying_users": paying_users,
            "mrr": round(mrr, 2),
        }
