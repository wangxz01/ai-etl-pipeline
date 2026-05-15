"""
销售分析模块测试套件

覆盖收入流水线（MRR/ARR）、管道速度和客户 LTV 的单元测试。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from src.etl.sales.cohort_ltv import CohortLTVAnalyzer, CohortLTVResult
from src.etl.sales.revenue_pipeline import (
    RevenueMetrics,
    RevenuePipeline,
    PipelineMetrics,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def sample_users() -> pd.DataFrame:
    """生成模拟用户数据。"""
    users: list[dict[str, Any]] = []
    for i in range(200):
        users.append({
            "user_id": str(uuid4()),
            "email": f"user{i}@example.com",
            "username": f"用户{i}",
            "phone": f"138{i:08d}",
            "signup_date": date(2025, 1, 1) + timedelta(days=i % 90),
            "plan_type": ["free", "starter", "pro", "enterprise"][i % 4],
            "region": ["CN", "US", "EU"][i % 3],
            "status": "active" if i % 5 != 0 else "churned",
            "mrr": [0, 99, 299, 999][i % 4],
            "timezone": "Asia/Shanghai",
            "created_at": datetime(2025, 1, 1),
            "updated_at": datetime(2025, 1, 1),
        })
    return pd.DataFrame(users)


@pytest.fixture
def sample_subscriptions(sample_users: pd.DataFrame) -> pd.DataFrame:
    """生成模拟订阅数据。"""
    subs: list[dict[str, Any]] = []
    paying = sample_users[sample_users["plan_type"] != "free"]

    for _, user in paying.iterrows():
        plan = user["plan_type"]
        prices = {"starter": 99, "pro": 299, "enterprise": 999}
        amount = prices[plan]
        subs.append({
            "sub_id": str(uuid4()),
            "user_id": user["user_id"],
            "plan": plan,
            "amount": float(amount),
            "currency": "CNY",
            "billing_cycle": "monthly",
            "started_at": datetime.combine(user["signup_date"], datetime.min.time()),
            "ended_at": None,
            "cancelled_at": None,
            "cancel_reason": None,
            "status": "active" if user["status"] == "active" else "cancelled",
            "mrr_amount": float(amount),
            "created_at": datetime(2025, 1, 1),
            "updated_at": datetime(2025, 1, 1),
        })
    return pd.DataFrame(subs)


@pytest.fixture
def sample_orders(sample_users: pd.DataFrame) -> pd.DataFrame:
    """生成模拟订单数据。"""
    orders: list[dict[str, Any]] = []
    products = [
        {"product_id": str(uuid4()), "price": 99.0, "category": "software"},
        {"product_id": str(uuid4()), "price": 299.0, "category": "addon"},
        {"product_id": str(uuid4()), "price": 2999.0, "category": "hardware"},
    ]

    for _, user in sample_users.iterrows():
        num_orders = np.random.randint(0, 4)
        for _ in range(num_orders):
            product = products[np.random.randint(0, len(products))]
            order_date = user["signup_date"] + timedelta(days=np.random.randint(1, 60))
            quantity = np.random.randint(1, 4)
            status = np.random.choice(
                ["pending", "paid", "delivered", "refunded", "cancelled"],
                p=[0.05, 0.15, 0.65, 0.08, 0.07],
            )
            amount = product["price"]
            discount = np.random.choice([0, 5, 10, 15], p=[0.5, 0.25, 0.15, 0.10])
            total = round(amount * quantity * (1 - discount / 100), 2)

            orders.append({
                "order_id": str(uuid4()),
                "user_id": user["user_id"],
                "product_id": product["product_id"],
                "amount": amount,
                "quantity": quantity,
                "discount": discount,
                "tax": round(amount * quantity * 0.06, 2),
                "shipping_cost": 0,
                "total_amount": total,
                "currency": "CNY",
                "payment_method": "alipay",
                "status": status,
                "order_date": order_date,
                "fulfilled_date": order_date + timedelta(days=3) if status == "delivered" else None,
                "created_at": datetime.combine(order_date, datetime.min.time()),
                "updated_at": datetime.combine(
                    order_date + timedelta(days=3 if status == "delivered" else 0),
                    datetime.min.time(),
                ),
            })
    return pd.DataFrame(orders)


@pytest.fixture
def sample_products() -> pd.DataFrame:
    """生成模拟产品数据。"""
    return pd.DataFrame([
        {"product_id": str(uuid4()), "name": "基础版", "category": "software", "price": 99.0, "cost": 15.0, "sku": "SaaS-001"},
        {"product_id": str(uuid4()), "name": "专业版", "category": "software", "price": 299.0, "cost": 40.0, "sku": "SaaS-002"},
        {"product_id": str(uuid4()), "name": "网关", "category": "hardware", "price": 2999.0, "cost": 1200.0, "sku": "HW-001"},
    ])


# ============================================================
# MRR/ARR 计算测试
# ============================================================


class TestMRRARRCalculation:
    """MRR/ARR 计算测试。"""

    def test_basic_mrr(
        self,
        sample_subscriptions: pd.DataFrame,
        sample_orders: pd.DataFrame,
        sample_users: pd.DataFrame,
    ) -> None:
        """测试基本 MRR 计算。"""
        pipeline = RevenuePipeline(metric_date=date(2025, 3, 1))
        result = pipeline.run(sample_subscriptions, sample_orders, sample_users)

        metrics = result["revenue_metrics"]
        assert isinstance(metrics, RevenueMetrics)
        assert metrics.mrr >= 0
        assert metrics.arr == metrics.mrr * 12
        assert metrics.metric_date == date(2025, 3, 1)

    def test_mrr_equals_active_subs(
        self,
        sample_subscriptions: pd.DataFrame,
        sample_orders: pd.DataFrame,
        sample_users: pd.DataFrame,
    ) -> None:
        """测试 MRR 等于活跃订阅月化金额之和。"""
        pipeline = RevenuePipeline(metric_date=date(2025, 3, 1))
        result = pipeline.run(sample_subscriptions, sample_orders, sample_users)
        metrics = result["revenue_metrics"]

        active = sample_subscriptions[
            (sample_subscriptions["status"] == "active")
        ]
        expected_mrr = float(active["mrr_amount"].sum())

        assert abs(metrics.mrr - expected_mrr) < 1.0

    def test_revenue_split(
        self,
        sample_subscriptions: pd.DataFrame,
        sample_orders: pd.DataFrame,
        sample_users: pd.DataFrame,
    ) -> None:
        """测试经常性 vs 一次性收入拆分。"""
        pipeline = RevenuePipeline(metric_date=date(2025, 3, 1))
        result = pipeline.run(sample_subscriptions, sample_orders, sample_users)
        metrics = result["revenue_metrics"]

        assert metrics.recurring_revenue >= 0
        assert metrics.one_time_revenue >= 0

    def test_mrr_growth_calculation(
        self,
        sample_subscriptions: pd.DataFrame,
        sample_orders: pd.DataFrame,
        sample_users: pd.DataFrame,
    ) -> None:
        """测试 MRR 增长指标。"""
        pipeline = RevenuePipeline(metric_date=date(2025, 3, 1))
        result = pipeline.run(sample_subscriptions, sample_orders, sample_users)
        metrics = result["revenue_metrics"]

        # net_mrr_growth = new_mrr - churned_mrr
        assert metrics.net_mrr_growth == round(
            metrics.new_mrr - metrics.churned_mrr, 2
        )

    def test_empty_subscriptions(
        self,
        sample_orders: pd.DataFrame,
        sample_users: pd.DataFrame,
    ) -> None:
        """测试空订阅数据时 MRR 为零。"""
        empty_subs = pd.DataFrame(columns=[
            "sub_id", "user_id", "plan", "amount", "billing_cycle",
            "started_at", "ended_at", "status", "mrr_amount",
        ])
        pipeline = RevenuePipeline(metric_date=date(2025, 3, 1))
        result = pipeline.run(empty_subs, sample_orders, sample_users)

        assert result["revenue_metrics"].mrr == 0.0
        assert result["revenue_metrics"].arr == 0.0


# ============================================================
# 管道速度测试
# ============================================================


class TestPipelineVelocity:
    """销售管道速度测试。"""

    def test_pipeline_stages(
        self,
        sample_orders: pd.DataFrame,
        sample_users: pd.DataFrame,
    ) -> None:
        """测试管道阶段聚合。"""
        pipeline = RevenuePipeline(metric_date=date(2025, 3, 1))
        result = pipeline.run(
            pd.DataFrame(), sample_orders, sample_users,
        )

        # 没有订阅数据时管道仍能计算
        pipeline_metrics = result["pipeline_metrics"]
        assert isinstance(pipeline_metrics, list)

        stages = {m.stage for m in pipeline_metrics}
        expected_stages = {"lead", "opportunity", "negotiation", "closed_won", "closed_lost"}
        assert stages == expected_stages

    def test_win_rate_calculation(
        self,
        sample_orders: pd.DataFrame,
        sample_users: pd.DataFrame,
    ) -> None:
        """测试赢单率计算。"""
        pipeline = RevenuePipeline(metric_date=date(2025, 3, 1))
        result = pipeline.run(
            pd.DataFrame(), sample_orders, sample_users,
        )
        win_rates = result["win_rates"]

        assert isinstance(win_rates, pd.DataFrame)
        if not win_rates.empty:
            assert "win_rate" in win_rates.columns
            assert "plan_type" in win_rates.columns
            assert "region" in win_rates.columns
            assert (win_rates["win_rate"] >= 0).all()
            assert (win_rates["win_rate"] <= 1).all()

    def test_deal_size_positive(
        self,
        sample_orders: pd.DataFrame,
        sample_users: pd.DataFrame,
    ) -> None:
        """测试平均交易规模为正值。"""
        pipeline = RevenuePipeline(metric_date=date(2025, 3, 1))
        result = pipeline.run(
            pd.DataFrame(), sample_orders, sample_users,
        )

        for metric in result["pipeline_metrics"]:
            if metric.count > 0:
                assert metric.avg_deal_size > 0
                assert metric.total_amount > 0

    def test_empty_orders(self) -> None:
        """测试空订单数据。"""
        pipeline = RevenuePipeline(metric_date=date(2025, 3, 1))
        empty_orders = pd.DataFrame(columns=[
            "order_id", "user_id", "product_id", "amount", "quantity",
            "status", "order_date", "created_at", "updated_at",
        ])
        empty_users = pd.DataFrame(columns=["user_id", "plan_type", "region"])
        metrics = pipeline._calculate_pipeline_metrics(empty_orders, empty_users)

        assert len(metrics) == 0


# ============================================================
# LTV 分析测试
# ============================================================


class TestCohortLTV:
    """客户 LTV 分析测试。"""

    def test_basic_ltv(
        self,
        sample_orders: pd.DataFrame,
        sample_users: pd.DataFrame,
    ) -> None:
        """测试基本 LTV 计算。"""
        analyzer = CohortLTVAnalyzer(projection_months=6)
        results = analyzer.run(sample_orders, users_df=sample_users)

        assert isinstance(results, list)
        if results:
            for r in results:
                assert isinstance(r, CohortLTVResult)
                assert r.cohort_size > 0
                assert r.actual_ltv >= 0
                assert r.projected_ltv >= 0

    def test_ltv_with_subscriptions(
        self,
        sample_orders: pd.DataFrame,
        sample_subscriptions: pd.DataFrame,
        sample_users: pd.DataFrame,
    ) -> None:
        """测试包含订阅数据的 LTV 计算。"""
        analyzer = CohortLTVAnalyzer(projection_months=6)
        results = analyzer.run(
            sample_orders,
            subscriptions_df=sample_subscriptions,
            users_df=sample_users,
        )

        for r in results:
            assert r.cohort_month is not None
            assert len(r.months_data) <= 7  # 0-6 months

    def test_projected_ltv_gte_actual(
        self,
        sample_orders: pd.DataFrame,
        sample_users: pd.DataFrame,
    ) -> None:
        """测试预测 LTV >= 实际累计 LTV。"""
        analyzer = CohortLTVAnalyzer(projection_months=12, min_cohort_size=3)
        results = analyzer.run(sample_orders, users_df=sample_users)

        for r in results:
            assert r.projected_ltv >= r.actual_ltv

    def test_cumulative_ltv_increasing(
        self,
        sample_orders: pd.DataFrame,
        sample_users: pd.DataFrame,
    ) -> None:
        """测试累计 LTV 随月份递增。"""
        analyzer = CohortLTVAnalyzer(projection_months=6)
        results = analyzer.run(sample_orders, users_df=sample_users)

        for r in results:
            if len(r.months_data) >= 2:
                cum_ltv_values = [m["cum_ltv"] for m in r.months_data]
                for i in range(1, len(cum_ltv_values)):
                    assert cum_ltv_values[i] >= cum_ltv_values[i - 1]

    def test_summary_generation(
        self,
        sample_orders: pd.DataFrame,
        sample_users: pd.DataFrame,
    ) -> None:
        """测试 LTV 摘要表生成。"""
        analyzer = CohortLTVAnalyzer(projection_months=6)
        results = analyzer.run(sample_orders, users_df=sample_users)
        summary = analyzer.compute_summary(results)

        assert isinstance(summary, pd.DataFrame)
        if not summary.empty:
            assert "cohort_month" in summary.columns
            assert "actual_ltv" in summary.columns
            assert "projected_ltv" in summary.columns
            assert "cohort_size" in summary.columns

    def test_min_cohort_size_filter(
        self,
        sample_users: pd.DataFrame,
    ) -> None:
        """测试最小群组大小过滤。"""
        # 创建极少的订单数据
        orders = pd.DataFrame([
            {"order_id": str(uuid4()), "user_id": sample_users.iloc[0]["user_id"],
             "product_id": str(uuid4()), "amount": 100.0, "quantity": 1,
             "discount": 0, "status": "paid",
             "order_date": date(2025, 1, 15),
             "created_at": datetime(2025, 1, 15), "updated_at": datetime(2025, 1, 15)},
        ])
        analyzer = CohortLTVAnalyzer(min_cohort_size=10)
        results = analyzer.run(orders, users_df=sample_users)

        for r in results:
            assert r.cohort_size >= 10

    def test_empty_orders(self) -> None:
        """测试空订单数据。"""
        analyzer = CohortLTVAnalyzer()
        empty_orders = pd.DataFrame(columns=[
            "order_id", "user_id", "product_id", "amount",
            "order_date", "created_at",
        ])
        results = analyzer.run(empty_orders)
        assert len(results) == 0


# ============================================================
# ARPU/ARPPU 测试
# ============================================================


class TestARPUARPPU:
    """ARPU/ARPPU 计算测试。"""

    def test_arpu_arppu(
        self,
        sample_subscriptions: pd.DataFrame,
        sample_users: pd.DataFrame,
    ) -> None:
        """测试 ARPU 和 ARPPU 计算。"""
        pipeline = RevenuePipeline(metric_date=date(2025, 3, 1))
        result = pipeline.compute_arpu_arppu(sample_subscriptions, sample_users)

        assert "arpu" in result
        assert "arppu" in result
        assert result["arpu"] >= 0
        assert result["arppu"] >= 0
        # ARPPU >= ARPU（付费用户人均 >= 全体用户人均）
        if result["paying_users"] > 0:
            assert result["arppu"] >= result["arpu"]

    def test_arpu_formula(
        self,
        sample_subscriptions: pd.DataFrame,
        sample_users: pd.DataFrame,
    ) -> None:
        """测试 ARPU 计算公式正确性。"""
        pipeline = RevenuePipeline(metric_date=date(2025, 3, 1))
        result = pipeline.compute_arpu_arppu(sample_subscriptions, sample_users)

        active = sample_users[sample_users["status"].isin(["active", "trial"])]
        total_active = len(active)

        if total_active > 0 and result["mrr"] > 0:
            expected_arpu = round(result["mrr"] / total_active, 2)
            assert abs(result["arpu"] - expected_arpu) < 1.0
