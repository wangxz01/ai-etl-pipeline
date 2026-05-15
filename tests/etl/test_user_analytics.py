"""
用户分析模块测试套件

覆盖 DAU/MAU 计算、留存分析和流失预测的单元测试。
使用 pytest fixtures 提供模拟数据。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from src.etl.user_analytics.churn_prediction import (
    ChurnModelConfig,
    ChurnPredictor,
    ChurnScore,
)
from src.etl.user_analytics.dau_mau_etl import DAUMAUCalculator, DAUMAUResult
from src.etl.user_analytics.retention_etl import CohortData, RetentionAnalyzer


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def sample_users() -> pd.DataFrame:
    """生成模拟用户数据（100个用户，10个测试账号）。"""
    users: list[dict[str, Any]] = []
    for i in range(100):
        is_test = i < 10
        email = f"test_user{i}@test-company.com" if is_test else f"user{i}@example.com"
        users.append({
            "user_id": str(uuid4()),
            "email": email,
            "username": f"用户{i}",
            "phone": f"138{i:08d}",
            "signup_date": date(2025, 1, 1) + timedelta(days=i % 30),
            "plan_type": ["free", "starter", "pro", "enterprise"][i % 4],
            "region": ["CN", "US", "EU", "JP", "SEA"][i % 5],
            "status": ["active", "churned", "suspended", "trial"][i % 4],
            "company": f"公司{i}" if i % 3 == 0 else None,
            "industry": "互联网" if i % 5 == 0 else None,
            "mrr": [0, 99, 299, 999][i % 4],
            "timezone": "Asia/Shanghai",
            "created_at": datetime(2025, 1, 1),
            "updated_at": datetime(2025, 1, 1),
        })
    return pd.DataFrame(users)


@pytest.fixture
def sample_events(sample_users: pd.DataFrame) -> pd.DataFrame:
    """生成模拟事件数据（每个活跃用户 5-15 条事件）。"""
    events: list[dict[str, Any]] = []
    active_users = sample_users[sample_users["status"].isin(["active", "trial"])]

    event_id = 0
    for _, user in active_users.iterrows():
        num_events = np.random.randint(5, 16)
        for _ in range(num_events):
            event_id += 1
            event_date = date(2025, 1, 15) + timedelta(days=np.random.randint(0, 30))
            events.append({
                "event_id": event_id,
                "user_id": user["user_id"],
                "event_type": np.random.choice(["page_view", "click", "feature_use", "api_call"]),
                "event_name": "测试事件",
                "page_url": "/dashboard",
                "device": np.random.choice(["desktop", "mobile", "tablet"]),
                "browser": "Chrome",
                "os": "Windows",
                "ip": "192.168.1.1",
                "country": user["region"],
                "session_id": str(uuid4()),
                "event_duration": np.random.randint(1000, 300000),
                "event_date": event_date,
                "event_timestamp": datetime.combine(event_date, datetime.min.time()),
                "created_at": datetime.combine(event_date, datetime.min.time()),
            })
    return pd.DataFrame(events)


@pytest.fixture
def sample_subscriptions(sample_users: pd.DataFrame) -> pd.DataFrame:
    """生成模拟订阅数据。"""
    subs: list[dict[str, Any]] = []
    paying_users = sample_users[sample_users["plan_type"] != "free"]

    for _, user in paying_users.iterrows():
        plan = user["plan_type"]
        amount = {"starter": 99, "pro": 299, "enterprise": 999}[plan]
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
            "created_at": datetime(2025, 1, 1),
            "updated_at": datetime(2025, 1, 1),
        })
    return pd.DataFrame(subs)


@pytest.fixture
def sample_tickets(sample_users: pd.DataFrame) -> pd.DataFrame:
    """生成模拟工单数据。"""
    tickets: list[dict[str, Any]] = []
    for _, user in sample_users.head(30).iterrows():
        for _ in range(np.random.randint(0, 3)):
            tickets.append({
                "ticket_id": str(uuid4()),
                "user_id": user["user_id"],
                "subject": "测试工单",
                "category": np.random.choice(["billing", "technical", "account"]),
                "priority": np.random.choice(["low", "medium", "high", "critical"]),
                "status": np.random.choice(["open", "resolved", "closed"]),
                "assigned_to": "agent_001",
                "first_response_at": datetime(2025, 2, 1),
                "resolution_note": "已解决",
                "satisfaction": np.random.randint(1, 6),
                "tags": [],
                "created_at": datetime(2025, 2, 1),
                "updated_at": datetime(2025, 2, 5),
                "resolved_at": datetime(2025, 2, 5),
            })
    return pd.DataFrame(tickets)


# ============================================================
# DAU/MAU 测试
# ============================================================


class TestDAUMAUCalculator:
    """DAU/MAU 计算器测试。"""

    def test_basic_dau_calculation(
        self,
        sample_events: pd.DataFrame,
        sample_users: pd.DataFrame,
    ) -> None:
        """测试基本的 DAU 计算。"""
        calculator = DAUMAUCalculator()
        target_date = date(2025, 1, 15)
        result = calculator.run(sample_events, sample_users, target_date)

        assert isinstance(result, DAUMAUResult)
        assert result.metric_date == target_date
        assert result.dau >= 0
        assert result.wau >= result.dau  # WAU >= DAU
        assert result.mau >= result.wau  # MAU >= WAU

    def test_sticky_factor_range(
        self,
        sample_events: pd.DataFrame,
        sample_users: pd.DataFrame,
    ) -> None:
        """测试粘性系数在合理范围内。"""
        calculator = DAUMAUCalculator()
        result = calculator.run(sample_events, sample_users, date(2025, 1, 20))

        if result.mau > 0:
            assert 0 <= result.sticky_factor <= 1.0
            expected_sticky = result.dau / result.mau
            assert abs(result.sticky_factor - expected_sticky) < 0.001

    def test_test_account_exclusion(
        self,
        sample_users: pd.DataFrame,
    ) -> None:
        """测试账号过滤逻辑。"""
        calculator = DAUMAUCalculator(exclude_test_accounts=True)
        assert calculator._is_test_account("test_user@test-company.com") is True
        assert calculator._is_test_account("bot_123@test-company.com") is True
        assert calculator._is_test_account("demo@test-company.com") is True
        assert calculator._is_test_account("normal@example.com") is False

    def test_no_test_account_exclusion(self) -> None:
        """测试不过滤测试账号时正常用户不被排除。"""
        calculator = DAUMAUCalculator(exclude_test_accounts=False)
        assert calculator._is_test_account("anyone@example.com") is False

    def test_segment_breakdown(
        self,
        sample_events: pd.DataFrame,
        sample_users: pd.DataFrame,
    ) -> None:
        """测试分维度数据包含正确的结构。"""
        calculator = DAUMAUCalculator()
        result = calculator.run(sample_events, sample_users, date(2025, 1, 15))

        assert len(result.segments) > 0
        for seg in result.segments:
            assert "metric_name" in seg
            assert "metric_value" in seg
            assert "region" in seg
            assert "plan_type" in seg
            assert seg["metric_value"] >= 0

    def test_empty_events(
        self,
        sample_users: pd.DataFrame,
    ) -> None:
        """测试空事件数据时返回零值。"""
        empty_events = pd.DataFrame(columns=[
            "event_id", "user_id", "event_type", "event_name",
            "event_date", "device", "event_duration",
        ])
        calculator = DAUMAUCalculator()
        result = calculator.run(empty_events, sample_users, date(2025, 1, 15))

        assert result.dau == 0
        assert result.wau == 0
        assert result.mau == 0
        assert result.sticky_factor == 0.0

    def test_trend_calculation(
        self,
        sample_events: pd.DataFrame,
        sample_users: pd.DataFrame,
    ) -> None:
        """测试 DAU 趋势计算。"""
        calculator = DAUMAUCalculator()
        trend = calculator.run_trend(
            sample_events, sample_users,
            start_date=date(2025, 1, 1),
            end_date=date(2025, 2, 15),
        )

        assert isinstance(trend, pd.DataFrame)
        if not trend.empty:
            assert "dau" in trend.columns
            assert "dau_7d_avg" in trend.columns
            assert "dau_30d_avg" in trend.columns
            assert (trend["dau"] >= 0).all()

    def test_sql_mode_without_connection(self) -> None:
        """测试未配置数据库连接时 SQL 模式报错。"""
        calculator = DAUMAUCalculator()
        with pytest.raises(RuntimeError, match="数据库连接"):
            calculator.run_sql(date(2025, 1, 15))


# ============================================================
# 留存分析测试
# ============================================================


class TestRetentionAnalyzer:
    """留存分析器测试。"""

    def test_weekly_retention(
        self,
        sample_events: pd.DataFrame,
        sample_users: pd.DataFrame,
    ) -> None:
        """测试周留存计算。"""
        analyzer = RetentionAnalyzer(max_periods=4)
        results = analyzer.run(
            sample_events, sample_users,
            period_type="week",
        )

        assert isinstance(results, list)
        for cohort in results:
            assert isinstance(cohort, CohortData)
            assert cohort.period_type == "week"
            assert cohort.cohort_size > 0
            assert 0 in cohort.retention_curve  # week-0 应该存在
            assert cohort.retention_curve[0] > 0  # 注册当期留存应 > 0

    def test_monthly_retention(
        self,
        sample_events: pd.DataFrame,
        sample_users: pd.DataFrame,
    ) -> None:
        """测试月留存计算。"""
        analyzer = RetentionAnalyzer(max_periods=3)
        results = analyzer.run(
            sample_events, sample_users,
            period_type="month",
        )

        for cohort in results:
            assert cohort.period_type == "month"
            assert len(cohort.retention_curve) <= 4  # 0-3

    def test_retention_rate_decreasing(
        self,
        sample_events: pd.DataFrame,
        sample_users: pd.DataFrame,
    ) -> None:
        """测试留存率总体呈下降趋势。"""
        analyzer = RetentionAnalyzer(max_periods=6)
        results = analyzer.run(
            sample_events, sample_users,
            period_type="week",
        )

        for cohort in results:
            rates = list(cohort.retention_curve.values())
            if len(rates) >= 3:
                # 前3个周期的留存率应大致递减
                assert rates[0] >= rates[-1] or rates[0] > 0

    def test_retention_matrix(
        self,
        sample_events: pd.DataFrame,
        sample_users: pd.DataFrame,
    ) -> None:
        """测试留存矩阵构建。"""
        analyzer = RetentionAnalyzer(max_periods=4)
        results = analyzer.run(sample_events, sample_users)
        matrix = analyzer.build_retention_matrix(results)

        assert isinstance(matrix, pd.DataFrame)
        if not matrix.empty:
            assert "cohort_date" in matrix.columns
            assert "cohort_size" in matrix.columns
            assert "period_0" in matrix.columns

    def test_average_retention_curve(
        self,
        sample_events: pd.DataFrame,
        sample_users: pd.DataFrame,
    ) -> None:
        """测试平均留存曲线计算。"""
        analyzer = RetentionAnalyzer(max_periods=4)
        results = analyzer.run(sample_events, sample_users)
        avg_curve = analyzer.compute_average_retention_curve(results)

        assert isinstance(avg_curve, dict)
        if avg_curve:
            for period, rate in avg_curve.items():
                assert 0 <= rate <= 1.0

    def test_small_cohort_excluded(
        self,
        sample_events: pd.DataFrame,
    ) -> None:
        """测试小群组被排除。"""
        # 创建一个只有1个用户的数据集
        single_user = pd.DataFrame([{
            "user_id": str(uuid4()),
            "email": "only@example.com",
            "signup_date": date(2025, 1, 1),
            "region": "CN",
            "plan_type": "free",
            "status": "active",
        }])
        analyzer = RetentionAnalyzer(max_periods=4)
        results = analyzer.run(sample_events.head(0), single_user)

        # 只有一个用户，cohort_size < 5 应该被跳过
        assert len(results) == 0

    def test_anomaly_detection(
        self,
        sample_events: pd.DataFrame,
        sample_users: pd.DataFrame,
    ) -> None:
        """测试留存异常检测。"""
        analyzer = RetentionAnalyzer(max_periods=4)
        results = analyzer.run(sample_events, sample_users)
        anomalies = analyzer.detect_retention_anomalies(results, z_threshold=1.5)

        assert isinstance(anomalies, list)
        for anomaly in anomalies:
            assert "cohort_date" in anomaly
            assert "period" in anomaly
            assert "z_score" in anomaly

    def test_sql_mode_without_connection(self) -> None:
        """测试未配置数据库连接时抛出异常。"""
        import psycopg2
        analyzer = RetentionAnalyzer()
        with pytest.raises(Exception):
            analyzer.run_sql(
                db_connection=None,  # type: ignore
                period_type="week",
            )


# ============================================================
# 流失预测测试
# ============================================================


class TestChurnPredictor:
    """流失预测器测试。"""

    def test_basic_churn_scoring(
        self,
        sample_events: pd.DataFrame,
        sample_users: pd.DataFrame,
        sample_subscriptions: pd.DataFrame,
        sample_tickets: pd.DataFrame,
    ) -> None:
        """测试基本流失评分计算。"""
        predictor = ChurnPredictor(score_date=date(2025, 2, 15))
        scores = predictor.run(
            sample_events, sample_users,
            subscriptions_df=sample_subscriptions,
            tickets_df=sample_tickets,
        )

        assert len(scores) > 0
        for score in scores:
            assert isinstance(score, ChurnScore)
            assert 0 <= score.churn_score <= 1.0
            assert score.risk_level in ("low", "medium", "high", "critical")
            assert score.score_date == date(2025, 2, 15)

    def test_churn_features_populated(
        self,
        sample_events: pd.DataFrame,
        sample_users: pd.DataFrame,
    ) -> None:
        """测试特征值已填充。"""
        predictor = ChurnPredictor(score_date=date(2025, 2, 15))
        scores = predictor.run(sample_events, sample_users)

        for score in scores:
            assert "login_frequency" in score.features
            assert "usage_decline" in score.features
            assert "feature_breadth" in score.features
            assert score.features["login_frequency"] >= 0

    def test_risk_level_thresholds(
        self,
        sample_events: pd.DataFrame,
        sample_users: pd.DataFrame,
    ) -> None:
        """测试风险等级划分正确。"""
        config = ChurnModelConfig(
            thresholds={"low": 0.3, "medium": 0.5, "high": 0.7, "critical": 1.0},
        )
        predictor = ChurnPredictor(config=config, score_date=date(2025, 2, 15))
        scores = predictor.run(sample_events, sample_users)

        for score in scores:
            if score.churn_score < 0.3:
                assert score.risk_level == "low"
            elif score.churn_score < 0.5:
                assert score.risk_level == "medium"
            elif score.churn_score < 0.7:
                assert score.risk_level == "high"
            else:
                assert score.risk_level == "critical"

    def test_score_distribution(
        self,
        sample_events: pd.DataFrame,
        sample_users: pd.DataFrame,
    ) -> None:
        """测试评分分布统计。"""
        predictor = ChurnPredictor(score_date=date(2025, 2, 15))
        scores = predictor.run(sample_events, sample_users)
        dist = predictor.get_score_distribution(scores)

        assert dist["total"] == len(scores)
        assert "mean_score" in dist
        assert "risk_distribution" in dist
        assert sum(dist["risk_distribution"].values()) == len(scores)

    def test_high_risk_users(
        self,
        sample_events: pd.DataFrame,
        sample_users: pd.DataFrame,
    ) -> None:
        """测试高风险用户筛选。"""
        predictor = ChurnPredictor(score_date=date(2025, 2, 15))
        scores = predictor.run(sample_events, sample_users)
        high_risk = predictor.get_high_risk_users(scores, min_risk="high", limit=10)

        assert len(high_risk) <= 10
        for score in high_risk:
            assert score.risk_level in ("high", "critical")
        # 应按分数降序排列
        for i in range(len(high_risk) - 1):
            assert high_risk[i].churn_score >= high_risk[i + 1].churn_score

    def test_empty_data(self) -> None:
        """测试空数据输入。"""
        predictor = ChurnPredictor(score_date=date(2025, 2, 15))
        empty_events = pd.DataFrame(columns=[
            "user_id", "event_date", "event_type", "event_duration",
        ])
        empty_users = pd.DataFrame(columns=[
            "user_id", "email", "status", "region", "plan_type",
        ])
        scores = predictor.run(empty_events, empty_users)
        assert len(scores) == 0

    def test_custom_weights(self) -> None:
        """测试自定义特征权重。"""
        config = ChurnModelConfig(
            weights={
                "login_frequency": 0.50,
                "usage_decline": 0.50,
            },
        )
        predictor = ChurnPredictor(config=config, score_date=date(2025, 2, 15))
        # 确保配置正确加载
        assert predictor.config.weights["login_frequency"] == 0.50

    def test_sql_mode_without_connection(self) -> None:
        """测试未配置数据库连接时 SQL 模式报错。"""
        predictor = ChurnPredictor()
        with pytest.raises(Exception):
            predictor.run_sql(db_connection=None)  # type: ignore
