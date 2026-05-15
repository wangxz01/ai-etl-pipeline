"""
数据质量模块测试套件

覆盖数据质量分析引擎和数据质量规则配置的单元测试。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from src.etl.data_quality.profiling_engine import (
    AnomalyAlert,
    ColumnProfile,
    DataProfilingEngine,
    TableProfile,
)
from src.etl.data_quality.dq_rules_config import (
    DQRule,
    DQRuleEngine,
    DQViolation,
    Severity,
    _email_format,
    _enum_values,
    _max_amount,
    _max_null_rate,
    _min_row_count,
    _no_future_date,
    _no_negative,
    _no_null,
    _phone_format,
    _referential_integrity,
    _string_length,
    _unique_column,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def clean_orders() -> pd.DataFrame:
    """生成干净（无质量问题）的订单数据。"""
    orders: list[dict[str, Any]] = []
    for i in range(100):
        orders.append({
            "order_id": str(uuid4()),
            "user_id": str(uuid4()),
            "product_id": str(uuid4()),
            "amount": round(np.random.uniform(10, 5000), 2),
            "quantity": np.random.randint(1, 10),
            "discount": np.random.choice([0, 5, 10]),
            "status": np.random.choice(["pending", "paid", "delivered"]),
            "order_date": date(2025, 1, 15) + timedelta(days=i % 30),
            "created_at": datetime(2025, 1, 15),
            "updated_at": datetime(2025, 1, 15),
        })
    return pd.DataFrame(orders)


@pytest.fixture
def dirty_orders() -> pd.DataFrame:
    """生成包含各种质量问题的订单数据。"""
    orders: list[dict[str, Any]] = []
    for i in range(100):
        order: dict[str, Any] = {
            "order_id": str(uuid4()) if i != 0 else str(uuid4()),  # 保留唯一性
            "user_id": str(uuid4()),
            "product_id": str(uuid4()),
            "amount": round(np.random.uniform(10, 5000), 2),
            "quantity": np.random.randint(1, 10),
            "discount": np.random.choice([0, 5, 10]),
            "status": np.random.choice(["pending", "paid", "delivered", "invalid_status"]),
            "order_date": date(2025, 1, 15) + timedelta(days=i % 30),
            "created_at": datetime(2025, 1, 15),
            "updated_at": datetime(2025, 1, 15),
        }

        # 注入数据质量问题
        if i < 5:
            order["amount"] = -100.0  # 负值
        if i < 10:
            order["discount"] = np.nan  # 空值
        if i < 3:
            order["status"] = "invalid_status"  # 非法枚举值

        orders.append(order)
    return pd.DataFrame(orders)


@pytest.fixture
def users_with_issues() -> pd.DataFrame:
    """生成包含质量问题的用户数据。"""
    users: list[dict[str, Any]] = []
    for i in range(50):
        user: dict[str, Any] = {
            "user_id": str(uuid4()),
            "email": f"user{i}@example.com",
            "username": f"用户{i}",
            "phone": f"138{i:08d}",
            "signup_date": date(2025, 1, 1) + timedelta(days=i),
            "plan_type": ["free", "starter", "pro", "enterprise"][i % 4],
            "status": "active",
        }

        # 注入问题
        if i == 0:
            user["email"] = "invalid-email"  # 无效邮箱格式
        if i == 1:
            user["email"] = None  # 空邮箱
        if i == 2:
            user["plan_type"] = "platinum"  # 非法枚举值

        users.append(user)
    return pd.DataFrame(users)


@pytest.fixture
def reference_users() -> pd.DataFrame:
    """生成参照用户表。"""
    return pd.DataFrame([
        {"user_id": "u1"},
        {"user_id": "u2"},
        {"user_id": "u3"},
    ])


# ============================================================
# 数据质量分析引擎测试
# ============================================================


class TestDataProfilingEngine:
    """数据质量分析引擎测试。"""

    def test_profile_clean_table(self, clean_orders: pd.DataFrame) -> None:
        """测试干净数据的分析结果。"""
        engine = DataProfilingEngine()
        profile = engine.profile_table(clean_orders, "orders")

        assert isinstance(profile, TableProfile)
        assert profile.table_name == "orders"
        assert profile.row_count == 100
        assert profile.column_count > 0
        assert profile.overall_quality_score > 80

    def test_profile_dirty_table(self, dirty_orders: pd.DataFrame) -> None:
        """测试脏数据的分析结果应发现异常。"""
        engine = DataProfilingEngine()
        profile = engine.profile_table(dirty_orders, "orders")

        assert len(profile.anomalies) > 0
        assert profile.overall_quality_score < 100

    def test_column_profiling(self, clean_orders: pd.DataFrame) -> None:
        """测试列级分析。"""
        engine = DataProfilingEngine()
        profile = engine.profile_table(clean_orders, "orders")

        for col_profile in profile.columns:
            assert isinstance(col_profile, ColumnProfile)
            assert col_profile.column_name is not None
            assert col_profile.total_count == 100
            assert 0 <= col_profile.null_rate <= 1.0
            assert col_profile.quality_score > 0

    def test_numeric_column_stats(self, clean_orders: pd.DataFrame) -> None:
        """测试数值列统计信息。"""
        engine = DataProfilingEngine()
        profile = engine.profile_table(clean_orders, "orders")

        amount_profile = next(
            (c for c in profile.columns if c.column_name == "amount"), None
        )
        assert amount_profile is not None
        assert amount_profile.mean_value is not None
        assert amount_profile.std_value is not None
        assert amount_profile.min_value is not None
        assert amount_profile.max_value is not None
        assert amount_profile.min_value <= amount_profile.max_value

    def test_null_rate_detection(self, dirty_orders: pd.DataFrame) -> None:
        """测试空值率检测。"""
        engine = DataProfilingEngine(null_rate_threshold=0.05)
        profile = engine.profile_table(dirty_orders, "orders")

        null_anomalies = [
            a for a in profile.anomalies
            if a.get("alert_type") == "high_null_rate"
        ]
        assert len(null_anomalies) > 0

    def test_negative_value_detection(self, dirty_orders: pd.DataFrame) -> None:
        """测试负值检测。"""
        engine = DataProfilingEngine()
        profile = engine.profile_table(dirty_orders, "orders")

        negative_anomalies = [
            a for a in profile.anomalies
            if a.get("alert_type") == "negative_value"
        ]
        assert len(negative_anomalies) > 0

    def test_freshness_check(self) -> None:
        """测试数据新鲜度检查。"""
        engine = DataProfilingEngine(freshness_sla_days=7)

        # 新鲜数据
        fresh_df = pd.DataFrame({
            "event_date": [datetime.now() - timedelta(hours=1)],
        })
        fresh_profile = engine.profile_table(fresh_df, "events", date_column="event_date")
        assert fresh_profile.freshness_score > 80

    def test_stale_data_detection(self) -> None:
        """测试过期数据检测。"""
        engine = DataProfilingEngine(freshness_sla_days=1)

        stale_df = pd.DataFrame({
            "event_date": [datetime(2020, 1, 1)],
        })
        profile = engine.profile_table(stale_df, "old_events", date_column="event_date")
        assert profile.freshness_score < 50

    def test_referential_integrity(
        self,
        reference_users: pd.DataFrame,
    ) -> None:
        """测试参照完整性检查。"""
        engine = DataProfilingEngine()

        # 有孤立值的订单
        orders_df = pd.DataFrame({
            "order_id": [str(uuid4()) for _ in range(5)],
            "user_id": ["u1", "u2", "orphan1", "orphan2", "u3"],
        })

        ref_data = {"user_id": reference_users}
        profile = engine.profile_table(
            orders_df, "orders", reference_data=ref_data,
        )

        ref_anomalies = [
            a for a in profile.anomalies
            if a.get("alert_type") == "referential_integrity"
        ]
        assert len(ref_anomalies) > 0

    def test_empty_dataframe(self) -> None:
        """测试空 DataFrame 分析。"""
        engine = DataProfilingEngine()
        empty_df = pd.DataFrame()
        profile = engine.profile_table(empty_df, "empty_table")

        assert profile.row_count == 0
        assert profile.overall_quality_score == 0.0

    def test_multiple_tables_profiling(self) -> None:
        """测试批量分析多个表。"""
        engine = DataProfilingEngine()
        tables = {
            "table_a": pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]}),
            "table_b": pd.DataFrame({"col1": [10, 20], "col2": [None, "x"]}),
        }

        profiles = engine.profile_multiple_tables(tables)
        assert len(profiles) == 2
        assert all(isinstance(p, TableProfile) for p in profiles)

    def test_report_generation(self) -> None:
        """测试报告生成。"""
        engine = DataProfilingEngine()
        profiles = engine.profile_multiple_tables({
            "t1": pd.DataFrame({"a": [1, 2, 3]}),
            "t2": pd.DataFrame({"b": [None, None, 3]}),
        })

        report = engine.generate_report(profiles)
        assert "total_tables" in report
        assert "average_quality_score" in report
        assert "total_anomalies" in report
        assert report["total_tables"] == 2

    def test_top_values_profiling(self) -> None:
        """测试高频值统计。"""
        engine = DataProfilingEngine()
        df = pd.DataFrame({"status": ["active"] * 60 + ["churned"] * 30 + ["suspended"] * 10})
        profile = engine.profile_table(df, "test")

        status_col = next(c for c in profile.columns if c.column_name == "status")
        assert len(status_col.top_values) > 0
        assert status_col.top_values[0]["value"] == "active"
        assert status_col.top_values[0]["count"] == 60

    def test_value_distribution(self, clean_orders: pd.DataFrame) -> None:
        """测试数值分布（分位数）。"""
        engine = DataProfilingEngine()
        profile = engine.profile_table(clean_orders, "orders")

        amount_col = next(c for c in profile.columns if c.column_name == "amount")
        assert "p25" in amount_col.value_distribution
        assert "p50" in amount_col.value_distribution
        assert "p75" in amount_col.value_distribution


# ============================================================
# DQ 规则引擎测试
# ============================================================


class TestDQRuleEngine:
    """数据质量规则引擎测试。"""

    def test_no_negative_rule(self) -> None:
        """测试非负值规则。"""
        rule = _no_negative("amount")
        df = pd.DataFrame({"amount": [10, -5, 20, -1, 30]})
        mask = rule.check_fn(df)
        assert int(mask.sum()) == 2

    def test_no_null_rule(self) -> None:
        """测试非空规则。"""
        rule = _no_null("email")
        df = pd.DataFrame({"email": ["a@b.com", None, "c@d.com", None]})
        mask = rule.check_fn(df)
        assert int(mask.sum()) == 2

    def test_email_format_rule(self) -> None:
        """测试邮箱格式规则。"""
        rule = _email_format("email")
        df = pd.DataFrame({
            "email": ["valid@example.com", "invalid", "also@valid.org", "@missing.com"],
        })
        mask = rule.check_fn(df)
        assert int(mask.sum()) == 2

    def test_enum_values_rule(self) -> None:
        """测试枚举值规则。"""
        rule = _enum_values("status", ["active", "churned", "suspended"])
        df = pd.DataFrame({"status": ["active", "invalid", "churned", None]})
        mask = rule.check_fn(df)
        assert int(mask.sum()) == 1  # None 不参与检查

    def test_unique_column_rule(self) -> None:
        """测试唯一性规则。"""
        rule = _unique_column("order_id")
        df = pd.DataFrame({"order_id": ["a", "b", "a", "c", "b"]})
        mask = rule.check_fn(df)
        assert int(mask.sum()) == 2  # 第二个 a 和 b

    def test_max_amount_rule(self) -> None:
        """测试金额上限规则。"""
        rule = _max_amount("amount", 10000)
        df = pd.DataFrame({"amount": [100, 50000, 200, 15000]})
        mask = rule.check_fn(df)
        assert int(mask.sum()) == 2

    def test_string_length_rule(self) -> None:
        """测试字符串长度规则。"""
        rule = _string_length("name", 2, 50)
        df = pd.DataFrame({"name": ["正常名称", "a", "x" * 100, "好的"]})
        mask = rule.check_fn(df)
        assert int(mask.sum()) == 2

    def test_no_future_date_rule(self) -> None:
        """测试不允许未来日期规则。"""
        rule = _no_future_date("order_date")
        df = pd.DataFrame({
            "order_date": [date(2020, 1, 1), date(2099, 1, 1), date(2025, 1, 1)],
        })
        mask = rule.check_fn(df)
        assert int(mask.sum()) == 1

    def test_phone_format_rule(self) -> None:
        """测试手机号格式规则。"""
        rule = _phone_format("phone")
        df = pd.DataFrame({
            "phone": ["13800138000", "12345678901", "abc", "13900139000"],
        })
        mask = rule.check_fn(df)
        assert int(mask.sum()) == 2

    def test_referential_integrity_rule(self) -> None:
        """测试参照完整性规则。"""
        ref_ids = {"u1", "u2", "u3"}
        rule = _referential_integrity("user_id", ref_ids)
        df = pd.DataFrame({"user_id": ["u1", "u2", "u4", "u5", "u3"]})
        mask = rule.check_fn(df)
        assert int(mask.sum()) == 2

    def test_min_row_count_rule(self) -> None:
        """测试最小行数规则。"""
        rule = _min_row_count(10)
        small_df = pd.DataFrame({"a": [1, 2, 3]})
        mask = rule.check_fn(small_df)
        assert bool(mask.iloc[0]) is True

        large_df = pd.DataFrame({"a": range(100)})
        mask = rule.check_fn(large_df)
        assert bool(mask.iloc[0]) is False

    def test_engine_validate_clean_data(self, clean_orders: pd.DataFrame) -> None:
        """测试规则引擎对干净数据的验证。"""
        engine = DQRuleEngine()
        violations = engine.validate(clean_orders, "orders")

        # 干净数据应该几乎没有违规
        error_violations = [v for v in violations if v.severity == Severity.ERROR]
        assert len(error_violations) == 0

    def test_engine_validate_dirty_data(self, dirty_orders: pd.DataFrame) -> None:
        """测试规则引擎对脏数据的验证。"""
        engine = DQRuleEngine()
        violations = engine.validate(dirty_orders, "orders")

        assert len(violations) > 0
        # 应检测到负值
        negative_violations = [v for v in violations if "negative" in v.rule_name]
        assert len(negative_violations) > 0

    def test_engine_validate_users(self, users_with_issues: pd.DataFrame) -> None:
        """测试规则引擎对用户数据的验证。"""
        engine = DQRuleEngine()
        violations = engine.validate(users_with_issues, "users")

        # 应检测到邮箱格式问题和枚举值问题
        violation_names = [v.rule_name for v in violations]
        assert any("email" in n for n in violation_names)
        assert any("enum" in n for n in violation_names)

    def test_violation_report(self, dirty_orders: pd.DataFrame) -> None:
        """测试违规报告生成。"""
        engine = DQRuleEngine()
        violations = engine.validate(dirty_orders, "orders")
        report = engine.generate_violation_report(violations)

        assert "total_violations" in report
        assert "severity_distribution" in report
        assert report["total_violations"] > 0
        assert isinstance(report["severity_distribution"], dict)

    def test_custom_rules(self, clean_orders: pd.DataFrame) -> None:
        """测试自定义规则。"""
        custom_rule = _max_amount("amount", 100)  # 非常低的阈值
        engine = DQRuleEngine(custom_rules=[custom_rule])
        violations = engine.validate(clean_orders, "orders")

        max_violations = [v for v in violations if "max_amount" in v.rule_name]
        assert len(max_violations) > 0

    def test_extra_rules(self, clean_orders: pd.DataFrame) -> None:
        """测试额外规则参数。"""
        extra = _string_length("status", 2, 10)
        engine = DQRuleEngine()
        violations = engine.validate(clean_orders, "orders", extra_rules=[extra])

        # status 列值都是短字符串，应该没有违规
        status_violations = [v for v in violations if v.column_name == "status" and "string_length" in v.rule_name]
        assert len(status_violations) == 0

    def test_get_rules_for_table(self) -> None:
        """测试获取表规则列表。"""
        engine = DQRuleEngine()
        rules = engine.get_rules_for_table("users")

        assert len(rules) > 0
        assert all("name" in r for r in rules)
        assert all("severity" in r for r in rules)

    def test_get_all_rules(self) -> None:
        """测试获取所有规则。"""
        engine = DQRuleEngine()
        all_rules = engine.get_all_rules()

        assert len(all_rules) > 0
        assert "users" in all_rules
        assert "orders" in all_rules
        assert "products" in all_rules

    def test_nonexistent_column_skipped(self) -> None:
        """测试不存在的列被跳过。"""
        engine = DQRuleEngine()
        df = pd.DataFrame({"a": [1, 2, 3]})
        # orders 规则集引用了很多不存在的列
        violations = engine.validate(df, "orders")

        # 不应抛异常，违规数应该很少（只有通用的 volume 检查等）
        assert isinstance(violations, list)

    def test_severity_enum(self) -> None:
        """测试严重等级枚举。"""
        assert Severity.INFO == "info"
        assert Severity.WARNING == "warning"
        assert Severity.ERROR == "error"
        assert Severity.CRITICAL == "critical"

    def test_max_null_rate_threshold(self) -> None:
        """测试空值率阈值规则。"""
        rule = _max_null_rate("col", max_rate=0.10)
        assert rule.params["max_rate"] == 0.10
        assert rule.severity == Severity.ERROR

        rule2 = _max_null_rate("col", max_rate=0.20)
        assert rule2.severity == Severity.WARNING
