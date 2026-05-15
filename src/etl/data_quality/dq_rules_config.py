"""
数据质量规则配置

定义可复用的数据质量规则，使用 Python dataclass 描述。
每条规则包含检查逻辑、严重等级和自动修复建议。

规则类型:
    - 格式校验（邮箱、电话、URL）
    - 范围校验（非负、日期范围、枚举值）
    - 参照完整性（外键存在性）
    - 时效性（数据新鲜度 SLA）
    - 业务逻辑（金额合理性、状态转换）
    - 统计特性（唯一性、分布均匀性）

使用方法:
    from etl.data_quality.dq_rules_config import DQRuleEngine
    engine = DQRuleEngine()
    violations = engine.validate(df, table_name="orders")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# 枚举和类型定义
# ============================================================


class Severity(str, Enum):
    """规则严重等级。"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DQViolation:
    """
    数据质量违规记录。

    Attributes:
        rule_name: 触发的规则名称
        table_name: 表名
        column_name: 列名（可选）
        severity: 严重等级
        message: 违规描述
        affected_rows: 受影响行数
        affected_percentage: 受影响比例
        remediation: 修复建议
        sample_values: 样本违规值
    """
    rule_name: str
    table_name: str
    column_name: str | None = None
    severity: Severity = Severity.WARNING
    message: str = ""
    affected_rows: int = 0
    affected_percentage: float = 0.0
    remediation: str = ""
    sample_values: list[Any] = field(default_factory=list)


# ============================================================
# 规则定义
# ============================================================


@dataclass
class DQRule:
    """
    数据质量规则定义。

    Attributes:
        name: 规则名称
        description: 规则描述
        severity: 严重等级
        check_type: 检查类型
        column: 目标列名（None 表示表级规则）
        params: 检查参数
        check_fn: 检查函数（接受 DataFrame，返回违规 Series）
        remediation: 自动修复建议
    """
    name: str
    description: str
    severity: Severity
    check_type: str
    column: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    check_fn: Callable[[pd.DataFrame], pd.Series] | None = None
    remediation: str = ""


# ============================================================
# 规则工厂函数
# ============================================================


def _no_negative(column: str) -> DQRule:
    """创建不允许负值的规则。"""
    def check(df: pd.DataFrame) -> pd.Series:
        return df[column].dropna() < 0
    return DQRule(
        name=f"no_negative_{column}",
        description=f"列 {column} 不允许负值",
        severity=Severity.ERROR,
        check_type="range",
        column=column,
        check_fn=check,
        remediation=f"检查 {column} 列的数据来源，修正负值为0或绝对值",
    )


def _no_null(column: str) -> DQRule:
    """创建不允许空值的规则。"""
    def check(df: pd.DataFrame) -> pd.Series:
        return df[column].isna()
    return DQRule(
        name=f"no_null_{column}",
        description=f"列 {column} 不允许空值",
        severity=Severity.ERROR,
        check_type="completeness",
        column=column,
        check_fn=check,
        remediation=f"检查 {column} 列的 ETL 逻辑，确保上游数据填充该字段",
    )


def _max_null_rate(column: str, max_rate: float = 0.05) -> DQRule:
    """创建空值率上限规则。"""
    def check(df: pd.DataFrame) -> pd.Series:
        return df[column].isna()
    return DQRule(
        name=f"max_null_rate_{column}",
        description=f"列 {column} 空值率不得超过 {max_rate:.0%}",
        severity=Severity.WARNING if max_rate >= 0.1 else Severity.ERROR,
        check_type="completeness",
        column=column,
        params={"max_rate": max_rate},
        check_fn=check,
        remediation=f"检查 {column} 的空值来源，考虑填充默认值或修复上游数据",
    )


def _email_format(column: str = "email") -> DQRule:
    """创建邮箱格式校验规则。"""
    email_pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    def check(df: pd.DataFrame) -> pd.Series:
        non_null = df[column].dropna().astype(str)
        mask = ~non_null.apply(lambda x: bool(email_pattern.match(x)))
        return mask.reindex(df.index, fill_value=False)
    return DQRule(
        name=f"email_format_{column}",
        description=f"列 {column} 必须符合邮箱格式",
        severity=Severity.WARNING,
        check_type="format",
        column=column,
        check_fn=check,
        remediation="检查邮箱格式，移除无效邮箱或要求用户更新",
    )


def _enum_values(column: str, allowed: list[str]) -> DQRule:
    """创建枚举值校验规则。"""
    def check(df: pd.DataFrame) -> pd.Series:
        return ~df[column].dropna().isin(allowed)
    return DQRule(
        name=f"enum_values_{column}",
        description=f"列 {column} 只允许以下值: {', '.join(allowed)}",
        severity=Severity.ERROR,
        check_type="range",
        column=column,
        params={"allowed_values": allowed},
        check_fn=check,
        remediation=f"检查 {column} 的数据源，确保只写入允许的枚举值",
    )


def _referential_integrity(
    fk_column: str,
    ref_values: set[str],
) -> DQRule:
    """创建参照完整性规则。"""
    def check(df: pd.DataFrame) -> pd.Series:
        non_null = df[fk_column].dropna().astype(str)
        mask = ~non_null.isin(ref_values)
        return mask.reindex(df.index, fill_value=False)
    return DQRule(
        name=f"ref_integrity_{fk_column}",
        description=f"外键列 {fk_column} 的值必须在参照表中存在",
        severity=Severity.ERROR,
        check_type="referential",
        column=fk_column,
        params={"ref_count": len(ref_values)},
        check_fn=check,
        remediation=f"检查 {fk_column} 对应的参照数据，可能需要同步更新或标记为孤立记录",
    )


def _recency_sla(column: str, max_age_days: int = 1) -> DQRule:
    """创建数据新鲜度 SLA 规则。"""
    def check(df: pd.DataFrame) -> pd.Series:
        dates = pd.to_datetime(df[column], errors="coerce")
        max_date = dates.max()
        if pd.isna(max_date):
            return pd.Series([True] * len(df))
        age = (pd.Timestamp.now() - max_date).total_seconds() / 86400
        return pd.Series([age > max_age_days] * len(df))
    return DQRule(
        name=f"recency_sla_{column}",
        description=f"列 {column} 的最新数据不得超过 {max_age_days} 天",
        severity=Severity.CRITICAL,
        check_type="freshness",
        column=column,
        params={"max_age_days": max_age_days},
        check_fn=check,
        remediation="检查 ETL 任务是否正常运行，手动触发数据刷新",
    )


def _min_row_count(min_rows: int = 1) -> DQRule:
    """创建最小行数规则。"""
    def check(df: pd.DataFrame) -> pd.Series:
        return pd.Series([len(df) < min_rows])
    return DQRule(
        name="min_row_count",
        description=f"表行数不得少于 {min_rows}",
        severity=Severity.CRITICAL,
        check_type="volume",
        params={"min_rows": min_rows},
        check_fn=check,
        remediation="检查数据加载流程，确认上游数据源正常产出",
    )


def _max_amount(column: str, max_val: float) -> DQRule:
    """创建金额上限规则。"""
    def check(df: pd.DataFrame) -> pd.Series:
        return df[column].dropna() > max_val
    return DQRule(
        name=f"max_amount_{column}",
        description=f"列 {column} 值不得超过 {max_val}",
        severity=Severity.WARNING,
        check_type="range",
        column=column,
        params={"max_value": max_val},
        check_fn=check,
        remediation=f"检查 {column} 的异常大值，可能是单位错误或数据录入错误",
    )


def _min_amount(column: str, min_val: float) -> DQRule:
    """创建金额下限规则（不含零）。"""
    def check(df: pd.DataFrame) -> pd.Series:
        return (df[column].dropna() < min_val) & (df[column].dropna() > 0)
    return DQRule(
        name=f"min_amount_{column}",
        description=f"列 {column} 非零值不得低于 {min_val}",
        severity=Severity.WARNING,
        check_type="range",
        column=column,
        params={"min_value": min_val},
        check_fn=check,
        remediation=f"检查 {column} 的异常小值，可能需要过滤或修正",
    )


def _unique_column(column: str) -> DQRule:
    """创建唯一性规则。"""
    def check(df: pd.DataFrame) -> pd.Series:
        duplicated = df[column].duplicated(keep="first")
        return duplicated
    return DQRule(
        name=f"unique_{column}",
        description=f"列 {column} 必须唯一",
        severity=Severity.ERROR,
        check_type="uniqueness",
        column=column,
        check_fn=check,
        remediation=f"检查 {column} 的重复值，执行去重或更新主键生成逻辑",
    )


def _date_range(column: str, start: date, end: date) -> DQRule:
    """创建日期范围规则。"""
    def check(df: pd.DataFrame) -> pd.Series:
        dates = pd.to_datetime(df[column], errors="coerce")
        return (dates < pd.Timestamp(start)) | (dates > pd.Timestamp(end))
    return DQRule(
        name=f"date_range_{column}",
        description=f"列 {column} 必须在 {start} 到 {end} 范围内",
        severity=Severity.WARNING,
        check_type="range",
        column=column,
        params={"start": str(start), "end": str(end)},
        check_fn=check,
        remediation=f"检查 {column} 超出范围的日期，可能是时区或格式问题",
    )


def _phone_format(column: str = "phone") -> DQRule:
    """创建中国手机号格式校验规则。"""
    cn_phone = re.compile(r"^1[3-9]\d{9}$")
    def check(df: pd.DataFrame) -> pd.Series:
        non_null = df[column].dropna().astype(str)
        mask = ~non_null.apply(lambda x: bool(cn_phone.match(x.strip())))
        return mask.reindex(df.index, fill_value=False)
    return DQRule(
        name=f"phone_format_{column}",
        description=f"列 {column} 必须符合中国手机号格式",
        severity=Severity.INFO,
        check_type="format",
        column=column,
        check_fn=check,
        remediation="检查电话号码格式，支持国际号码需要调整正则",
    )


def _no_future_date(column: str) -> DQRule:
    """创建不允许未来日期的规则。"""
    def check(df: pd.DataFrame) -> pd.Series:
        dates = pd.to_datetime(df[column], errors="coerce")
        return dates > pd.Timestamp.now()
    return DQRule(
        name=f"no_future_date_{column}",
        description=f"列 {column} 不允许未来日期",
        severity=Severity.ERROR,
        check_type="range",
        column=column,
        check_fn=check,
        remediation=f"检查 {column} 的日期来源，可能是时区配置错误",
    )


def _consistent_status_transition(
    status_col: str,
    valid_transitions: dict[str, list[str]],
) -> DQRule:
    """创建状态转换一致性规则。"""
    def check(df: pd.DataFrame) -> pd.Series:
        # 表级检查：验证每个状态值是否在合法集合中
        valid_statuses = set(valid_transitions.keys()) | {"pending"}
        mask = ~df[status_col].dropna().isin(valid_statuses)
        return mask.reindex(df.index, fill_value=False)
    return DQRule(
        name=f"valid_status_{status_col}",
        description=f"列 {status_col} 的值必须是合法状态",
        severity=Severity.ERROR,
        check_type="business_logic",
        column=status_col,
        params={"valid_transitions": valid_transitions},
        check_fn=check,
        remediation=f"检查 {status_col} 的状态写入逻辑，确保只使用合法状态值",
    )


def _amount_consistency(amount_col: str, quantity_col: str, total_col: str) -> DQRule:
    """创建金额一致性规则（总价 ≈ 单价 × 数量）。"""
    def check(df: pd.DataFrame) -> pd.Series:
        if amount_col not in df.columns or quantity_col not in df.columns or total_col not in df.columns:
            return pd.Series([False] * len(df))
        expected = df[amount_col] * df[quantity_col]
        diff = (df[total_col] - expected).abs()
        threshold = expected * 0.1 + 0.01  # 允许10%误差
        return diff > threshold
    return DQRule(
        name=f"amount_consistency_{total_col}",
        description=f"列 {total_col} 应约等于 {amount_col} × {quantity_col}",
        severity=Severity.WARNING,
        check_type="business_logic",
        column=total_col,
        params={"amount_col": amount_col, "quantity_col": quantity_col},
        check_fn=check,
        remediation="检查订单金额计算逻辑，确认折扣、税费等已正确处理",
    )


def _string_length(column: str, min_len: int, max_len: int) -> DQRule:
    """创建字符串长度规则。"""
    def check(df: pd.DataFrame) -> pd.Series:
        lengths = df[column].dropna().astype(str).str.len()
        return (lengths < min_len) | (lengths > max_len)
    return DQRule(
        name=f"string_length_{column}",
        description=f"列 {column} 长度必须在 {min_len} 到 {max_len} 之间",
        severity=Severity.WARNING,
        check_type="format",
        column=column,
        params={"min_len": min_len, "max_len": max_len},
        check_fn=check,
        remediation=f"检查 {column} 的字符串长度异常，可能需要截断或补全",
    )


# ============================================================
# 规则引擎
# ============================================================


# 预定义规则集：各表的默认规则
TABLE_RULES: dict[str, list[DQRule]] = {
    "users": [
        _no_null("user_id"),
        _unique_column("user_id"),
        _email_format("email"),
        _no_null("email"),
        _enum_values("plan_type", ["free", "starter", "pro", "enterprise"]),
        _enum_values("status", ["active", "suspended", "churned", "trial"]),
        _enum_values("region", ["CN", "US", "EU", "JP", "SEA"]),
        _no_negative("mrr"),
        _no_future_date("signup_date"),
    ],
    "user_events": [
        _no_null("event_id"),
        _no_null("user_id"),
        _no_null("event_date"),
        _enum_values("event_type", ["page_view", "click", "feature_use", "api_call"]),
        _enum_values("device", ["desktop", "mobile", "tablet"]),
        _no_negative("event_duration"),
        _min_row_count(100),
    ],
    "subscriptions": [
        _no_null("sub_id"),
        _no_null("user_id"),
        _enum_values("plan", ["free", "starter", "pro", "enterprise"]),
        _enum_values("status", ["active", "trialing", "past_due", "cancelled", "expired"]),
        _no_negative("amount"),
        _no_future_date("started_at"),
    ],
    "orders": [
        _no_null("order_id"),
        _no_null("user_id"),
        _no_negative("amount"),
        _no_negative("quantity"),
        _max_amount("amount", 1000000),
        _enum_values("status", ["pending", "paid", "shipped", "delivered", "refunded", "cancelled"]),
        _no_future_date("order_date"),
        _amount_consistency("amount", "quantity", "total_amount") if True else None,
    ],
    "products": [
        _unique_column("sku"),
        _no_null("product_id"),
        _no_negative("price"),
        _no_negative("cost"),
        _string_length("name", 1, 200),
        _enum_values("category", ["software", "hardware", "service", "addon", "training"]),
    ],
    "support_tickets": [
        _no_null("ticket_id"),
        _no_null("user_id"),
        _enum_values("priority", ["low", "medium", "high", "critical"]),
        _enum_values("status", ["open", "in_progress", "waiting_customer", "resolved", "closed"]),
        _enum_values("category", ["billing", "technical", "account", "feature_request", "bug", "other"]),
    ],
}


class DQRuleEngine:
    """
    数据质量规则引擎。

    对 DataFrame 应用预定义或自定义规则，返回违规记录。
    支持按表名加载默认规则集，也支持添加自定义规则。

    Usage:
        engine = DQRuleEngine()
        violations = engine.validate(df, table_name="orders")
        report = engine.generate_violation_report(violations)
    """

    def __init__(self, custom_rules: list[DQRule] | None = None) -> None:
        """
        初始化规则引擎。

        Args:
            custom_rules: 自定义规则列表（追加到预定义规则之后）
        """
        self.custom_rules = custom_rules or []

    def validate(
        self,
        df: pd.DataFrame,
        table_name: str,
        extra_rules: list[DQRule] | None = None,
    ) -> list[DQViolation]:
        """
        对 DataFrame 执行数据质量规则检查。

        Args:
            df: 待检查的 DataFrame
            table_name: 表名（用于加载默认规则集）
            extra_rules: 额外规则列表

        Returns:
            DQViolation 违规记录列表
        """
        logger.info("开始数据质量规则检查: %s (%d 行)", table_name, len(df))

        # 收集适用规则
        rules: list[DQRule] = []
        if table_name in TABLE_RULES:
            rules.extend([r for r in TABLE_RULES[table_name] if r is not None])
        rules.extend(self.custom_rules)
        if extra_rules:
            rules.extend(extra_rules)

        violations: list[DQViolation] = []

        for rule in rules:
            try:
                # 跳过不存在的列
                if rule.column and rule.column not in df.columns:
                    logger.debug("规则 %s 跳过: 列 %s 不存在", rule.name, rule.column)
                    continue

                if rule.check_fn is None:
                    continue

                mask = rule.check_fn(df)
                affected = int(mask.sum())

                if affected > 0:
                    # 检查是否为阈值类规则
                    if rule.check_type == "completeness" and "max_rate" in rule.params:
                        rate = affected / len(df)
                        if rate <= rule.params["max_rate"]:
                            continue

                    # 检查是否为表级规则（如 min_row_count）
                    if rule.check_type == "volume":
                        if affected > 0:
                            violations.append(DQViolation(
                                rule_name=rule.name,
                                table_name=table_name,
                                severity=rule.severity,
                                message=rule.description,
                                affected_rows=0,
                                remediation=rule.remediation,
                            ))
                        continue

                    percentage = round(affected / len(df), 4) if len(df) > 0 else 0.0

                    # 采集样本值
                    sample_values: list[Any] = []
                    if rule.column:
                        violation_rows = df[mask.reindex(df.index, fill_value=False)]
                        if not violation_rows.empty:
                            sample_values = violation_rows[rule.column].head(5).tolist()

                    violations.append(DQViolation(
                        rule_name=rule.name,
                        table_name=table_name,
                        column_name=rule.column,
                        severity=rule.severity,
                        message=rule.description,
                        affected_rows=affected,
                        affected_percentage=percentage,
                        remediation=rule.remediation,
                        sample_values=sample_values,
                    ))

            except Exception as e:
                logger.error("规则 %s 执行失败: %s", rule.name, e)
                violations.append(DQViolation(
                    rule_name=rule.name,
                    table_name=table_name,
                    severity=Severity.ERROR,
                    message=f"规则执行异常: {e}",
                    remediation="检查规则实现和数据格式",
                ))

        logger.info("规则检查完成: %d 条违规", len(violations))
        return violations

    def generate_violation_report(
        self,
        violations: list[DQViolation],
    ) -> dict[str, Any]:
        """
        生成违规报告摘要。

        Args:
            violations: 违规记录列表

        Returns:
            报告摘要字典
        """
        severity_counts: dict[str, int] = {"info": 0, "warning": 0, "error": 0, "critical": 0}
        type_counts: dict[str, int] = {}

        for v in violations:
            sev = v.severity.value if isinstance(v.severity, Severity) else v.severity
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            rule_type = v.rule_name.split("_")[0]
            type_counts[rule_type] = type_counts.get(rule_type, 0) + 1

        top_violations = sorted(
            violations,
            key=lambda v: v.affected_percentage,
            reverse=True,
        )[:10]

        return {
            "total_violations": len(violations),
            "severity_distribution": severity_counts,
            "type_distribution": type_counts,
            "top_violations": [
                {
                    "rule": v.rule_name,
                    "table": v.table_name,
                    "column": v.column_name,
                    "severity": v.severity.value if isinstance(v.severity, Severity) else v.severity,
                    "affected": v.affected_rows,
                    "percentage": f"{v.affected_percentage:.2%}",
                    "remediation": v.remediation,
                }
                for v in top_violations
            ],
        }

    def get_rules_for_table(self, table_name: str) -> list[dict[str, Any]]:
        """
        获取指定表的规则列表。

        Args:
            table_name: 表名

        Returns:
            规则描述列表
        """
        rules = TABLE_RULES.get(table_name, [])
        return [
            {
                "name": r.name,
                "description": r.description,
                "severity": r.severity.value,
                "check_type": r.check_type,
                "column": r.column,
            }
            for r in rules
            if r is not None
        ]

    def get_all_rules(self) -> dict[str, list[dict[str, Any]]]:
        """
        获取所有预定义规则。

        Returns:
            {表名: 规则列表} 字典
        """
        return {
            table: self.get_rules_for_table(table)
            for table in TABLE_RULES
        }
