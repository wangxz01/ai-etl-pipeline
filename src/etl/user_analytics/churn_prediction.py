"""
用户流失预测引擎

基于多维特征工程和加权评分模型预测用户流失风险。
包含 SQL 特征提取 CTE 和 pandas 特征矩阵构建两种实现。

特征维度:
    1. 登录频率（login_frequency）: 最近30天登录天数占比
    2. 工单数量（ticket_count）: 最近90天工单数量
    3. 付款延迟（payment_delay）: 订阅付款平均延迟天数
    4. 使用下降率（usage_decline）: 近30天 vs 前30天活跃度变化
    5. 功能使用广度（feature_breadth）: 使用的不同功能类型数
    6. 会话时长趋势（session_trend）: 平均会话时长变化
    7. 支持满意度（support_satisfaction）: 工单平均满意度

流失评分 = 加权组合各特征，输出 0-1 之间的风险分数。

使用方法:
    from etl.user_analytics.churn_prediction import ChurnPredictor
    predictor = ChurnPredictor()
    scores = predictor.run(events_df, users_df, subscriptions_df, tickets_df)
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
class ChurnScore:
    """
    单个用户的流失评分。

    Attributes:
        user_id: 用户ID
        score_date: 评分日期
        churn_score: 流失风险分数（0-1）
        risk_level: 风险等级（low/medium/high/critical）
        features: 各特征原始值
    """
    user_id: str
    score_date: date
    churn_score: float
    risk_level: str
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class ChurnModelConfig:
    """
    流失模型配置。

    Attributes:
        lookback_days: 特征计算的回溯天数
        weights: 各特征的权重
        thresholds: 风险等级阈值
    """
    lookback_days: int = 90
    recent_window_days: int = 30
    comparison_window_days: int = 30
    weights: dict[str, float] = field(default_factory=lambda: {
        "login_frequency": 0.20,
        "ticket_count": 0.15,
        "ticket_sentiment": 0.10,
        "payment_delay": 0.15,
        "usage_decline": 0.20,
        "feature_breadth": 0.10,
        "session_trend": 0.10,
    })
    thresholds: dict[str, float] = field(default_factory=lambda: {
        "low": 0.3,
        "medium": 0.5,
        "high": 0.7,
        "critical": 1.0,
    })


# ============================================================
# SQL 特征工程查询
# ============================================================

SQL_CHURN_FEATURES = """
WITH user_base AS (
    -- 活跃用户基础信息
    SELECT user_id, email, plan_type, region, signup_date, status
    FROM users
    WHERE status IN ('active', 'trial')
      AND email NOT LIKE 'test_%%'
      AND email NOT LIKE 'bot_%%'
),

-- 特征1: 登录频率（最近30天活跃天数 / 30）
login_freq AS (
    SELECT
        e.user_id,
        COUNT(DISTINCT e.event_date)::NUMERIC / 30.0 AS login_frequency
    FROM user_events e
    INNER JOIN user_base ub ON e.user_id = ub.user_id
    WHERE e.event_date BETWEEN %(recent_start)s AND %(score_date)s
    GROUP BY e.user_id
),

-- 特征2: 工单数量和满意度（最近90天）
ticket_stats AS (
    SELECT
        t.user_id,
        COUNT(*) AS ticket_count,
        AVG(t.satisfaction)::NUMERIC AS avg_satisfaction,
        COUNT(*) FILTER (WHERE t.priority IN ('high', 'critical'))::NUMERIC AS high_priority_count
    FROM support_tickets t
    INNER JOIN user_base ub ON t.user_id = ub.user_id
    WHERE t.created_at >= %(lookback_start)s
    GROUP BY t.user_id
),

-- 特征3: 付款延迟（订阅续费延迟天数）
payment_delays AS (
    SELECT
        s.user_id,
        CASE
            WHEN COUNT(*) = 0 THEN 0
            ELSE AVG(EXTRACT(EPOCH FROM (s.started_at - LAG(s.ended_at) OVER (PARTITION BY s.user_id ORDER BY s.started_at))) / 86400)::NUMERIC
        END AS avg_payment_delay
    FROM subscriptions s
    INNER JOIN user_base ub ON s.user_id = ub.user_id
    WHERE s.status IN ('active', 'past_due', 'cancelled')
    GROUP BY s.user_id
),

-- 特征4: 使用下降率（近30天 vs 前30天事件数）
usage_comparison AS (
    SELECT
        recent.user_id,
        CASE
            WHEN prev.event_count > 0
            THEN ROUND((prev.event_count - recent.event_count)::NUMERIC / prev.event_count, 4)
            ELSE 0
        END AS usage_decline
    FROM (
        SELECT user_id, COUNT(*) AS event_count
        FROM user_events
        WHERE event_date BETWEEN %(recent_start)s AND %(score_date)s
        GROUP BY user_id
    ) recent
    INNER JOIN (
        SELECT user_id, COUNT(*) AS event_count
        FROM user_events
        WHERE event_date BETWEEN %(comparison_start)s AND %(recent_start)s - INTERVAL '1 day'
        GROUP BY user_id
    ) prev ON recent.user_id = prev.user_id
),

-- 特征5: 功能使用广度（不同 event_type 数量）
feature_breadth AS (
    SELECT
        e.user_id,
        COUNT(DISTINCT e.event_type)::NUMERIC AS feature_types
    FROM user_events e
    INNER JOIN user_base ub ON e.user_id = ub.user_id
    WHERE e.event_date >= %(lookback_start)s
    GROUP BY e.user_id
),

-- 特征6: 会话时长趋势
session_trends AS (
    SELECT
        recent.user_id,
        CASE
            WHEN prev_avg > 0
            THEN ROUND((recent_avg - prev_avg)::NUMERIC / prev_avg, 4)
            ELSE 0
        END AS session_trend
    FROM (
        SELECT user_id, AVG(event_duration) AS recent_avg
        FROM user_events
        WHERE event_date BETWEEN %(recent_start)s AND %(score_date)s
        GROUP BY user_id
    ) recent
    INNER JOIN (
        SELECT user_id, AVG(event_duration) AS prev_avg
        FROM user_events
        WHERE event_date BETWEEN %(comparison_start)s AND %(recent_start)s - INTERVAL '1 day'
        GROUP BY user_id
    ) prev ON recent.user_id = prev.user_id
)

-- 合并所有特征
SELECT
    ub.user_id,
    ub.plan_type,
    ub.region,
    COALESCE(lf.login_frequency, 0) AS login_frequency,
    COALESCE(ts.ticket_count, 0) AS ticket_count,
    COALESCE(ts.avg_satisfaction, 3) AS avg_satisfaction,
    COALESCE(ts.high_priority_count, 0) AS high_priority_count,
    COALESCE(pd.avg_payment_delay, 0) AS payment_delay,
    COALESCE(uc.usage_decline, 0) AS usage_decline,
    COALESCE(fb.feature_types, 0) AS feature_breadth,
    COALESCE(st.session_trend, 0) AS session_trend
FROM user_base ub
LEFT JOIN login_freq lf ON ub.user_id = lf.user_id
LEFT JOIN ticket_stats ts ON ub.user_id = ts.user_id
LEFT JOIN payment_delays pd ON ub.user_id = pd.user_id
LEFT JOIN usage_comparison uc ON ub.user_id = uc.user_id
LEFT JOIN feature_breadth fb ON ub.user_id = fb.user_id
LEFT JOIN session_trends st ON ub.user_id = st.user_id;
"""


# ============================================================
# pandas 特征工程实现
# ============================================================


class ChurnPredictor:
    """
    用户流失预测器。

    基于多维特征和加权评分模型计算用户流失风险。
    特征经过归一化后按权重组合，输出 0-1 之间的风险分数。

    Usage:
        predictor = ChurnPredictor()
        scores = predictor.run(events_df, users_df, subscriptions_df, tickets_df)
    """

    def __init__(
        self,
        config: ChurnModelConfig | None = None,
        score_date: date | None = None,
        exclude_test_accounts: bool = True,
    ) -> None:
        """
        初始化流失预测器。

        Args:
            config: 模型配置（使用默认配置如果为None）
            score_date: 评分日期（默认为今天）
            exclude_test_accounts: 是否排除测试账号
        """
        self.config = config or ChurnModelConfig()
        self.score_date = score_date or date.today()
        self.exclude_test_accounts = exclude_test_accounts
        self.test_patterns = ["test_", "bot_", "demo_", "qa_", "@test-company.com"]

    def run(
        self,
        events_df: pd.DataFrame,
        users_df: pd.DataFrame,
        subscriptions_df: pd.DataFrame | None = None,
        tickets_df: pd.DataFrame | None = None,
    ) -> list[ChurnScore]:
        """
        执行完整的流失预测流程。

        Args:
            events_df: 用户事件数据
            users_df: 用户数据
            subscriptions_df: 订阅数据（可选）
            tickets_df: 工单数据（可选）

        Returns:
            ChurnScore 列表
        """
        logger.info("开始流失预测，评分日期: %s", self.score_date)

        # 预处理
        events = self._preprocess_events(events_df)
        users = self._preprocess_users(users_df)

        # 过滤测试账号和非活跃用户
        if self.exclude_test_accounts:
            test_mask = users["email"].apply(self._is_test_account)
            users = users[~test_mask]
        active_users = users[users["status"].isin(["active", "trial"])]
        events = events[events["user_id"].isin(active_users["user_id"])]

        # 构建特征矩阵
        feature_matrix = self._build_feature_matrix(
            events, active_users, subscriptions_df, tickets_df,
        )

        # 计算流失评分
        scores = self._compute_scores(feature_matrix)

        logger.info(
            "流失预测完成，%d 用户评分，高风险: %d",
            len(scores),
            sum(1 for s in scores if s.risk_level in ("high", "critical")),
        )
        return scores

    def _build_feature_matrix(
        self,
        events: pd.DataFrame,
        users: pd.DataFrame,
        subscriptions: pd.DataFrame | None,
        tickets: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """
        构建用户流失特征矩阵。

        Args:
            events: 事件数据
            users: 用户数据
            subscriptions: 订阅数据
            tickets: 工单数据

        Returns:
            特征矩阵 DataFrame，每行一个用户
        """
        score_date = self.score_date
        recent_start = score_date - timedelta(days=self.config.recent_window_days)
        comparison_start = recent_start - timedelta(days=self.config.comparison_window_days)
        lookback_start = score_date - timedelta(days=self.config.lookback_days)

        # 基础用户列表
        feature_rows: list[dict[str, Any]] = []

        for _, user in users.iterrows():
            uid = user["user_id"]
            features: dict[str, float] = {
                "user_id": uid,
                "plan_type": user.get("plan_type", "free"),
                "region": user.get("region", "CN"),
            }

            user_events = events[events["user_id"] == uid]

            # ---- 特征1: 登录频率 ----
            recent_events = user_events[
                (user_events["event_date"] >= recent_start) &
                (user_events["event_date"] <= score_date)
            ]
            active_days = recent_events["event_date"].nunique()
            features["login_frequency"] = round(active_days / self.config.recent_window_days, 4)

            # ---- 特征4: 使用下降率 ----
            recent_count = len(recent_events)
            prev_events = user_events[
                (user_events["event_date"] >= comparison_start) &
                (user_events["event_date"] < recent_start)
            ]
            prev_count = len(prev_events)
            if prev_count > 0:
                features["usage_decline"] = round(
                    max(0, (prev_count - recent_count) / prev_count), 4
                )
            else:
                features["usage_decline"] = 1.0 if recent_count == 0 else 0.0

            # ---- 特征5: 功能使用广度 ----
            lookback_events = user_events[
                user_events["event_date"] >= lookback_start
            ]
            features["feature_breadth"] = float(lookback_events["event_type"].nunique())

            # ---- 特征6: 会话时长趋势 ----
            recent_avg_duration = recent_events["event_duration"].mean() if len(recent_events) > 0 else 0.0
            prev_avg_duration = prev_events["event_duration"].mean() if len(prev_events) > 0 else 0.0
            if prev_avg_duration > 0:
                features["session_trend"] = round(
                    (recent_avg_duration - prev_avg_duration) / prev_avg_duration, 4
                )
            else:
                features["session_trend"] = 0.0

            # ---- 特征2&3: 工单相关 ----
            features["ticket_count"] = 0.0
            features["ticket_sentiment"] = 3.0  # 中性默认值
            features["high_priority_count"] = 0.0

            if tickets is not None and not tickets.empty:
                user_tickets = tickets[
                    (tickets["user_id"] == uid) &
                    (pd.to_datetime(tickets["created_at"]) >= pd.Timestamp(lookback_start))
                ]
                features["ticket_count"] = float(len(user_tickets))
                if len(user_tickets) > 0 and "satisfaction" in user_tickets.columns:
                    valid_sat = user_tickets["satisfaction"].dropna()
                    if len(valid_sat) > 0:
                        # 满意度反转：低满意度=高风险
                        features["ticket_sentiment"] = float(valid_sat.mean())
                    high_prio = user_tickets[
                        user_tickets["priority"].isin(["high", "critical"])
                    ]
                    features["high_priority_count"] = float(len(high_prio))

            # ---- 特征: 付款延迟 ----
            features["payment_delay"] = 0.0
            if subscriptions is not None and not subscriptions.empty:
                user_subs = subscriptions[
                    subscriptions["user_id"] == uid
                ].sort_values("started_at")
                if len(user_subs) >= 2:
                    delays: list[float] = []
                    started = pd.to_datetime(user_subs["started_at"])
                    ended = pd.to_datetime(user_subs["ended_at"])
                    for i in range(1, len(user_subs)):
                        if pd.notna(ended.iloc[i - 1]) and pd.notna(started.iloc[i]):
                            delay = (started.iloc[i] - ended.iloc[i - 1]).days
                            if delay > 0:
                                delays.append(float(delay))
                    if delays:
                        features["payment_delay"] = round(
                            sum(delays) / len(delays), 2
                        )

            feature_rows.append(features)

        matrix = pd.DataFrame(feature_rows)
        logger.info("特征矩阵构建完成，%d 用户 x %d 特征", len(matrix), len(matrix.columns) - 3)
        return matrix

    def _compute_scores(self, feature_matrix: pd.DataFrame) -> list[ChurnScore]:
        """
        基于特征矩阵计算流失评分。

        评分逻辑:
            1. 对每个特征进行归一化到 [0, 1]
            2. 反转"正向"特征（如登录频率高 = 低风险）
            3. 加权求和得到最终分数
            4. 根据阈值划分风险等级

        Args:
            feature_matrix: 特征矩阵

        Returns:
            ChurnScore 列表
        """
        weights = self.config.weights
        thresholds = self.config.thresholds
        scores: list[ChurnScore] = []

        for _, row in feature_matrix.iterrows():
            uid = str(row["user_id"])

            # 归一化各特征
            # login_frequency: 0-1（高=好），反转为风险
            login_risk = 1.0 - min(float(row.get("login_frequency", 0)), 1.0)

            # ticket_count: 0-10（多=差），线性归一化
            ticket_risk = min(float(row.get("ticket_count", 0)) / 10.0, 1.0)

            # ticket_sentiment: 1-5（低=差），反转为风险
            sentiment = float(row.get("ticket_sentiment", 3.0))
            sentiment_risk = max(0, (5.0 - sentiment) / 4.0)

            # payment_delay: 0-30天（高=差）
            delay_risk = min(float(row.get("payment_delay", 0)) / 30.0, 1.0)

            # usage_decline: 0-1（高=差）
            decline_risk = min(float(row.get("usage_decline", 0)), 1.0)

            # feature_breadth: 0-5（低=差），反转为风险
            breadth = float(row.get("feature_breadth", 0))
            breadth_risk = max(0, 1.0 - breadth / 5.0)

            # session_trend: -1到1（负=差），转为 0-1
            trend = float(row.get("session_trend", 0))
            trend_risk = max(0, min(1, (1.0 - trend) / 2.0))

            # 加权求和
            churn_score = (
                weights.get("login_frequency", 0.20) * login_risk
                + weights.get("ticket_count", 0.15) * ticket_risk
                + weights.get("ticket_sentiment", 0.10) * sentiment_risk
                + weights.get("payment_delay", 0.15) * delay_risk
                + weights.get("usage_decline", 0.20) * decline_risk
                + weights.get("feature_breadth", 0.10) * breadth_risk
                + weights.get("session_trend", 0.10) * trend_risk
            )
            churn_score = round(max(0.0, min(1.0, churn_score)), 4)

            # 风险等级划分
            if churn_score < thresholds["low"]:
                risk_level = "low"
            elif churn_score < thresholds["medium"]:
                risk_level = "medium"
            elif churn_score < thresholds["high"]:
                risk_level = "high"
            else:
                risk_level = "critical"

            features = {
                "login_frequency": float(row.get("login_frequency", 0)),
                "ticket_count": float(row.get("ticket_count", 0)),
                "payment_delay": float(row.get("payment_delay", 0)),
                "usage_decline": float(row.get("usage_decline", 0)),
                "feature_breadth": float(row.get("feature_breadth", 0)),
                "session_trend": float(row.get("session_trend", 0)),
            }

            scores.append(ChurnScore(
                user_id=uid,
                score_date=self.score_date,
                churn_score=churn_score,
                risk_level=risk_level,
                features=features,
            ))

        return scores

    def run_sql(
        self,
        db_connection: DBConnection,
    ) -> pd.DataFrame:
        """
        使用 SQL 执行特征提取（需要后续 Python 评分）。

        Args:
            db_connection: 数据库连接

        Returns:
            特征 DataFrame
        """
        score_date = self.score_date
        recent_start = score_date - timedelta(days=self.config.recent_window_days)
        comparison_start = recent_start - timedelta(days=self.config.comparison_window_days)
        lookback_start = score_date - timedelta(days=self.config.lookback_days)

        cursor = db_connection.cursor()
        try:
            cursor.execute(SQL_CHURN_FEATURES, {
                "score_date": score_date,
                "recent_start": recent_start,
                "comparison_start": comparison_start,
                "lookback_start": lookback_start,
            })
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            result_df = pd.DataFrame(rows, columns=columns)
            logger.info("SQL 特征提取完成，%d 用户", len(result_df))
            return result_df
        except Exception as e:
            logger.error("SQL 流失特征提取失败: %s", e)
            raise
        finally:
            cursor.close()

    def get_score_distribution(self, scores: list[ChurnScore]) -> dict[str, Any]:
        """
        获取流失评分分布统计。

        Args:
            scores: 评分列表

        Returns:
            分布统计字典
        """
        if not scores:
            return {"total": 0}

        score_values = [s.churn_score for s in scores]
        risk_counts: dict[str, int] = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for s in scores:
            risk_counts[s.risk_level] += 1

        return {
            "total": len(scores),
            "mean_score": round(float(np.mean(score_values)), 4),
            "median_score": round(float(np.median(score_values)), 4),
            "std_score": round(float(np.std(score_values)), 4),
            "risk_distribution": risk_counts,
            "score_date": self.score_date.isoformat(),
        }

    def get_high_risk_users(
        self,
        scores: list[ChurnScore],
        min_risk: str = "high",
        limit: int = 100,
    ) -> list[ChurnScore]:
        """
        获取高风险用户列表。

        Args:
            scores: 评分列表
            min_risk: 最低风险等级
            limit: 返回数量上限

        Returns:
            高风险用户评分列表（按分数降序）
        """
        risk_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        min_level = risk_order.get(min_risk, 2)

        filtered = [
            s for s in scores
            if risk_order.get(s.risk_level, 0) >= min_level
        ]
        filtered.sort(key=lambda s: s.churn_score, reverse=True)
        return filtered[:limit]

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
        if "event_duration" not in result.columns:
            result["event_duration"] = 0
        return result

    def _preprocess_users(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理用户数据。"""
        result = df.copy()
        result["user_id"] = result["user_id"].astype(str)
        return result

    def _is_test_account(self, email: str) -> bool:
        """判断是否为测试账号。"""
        if not isinstance(email, str):
            return False
        return any(p in email.lower() for p in self.test_patterns)
