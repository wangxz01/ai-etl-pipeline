"""
测试数据生成脚本

使用 Faker 库生成真实感测试数据，覆盖用户、事件、订阅、订单、产品和工单表。
支持生成 10000+ 条记录，数据模式符合中国 SaaS 业务特征。

使用方法:
    python -m data.seed_data --records 10000 --output-dir ./data/output

依赖:
    pip install faker pandas numpy
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from faker import Faker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# 常量配置
# ============================================================

# 中国城市与地区映射
REGION_WEIGHTS: dict[str, float] = {
    "CN": 0.60,
    "US": 0.15,
    "EU": 0.10,
    "JP": 0.10,
    "SEA": 0.05,
}

# 套餐类型分布
PLAN_WEIGHTS: dict[str, float] = {
    "free": 0.45,
    "starter": 0.25,
    "pro": 0.20,
    "enterprise": 0.10,
}

# 套餐价格（CNY）
PLAN_PRICES: dict[str, dict[str, float]] = {
    "free": {"monthly": 0, "quarterly": 0, "yearly": 0},
    "starter": {"monthly": 99, "quarterly": 259, "yearly": 899},
    "pro": {"monthly": 299, "quarterly": 799, "yearly": 2799},
    "enterprise": {"monthly": 999, "quarterly": 2699, "yearly": 9599},
}

# 事件类型及其权重
EVENT_TYPES: dict[str, dict[str, float]] = {
    "page_view": {"weight": 0.40, "names": [
        "首页浏览", "产品页浏览", "定价页浏览", "文档页浏览",
        "博客浏览", "案例页浏览", "设置页浏览", "仪表盘浏览",
    ]},
    "click": {"weight": 0.25, "names": [
        "按钮点击", "导航点击", "链接点击", "菜单点击",
        "搜索点击", "标签切换", "下拉选择",
    ]},
    "feature_use": {"weight": 0.20, "names": [
        "报表生成", "数据导出", "仪表盘创建", "告警设置",
        "API调用", "团队邀请", "权限配置", "数据导入",
        "自定义查询", "邮件通知配置", "Webhook设置",
    ]},
    "api_call": {"weight": 0.15, "names": [
        "GET /api/v1/users", "POST /api/v1/reports", "GET /api/v1/metrics",
        "PUT /api/v1/settings", "POST /api/v1/exports", "GET /api/v1/dashboards",
        "DELETE /api/v1/alerts", "POST /api/v1/webhooks",
    ]},
}

# 产品目录
PRODUCT_CATALOG: list[dict[str, Any]] = [
    {"name": "基础版SaaS订阅", "category": "software", "price": 99.0, "cost": 15.0, "sku": "SaaS-001"},
    {"name": "专业版SaaS订阅", "category": "software", "price": 299.0, "cost": 40.0, "sku": "SaaS-002"},
    {"name": "企业版SaaS订阅", "category": "software", "price": 999.0, "cost": 120.0, "sku": "SaaS-003"},
    {"name": "数据采集网关", "category": "hardware", "price": 2999.0, "cost": 1200.0, "sku": "HW-001"},
    {"name": "边缘计算盒子", "category": "hardware", "price": 4999.0, "cost": 2000.0, "sku": "HW-002"},
    {"name": "实施部署服务", "category": "service", "price": 15000.0, "cost": 5000.0, "sku": "SVC-001"},
    {"name": "定制开发服务", "category": "service", "price": 30000.0, "cost": 12000.0, "sku": "SVC-002"},
    {"name": "7×24技术支持", "category": "service", "price": 5999.0, "cost": 2000.0, "sku": "SVC-003"},
    {"name": "高级分析模块", "category": "addon", "price": 199.0, "cost": 25.0, "sku": "ADD-001"},
    {"name": "AI智能预警模块", "category": "addon", "price": 399.0, "cost": 50.0, "sku": "ADD-002"},
    {"name": "多租户管理模块", "category": "addon", "price": 499.0, "cost": 60.0, "sku": "ADD-003"},
    {"name": "管理员培训课程", "category": "training", "price": 2999.0, "cost": 800.0, "sku": "TRN-001"},
    {"name": "数据分析师认证", "category": "training", "price": 1999.0, "cost": 500.0, "sku": "TRN-002"},
    {"name": "开发者高级课程", "category": "training", "price": 3499.0, "cost": 900.0, "sku": "TRN-003"},
]

# 工单类别
TICKET_CATEGORIES: list[str] = ["billing", "technical", "account", "feature_request", "bug", "other"]
TICKET_PRIORITY_WEIGHTS: dict[str, float] = {"low": 0.25, "medium": 0.40, "high": 0.25, "critical": 0.10}

# 中国公司名前缀和后缀
COMPANY_PREFIXES: list[str] = [
    "华云", "鼎新", "中科", "银河", "量子", "星辰", "蓝桥", "紫光",
    "金蝶", "用友", "浪潮", "宝信", "启明", "拓尔思", "海量", "数美",
]
COMPANY_SUFFIXES: list[str] = [
    "科技有限公司", "信息技术有限公司", "数据服务有限公司",
    "云计算有限公司", "智能科技有限公司", "网络科技有限公司",
]

# 中国行业分类
INDUSTRIES: list[str] = [
    "互联网", "金融", "制造业", "零售", "教育", "医疗健康",
    "物流", "房地产", "能源", "政府", "电信", "媒体",
]


class SeedDataGenerator:
    """
    测试数据生成器

    生成符合中国 SaaS 业务特征的测试数据，包括用户、事件、
    订阅、订单、产品和客服工单。所有数据可导出为 CSV 或
    SQL INSERT 语句。
    """

    def __init__(
        self,
        num_users: int = 2000,
        num_events_per_user: tuple[int, int] = (5, 50),
        start_date: date = date(2024, 1, 1),
        end_date: date = date(2025, 12, 31),
        seed: int = 42,
    ) -> None:
        """
        初始化数据生成器。

        Args:
            num_users: 生成的用户数量
            num_events_per_user: 每个用户的事件数量范围（min, max）
            start_date: 数据起始日期
            end_date: 数据结束日期
            seed: 随机种子，确保可复现
        """
        self.num_users = num_users
        self.events_range = num_events_per_user
        self.start_date = start_date
        self.end_date = end_date
        self.seed = seed

        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)

        # 初始化 Faker（中文和英文）
        self.faker_cn = Faker("zh_CN")
        self.faker_en = Faker("en_US")
        Faker.seed(seed)

        # 数据存储
        self.users: list[dict[str, Any]] = []
        self.user_events: list[dict[str, Any]] = []
        self.subscriptions: list[dict[str, Any]] = []
        self.orders: list[dict[str, Any]] = []
        self.products: list[dict[str, Any]] = []
        self.support_tickets: list[dict[str, Any]] = []

        # 索引缓存
        self._user_ids: list[str] = []
        self._product_ids: list[str] = []
        self._user_regions: dict[str, str] = {}
        self._user_plans: dict[str, str] = {}
        self._user_signup: dict[str, date] = {}

    def generate_all(self) -> dict[str, pd.DataFrame]:
        """
        执行完整的数据生成流程。

        Returns:
            包含所有表数据的 DataFrame 字典
        """
        logger.info("开始生成测试数据（seed=%d）...", self.seed)

        self._generate_products()
        logger.info("产品数据: %d 条", len(self.products))

        self._generate_users()
        logger.info("用户数据: %d 条", len(self.users))

        self._generate_subscriptions()
        logger.info("订阅数据: %d 条", len(self.subscriptions))

        self._generate_events()
        logger.info("事件数据: %d 条", len(self.user_events))

        self._generate_orders()
        logger.info("订单数据: %d 条", len(self.orders))

        self._generate_tickets()
        logger.info("工单数据: %d 条", len(self.support_tickets))

        return {
            "users": pd.DataFrame(self.users),
            "user_events": pd.DataFrame(self.user_events),
            "subscriptions": pd.DataFrame(self.subscriptions),
            "orders": pd.DataFrame(self.orders),
            "products": pd.DataFrame(self.products),
            "support_tickets": pd.DataFrame(self.support_tickets),
        }

    def _weighted_choice(self, weights: dict[str, float]) -> str:
        """根据权重字典随机选择一个键。"""
        keys = list(weights.keys())
        vals = list(weights.values())
        return random.choices(keys, weights=vals, k=1)[0]

    def _random_date_in_range(self, start: date | None = None, end: date | None = None) -> date:
        """在指定范围内生成随机日期。"""
        start = start or self.start_date
        end = end or self.end_date
        delta = (end - start).days
        if delta <= 0:
            return start
        return start + timedelta(days=random.randint(0, delta))

    def _generate_products(self) -> None:
        """生成产品目录数据。"""
        for prod in PRODUCT_CATALOG:
            self.products.append({
                "product_id": str(uuid.uuid4()),
                "name": prod["name"],
                "category": prod["category"],
                "subcategory": None,
                "price": prod["price"],
                "cost": prod["cost"],
                "sku": prod["sku"],
                "is_active": True,
                "launch_date": self._random_date_in_range(
                    date(2023, 1, 1), date(2024, 6, 30)
                ),
                "description": f"{prod['name']}，高质量产品，值得信赖",
                "created_at": datetime(2024, 1, 1),
                "updated_at": datetime(2024, 1, 1),
            })
            self._product_ids.append(self.products[-1]["product_id"])

    def _generate_users(self) -> None:
        """
        生成用户数据。

        根据地区分布和套餐权重生成具有中国业务特征的用户。
        包含少量测试账号（以 test_ 或 bot_ 开头的邮箱）。
        """
        # 中国姓氏和名字
        cn_surnames = ["王", "李", "张", "刘", "陈", "杨", "赵", "黄", "周", "吴",
                       "徐", "孙", "胡", "朱", "高", "林", "何", "郭", "马", "罗"]
        cn_given_names = ["伟", "芳", "娜", "秀英", "敏", "静", "丽", "强", "磊", "洋",
                          "勇", "艳", "杰", "娟", "涛", "明", "超", "秀兰", "霞", "平",
                          "鑫", "浩", "宇", "晨", "思远", "子涵", "雨泽", "博文", "梓轩", "思琪"]

        test_account_ratio = 0.02  # 2% 测试账号

        for i in range(self.num_users):
            region = self._weighted_choice(REGION_WEIGHTS)
            plan = self._weighted_choice(PLAN_WEIGHTS)
            signup = self._random_date_in_range()

            # 生成邮箱
            is_test = random.random() < test_account_ratio
            if is_test:
                email_prefix = random.choice(["test_", "bot_", "demo_", "qa_"])
                email = f"{email_prefix}user{i:05d}@test-company.com"
            elif region == "CN":
                surname = random.choice(cn_surnames)
                given = random.choice(cn_given_names)
                pinyin_first = self.faker_en.first_name().lower()
                pinyin_last = self.faker_en.last_name().lower()
                domains = ["163.com", "qq.com", "126.com", "gmail.com", "outlook.com", "company.cn"]
                email = f"{pinyin_last}{pinyin_first}{random.randint(10, 9999)}@{random.choice(domains)}"
            else:
                email = self.faker_en.email()

            # 生成用户名
            if region == "CN":
                username = f"{random.choice(cn_surnames)}{random.choice(cn_given_names)}"
            else:
                username = self.faker_en.user_name()

            # 电话号码
            if region == "CN":
                phone = f"1{random.choice(['3', '5', '7', '8', '9'])}{random.randint(100000000, 999999999)}"
            else:
                phone = self.faker_en.phone_number()[:20]

            # 状态：大部分活跃，部分流失或暂停
            status = random.choices(
                ["active", "suspended", "churned", "trial"],
                weights=[0.65, 0.05, 0.20, 0.10],
                k=1,
            )[0]

            # 公司和行业（企业用户才有）
            company = None
            industry = None
            if plan in ("pro", "enterprise") and random.random() < 0.85:
                company = f"{random.choice(COMPANY_PREFIXES)}{random.choice(COMPANY_SUFFIXES)}"
                industry = random.choice(INDUSTRIES)

            # MRR
            mrr = PLAN_PRICES[plan]["monthly"] if status == "active" else 0

            # 时区
            tz_map = {"CN": "Asia/Shanghai", "US": "America/New_York",
                      "EU": "Europe/Berlin", "JP": "Asia/Tokyo", "SEA": "Asia/Singapore"}

            user_id = str(uuid.uuid4())
            self.users.append({
                "user_id": user_id,
                "email": email,
                "username": username,
                "phone": phone,
                "signup_date": signup,
                "plan_type": plan,
                "region": region,
                "status": status,
                "company": company,
                "industry": industry,
                "mrr": mrr,
                "timezone": tz_map.get(region, "UTC"),
                "created_at": datetime.combine(signup, datetime.min.time()),
                "updated_at": datetime.combine(signup, datetime.min.time()),
            })
            self._user_ids.append(user_id)
            self._user_regions[user_id] = region
            self._user_plans[user_id] = plan
            self._user_signup[user_id] = signup

    def _generate_subscriptions(self) -> None:
        """
        生成订阅数据。

        每个付费用户至少一条订阅记录，部分用户有升级/降级历史。
        """
        for user_id in self._user_ids:
            plan = self._user_plans[user_id]
            signup = self._user_signup[user_id]

            # 免费用户跳过
            if plan == "free":
                continue

            # 主要订阅
            billing_cycle = random.choices(
                ["monthly", "quarterly", "yearly"],
                weights=[0.50, 0.30, 0.20],
                k=1,
            )[0]
            amount = PLAN_PRICES[plan][billing_cycle]
            currency = "CNY" if self._user_regions[user_id] == "CN" else random.choice(["USD", "EUR"])

            started_at = datetime.combine(signup, datetime.min.time()) + timedelta(
                days=random.randint(0, 3)
            )

            # 是否已取消
            is_cancelled = random.random() < 0.20
            ended_at = None
            cancelled_at = None
            cancel_reason = None
            sub_status = "active"

            if is_cancelled:
                duration_map = {"monthly": 30, "quarterly": 90, "yearly": 365}
                sub_duration = timedelta(days=random.randint(30, duration_map[billing_cycle] * 3))
                ended_at = started_at + sub_duration
                cancelled_at = ended_at - timedelta(days=random.randint(0, 7))
                cancel_reason = random.choice([
                    "价格过高", "功能不满足需求", "转向竞品",
                    "项目结束", "预算缩减", "团队调整",
                ])
                sub_status = random.choice(["cancelled", "expired"])

            self.subscriptions.append({
                "sub_id": str(uuid.uuid4()),
                "user_id": user_id,
                "plan": plan,
                "amount": amount,
                "currency": currency,
                "billing_cycle": billing_cycle,
                "started_at": started_at,
                "ended_at": ended_at,
                "cancelled_at": cancelled_at,
                "cancel_reason": cancel_reason,
                "status": sub_status,
                "created_at": started_at,
                "updated_at": ended_at or started_at,
            })

            # 部分用户有历史升级/降级（30%概率）
            if random.random() < 0.30:
                plans_ordered = ["starter", "pro", "enterprise"]
                current_idx = plans_ordered.index(plan) if plan in plans_ordered else 0
                if current_idx > 0:
                    old_plan = plans_ordered[current_idx - 1]
                    old_amount = PLAN_PRICES[old_plan]["monthly"]
                    old_start = started_at - timedelta(days=random.randint(30, 180))
                    self.subscriptions.append({
                        "sub_id": str(uuid.uuid4()),
                        "user_id": user_id,
                        "plan": old_plan,
                        "amount": old_amount,
                        "currency": currency,
                        "billing_cycle": "monthly",
                        "started_at": old_start,
                        "ended_at": started_at,
                        "cancelled_at": started_at,
                        "cancel_reason": "套餐升级",
                        "status": "expired",
                        "created_at": old_start,
                        "updated_at": started_at,
                    })

    def _generate_events(self) -> None:
        """
        生成用户行为事件数据。

        每个用户生成 5-50 条事件，活跃用户事件更多。
        事件时间集中在工作时间和工作日。
        """
        event_id = 0
        for user_id in self._user_ids:
            signup = self._user_signup[user_id]
            user_status = next(
                (u["status"] for u in self.users if u["user_id"] == user_id), "active"
            )

            # 根据用户状态调整事件数量
            if user_status == "churned":
                num_events = random.randint(2, 10)
            elif user_status == "active":
                num_events = random.randint(self.events_range[0], self.events_range[1])
            else:
                num_events = random.randint(3, 20)

            for _ in range(num_events):
                # 事件日期在注册后到当前范围内
                event_date = self._random_date_in_range(
                    signup, min(self.end_date, signup + timedelta(days=365))
                )

                # 事件时间偏向工作时间（9-18点）
                hour = int(np.clip(np.random.normal(14, 4), 0, 23))
                minute = random.randint(0, 59)
                second = random.randint(0, 59)
                event_ts = datetime(
                    event_date.year, event_date.month, event_date.day,
                    hour, minute, second,
                )

                # 选择事件类型
                type_weights = {k: v["weight"] for k, v in EVENT_TYPES.items()}
                event_type = self._weighted_choice(type_weights)
                event_name = random.choice(EVENT_TYPES[event_type]["names"])

                # 页面URL
                pages = [
                    "/dashboard", "/products", "/pricing", "/docs/api",
                    "/settings/profile", "/settings/billing", "/reports",
                    "/analytics", "/team", "/integrations", "/login",
                    "/signup", "/features", "/blog", "/about",
                ]
                page_url = random.choice(pages) if event_type == "page_view" else None

                # 设备类型
                device = random.choices(
                    ["desktop", "mobile", "tablet"],
                    weights=[0.60, 0.30, 0.10],
                    k=1,
                )[0]

                # 浏览器和操作系统
                browsers = ["Chrome", "Firefox", "Safari", "Edge", "WeChat"]
                os_list = ["Windows 10", "macOS", "Linux", "iOS", "Android"]
                if device == "mobile":
                    browser = random.choice(["Chrome Mobile", "Safari", "WeChat"])
                    os_name = random.choice(["iOS", "Android"])
                else:
                    browser = random.choice(browsers[:4])
                    os_name = random.choice(os_list[:3])

                # 会话ID（同一用户同一日期共享会话的概率40%）
                session_id = str(uuid.uuid4())

                # IP地址
                region = self._user_regions[user_id]
                if region == "CN":
                    ip = f"{random.choice(['116', '117', '183', '202', '211', '223'])}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
                else:
                    ip = str(self.faker_en.ipv4_public())

                # 事件持续时长
                if event_type == "page_view":
                    duration = random.randint(1000, 300000)  # 1秒到5分钟
                elif event_type == "feature_use":
                    duration = random.randint(5000, 600000)  # 5秒到10分钟
                else:
                    duration = random.randint(50, 5000)

                event_id += 1
                self.user_events.append({
                    "event_id": event_id,
                    "user_id": user_id,
                    "event_type": event_type,
                    "event_name": event_name,
                    "page_url": page_url,
                    "referrer_url": None,
                    "device": device,
                    "browser": browser,
                    "os": os_name,
                    "ip": ip,
                    "country": region,
                    "session_id": session_id,
                    "event_duration": duration,
                    "event_date": event_date,
                    "event_timestamp": event_ts,
                    "created_at": event_ts,
                })

    def _generate_orders(self) -> None:
        """
        生成订单数据。

        每个付费用户有 1-8 笔订单，订单状态分布符合真实业务。
        """
        for user_id in self._user_ids:
            plan = self._user_plans[user_id]
            signup = self._user_signup[user_id]

            # 订单数量取决于套餐等级
            if plan == "free":
                num_orders = random.choices([0, 1], weights=[0.80, 0.20], k=1)[0]
            elif plan == "starter":
                num_orders = random.randint(1, 3)
            elif plan == "pro":
                num_orders = random.randint(2, 5)
            else:
                num_orders = random.randint(3, 8)

            for _ in range(num_orders):
                product = random.choice(self.products)
                order_date = self._random_date_in_range(
                    signup, min(self.end_date, signup + timedelta(days=365))
                )

                quantity = random.choices(
                    [1, 2, 3, 5, 10],
                    weights=[0.55, 0.20, 0.10, 0.10, 0.05],
                    k=1,
                )[0]

                # 折扣
                discount = random.choices(
                    [0, 5, 10, 15, 20, 30],
                    weights=[0.40, 0.20, 0.15, 0.10, 0.10, 0.05],
                    k=1,
                )[0]

                # 订单状态
                status = random.choices(
                    ["pending", "paid", "shipped", "delivered", "refunded", "cancelled"],
                    weights=[0.05, 0.10, 0.05, 0.65, 0.08, 0.07],
                    k=1,
                )[0]

                # 支付方式
                payment_methods = [
                    "alipay", "wechat_pay", "bank_transfer",
                    "credit_card", "paypal",
                ]
                region = self._user_regions[user_id]
                if region == "CN":
                    payment = random.choices(
                        payment_methods[:3], weights=[0.40, 0.35, 0.25], k=1,
                    )[0]
                else:
                    payment = random.choices(
                        payment_methods[3:], weights=[0.50, 0.50], k=1,
                    )[0]

                fulfilled_date = None
                if status in ("shipped", "delivered"):
                    fulfilled_date = order_date + timedelta(days=random.randint(1, 7))

                amount = product["price"]
                tax = round(amount * quantity * 0.06, 2)  # 6% 税率
                shipping = random.choice([0, 0, 0, 15, 25])  # 大部分免运费

                self.orders.append({
                    "order_id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "product_id": product["product_id"],
                    "amount": amount,
                    "quantity": quantity,
                    "discount": discount,
                    "tax": tax,
                    "shipping_cost": shipping,
                    "currency": "CNY" if region == "CN" else "USD",
                    "payment_method": payment,
                    "status": status,
                    "order_date": order_date,
                    "fulfilled_date": fulfilled_date,
                    "created_at": datetime.combine(order_date, datetime.min.time()),
                    "updated_at": datetime.combine(order_date, datetime.min.time()),
                })

    def _generate_tickets(self) -> None:
        """
        生成客服工单数据。

        约 30% 的用户会提交工单，活跃用户和高套餐用户工单略多。
        """
        for user_id in self._user_ids:
            plan = self._user_plans[user_id]
            signup = self._user_signup[user_id]

            # 工单概率
            ticket_prob = {"free": 0.10, "starter": 0.25, "pro": 0.35, "enterprise": 0.45}
            if random.random() > ticket_prob.get(plan, 0.20):
                continue

            num_tickets = random.randint(1, 4)
            for _ in range(num_tickets):
                category = random.choice(TICKET_CATEGORIES)
                priority = self._weighted_choice(TICKET_PRIORITY_WEIGHTS)

                created = datetime.combine(
                    self._random_date_in_range(
                        signup, min(self.end_date, signup + timedelta(days=300))
                    ),
                    datetime.min.time(),
                ) + timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))

                # 工单主题
                subjects_map: dict[str, list[str]] = {
                    "billing": ["发票开具问题", "退款申请", "账单金额疑问", "付款失败", "升级套餐咨询"],
                    "technical": ["页面加载缓慢", "数据导出失败", "API返回500错误", "报表显示异常", "登录验证码收不到"],
                    "account": ["密码重置", "账号冻结", "权限变更申请", "团队成员管理", "域名绑定问题"],
                    "feature_request": ["希望增加数据对比功能", "建议增加暗黑模式", "移动端体验优化", "多语言支持"],
                    "bug": ["仪表盘数据不刷新", "通知未正常推送", "时区显示错误", "导出文件乱码"],
                    "other": ["合作咨询", "资质查询", "数据安全合规", "SLA保障"],
                }
                subject = random.choice(subjects_map.get(category, ["一般咨询"]))

                # 状态和解决时间
                status = random.choices(
                    ["open", "in_progress", "waiting_customer", "resolved", "closed"],
                    weights=[0.05, 0.10, 0.05, 0.40, 0.40],
                    k=1,
                )[0]

                resolved_at = None
                first_response_hours = None
                satisfaction = None

                if status in ("resolved", "closed"):
                    resolve_hours = random.randint(1, 168)  # 1小时到7天
                    resolved_at = created + timedelta(hours=resolve_hours)
                    first_response_hours = random.randint(1, 24)
                    satisfaction = random.choices(
                        [1, 2, 3, 4, 5],
                        weights=[0.05, 0.10, 0.20, 0.35, 0.30],
                        k=1,
                    )[0]

                self.support_tickets.append({
                    "ticket_id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "subject": subject,
                    "category": category,
                    "priority": priority,
                    "status": status,
                    "assigned_to": f"agent_{random.randint(1, 10):03d}",
                    "first_response_at": (
                        created + timedelta(hours=first_response_hours)
                        if first_response_hours else None
                    ),
                    "resolution_note": "已解决" if resolved_at else None,
                    "satisfaction": satisfaction,
                    "tags": [],
                    "created_at": created,
                    "updated_at": resolved_at or created,
                    "resolved_at": resolved_at,
                })

    def export_csv(self, output_dir: str | Path) -> None:
        """
        将所有数据导出为 CSV 文件。

        Args:
            output_dir: 输出目录路径
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        data = self.generate_all()
        for table_name, df in data.items():
            file_path = output_dir / f"{table_name}.csv"
            df.to_csv(file_path, index=False, encoding="utf-8-sig")
            logger.info("导出 %s -> %s (%d 行)", table_name, file_path, len(df))

    def export_sql(self, output_dir: str | Path) -> None:
        """
        将所有数据导出为 SQL INSERT 语句文件。

        Args:
            output_dir: 输出目录路径
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        data = self.generate_all()
        for table_name, df in data.items():
            file_path = output_dir / f"{table_name}_insert.sql"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"-- {table_name} 数据插入语句\n")
                f.write(f"-- 共 {len(df)} 条记录\n\n")
                for _, row in df.iterrows():
                    cols = ", ".join(df.columns)
                    vals = []
                    for v in row:
                        if v is None:
                            vals.append("NULL")
                        elif isinstance(v, (int, float)):
                            vals.append(str(v))
                        elif isinstance(v, date):
                            vals.append(f"'{v.isoformat()}'")
                        elif isinstance(v, datetime):
                            vals.append(f"'{v.isoformat()}'")
                        else:
                            escaped = str(v).replace("'", "''")
                            vals.append(f"'{escaped}'")
                    f.write(f"INSERT INTO {table_name} ({cols}) VALUES ({', '.join(vals)});\n")
            logger.info("导出 SQL %s -> %s", table_name, file_path)

    def get_statistics(self) -> dict[str, Any]:
        """
        返回生成数据的统计摘要。

        Returns:
            各表的行数和关键统计信息
        """
        data = self.generate_all()
        stats: dict[str, Any] = {}
        for table_name, df in data.items():
            stats[table_name] = {
                "row_count": len(df),
                "columns": list(df.columns),
            }
            if table_name == "users":
                stats[table_name]["plan_distribution"] = df["plan_type"].value_counts().to_dict()
                stats[table_name]["region_distribution"] = df["region"].value_counts().to_dict()
                stats[table_name]["status_distribution"] = df["status"].value_counts().to_dict()
            elif table_name == "user_events":
                stats[table_name]["type_distribution"] = df["event_type"].value_counts().to_dict()
            elif table_name == "orders":
                stats[table_name]["total_revenue"] = float(df["amount"].sum())
                stats[table_name]["avg_order_value"] = float(df["amount"].mean())
        return stats


def main() -> None:
    """命令行入口函数。"""
    parser = argparse.ArgumentParser(description="SaaS 业务测试数据生成器")
    parser.add_argument(
        "--users", type=int, default=2000,
        help="用户数量（默认 2000）",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./data/output",
        help="输出目录（默认 ./data/output）",
    )
    parser.add_argument(
        "--format", type=str, choices=["csv", "sql", "both"], default="csv",
        help="输出格式（默认 csv）",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（默认 42）",
    )
    parser.add_argument(
        "--stats-only", action="store_true",
        help="仅输出统计信息，不导出文件",
    )

    args = parser.parse_args()

    generator = SeedDataGenerator(
        num_users=args.users,
        seed=args.seed,
    )

    if args.stats_only:
        stats = generator.get_statistics()
        print(json.dumps(stats, indent=2, ensure_ascii=False, default=str))
        return

    if args.format in ("csv", "both"):
        generator.export_csv(args.output_dir)
    if args.format in ("sql", "both"):
        generator.export_sql(args.output_dir)

    # 输出统计摘要
    stats = generator.get_statistics()
    print("\n=== 数据生成统计 ===")
    for table, info in stats.items():
        print(f"  {table}: {info['row_count']} 行")
    print(f"\n总记录数: {sum(s['row_count'] for s in stats.values())}")


if __name__ == "__main__":
    main()
