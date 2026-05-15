-- ============================================================
-- SaaS 业务数据库模式定义
-- 版本: 1.0.0
-- 描述: 包含用户、事件、订阅、订单、产品、客服工单等核心业务表
-- ============================================================

-- 启用必要扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- 模糊搜索支持

-- ============================================================
-- 用户表：存储核心用户信息
-- ============================================================
DROP TABLE IF EXISTS user_events CASCADE;
DROP TABLE IF EXISTS support_tickets CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS subscriptions CASCADE;
DROP TABLE IF EXISTS products CASCADE;
DROP TABLE IF EXISTS users CASCADE;

CREATE TABLE users (
    user_id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email           VARCHAR(320) NOT NULL,
    username        VARCHAR(100),
    phone           VARCHAR(20),
    signup_date     DATE NOT NULL DEFAULT CURRENT_DATE,
    plan_type       VARCHAR(20) NOT NULL DEFAULT 'free'
                        CHECK (plan_type IN ('free', 'starter', 'pro', 'enterprise')),
    region          VARCHAR(10) NOT NULL DEFAULT 'CN'
                        CHECK (region IN ('CN', 'US', 'EU', 'JP', 'SEA')),
    status          VARCHAR(20) NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active', 'suspended', 'churned', 'trial')),
    company         VARCHAR(200),           -- 公司名
    industry        VARCHAR(100),           -- 行业
    mrr             NUMERIC(12, 2) DEFAULT 0,  -- 月经常性收入
    timezone        VARCHAR(50) DEFAULT 'Asia/Shanghai',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- 约束：邮箱格式校验
    CONSTRAINT uq_users_email UNIQUE (email),
    CONSTRAINT ck_users_email_format CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
);

-- 用户表索引
CREATE INDEX idx_users_signup_date ON users (signup_date);
CREATE INDEX idx_users_plan_type ON users (plan_type);
CREATE INDEX idx_users_region ON users (region);
CREATE INDEX idx_users_status ON users (status);
CREATE INDEX idx_users_email_trgm ON users USING gin (email gin_trgm_ops);  -- 邮箱模糊搜索
CREATE INDEX idx_users_created_at ON users (created_at);

COMMENT ON TABLE users IS '核心用户表，包含注册信息、套餐、地区和状态';
COMMENT ON COLUMN users.mrr IS '用户当前月经常性收入，由订阅聚合计算';
COMMENT ON COLUMN users.timezone IS '用户所在时区，用于事件时间规范化';

-- ============================================================
-- 用户事件表：记录用户所有行为事件（页面访问、点击、功能使用等）
-- ============================================================
CREATE TABLE user_events (
    event_id        BIGSERIAL PRIMARY KEY,
    user_id         UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    event_type      VARCHAR(50) NOT NULL,   -- 事件类型：page_view, click, feature_use, api_call
    event_name      VARCHAR(200) NOT NULL,  -- 具体事件名称
    page_url        VARCHAR(2000),          -- 页面URL
    referrer_url    VARCHAR(2000),          -- 来源URL
    device          VARCHAR(20) NOT NULL DEFAULT 'desktop'
                        CHECK (device IN ('desktop', 'mobile', 'tablet')),
    browser         VARCHAR(50),            -- 浏览器类型
    os              VARCHAR(50),            -- 操作系统
    ip              INET,                   -- 客户端IP地址
    country         VARCHAR(10),            -- 国家代码
    session_id      UUID,                   -- 会话ID
    event_duration  INTEGER DEFAULT 0,      -- 事件持续时长（毫秒）
    event_date      DATE NOT NULL,          -- 事件日期（用于分区和聚合）
    event_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- 约束
    CONSTRAINT ck_event_duration_nonneg CHECK (event_duration >= 0)
);

-- 事件表索引（高频查询优化）
CREATE INDEX idx_events_user_id ON user_events (user_id);
CREATE INDEX idx_events_event_date ON user_events (event_date);
CREATE INDEX idx_events_event_type ON user_events (event_type);
CREATE INDEX idx_events_user_date ON user_events (user_id, event_date);
CREATE INDEX idx_events_type_date ON user_events (event_type, event_date);
CREATE INDEX idx_events_session ON user_events (session_id);
CREATE INDEX idx_events_timestamp ON user_events (event_timestamp);

COMMENT ON TABLE user_events IS '用户行为事件表，支撑DAU/MAU、留存、漏斗等分析';
COMMENT ON COLUMN user_events.event_type IS '事件类型分类：page_view=页面浏览, click=点击, feature_use=功能使用, api_call=API调用';
COMMENT ON COLUMN user_events.session_id IS '会话标识，30分钟无活动自动过期';

-- ============================================================
-- 订阅表：记录用户订阅历史和当前订阅状态
-- ============================================================
CREATE TABLE subscriptions (
    sub_id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    plan            VARCHAR(20) NOT NULL
                        CHECK (plan IN ('free', 'starter', 'pro', 'enterprise')),
    amount          NUMERIC(12, 2) NOT NULL CHECK (amount >= 0),
    currency        VARCHAR(3) NOT NULL DEFAULT 'CNY'
                        CHECK (currency IN ('CNY', 'USD', 'EUR', 'JPY')),
    billing_cycle   VARCHAR(10) NOT NULL DEFAULT 'monthly'
                        CHECK (billing_cycle IN ('monthly', 'quarterly', 'yearly')),
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at        TIMESTAMPTZ,            -- NULL 表示当前有效订阅
    cancelled_at    TIMESTAMPTZ,            -- 取消时间（区别于到期时间）
    cancel_reason   VARCHAR(200),           -- 取消原因
    status          VARCHAR(20) NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active', 'trialing', 'past_due', 'cancelled', 'expired')),
    mrr_amount      NUMERIC(12, 2) GENERATED ALWAYS AS (
                        CASE
                            WHEN billing_cycle = 'monthly' THEN amount
                            WHEN billing_cycle = 'quarterly' THEN amount / 3.0
                            WHEN billing_cycle = 'yearly' THEN amount / 12.0
                            ELSE amount
                        END
                    ) STORED,               -- 自动计算月化金额
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- 约束：结束时间必须晚于开始时间
    CONSTRAINT ck_sub_date_range CHECK (ended_at IS NULL OR ended_at > started_at)
);

-- 订阅表索引
CREATE INDEX idx_subs_user_id ON subscriptions (user_id);
CREATE INDEX idx_subs_status ON subscriptions (status);
CREATE INDEX idx_subs_plan ON subscriptions (plan);
CREATE INDEX idx_subs_started ON subscriptions (started_at);
CREATE INDEX idx_subs_ended ON subscriptions (ended_at);
CREATE INDEX idx_subs_user_status ON subscriptions (user_id, status);
CREATE INDEX idx_subs_date_range ON subscriptions (started_at, ended_at);

COMMENT ON TABLE subscriptions IS '用户订阅表，支撑MRR/ARR计算和流失分析';
COMMENT ON COLUMN subscriptions.mrr_amount IS '月化金额，由数据库自动计算';

-- ============================================================
-- 产品表：产品目录
-- ============================================================
CREATE TABLE products (
    product_id      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            VARCHAR(200) NOT NULL,
    category        VARCHAR(50) NOT NULL
                        CHECK (category IN ('software', 'hardware', 'service', 'addon', 'training')),
    subcategory     VARCHAR(50),
    price           NUMERIC(12, 2) NOT NULL CHECK (price >= 0),
    cost            NUMERIC(12, 2) NOT NULL DEFAULT 0 CHECK (cost >= 0),
    sku             VARCHAR(50) NOT NULL UNIQUE,
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    launch_date     DATE DEFAULT CURRENT_DATE,
    description     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 产品表索引
CREATE INDEX idx_products_category ON products (category);
CREATE INDEX idx_products_sku ON products (sku);
CREATE INDEX idx_products_active ON products (is_active) WHERE is_active = TRUE;

COMMENT ON TABLE products IS '产品目录，包含软件、硬件、服务和培训等类型';

-- ============================================================
-- 订单表：记录所有交易订单
-- ============================================================
CREATE TABLE orders (
    order_id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    product_id      UUID NOT NULL REFERENCES products(product_id),
    amount          NUMERIC(12, 2) NOT NULL CHECK (amount >= 0),
    quantity        INTEGER NOT NULL DEFAULT 1 CHECK (quantity > 0),
    discount        NUMERIC(5, 2) NOT NULL DEFAULT 0 CHECK (discount BETWEEN 0 AND 100),
    tax             NUMERIC(12, 2) NOT NULL DEFAULT 0,
    shipping_cost   NUMERIC(12, 2) NOT NULL DEFAULT 0,
    total_amount    NUMERIC(12, 2) GENERATED ALWAYS AS (
                        ROUND(amount * quantity * (1 - discount / 100.0) + tax + shipping_cost, 2)
                    ) STORED,               -- 自动计算总价
    currency        VARCHAR(3) NOT NULL DEFAULT 'CNY',
    payment_method  VARCHAR(30),            -- 支付方式
    status          VARCHAR(20) NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'paid', 'shipped', 'delivered', 'refunded', 'cancelled')),
    order_date      DATE NOT NULL DEFAULT CURRENT_DATE,
    fulfilled_date  DATE,                   -- 发货日期
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 订单表索引
CREATE INDEX idx_orders_user_id ON orders (user_id);
CREATE INDEX idx_orders_product_id ON orders (product_id);
CREATE INDEX idx_orders_status ON orders (status);
CREATE INDEX idx_orders_date ON orders (order_date);
CREATE INDEX idx_orders_user_date ON orders (user_id, order_date);
CREATE INDEX idx_orders_created_at ON orders (created_at);

COMMENT ON TABLE orders IS '交易订单表，包含金额、折扣、支付状态等信息';
COMMENT ON COLUMN orders.total_amount IS '订单总金额，由系统自动计算';

-- ============================================================
-- 客服工单表：用户支持请求
-- ============================================================
CREATE TABLE support_tickets (
    ticket_id       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    subject         VARCHAR(500) NOT NULL,
    category        VARCHAR(50) NOT NULL
                        CHECK (category IN ('billing', 'technical', 'account', 'feature_request', 'bug', 'other')),
    priority        VARCHAR(10) NOT NULL DEFAULT 'medium'
                        CHECK (priority IN ('low', 'medium', 'high', 'critical')),
    status          VARCHAR(20) NOT NULL DEFAULT 'open'
                        CHECK (status IN ('open', 'in_progress', 'waiting_customer', 'resolved', 'closed')),
    assigned_to     VARCHAR(100),           -- 客服人员
    first_response_at TIMESTAMPTZ,          -- 首次响应时间
    resolution_note TEXT,                   -- 解决方案说明
    satisfaction    SMALLINT CHECK (satisfaction BETWEEN 1 AND 5),  -- 满意度评分
    tags            VARCHAR(200)[],         -- 标签数组
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at     TIMESTAMPTZ,            -- 解决时间

    -- 约束：解决时间必须晚于创建时间
    CONSTRAINT ck_ticket_resolved_after_created CHECK (resolved_at IS NULL OR resolved_at >= created_at)
);

-- 工单表索引
CREATE INDEX idx_tickets_user_id ON support_tickets (user_id);
CREATE INDEX idx_tickets_category ON support_tickets (category);
CREATE INDEX idx_tickets_priority ON support_tickets (priority);
CREATE INDEX idx_tickets_status ON support_tickets (status);
CREATE INDEX idx_tickets_created ON support_tickets (created_at);
CREATE INDEX idx_tickets_resolved ON support_tickets (resolved_at) WHERE resolved_at IS NOT NULL;
CREATE INDEX idx_tickets_user_created ON support_tickets (user_id, created_at);

COMMENT ON TABLE support_tickets IS '客服工单表，追踪用户支持请求和满意度';
COMMENT ON COLUMN support_tickets.satisfaction IS '用户满意度评分，1=非常不满意，5=非常满意';

-- ============================================================
-- ETL 输出表：聚合指标存储
-- ============================================================

-- DAU/MAU 指标表
CREATE TABLE IF NOT EXISTS metrics.dau_mau (
    metric_date     DATE NOT NULL,
    metric_name     VARCHAR(50) NOT NULL,   -- dau, wau, mau, sticky_factor
    metric_value    BIGINT NOT NULL,
    region          VARCHAR(10) DEFAULT 'ALL',
    plan_type       VARCHAR(20) DEFAULT 'ALL',
    device          VARCHAR(20) DEFAULT 'ALL',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (metric_date, metric_name, region, plan_type, device)
);

COMMENT ON TABLE metrics.dau_mau IS 'DAU/MAU 聚合指标表，按地区、套餐、设备维度拆分';

-- 留存分析表
CREATE TABLE IF NOT EXISTS metrics.retention (
    cohort_date     DATE NOT NULL,          -- 群组起始日期
    cohort_size     INTEGER NOT NULL,       -- 群组初始人数
    period_type     VARCHAR(10) NOT NULL,   -- week / month
    period_number   INTEGER NOT NULL,       -- 第N个周期（0=注册当期）
    retained_users  INTEGER NOT NULL,       -- 留存人数
    retention_rate  NUMERIC(5, 4) NOT NULL, -- 留存率
    region          VARCHAR(10) DEFAULT 'ALL',
    plan_type       VARCHAR(20) DEFAULT 'ALL',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (cohort_date, period_type, period_number, region, plan_type)
);

COMMENT ON TABLE metrics.retention IS '用户留存分析表，按周/月群组追踪留存率';

-- 流失评分表
CREATE TABLE IF NOT EXISTS metrics.churn_scores (
    user_id         UUID NOT NULL REFERENCES users(user_id),
    score_date      DATE NOT NULL,
    churn_score     NUMERIC(5, 4) NOT NULL CHECK (churn_score BETWEEN 0 AND 1),
    risk_level      VARCHAR(10) NOT NULL CHECK (risk_level IN ('low', 'medium', 'high', 'critical')),
    login_frequency NUMERIC(8, 2),          -- 登录频率特征
    ticket_count    INTEGER,                -- 工单数量特征
    payment_delay   NUMERIC(8, 2),          -- 付款延迟特征
    usage_decline   NUMERIC(5, 4),          -- 使用下降率
    features_used   JSONB,                  -- 使用的功能列表
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, score_date)
);

COMMENT ON TABLE metrics.churn_scores IS '用户流失风险评分表，基于多维度特征';

-- 收入指标表
CREATE TABLE IF NOT EXISTS metrics.revenue (
    metric_date     DATE NOT NULL,
    metric_name     VARCHAR(50) NOT NULL,   -- mrr, arr, revenue, arpu, arppu
    metric_value    NUMERIC(15, 2) NOT NULL,
    currency        VARCHAR(3) DEFAULT 'CNY',
    plan_type       VARCHAR(20) DEFAULT 'ALL',
    region          VARCHAR(10) DEFAULT 'ALL',
    segment         VARCHAR(50) DEFAULT 'ALL',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (metric_date, metric_name, currency, plan_type, region, segment)
);

COMMENT ON TABLE metrics.revenue IS '收入相关指标聚合表，包含MRR/ARR/ARPU等';

-- LTV 群组表
CREATE TABLE IF NOT EXISTS metrics.cohort_ltv (
    cohort_month    DATE NOT NULL,
    months_since    INTEGER NOT NULL,       -- 距群组起始的月数
    cohort_size     INTEGER NOT NULL,
    active_users    INTEGER NOT NULL,
    total_revenue   NUMERIC(15, 2) NOT NULL,
    arpu            NUMERIC(12, 2),
    cum_ltv         NUMERIC(12, 2),         -- 累计LTV
    projected_ltv   NUMERIC(12, 2),         -- 预测LTV
    plan_type       VARCHAR(20) DEFAULT 'ALL',
    region          VARCHAR(10) DEFAULT 'ALL',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (cohort_month, months_since, plan_type, region)
);

COMMENT ON TABLE metrics.cohort_ltv IS '群组LTV分析表，追踪不同群组的累计和预测LTV';

-- 数据质量报告表
CREATE TABLE IF NOT EXISTS metrics.data_quality (
    report_date     DATE NOT NULL,
    table_name      VARCHAR(100) NOT NULL,
    column_name     VARCHAR(100),
    quality_score   NUMERIC(5, 2) CHECK (quality_score BETWEEN 0 AND 100),
    check_type      VARCHAR(50) NOT NULL,   -- null_rate, distinct_count, anomaly, etc.
    check_result    JSONB NOT NULL,
    severity        VARCHAR(10) DEFAULT 'info',
    message         TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (report_date, table_name, column_name, check_type)
);

COMMENT ON TABLE metrics.data_quality IS '数据质量检查结果表';
