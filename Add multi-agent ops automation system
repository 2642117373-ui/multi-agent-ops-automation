"""
Multi-Agent 协同运营自动化系统（单文件 MVP）
====================================================

功能：
1. 多 Agent 协同：Planner / Executor / Reviewer / Reporter
2. 支持运营任务自动拆解、执行、质检、生成日报
3. SQLite 持久化任务、日志、执行结果
4. FastAPI 提供接口：创建任务、查看任务、运行任务、查看报告
5. 支持两种 LLM 模式：
   - mock：无需 API Key，可直接运行演示
   - openai：设置 OPENAI_API_KEY 后调用真实模型

安装：
    pip install fastapi uvicorn pydantic python-dotenv openai

运行：
    python multi_agent_ops_automation_system.py

访问：
    http://127.0.0.1:8000/docs

环境变量可选：
    LLM_PROVIDER=mock 或 openai
    OPENAI_API_KEY=你的key
    OPENAI_MODEL=gpt-4.1-mini

示例任务：
    帮我策划并执行一次小红书账号的周运营，包括选题、内容排期、标题优化、发布检查和复盘报告。
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

DB_PATH = os.getenv("DB_PATH", "multi_agent_ops.db")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


# =========================
# 数据模型
# =========================

class TaskStatus(str, Enum):
    CREATED = "created"
    PLANNED = "planned"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"


class SubTaskStatus(str, Enum):
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"


@dataclass
class AgentMessage:
    agent: str
    role: str
    content: str
    created_at: str


@dataclass
class SubTask:
    id: str
    title: str
    objective: str
    owner_agent: str
    expected_output: str
    status: str = SubTaskStatus.PENDING.value
    result: Optional[str] = None


@dataclass
class OpsTask:
    id: str
    title: str
    description: str
    status: str
    created_at: str
    updated_at: str
    plan: Optional[List[Dict[str, Any]]] = None
    result: Optional[str] = None
    review: Optional[str] = None
    report: Optional[str] = None


class CreateTaskRequest(BaseModel):
    title: str = Field(..., description="任务标题")
    description: str = Field(..., description="任务描述，例如运营目标、平台、周期、约束")


class RunTaskRequest(BaseModel):
    task_id: str


# =========================
# 数据库
# =========================

class Database:
    def __init__(self, path: str):
        self.path = path
        self.init_db()

    def connect(self):
        return sqlite3.connect(self.path)

    def init_db(self):
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    plan TEXT,
                    result TEXT,
                    review TEXT,
                    report TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    agent TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.commit()

    def create_task(self, title: str, description: str) -> OpsTask:
        now = datetime.now().isoformat(timespec="seconds")
        task = OpsTask(
            id=str(uuid.uuid4()),
            title=title,
            description=description,
            status=TaskStatus.CREATED.value,
            created_at=now,
            updated_at=now,
        )
        with self.connect() as conn:
            conn.execute("""
                INSERT INTO tasks (id, title, description, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (task.id, task.title, task.description, task.status, task.created_at, task.updated_at))
            conn.commit()
        return task

    def get_task(self, task_id: str) -> OpsTask:
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
            row = cur.fetchone()
        if not row:
            raise KeyError("Task not found")
        return self._row_to_task(row)

    def list_tasks(self) -> List[OpsTask]:
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM tasks ORDER BY created_at DESC")
            rows = cur.fetchall()
        return [self._row_to_task(row) for row in rows]

    def update_task(self, task_id: str, **fields: Any) -> OpsTask:
        if not fields:
            return self.get_task(task_id)
        fields["updated_at"] = datetime.now().isoformat(timespec="seconds")
        columns = []
        values = []
        for key, value in fields.items():
            columns.append(f"{key} = ?")
            if key == "plan" and isinstance(value, list):
                values.append(json.dumps(value, ensure_ascii=False, indent=2))
            else:
                values.append(value)
        values.append(task_id)
        sql = f"UPDATE tasks SET {', '.join(columns)} WHERE id = ?"
        with self.connect() as conn:
            conn.execute(sql, values)
            conn.commit()
        return self.get_task(task_id)

    def add_log(self, task_id: str, message: AgentMessage):
        with self.connect() as conn:
            conn.execute("""
                INSERT INTO logs (id, task_id, agent, role, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()),
                task_id,
                message.agent,
                message.role,
                message.content,
                message.created_at,
            ))
            conn.commit()

    def get_logs(self, task_id: str) -> List[AgentMessage]:
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT agent, role, content, created_at
                FROM logs
                WHERE task_id = ?
                ORDER BY created_at ASC
            """, (task_id,))
            rows = cur.fetchall()
        return [AgentMessage(agent=r[0], role=r[1], content=r[2], created_at=r[3]) for r in rows]

    def _row_to_task(self, row: tuple) -> OpsTask:
        plan = json.loads(row[6]) if row[6] else None
        return OpsTask(
            id=row[0],
            title=row[1],
            description=row[2],
            status=row[3],
            created_at=row[4],
            updated_at=row[5],
            plan=plan,
            result=row[7],
            review=row[8],
            report=row[9],
        )


# =========================
# LLM 抽象层
# =========================

class LLMClient(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        pass


class MockLLMClient(LLMClient):
    """无需 API Key 的演示模型，用规则生成结果。"""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if "拆解" in system_prompt or "Planner" in system_prompt:
            return json.dumps([
                {
                    "title": "明确运营目标与用户画像",
                    "objective": "提炼本次运营任务的目标、平台、受众、转化路径和约束条件。",
                    "owner_agent": "strategy_agent",
                    "expected_output": "运营目标说明、目标用户画像、核心指标。"
                },
                {
                    "title": "生成内容选题与排期",
                    "objective": "根据目标用户生成 7 天内容主题、标题方向、发布节奏。",
                    "owner_agent": "content_agent",
                    "expected_output": "一周内容日历，包括选题、标题、发布时间、内容形式。"
                },
                {
                    "title": "执行发布前检查",
                    "objective": "检查标题吸引力、内容结构、行动引导、平台规范风险。",
                    "owner_agent": "qa_agent",
                    "expected_output": "发布检查清单与优化建议。"
                },
                {
                    "title": "生成复盘报告",
                    "objective": "基于执行结果总结亮点、问题、下一步优化方向。",
                    "owner_agent": "report_agent",
                    "expected_output": "运营复盘报告。"
                }
            ], ensure_ascii=False, indent=2)

        if "审核" in system_prompt or "Reviewer" in system_prompt:
            return (
                "质检结论：整体任务拆解完整，已覆盖目标定位、内容生产、发布检查和复盘。\n"
                "主要风险：1）缺少真实平台数据接入；2）执行结果目前偏策略文本，未形成自动发布闭环；"
                "3）建议后续接入日历、表格、内容平台 API 和数据看板。\n"
                "改进建议：增加指标字段，例如曝光量、点击率、收藏率、转化率，并形成周期性对比。"
            )

        if "报告" in system_prompt or "Reporter" in system_prompt:
            return (
                "# 多 Agent 运营自动化日报\n\n"
                "## 1. 今日完成\n"
                "- 已完成运营目标拆解。\n"
                "- 已生成内容排期与执行检查清单。\n"
                "- 已完成质量审核与风险提示。\n\n"
                "## 2. 当前产出\n"
                "形成了从目标分析、内容策划、执行检查到复盘报告的自动化流程。\n\n"
                "## 3. 后续优化\n"
                "建议接入真实数据源，实现自动抓取数据、自动对比指标和自动提出下周期计划。"
            )

        return (
            "执行结果：已根据任务目标生成可执行运营方案。\n"
            "包括：用户画像、内容主题、标题方向、发布时间建议、检查清单和复盘要点。"
        )


class OpenAILLMClient(LLMClient):
    def __init__(self, model: str):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content or ""


def build_llm() -> LLMClient:
    if LLM_PROVIDER.lower() == "openai":
        return OpenAILLMClient(model=OPENAI_MODEL)
    return MockLLMClient()


# =========================
# Agent 基类与具体 Agent
# =========================

class Agent(ABC):
    def __init__(self, name: str, role: str, llm: LLMClient, db: Database):
        self.name = name
        self.role = role
        self.llm = llm
        self.db = db

    def log(self, task_id: str, content: str):
        msg = AgentMessage(
            agent=self.name,
            role=self.role,
            content=content,
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        self.db.add_log(task_id, msg)

    @abstractmethod
    def run(self, task: OpsTask, context: Dict[str, Any]) -> Any:
        pass


class PlannerAgent(Agent):
    def run(self, task: OpsTask, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        system_prompt = """
你是 Planner Agent，负责把运营自动化任务拆解为多个可执行子任务。
要求：
1. 输出 JSON 数组。
2. 每个子任务包含 title、objective、owner_agent、expected_output。
3. 子任务要覆盖策略、内容、执行、质检、复盘。
4. 不要输出 JSON 之外的解释。
"""
        user_prompt = f"任务标题：{task.title}\n任务描述：{task.description}"
        raw = self.llm.complete(system_prompt, user_prompt)
        self.log(task.id, raw)

        try:
            plan = json.loads(raw)
        except json.JSONDecodeError:
            plan = [
                {
                    "title": "任务拆解失败后的默认执行任务",
                    "objective": task.description,
                    "owner_agent": "executor_agent",
                    "expected_output": "完整执行方案"
                }
            ]
        return plan


class ExecutorAgent(Agent):
    def run(self, task: OpsTask, context: Dict[str, Any]) -> str:
        plan = context.get("plan", [])
        system_prompt = """
你是 Executor Agent，负责根据 Planner Agent 的拆解结果执行运营任务。
你需要输出具体、可落地的执行结果，而不是泛泛建议。
输出结构：
1. 任务理解
2. 分步骤执行结果
3. 可直接使用的运营物料或清单
4. 下一步动作
"""
        user_prompt = f"原始任务：{task.description}\n\n任务拆解：{json.dumps(plan, ensure_ascii=False, indent=2)}"
        result = self.llm.complete(system_prompt, user_prompt)
        self.log(task.id, result)
        return result


class ReviewerAgent(Agent):
    def run(self, task: OpsTask, context: Dict[str, Any]) -> str:
        system_prompt = """
你是 Reviewer Agent，负责审核 Executor Agent 的执行结果。
请从完整性、可执行性、风险、指标闭环、自动化程度五个角度检查。
输出：
1. 质检结论
2. 主要问题
3. 修改建议
4. 是否通过
"""
        user_prompt = f"任务：{task.description}\n\n执行结果：{context.get('result')}"
        review = self.llm.complete(system_prompt, user_prompt)
        self.log(task.id, review)
        return review


class ReporterAgent(Agent):
    def run(self, task: OpsTask, context: Dict[str, Any]) -> str:
        system_prompt = """
你是 Reporter Agent，负责把多 Agent 协同过程整理成运营日报/复盘报告。
报告要结构化、简洁、可以直接发给团队成员。
"""
        user_prompt = f"""
任务：{task.title}
描述：{task.description}
计划：{json.dumps(context.get('plan'), ensure_ascii=False, indent=2)}
执行结果：{context.get('result')}
审核意见：{context.get('review')}
"""
        report = self.llm.complete(system_prompt, user_prompt)
        self.log(task.id, report)
        return report


# =========================
# 工具层：模拟运营工具
# =========================

class OpsTools:
    """后续可替换为真实 API，例如飞书、Notion、小红书、抖音、公众号、Google Sheets。"""

    @staticmethod
    def generate_content_calendar(topic: str, days: int = 7) -> List[Dict[str, str]]:
        calendar = []
        for i in range(1, days + 1):
            calendar.append({
                "day": f"Day {i}",
                "topic": f"{topic} 选题 {i}",
                "format": "图文 / 短视频 / 长文",
                "publish_time": "20:00",
                "goal": "提升曝光、收藏或转化"
            })
        return calendar

    @staticmethod
    def check_publish_risk(content: str) -> Dict[str, Any]:
        risky_words = ["绝对", "保证", "最强", "第一", "稳赚"]
        hits = [w for w in risky_words if w in content]
        return {
            "risk_level": "high" if hits else "low",
            "risky_words": hits,
            "suggestion": "删除绝对化表达，补充客观依据。" if hits else "暂无明显风险。"
        }


# =========================
# 编排器 Orchestrator
# =========================

class MultiAgentOrchestrator:
    def __init__(self, db: Database, llm: LLMClient):
        self.db = db
        self.llm = llm
        self.planner = PlannerAgent("planner_agent", "任务规划", llm, db)
        self.executor = ExecutorAgent("executor_agent", "任务执行", llm, db)
        self.reviewer = ReviewerAgent("reviewer_agent", "质量审核", llm, db)
        self.reporter = ReporterAgent("reporter_agent", "复盘报告", llm, db)

    def run_task(self, task_id: str) -> OpsTask:
        try:
            task = self.db.get_task(task_id)
            context: Dict[str, Any] = {}

            self.db.update_task(task_id, status=TaskStatus.PLANNED.value)
            plan = self.planner.run(task, context)
            context["plan"] = plan
            self.db.update_task(task_id, plan=plan)

            self.db.update_task(task_id, status=TaskStatus.EXECUTING.value)
            result = self.executor.run(task, context)
            context["result"] = result
            self.db.update_task(task_id, result=result)

            self.db.update_task(task_id, status=TaskStatus.REVIEWING.value)
            review = self.reviewer.run(task, context)
            context["review"] = review
            self.db.update_task(task_id, review=review)

            report = self.reporter.run(task, context)
            context["report"] = report
            completed_task = self.db.update_task(
                task_id,
                status=TaskStatus.COMPLETED.value,
                report=report,
            )
            return completed_task
        except Exception as e:
            self.db.update_task(task_id, status=TaskStatus.FAILED.value, result=str(e))
            raise


# =========================
# FastAPI 服务
# =========================

db = Database(DB_PATH)
llm = build_llm()
orchestrator = MultiAgentOrchestrator(db, llm)
app = FastAPI(title="Multi-Agent 协同运营自动化系统", version="1.0.0")


@app.get("/")
def home():
    return {
        "name": "Multi-Agent 协同运营自动化系统",
        "status": "running",
        "docs": "/docs",
        "llm_provider": LLM_PROVIDER,
    }


@app.post("/tasks")
def create_task(req: CreateTaskRequest):
    task = db.create_task(req.title, req.description)
    return asdict(task)


@app.get("/tasks")
def list_tasks():
    return [asdict(t) for t in db.list_tasks()]


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    try:
        return asdict(db.get_task(task_id))
    except KeyError:
        raise HTTPException(status_code=404, detail="Task not found")


@app.post("/tasks/run")
def run_task(req: RunTaskRequest):
    try:
        task = orchestrator.run_task(req.task_id)
        return asdict(task)
    except KeyError:
        raise HTTPException(status_code=404, detail="Task not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks/{task_id}/logs")
def get_task_logs(task_id: str):
    try:
        db.get_task(task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Task not found")
    return [asdict(log) for log in db.get_logs(task_id)]


@app.post("/demo")
def demo():
    task = db.create_task(
        title="小红书账号周运营自动化",
        description=(
            "请为一个面向职场新人的个人成长账号设计一周运营方案。"
            "要求包含用户画像、选题方向、标题、发布节奏、发布前检查和复盘报告。"
        ),
    )
    completed = orchestrator.run_task(task.id)
    return asdict(completed)


# =========================
# 命令行入口
# =========================

if __name__ == "__main__":
    import uvicorn
    print("启动 Multi-Agent 协同运营自动化系统...")
    print("接口文档：http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)
