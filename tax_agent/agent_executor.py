"""Tax Agent — AgentExecutor bridge between A2A SDK and LangGraph."""

from __future__ import annotations

import logging
from uuid import uuid4

from langchain_core.messages import HumanMessage

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, TextPart

from tax_agent.graph import create_graph

logger = logging.getLogger(__name__)

_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = create_graph()
    return _graph


class TaxAgentExecutor(AgentExecutor):
    """Bridges A2A RequestContext to the Tax LangGraph agent."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Extract question from message parts
        question = self._extract_question(context)
        context_id = context.context_id or str(uuid4())
        task_id = context.task_id or str(uuid4())
        metadata = context.message.metadata or {} if context.message else {}
        trace_id = metadata.get("trace_id", str(uuid4()))
        depth = int(metadata.get("delegation_depth", 0))

        import time
        from common.logging_utils import parallel_var, multi_model_var, trace_id_var, agent_name_var, log_event

        start_time = time.time()
        parallel_token = parallel_var.set(metadata.get("parallel", False))
        multi_model_token = multi_model_var.set(metadata.get("multi_model", False))
        trace_id_token = trace_id_var.set(trace_id)
        agent_name_token = agent_name_var.set("tax_agent")

        logger.info(
            "TaxAgent executing | task=%s context=%s trace=%s depth=%d",
            task_id, context_id, trace_id, depth,
        )

        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.submit()
        await updater.start_work()

        try:
            result = await _get_graph().ainvoke(
                {"messages": [HumanMessage(content=question)]},
                config={"configurable": {"thread_id": context_id}},
            )

            # Extract the last AI message
            answer = ""
            for msg in reversed(result.get("messages", [])):
                if hasattr(msg, "content") and msg.content:
                    if not isinstance(msg, HumanMessage):
                        answer = msg.content
                        break

            if not answer:
                answer = "I was unable to generate a tax analysis at this time."

            await updater.add_artifact(
                parts=[Part(root=TextPart(text=answer))],
                name="tax_analysis",
            )
            await updater.complete()

            log_event("agent_execution", {
                "duration_seconds": time.time() - start_time,
                "status": "success",
                "parallel": metadata.get("parallel", False),
                "multi_model": metadata.get("multi_model", False)
            })

        except Exception as exc:
            log_event("agent_execution", {
                "duration_seconds": time.time() - start_time,
                "status": "failed",
                "error": str(exc),
                "parallel": metadata.get("parallel", False),
                "multi_model": metadata.get("multi_model", False)
            })
            logger.exception("TaxAgent execution error: %s", exc)
            await updater.failed(
                updater.new_agent_message(
                    parts=[Part(root=TextPart(text=f"Tax analysis failed: {exc}"))]
                )
            )
        finally:
            parallel_var.reset(parallel_token)
            multi_model_var.reset(multi_model_token)
            trace_id_var.reset(trace_id_token)
            agent_name_var.reset(agent_name_token)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or str(uuid4())
        context_id = context.context_id or str(uuid4())
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.cancel()

    @staticmethod
    def _extract_question(context: RequestContext) -> str:
        if context.message and context.message.parts:
            parts = []
            for part in context.message.parts:
                inner = getattr(part, "root", part)
                text = getattr(inner, "text", None)
                if text:
                    parts.append(text)
            return "\n".join(parts)
        return ""