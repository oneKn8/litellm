"""Unit tests for the shadow-eval logger: sampling, unmasking, the hook's skip chain,
the detached pipeline's single attempt-row write, and the cache-first job lookup."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.integrations.shadow_eval_logger import (
    _MAX_CONCURRENT_SHADOW_TASKS,
    _MAX_JUDGE_PROMPT_CHARS,
    JUDGE_MAX_OUTPUT_TOKENS,
    ActiveShadowEvalJob,
    ShadowEvalLogger,
    _judge_user_prompt,
    _sample_hits,
    _unmask_preference,
)
from litellm.types.utils import SHADOW_EVAL_JUDGE_CALL_ORIGIN, SHADOW_EVAL_ROUTER_CALL_ORIGIN


def _job(**overrides) -> ActiveShadowEvalJob:
    defaults = dict(
        id="job-1",
        router_name="my-router",
        shadow_percentage=100.0,
        judge_model="judge-model",
        max_turns=200,
        ends_at=datetime.now(timezone.utc) + timedelta(days=1),
        attempts=0,
    )
    return ActiveShadowEvalJob(**{**defaults, **overrides})


def _prisma(jobs=(), attempt_counts=()) -> MagicMock:
    prisma = MagicMock()
    prisma.db.litellm_shadowevaljob.find_many = AsyncMock(return_value=list(jobs))
    prisma.db.litellm_shadowevalattempt.group_by = AsyncMock(
        return_value=[{"job_id": job_id, "_count": {"_all": count}} for job_id, count in attempt_counts]
    )
    prisma.db.litellm_shadowevalattempt.create = AsyncMock()
    return prisma


def _job_record(job: ActiveShadowEvalJob, api_key_id="key-hash") -> MagicMock:
    record = MagicMock()
    for field, value in dict(
        id=job.id,
        api_key_id=api_key_id,
        router_name=job.router_name,
        shadow_percentage=job.shadow_percentage,
        judge_model=job.judge_model,
        max_turns=job.max_turns,
        ends_at=job.ends_at,
    ).items():
        setattr(record, field, value)
    return record


def _router(shadow_text="shadow answer", judge_json='{"preference": "A", "confidence": 0.9, "reasoning": "x"}'):
    """One mock router serving the shadow call first, the judge call second. The shadow
    call's metadata receives the routing decision write-back, like the real router."""
    router = MagicMock()
    router.model_group_alias = {}
    router.get_model_list = MagicMock(return_value=[{"litellm_params": {"model": "openai/gpt-4o-mini"}}])

    async def acompletion(**kwargs):
        if kwargs["model"] == "my-router":
            kwargs["metadata"]["routing_decision"] = {"tier_label": "SIMPLE", "routed_model": "cheap-model"}
            return {"choices": [{"message": {"content": shadow_text}}], "usage": {"completion_tokens": 5}}
        return {"choices": [{"message": {"content": judge_json}}]}

    router.acompletion = MagicMock(side_effect=acompletion)
    return router


def _logger(router=None, prisma=None, job=None) -> ShadowEvalLogger:
    cache = InMemoryCache(max_size_in_memory=4, default_ttl=60)
    logger = ShadowEvalLogger(
        router_provider=lambda: router,
        prisma_provider=lambda: prisma,
        jobs_cache=cache,
    )
    if job is not None:
        cache.set_cache("shadow_eval:active_jobs", {"key-hash": job})
    return logger


def _success_kwargs(request_id="req-1", api_key_hash="key-hash", request_metadata=None, call_type="acompletion"):
    return {
        "standard_logging_object": {
            "id": request_id,
            "call_type": call_type,
            "model": "claude-opus",
            "metadata": {"user_api_key_hash": api_key_hash},
            "model_parameters": {"temperature": 0.5, "stream": True},
        },
        "litellm_params": {"metadata": request_metadata or {}},
        "messages": [{"role": "user", "content": "what is 2+2"}],
    }


RESPONSE = {"choices": [{"message": {"content": "real answer"}}]}

RESPONSES_API_RESPONSE = {
    "id": "resp_1",
    "created_at": 1,
    "model": "gpt-5",
    "object": "response",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "real answer", "annotations": []}],
        }
    ],
    "parallel_tool_calls": True,
    "error": None,
    "incomplete_details": None,
    "instructions": None,
    "metadata": None,
    "temperature": None,
    "tool_choice": "auto",
    "tools": [],
    "top_p": None,
    "status": "completed",
}


async def _drain(logger: ShadowEvalLogger, target: int = 0):
    for _ in range(100):
        if logger._inflight_shadow_tasks == target:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("shadow tasks never drained")


@pytest.mark.asyncio
class TestSurfaceNormalization:
    """/v1/messages and /v1/responses arms: the hook normalizes each surface's logged
    request through litellm's own transformations and judges only text-final turns."""

    async def _drive(self, hook_kwargs, response_obj):
        prisma = _prisma()
        router = _router()
        logger = _logger(router=router, prisma=prisma, job=_job())
        await logger.async_log_success_event(hook_kwargs, response_obj, None, None)
        await _drain(logger)
        return prisma, router

    async def test_anthropic_messages_arm_normalizes_blocks_and_system(self):
        hook_kwargs = _success_kwargs(call_type="anthropic_messages")
        hook_kwargs["messages"] = [{"role": "user", "content": [{"type": "text", "text": "what is 2+2"}]}]
        hook_kwargs["system"] = "you are terse"

        prisma, router = await self._drive(hook_kwargs, RESPONSE)

        shadow_messages = router.acompletion.call_args_list[0].kwargs["messages"]
        assert shadow_messages[0]["role"] == "system"
        assert shadow_messages[0]["content"] == "you are terse"
        assert shadow_messages[1]["role"] == "user"
        prisma.db.litellm_shadowevalattempt.create.assert_called_once()

    async def test_anthropic_bridge_path_recovers_system_from_proxy_wire_body(self):
        """On the openai-compatible bridge path kwargs carry no system (live-probed:
        kwargs["system"] is None and complete_input_dict is empty); the proxy's snapshot
        of the client's wire body is the only remaining source."""
        hook_kwargs = _success_kwargs(call_type="anthropic_messages")
        hook_kwargs["messages"] = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        hook_kwargs["litellm_params"]["proxy_server_request"] = {
            "body": {"model": "gpt-5", "max_tokens": 100, "system": "from the wire body", "messages": []}
        }

        _, router = await self._drive(hook_kwargs, RESPONSE)

        shadow_messages = router.acompletion.call_args_list[0].kwargs["messages"]
        assert shadow_messages[0] == {"role": "system", "content": "from the wire body"}

    async def test_anthropic_arm_translates_wire_body_params_not_logged_optional_params(self):
        """The wire body is the only surface-native param source on both provider paths
        (the bridge's inner completion rewrites the logged optional_params to chat
        shape); anthropic tools and stop_sequences reach the shadow call translated,
        transport and litellm keys never do."""
        hook_kwargs = _success_kwargs(call_type="anthropic_messages")
        hook_kwargs["messages"] = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        hook_kwargs["standard_logging_object"]["model_parameters"] = {"temperature": 0.9}
        hook_kwargs["litellm_params"]["proxy_server_request"] = {
            "body": {
                "model": "claude-x",
                "messages": [],
                "system": "you are terse",
                "max_tokens": 100,
                "temperature": 0.1,
                "top_k": 5,
                "stop_sequences": ["END"],
                "stream": True,
                "tools": [
                    {"name": "get_weather", "description": "d", "input_schema": {"type": "object", "properties": {}}}
                ],
                "litellm_metadata": {"user_api_key_hash": "key-hash"},
            }
        }

        _, router = await self._drive(hook_kwargs, RESPONSE)

        shadow_call = router.acompletion.call_args_list[0].kwargs
        assert shadow_call["max_tokens"] == 100
        assert shadow_call["temperature"] == 0.1
        assert shadow_call["top_k"] == 5
        assert shadow_call["stop"] == ["END"]
        assert shadow_call["tools"][0]["type"] == "function"
        assert shadow_call["tools"][0]["function"]["name"] == "get_weather"
        assert "stop_sequences" not in shadow_call
        assert "stream" not in shadow_call
        assert shadow_call["metadata"][INTERNAL_CALL_ORIGIN_METADATA_KEY] == SHADOW_EVAL_ROUTER_CALL_ORIGIN

    async def test_responses_arm_translates_wire_body_params_and_drops_surface_only_keys(self):
        from litellm.types.llms.openai import ResponsesAPIResponse

        hook_kwargs = _success_kwargs(call_type="aresponses")
        hook_kwargs["messages"] = "what is 8+8"
        hook_kwargs["litellm_params"]["proxy_server_request"] = {
            "body": {
                "model": "gpt-5",
                "input": "what is 8+8",
                "instructions": "you are terse",
                "max_output_tokens": 128,
                "temperature": 0.3,
                "previous_response_id": "resp_0",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "d",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            }
        }
        response = ResponsesAPIResponse.model_validate(RESPONSES_API_RESPONSE)

        _, router = await self._drive(hook_kwargs, response)

        shadow_call = router.acompletion.call_args_list[0].kwargs
        assert shadow_call["messages"][0] == {"role": "system", "content": "you are terse"}
        assert shadow_call["max_tokens"] == 128
        assert shadow_call["temperature"] == 0.3
        assert shadow_call["tools"][0]["function"]["name"] == "get_weather"
        assert "max_output_tokens" not in shadow_call
        assert "previous_response_id" not in shadow_call
        assert "instructions" not in shadow_call

    @pytest.mark.parametrize("call_type", ["aresponses", "responses"])
    async def test_responses_arms_normalize_bare_string_input_and_instructions(self, call_type):
        from litellm.types.llms.openai import ResponsesAPIResponse

        hook_kwargs = _success_kwargs(call_type=call_type)
        hook_kwargs["messages"] = "what is 8+8"
        hook_kwargs["instructions"] = "you are terse"
        response = ResponsesAPIResponse.model_validate(RESPONSES_API_RESPONSE)

        prisma, router = await self._drive(hook_kwargs, response)

        shadow_call = router.acompletion.call_args_list[0].kwargs
        shadow_messages = shadow_call["messages"]
        assert shadow_messages[0]["role"] == "system"
        assert shadow_messages[1]["role"] == "user"
        assert shadow_messages[1]["content"] == "what is 8+8"
        assert "tools" not in shadow_call
        prisma.db.litellm_shadowevalattempt.create.assert_called_once()

    @pytest.mark.parametrize(
        "response_mutation,kwargs_mutation",
        [
            ("chat-tool-calls", {}),
            ("responses-function-call", {"call_type": "aresponses"}),
        ],
        ids=["tool-final-chat-turn", "tool-final-responses-turn"],
    )
    async def test_unjudgeable_turns_are_skipped_without_consuming_budget(self, response_mutation, kwargs_mutation):
        from litellm.types.llms.openai import ResponsesAPIResponse

        hook_kwargs = _success_kwargs(**({"call_type": "acompletion"} | kwargs_mutation))
        response = RESPONSE
        if response_mutation == "chat-tool-calls":
            response = {
                "choices": [
                    {
                        "message": {
                            "content": "let me check",
                            "tool_calls": [
                                {"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                            ],
                        }
                    }
                ]
            }
        elif response_mutation == "responses-function-call":
            hook_kwargs["messages"] = "do the thing"
            response = ResponsesAPIResponse.model_validate(
                RESPONSES_API_RESPONSE
                | {
                    "output": [
                        {
                            "type": "function_call",
                            "name": "f",
                            "arguments": "{}",
                            "call_id": "c1",
                            "id": "fc1",
                            "status": "completed",
                        }
                    ]
                }
            )

        prisma, router = await self._drive(hook_kwargs, response)

        router.acompletion.assert_not_called()
        prisma.db.litellm_shadowevalattempt.create.assert_not_called()

    @pytest.mark.parametrize(
        "call_type,messages,response_obj",
        [
            ("anthropic_messages", "not-a-message-list", RESPONSE),
            ("acompletion", [{"role": "user", "content": "hi"}], {"unexpected": "shape"}),
            ("aresponses", "hi", RESPONSE),
        ],
        ids=["rejected-request-shape", "malformed-chat-response", "responses-response-without-output"],
    )
    async def test_unsampleable_shapes_fail_closed(self, call_type, messages, response_obj):
        """A request or response shape the normalizers reject is skipped without a
        provider call or an attempt row, never raised."""
        hook_kwargs = _success_kwargs(call_type=call_type)
        hook_kwargs["messages"] = messages

        prisma, router = await self._drive(hook_kwargs, response_obj)

        router.acompletion.assert_not_called()
        prisma.db.litellm_shadowevalattempt.create.assert_not_called()


class TestSampling:
    def test_boundaries_and_determinism(self):
        assert not any(_sample_hits(f"req-{i}", "job", 0.0) for i in range(100))
        assert all(_sample_hits(f"req-{i}", "job", 100.0) for i in range(100))
        assert len({_sample_hits("req-1", "job-1", 50.0) for _ in range(10)}) == 1

    def test_distribution_close_to_percentage(self):
        hits = sum(_sample_hits(f"req-{i}", "job-x", 10.0) for i in range(10_000))
        assert 800 < hits < 1200

    def test_different_jobs_sample_independently(self):
        agreements = sum(
            _sample_hits(f"req-{i}", "job-a", 50.0) == _sample_hits(f"req-{i}", "job-b", 50.0) for i in range(1000)
        )
        assert 300 < agreements < 700


@pytest.mark.parametrize(
    "raw,real_is_a,expected",
    [
        ("A", True, "real"),
        ("a", True, "real"),
        ("A", False, "shadow"),
        ("B", True, "shadow"),
        ("B", False, "real"),
        ("tie", True, "tie"),
        ("garbage", True, "tie"),
        ("", False, "tie"),
    ],
)
def test_unmask_preference(raw, real_is_a, expected):
    assert _unmask_preference(raw, real_is_a) == expected


def test_judge_prompt_is_bounded_however_large_the_inputs():
    prompt = _judge_user_prompt("c" * 200_000, "a" * 200_000, "b" * 200_000)
    assert len(prompt) < _MAX_JUDGE_PROMPT_CHARS + 100
    assert prompt.endswith("Which response is better?")
    small = _judge_user_prompt("conv", "alpha", "beta")
    assert "conv" in small and "alpha" in small and "beta" in small


@pytest.mark.asyncio
class TestSuccessHookSkipChain:
    async def test_happy_path_writes_exactly_one_attempt_row(self, monkeypatch: pytest.MonkeyPatch):
        import litellm as litellm_module

        monkeypatch.setattr(litellm_module, "completion_cost", lambda completion_response: 0.005)
        prisma = _prisma()
        router = _router()
        logger = _logger(router=router, prisma=prisma, job=_job())

        await logger.async_log_success_event(_success_kwargs(), RESPONSE, None, None)
        await _drain(logger)

        shadow_call = router.acompletion.call_args_list[0].kwargs
        assert shadow_call["temperature"] == 0.5
        assert "stream" not in shadow_call
        create = prisma.db.litellm_shadowevalattempt.create
        create.assert_awaited_once()
        row = create.call_args.kwargs["data"]
        assert row["job_id"] == "job-1"
        assert row["request_id"] == "req-1"
        assert row["outcome"] in ("real", "shadow")
        assert row["tier"] == "SIMPLE"
        assert row["real_model"] == "claude-opus"
        assert row["shadow_model"] == "cheap-model"
        assert row["confidence"] == 0.9
        assert row["judge_cost"] == 0.005
        assert row["error"] is None
        assert prisma.db.litellm_shadowevaljob.find_many.await_count == 0

    @pytest.mark.parametrize(
        "kwargs_mutation,job_mutation",
        [
            ({"request_metadata": {INTERNAL_CALL_ORIGIN_METADATA_KEY: "shadow_eval_router"}}, {}),
            ({"api_key_hash": "other-key"}, {}),
            ({"call_type": "aembedding"}, {}),
            ({"call_type": None}, {}),
            ({"request_metadata": {"routing_decision": {"router_model_name": "my-router"}}}, {}),
            ({}, {"ends_at": datetime.now(timezone.utc) - timedelta(seconds=1)}),
            ({}, {"attempts": 200}),
            ({}, {"attempts": 199, "max_turns": 200, "_starts": 1}),
        ],
        ids=[
            "internal-origin",
            "no-job-for-key",
            "non-chat",
            "missing-call-type",
            "self-shadow",
            "past-end",
            "turn-budget-reached",
            "budget-consumed-by-started-tasks",
        ],
    )
    async def test_skip_paths_store_nothing(self, kwargs_mutation, job_mutation):
        starts = job_mutation.pop("_starts", 0)
        prisma = _prisma()
        logger = _logger(router=_router(), prisma=prisma, job=_job(**job_mutation))
        logger._job_starts = {"job-1": starts}

        await logger.async_log_success_event(_success_kwargs(**kwargs_mutation), RESPONSE, None, None)
        await _drain(logger)

        prisma.db.litellm_shadowevalattempt.create.assert_not_called()
        assert logger._job_starts.get("job-1", 0) == starts

    async def test_completed_pipelines_hold_turn_budget_within_a_cache_generation(self):
        """A finished pipeline frees its concurrency slot but not its slice of the turn
        budget; the budget only reopens when a cache refill absorbs the written rows."""
        prisma = _prisma()
        logger = _logger(router=_router(), prisma=prisma, job=_job(attempts=199, max_turns=200))

        await logger.async_log_success_event(_success_kwargs(request_id="req-1"), RESPONSE, None, None)
        await _drain(logger)
        await logger.async_log_success_event(_success_kwargs(request_id="req-2"), RESPONSE, None, None)
        await _drain(logger)

        assert prisma.db.litellm_shadowevalattempt.create.await_count == 1

    async def test_v1_messages_surface_forwards_identity_from_litellm_metadata(self):
        """/v1/messages stores identity in litellm_params.litellm_metadata, so the hook
        resolves the bucket through the shared helper; every surface forwards the same
        identity to the shadow and judge calls."""
        prisma = _prisma()
        router = _router()
        logger = _logger(router=router, prisma=prisma, job=_job())

        hook_kwargs = _success_kwargs()
        hook_kwargs["litellm_params"] = {
            "litellm_metadata": {"user_api_key_hash": "key-hash", "user_api_key_team_id": "team-1"}
        }
        await logger.async_log_success_event(hook_kwargs, RESPONSE, None, None)
        await _drain(logger)

        shadow_call = router.acompletion.call_args_list[0].kwargs
        assert shadow_call["metadata"]["user_api_key_hash"] == "key-hash"
        assert shadow_call["metadata"]["user_api_key_team_id"] == "team-1"

    async def test_redacted_requests_are_never_shadowed(self):
        """Redaction rewrites the logged messages before callbacks run, so this hook only
        ever sees placeholders for opted-out traffic; the skip uses the redactor's own
        predicate, so every redaction source counts."""
        prisma = _prisma()
        router = _router()
        logger = _logger(router=router, prisma=prisma, job=_job())

        hook_kwargs = _success_kwargs()
        hook_kwargs["standard_callback_dynamic_params"] = {"turn_off_message_logging": True}
        await logger.async_log_success_event(hook_kwargs, RESPONSE, None, None)
        await _drain(logger)

        router.acompletion.assert_not_called()
        prisma.db.litellm_shadowevalattempt.create.assert_not_called()

    async def test_inflight_cap_sheds_instead_of_queueing(self):
        prisma = _prisma()
        logger = _logger(router=_router(), prisma=prisma, job=_job())
        logger._inflight_shadow_tasks = _MAX_CONCURRENT_SHADOW_TASKS

        await logger.async_log_success_event(_success_kwargs(), RESPONSE, None, None)

        assert logger._inflight_shadow_tasks == _MAX_CONCURRENT_SHADOW_TASKS
        prisma.db.litellm_shadowevalattempt.create.assert_not_called()


@pytest.mark.asyncio
class TestActiveJobsCache:
    async def test_cache_miss_reads_db_once_then_serves_from_cache(self):
        job = _job()
        prisma = _prisma(jobs=[_job_record(job)], attempt_counts=[("job-1", 7)])
        logger = ShadowEvalLogger(
            router_provider=lambda: None,
            prisma_provider=lambda: prisma,
            jobs_cache=InMemoryCache(max_size_in_memory=4, default_ttl=60),
        )

        first = await logger._active_jobs()
        second = await logger._active_jobs()

        assert first["key-hash"].id == "job-1"
        assert second["key-hash"].attempts == 7
        assert prisma.db.litellm_shadowevaljob.find_many.await_count == 1
        where = prisma.db.litellm_shadowevaljob.find_many.call_args.kwargs["where"]
        assert where["stopped_at"] is None
        assert "gt" in where["ends_at"]
        count_where = prisma.db.litellm_shadowevalattempt.group_by.call_args.kwargs["where"]
        assert count_where == {"job_id": {"in": ["job-1"]}}

    async def test_no_active_jobs_is_cached_too(self):
        prisma = _prisma(jobs=[])
        logger = ShadowEvalLogger(
            router_provider=lambda: None,
            prisma_provider=lambda: prisma,
            jobs_cache=InMemoryCache(max_size_in_memory=4, default_ttl=60),
        )

        assert await logger._active_jobs() == {}
        assert await logger._active_jobs() == {}
        assert prisma.db.litellm_shadowevaljob.find_many.await_count == 1
        prisma.db.litellm_shadowevalattempt.group_by.assert_not_called()

    async def test_db_fault_returns_empty_without_caching_the_fault(self):
        prisma = _prisma()
        prisma.db.litellm_shadowevaljob.find_many = AsyncMock(side_effect=RuntimeError("db blip"))
        logger = ShadowEvalLogger(
            router_provider=lambda: None,
            prisma_provider=lambda: prisma,
            jobs_cache=InMemoryCache(max_size_in_memory=4, default_ttl=60),
        )

        assert await logger._active_jobs() == {}
        assert await logger._active_jobs() == {}
        assert prisma.db.litellm_shadowevaljob.find_many.await_count == 2

    async def test_cache_refill_resets_the_starts_counter(self):
        job = _job()
        prisma = _prisma(jobs=[_job_record(job)], attempt_counts=[("job-1", 7)])
        logger = ShadowEvalLogger(
            router_provider=lambda: None,
            prisma_provider=lambda: prisma,
            jobs_cache=InMemoryCache(max_size_in_memory=4, default_ttl=60),
        )
        logger._job_starts = {"job-1": 5}

        await logger._active_jobs()

        assert logger._job_starts == {}


@pytest.mark.asyncio
class TestShadowPipeline:
    async def test_no_prisma_means_no_provider_spend(self):
        router = _router()
        logger = _logger(router=router, prisma=None)

        await logger._run_shadow_eval(
            job=_job(),
            request_id="req-1",
            messages=({"role": "user", "content": "hi"},),
            real_text="real answer",
            real_model="claude-opus",
            shadow_params={},
            parent_metadata={},
        )

        router.acompletion.assert_not_called()

    async def test_over_budget_key_skips_before_any_call(self, monkeypatch: pytest.MonkeyPatch):
        """The gate delegates to the auth path's own budget owner, so an over-budget
        verdict there (BudgetExceededError) skips the shadow before any provider call."""
        import litellm.proxy.auth.auth_checks as auth_checks
        from litellm.exceptions import BudgetExceededError
        from litellm.proxy._types import UserAPIKeyAuth

        monkeypatch.setattr(
            auth_checks,
            "_virtual_key_max_budget_check",
            AsyncMock(side_effect=BudgetExceededError(current_cost=11.0, max_budget=10.0)),
        )
        router = _router()
        prisma = _prisma()
        logger = _logger(router=router, prisma=prisma)

        await logger._run_shadow_eval(
            job=_job(),
            request_id="req-1",
            messages=({"role": "user", "content": "hi"},),
            real_text="real answer",
            real_model="claude-opus",
            shadow_params={},
            parent_metadata={"user_api_key_auth": UserAPIKeyAuth(api_key="sk-abc", max_budget=10.0)},
        )

        router.acompletion.assert_not_called()
        prisma.db.litellm_shadowevalattempt.create.assert_not_called()

    @pytest.mark.parametrize(
        "router_factory,expected_error,expected_cost",
        [
            (lambda: _failing_router(), "provider exploded", 0.0),
            (lambda: _router(judge_json="I prefer response A, definitely"), "unparseable judge verdict", 0.007),
        ],
        ids=["shadow-call-fails", "judge-verdict-unparseable"],
    )
    async def test_failures_become_error_rows_and_keep_billed_judge_cost(
        self, router_factory, expected_error, expected_cost, monkeypatch: pytest.MonkeyPatch
    ):
        import litellm as litellm_module

        monkeypatch.setattr(litellm_module, "completion_cost", lambda completion_response: 0.007)
        prisma = _prisma()
        logger = _logger(router=router_factory(), prisma=prisma)

        await logger._run_shadow_eval(
            job=_job(),
            request_id="req-1",
            messages=({"role": "user", "content": "hi"},),
            real_text="real answer",
            real_model="claude-opus",
            shadow_params={},
            parent_metadata={},
        )

        row = prisma.db.litellm_shadowevalattempt.create.call_args.kwargs["data"]
        assert row["outcome"] == "error"
        assert expected_error in row["error"]
        assert row["confidence"] is None
        assert row["judge_cost"] == expected_cost

    async def test_sub_calls_carry_identity_and_origin_but_never_parent_request_state(self):
        prisma = _prisma()
        router = _router()
        logger = _logger(router=router, prisma=prisma)
        parent_metadata = {
            "user_api_key_hash": "key-hash",
            "user_api_key_team_id": "team-1",
            "user_api_key_budget_reservation": {"amount": 1.0},
            "routing_decision": {"router_model_name": "other-router"},
        }

        await logger._run_shadow_eval(
            job=_job(),
            request_id="req-1",
            messages=({"role": "user", "content": "hi"},),
            real_text="real answer",
            real_model="claude-opus",
            shadow_params={"temperature": 0.2},
            parent_metadata=parent_metadata,
        )

        shadow_call = router.acompletion.call_args_list[0].kwargs
        judge_call = router.acompletion.call_args_list[1].kwargs
        for call in (shadow_call, judge_call):
            assert call["num_retries"] == 0
            assert call["fallbacks"] == []
            assert call["metadata"]["user_api_key_hash"] == "key-hash"
            assert call["metadata"]["user_api_key_team_id"] == "team-1"
            assert "user_api_key_budget_reservation" not in call["metadata"]
        assert shadow_call["metadata"][INTERNAL_CALL_ORIGIN_METADATA_KEY] == SHADOW_EVAL_ROUTER_CALL_ORIGIN
        assert judge_call["metadata"][INTERNAL_CALL_ORIGIN_METADATA_KEY] == SHADOW_EVAL_JUDGE_CALL_ORIGIN
        assert "routing_decision" not in judge_call["metadata"]
        assert shadow_call["temperature"] == 0.2
        assert judge_call["max_tokens"] == JUDGE_MAX_OUTPUT_TOKENS


def _failing_router():
    router = MagicMock()
    router.model_group_alias = {}
    router.get_model_list = MagicMock(return_value=None)
    router.acompletion = AsyncMock(side_effect=RuntimeError("provider exploded"))
    return router
