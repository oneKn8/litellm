import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig
from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
    AmazonInvokeConfig,
)
from litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation import (
    AmazonAnthropicClaudeMessagesConfig,
)
from litellm.llms.bedrock.request_metadata import (
    BEDROCK_REQUEST_METADATA_HEADER,
    BEDROCK_REQUEST_METADATA_MAX_PAIRS,
    resolve_bedrock_request_metadata,
)

MODEL = "anthropic.claude-3-5-sonnet-20240620-v1:0"
MESSAGES = [{"role": "user", "content": "hi"}]
ALL_FIELDS = [
    "user_api_key_alias",
    "user_api_key_team_alias",
    "user_api_key_user_email",
    "spend_logs_metadata",
]
IDENTITY = {"user_api_key_alias": "prod-key", "user_api_key_team_alias": "platform"}


@pytest.fixture(autouse=True)
def reset_setting():
    previous = litellm.bedrock_request_metadata_fields
    yield
    litellm.bedrock_request_metadata_fields = previous


def litellm_params(metadata_key, **metadata):
    return {metadata_key: dict(metadata)}


def converse_body(litellm_params_value, optional_params=None):
    return AmazonConverseConfig()._transform_request(
        model=MODEL,
        messages=MESSAGES,
        optional_params=dict(optional_params or {}),
        litellm_params=dict(litellm_params_value),
    )


@pytest.mark.parametrize("setting", [None, []])
def test_feature_off_by_default_leaves_body_and_headers_untouched(setting):
    litellm.bedrock_request_metadata_fields = setting
    params = litellm_params("metadata", spend_logs_metadata={"team": "x"}, **IDENTITY)

    assert "requestMetadata" not in converse_body(params)
    assert BEDROCK_REQUEST_METADATA_HEADER not in AmazonInvokeConfig().validate_environment(
        headers={}, model=MODEL, messages=MESSAGES, optional_params={}, litellm_params=dict(params)
    )
    messages_headers, _ = AmazonAnthropicClaudeMessagesConfig().validate_anthropic_messages_environment(
        headers={}, model=MODEL, messages=MESSAGES, optional_params={}, litellm_params=dict(params)
    )
    assert BEDROCK_REQUEST_METADATA_HEADER not in messages_headers


@pytest.mark.parametrize("metadata_key", ["metadata", "litellm_metadata"])
def test_resolver_reads_both_metadata_variable_names(metadata_key):
    """`/v1/chat/completions` populates `metadata`; the LITELLM_METADATA_ROUTES populate
    `litellm_metadata`. Reading only one silently forwards nothing on the other route."""
    litellm.bedrock_request_metadata_fields = ALL_FIELDS
    params = litellm_params(metadata_key, spend_logs_metadata={"cost_center": "cc-1"}, **IDENTITY)

    assert converse_body(params)["requestMetadata"] == {**IDENTITY, "cost_center": "cc-1"}


@pytest.mark.parametrize("metadata_key", ["metadata", "litellm_metadata"])
def test_invoke_messages_header_reads_both_metadata_variable_names(metadata_key):
    litellm.bedrock_request_metadata_fields = ALL_FIELDS
    params = litellm_params(metadata_key, **IDENTITY)

    headers, _ = AmazonAnthropicClaudeMessagesConfig().validate_anthropic_messages_environment(
        headers={}, model=MODEL, messages=MESSAGES, optional_params={}, litellm_params=params
    )

    assert json.loads(headers[BEDROCK_REQUEST_METADATA_HEADER]) == IDENTITY


@pytest.mark.parametrize("reverse_client_keys", [False, True])
@pytest.mark.parametrize("field_order", [ALL_FIELDS, list(reversed(ALL_FIELDS))])
@pytest.mark.parametrize("client_source", ["spend_logs_metadata", "requestMetadata"])
def test_identity_survives_a_caller_filling_every_slot(reverse_client_keys, field_order, client_source):
    """A caller sending 16 keys of its own must not evict the identity the feature exists to
    produce. Driven over every input ordering so the invariant is not an accident of one."""
    litellm.bedrock_request_metadata_fields = field_order
    client_keys = [f"client_{index:02d}" for index in range(BEDROCK_REQUEST_METADATA_MAX_PAIRS)]
    client_pairs = {key: "v" for key in (reversed(client_keys) if reverse_client_keys else client_keys)}
    if client_source == "spend_logs_metadata":
        params, optional_params = litellm_params("metadata", spend_logs_metadata=client_pairs, **IDENTITY), {}
    else:
        params, optional_params = litellm_params("metadata", **IDENTITY), {"requestMetadata": client_pairs}

    resolved = converse_body(params, optional_params)["requestMetadata"]

    assert len(resolved) == BEDROCK_REQUEST_METADATA_MAX_PAIRS
    for key, value in IDENTITY.items():
        assert resolved[key] == value
    assert len([key for key in resolved if key.startswith("client_")]) == (
        BEDROCK_REQUEST_METADATA_MAX_PAIRS - len(IDENTITY)
    )


@pytest.mark.parametrize(
    "field_order",
    [
        ["user_api_key_alias", "user_api_key_alias", "user_api_key_team_alias", "spend_logs_metadata"],
        ["user_api_key_alias", "user_api_key_team_alias", "user_api_key_alias", "spend_logs_metadata"],
        ["user_api_key_alias", "user_api_key_team_alias", "spend_logs_metadata", "user_api_key_team_alias"],
    ],
)
def test_a_field_repeated_in_the_allow_list_does_not_consume_a_client_slot(field_order):
    """An operator repeating a field in YAML must not inflate the reserved count and shrink the
    client budget. Asserts the client keys that should have fitted actually reach the wire, since
    asserting only that identity survives passes with or without the deduplication."""
    litellm.bedrock_request_metadata_fields = field_order
    client_keys = [f"client_{index:02d}" for index in range(BEDROCK_REQUEST_METADATA_MAX_PAIRS - 1)]
    params = litellm_params("metadata", spend_logs_metadata={key: "v" for key in client_keys}, **IDENTITY)

    resolved = converse_body(params)["requestMetadata"]

    expected_client_slots = BEDROCK_REQUEST_METADATA_MAX_PAIRS - len(IDENTITY)
    assert resolved == {**IDENTITY, **{key: "v" for key in client_keys[:expected_client_slots]}}
    assert len(resolved) == BEDROCK_REQUEST_METADATA_MAX_PAIRS
    assert client_keys[expected_client_slots - 1] in resolved


@pytest.mark.parametrize("client_source", ["spend_logs_metadata", "requestMetadata"])
@pytest.mark.parametrize(
    "forged_key",
    ["user_api_key_team_alias", "user_api_key_org_alias", "user_api_key_hash"],
)
def test_caller_cannot_forge_or_shadow_a_reserved_identity_key(forged_key, client_source):
    """`user_api_key_org_alias` and `user_api_key_hash` are names the proxy does not set here,
    so an exact-key reservation would let the forged value through under a name that reads as
    proxy-authoritative in the AWS billing record."""
    litellm.bedrock_request_metadata_fields = ALL_FIELDS
    forged = {forged_key: "attacker-controlled"}
    if client_source == "spend_logs_metadata":
        params, optional_params = litellm_params("metadata", spend_logs_metadata=forged, **IDENTITY), {}
    else:
        params, optional_params = litellm_params("metadata", **IDENTITY), {"requestMetadata": forged}

    resolved = converse_body(params, optional_params)["requestMetadata"]

    assert resolved == IDENTITY
    assert "attacker-controlled" not in resolved.values()


def test_identity_violating_the_character_class_is_dropped_and_the_request_succeeds():
    """A team alias with an apostrophe must not turn a working request into a 400 the moment
    an operator flips the setting on."""
    litellm.bedrock_request_metadata_fields = ALL_FIELDS
    params = litellm_params(
        "metadata",
        user_api_key_alias="prod-key",
        user_api_key_team_alias="O'Brien's team",
        user_api_key_user_email="x" * 300,
    )

    body = converse_body(params)

    assert body["requestMetadata"] == {"user_api_key_alias": "prod-key"}
    assert body["messages"]


def test_caller_supplied_violation_still_raises_bad_request():
    litellm.bedrock_request_metadata_fields = ALL_FIELDS

    with pytest.raises(litellm.exceptions.BadRequestError):
        converse_body(
            litellm_params("metadata", **IDENTITY),
            {"requestMetadata": {"team": "O'Brien's team"}},
        )


def test_non_string_and_absent_identity_values_are_dropped():
    litellm.bedrock_request_metadata_fields = ALL_FIELDS + ["user_api_key_spend"]
    params = litellm_params("metadata", user_api_key_alias="prod-key", user_api_key_spend=1.25)

    assert converse_body(params)["requestMetadata"] == {"user_api_key_alias": "prod-key"}


def test_email_is_separately_opt_in():
    """PII crossing into CloudTrail only when the operator names the field."""
    identity_with_email = {**IDENTITY, "user_api_key_user_email": "owner@example.com"}
    litellm.bedrock_request_metadata_fields = ["user_api_key_alias", "user_api_key_team_alias"]
    assert (
        "user_api_key_user_email"
        not in converse_body(litellm_params("metadata", **identity_with_email))["requestMetadata"]
    )

    litellm.bedrock_request_metadata_fields = ALL_FIELDS
    assert converse_body(litellm_params("metadata", **identity_with_email))["requestMetadata"] == identity_with_email


def test_resolver_returns_none_when_nothing_survives():
    litellm.bedrock_request_metadata_fields = ALL_FIELDS
    assert resolve_bedrock_request_metadata(litellm_params=None) is None
    assert resolve_bedrock_request_metadata(litellm_params={"metadata": {"unrelated": "x"}}) is None


def test_invoke_header_is_json_encoded_and_signed():
    litellm.bedrock_request_metadata_fields = ALL_FIELDS
    params = litellm_params("metadata", spend_logs_metadata={"cost_center": "cc-1"}, **IDENTITY)

    headers = AmazonInvokeConfig().validate_environment(
        headers={"anthropic-version": "bedrock-2023-05-31"},
        model=MODEL,
        messages=MESSAGES,
        optional_params={},
        litellm_params=params,
    )

    assert json.loads(headers[BEDROCK_REQUEST_METADATA_HEADER]) == {**IDENTITY, "cost_center": "cc-1"}
    signed = BaseAWSLLM()._filter_headers_for_aws_signature(headers)
    assert BEDROCK_REQUEST_METADATA_HEADER in signed
    assert "anthropic-version" not in signed


def test_invoke_header_does_not_displace_guardrail_headers_or_caller_headers():
    litellm.bedrock_request_metadata_fields = ALL_FIELDS
    caller_supplied = {BEDROCK_REQUEST_METADATA_HEADER: "caller-set"}

    headers = AmazonInvokeConfig().validate_environment(
        headers=dict(caller_supplied),
        model=MODEL,
        messages=MESSAGES,
        optional_params={"guardrailConfig": {"guardrailIdentifier": "gid", "guardrailVersion": "DRAFT"}},
        litellm_params=litellm_params("metadata", **IDENTITY),
    )

    assert headers[BEDROCK_REQUEST_METADATA_HEADER] == "caller-set"
    assert headers["X-Amzn-Bedrock-GuardrailIdentifier"] == "gid"
