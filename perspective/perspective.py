# MIT License
#
# Copyright (c) 2022-present HyperGH
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import abc
import asyncio
import enum
import json
import os
import sys
import traceback
import typing as t

import aiohttp
import attr

__all__ = [
    "AttributeName",
    "ScoreType",
    "Attribute",
    "AnalysisResponse",
    "AttributeScore",
    "Score",
    "SummaryScore",
    "SpanScore",
    "Client",
]

perspective_url = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={api_key}"


class AttributeName(str, enum.Enum):
    """An enum of possible comment attributes."""

    # Stable Attributes
    TOXICITY = "TOXICITY"
    SEVERE_TOXICITY = "SEVERE_TOXICITY"
    IDENTITY_ATTACK = "IDENTITY_ATTACK"
    INSULT = "INSULT"
    PROFANITY = "PROFANITY"
    THREAT = "THREAT"
    # Experimental Attributes
    TOXICITY_EXPERIMENTAL = "TOXICITY_EXPERIMENTAL"
    SEVERE_TOXICITY_EXPERIMENTAL = "SEVERE_TOXICITY_EXPERIMENTAL"
    IDENTITY_ATTACK_EXPERIMENTAL = "IDENTITY_ATTACK_EXPERIMENTAL"
    INSULT_EXPERIMENTAL = "INSULT_EXPERIMENTAL"
    PROFANITY_EXPERIMENTAL = "PROFANITY_EXPERIMENTAL"
    THREAT_EXPERIMENTAL = "THREAT_EXPERIMENTAL"
    SEXUALLY_EXPLICIT = "SEXUALLY_EXPLICIT"
    FLIRTATION = "FLIRTATION"
    # New York Times Attributes
    # These only support "en" as the language.
    ATTACK_ON_AUTHOR = "ATTACK_ON_AUTHOR"
    ATTACK_ON_COMMENTER = "ATTACK_ON_COMMENTER"
    INCOHERENT = "INCOHERENT"
    INFLAMMATORY = "INFLAMMATORY"
    LIKELY_TO_REJECT = "LIKELY_TO_REJECT"
    OBSCENE = "OBSCENE"
    SPAM = "SPAM"
    UNSUBSTANTIAL = "UNSUBSTANTIAL"


class ScoreType(str, enum.Enum):
    """An enum that contains alls possible score types."""

    SPAN = "SPAN"
    SUMMARY = "SUMMARY"


@attr.define()
class Attribute:
    """Represents a Perspective Attribute that can be requested."""

    name: t.Union[AttributeName, str]
    score_type: str = "PROBABILITY"
    score_threshold: t.Optional[float] = None

    def to_dict(self) -> t.Dict[str, t.Any]:
        """Convert this attribute to a dict before sending it to the API."""
        payload = {str(self.name): {"scoreType": self.score_type, "scoreThreshold": self.score_threshold}}
        return payload


@attr.frozen(weakref_slot=False)
class AnalysisResponse:
    """Represents an Analysis Response received through the API."""

    response: t.Dict[str, t.Any]
    languages: t.List[str]
    detected_languages: t.List[str]
    attribute_scores: t.List[AttributeScore]
    client_token: t.Optional[str] = None

    @classmethod
    def from_dict(cls, resp: t.Dict[str, t.Any]) -> AnalysisResponse:
        scores = []

        for name, data in resp["attributeScores"].items():
            scores.append(AttributeScore.from_data(name, data))
        return cls(
            response=resp,
            languages=resp["languages"],
            detected_languages=resp.get("detected_languages", []),
            client_token=resp.get("clientToken", None),
            attribute_scores=scores,
        )


@attr.frozen(weakref_slot=False)
class AttributeScore:

    name: AttributeName
    summary: SummaryScore
    span: t.List[SpanScore] = []

    @classmethod
    def from_data(cls, name: str, data: t.Dict[str, t.Any]) -> AttributeScore:
        if raw_span := data.get("spanScores"):
            span = [SpanScore.from_data(data) for data in raw_span]

        return cls(
            name=AttributeName(name),
            span=span,
            summary=SummaryScore.from_data(data["summaryScore"]),
        )


class Score(abc.ABC):
    """Generic base class for scores."""

    @property
    @abc.abstractmethod
    def score_type(self) -> ScoreType:
        """The ScoreType of this object."""
        ...


@attr.frozen(weakref_slot=False)
class SummaryScore(Score):
    """Represents a summary score rating for an AttributeScore."""

    value: float
    type: str

    @property
    def score_type(self) -> ScoreType:
        return ScoreType.SUMMARY

    @classmethod
    def from_data(cls, data: t.Dict[str, t.Any]) -> SummaryScore:
        return cls(value=data["value"], type=data["type"])


@attr.frozen(weakref_slot=False)
class SpanScore(Score):
    """Represents a summary score rating for an AttributeScore."""

    value: float
    type: str
    begin: t.Optional[int] = None
    end: t.Optional[int] = None

    @property
    def score_type(self) -> ScoreType:
        return ScoreType.SPAN

    @classmethod
    def from_data(cls, data: t.Dict[str, t.Any]) -> SpanScore:
        return cls(
            value=data["value"],
            type=data["type"],
            begin=data["begin"] if "begin" in data.keys() else None,
            end=data["end"] if "end" in data.keys() else None,
        )


class Client:
    """The client that handles making requests to the Perspective API.

    Parameters
    ----------
    api_key : str
        The API key provided by Perspective.
    qps : int
        The maximum allowed amount of requests per second
        set in the Google Cloud Console. Defaults to 1.
    do_not_store : bool
        If True, sends a doNotStore request with the payload.
        This should be used when handling confidential data,
        or data of persons under the age of 13.
    """

    def __init__(self, api_key: str, qps: int = 1, do_not_store: bool = False) -> None:

        self.api_key: str = api_key
        self.qps: int = qps
        self.do_not_store: bool = do_not_store
        self._queue: t.List[t.Dict[str, t.Tuple[t.Awaitable[AnalysisResponse], asyncio.Event]]] = []
        self._values: t.Dict[str, AnalysisResponse] = {}
        self._current_task: t.Optional[asyncio.Task[t.Any]] = None

    async def _iter_queue(self) -> None:
        """Iterate queue and return values to _values"""
        try:
            while len(self._queue) > 0:
                queue_data: t.Mapping[str, t.Tuple[t.Awaitable[AnalysisResponse], asyncio.Event]] = self._queue.pop(0)
                key: str = list(queue_data.keys())[0]
                data: t.Tuple[t.Awaitable[AnalysisResponse], asyncio.Event] = queue_data[key]

                coro: t.Awaitable[AnalysisResponse] = data[0]
                event: asyncio.Event = data[1]

                resp = await coro
                self._values[key] = resp

                event.set()
                await asyncio.sleep(1 / self.qps)
            self._current_task = None

        except Exception as e:
            print(f"Ignoring error in perspective._iter_queue: {e}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

    async def _execute_ratelimited(self, coro: t.Awaitable[AnalysisResponse]) -> AnalysisResponse:
        """Execute a function with the ratelimits in mind."""
        key = os.urandom(16).hex()  # Identifies value in _values
        event = asyncio.Event()

        self._queue.append({key: (coro, event)})

        if self._current_task is None:
            self._current_task = asyncio.create_task(self._iter_queue())

        await event.wait()
        return self._values.pop(key)

    async def analyze(
        self,
        text: str,
        languages: t.Union[t.List[str], str],
        requested_attributes: t.Union[t.List[Attribute], Attribute],
        *,
        session_id: t.Optional[str] = None,
        client_token: t.Optional[str] = None,
    ) -> AnalysisResponse:
        lang = [languages] if isinstance(languages, str) else languages
        attrib = [requested_attributes] if isinstance(requested_attributes, Attribute) else requested_attributes

        return await self._execute_ratelimited(
            self._make_request(text, lang, attrib, session_id=session_id, client_token=client_token)
        )

    async def _make_request(
        self,
        text: str,
        languages: t.List[str],
        requested_attributes: t.List[Attribute],
        *,
        session_id: t.Optional[str] = None,
        client_token: t.Optional[str] = None,
    ) -> AnalysisResponse:
        # TODO: Reuse session
        async with aiohttp.ClientSession() as session:

            attributes = {}
            for attribute in requested_attributes:
                attributes.update(attribute.to_dict())

            payload = {
                "comment": {
                    "text": text,
                    "type": "PLAIN_TEXT",
                },
                "languages": languages,
                "requestedAttributes": attributes,
                "doNotStore": self.do_not_store,
                "sessionId": session_id,
                "clientToken": client_token,
            }

            url = perspective_url.format(api_key=self.api_key)

            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    return AnalysisResponse.from_dict(await resp.json())
                raise ConnectionError(
                    f"Connection to Perspective API failed:\nResponse code: {resp.status}\n\n{json.dumps(await resp.json(), indent=4)}"
                )
