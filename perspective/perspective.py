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
from typing import Any
from typing import Coroutine
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Union

import aiohttp

__all__ = [
    "AttributeType",
    "ScoreType",
    "Attribute",
    "AnalysisResponse",
    "AttributeScore",
    "Score",
    "SummaryScore",
    "SpanScore",
    "Client",
]


class AttributeType(enum.Enum):
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


class ScoreType(enum.Enum):
    """An enum that contains alls possible score types."""

    NONE = "NONE"
    SPAN = "SPAN"
    SUMMARY = "SUMMARY"


class Attribute:
    """Represents a Perspective Attribute that can be requested."""

    def __init__(
        self,
        name: Union[AttributeType, str],
        *,
        score_type: str = "PROBABILITY",
        score_threshold: Optional[float] = None,
    ) -> None:
        self.name = AttributeType(name)
        self.score_type: str = str(score_type)
        self.score_threshold: Optional[float] = float(score_threshold) if score_threshold else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert this attribute to a dict before sending it to the API."""
        payload = {self.name.value: {"scoreType": self.score_type, "scoreThreshold": self.score_threshold}}
        return payload


class AnalysisResponse:
    """Represents an Analysis Response received through the API."""

    def __init__(self, response: Dict[str, Any]) -> None:
        self.response: Dict[str, Any] = response

        self.languages: List[str] = self.response["languages"]

        self.detected_languages: Optional[List[str]] = (
            self.response["detected_languages"] if "detected_languages" in self.response.keys() else None
        )

        self.client_token: str = self.response["clientToken"] if "clientToken" in self.response.keys() else None

        self.attribute_scores: List[AttributeScore] = []

        for name, data in self.response["attributeScores"].items():
            self.attribute_scores.append(AttributeScore(name, data))


class AttributeScore:
    """Represents an AttributeScore received through the API."""

    def __init__(self, name: str, score_data: Dict[str, Any]) -> None:
        self.name: AttributeType = AttributeType(name)
        self.span: List[SpanScore] = []
        for score_type, data in score_data.items():

            if score_type == "spanScores":
                for span_data in data:
                    self.span.append(SpanScore(span_data))

            elif score_type == "summaryScore":
                self.summary: SummaryScore = SummaryScore(data)


class Score(abc.ABC):
    """Generic base class for scores."""

    def __init__(self) -> None:
        self.score_type: ScoreType = ScoreType.NONE


class SummaryScore(Score):
    """Represents a summary score rating for an AttributeScore."""

    def __init__(self, score_data: Dict[str, Any]) -> None:
        super().__init__()
        self.score_type: ScoreType = ScoreType.SUMMARY
        self.value: float = score_data["value"]
        self.type: str = score_data["type"]


class SpanScore(Score):
    """Represents a span score rating for an AttributeScore."""

    def __init__(self, score_data: Dict[str, Any]) -> None:
        super().__init__()
        self.score_type: ScoreType = ScoreType.SPAN
        self.value: float = score_data["score"]["value"]
        self.type: str = score_data["score"]["type"]
        self.begin: Optional[int] = score_data["begin"] if "begin" in score_data.keys() else None
        self.end: Optional[int] = score_data["end"] if "end" in score_data.keys() else None


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
        self._queue: List[Dict[str, Tuple[Coroutine[Any, Any, AnalysisResponse], asyncio.Event]]] = []
        self._values: Dict[str, AnalysisResponse] = {}
        self._current_task: Optional[asyncio.Task[Any]] = None

    async def _iter_queue(self) -> None:
        """Iterate queue and return values to _values"""
        try:
            while len(self._queue) > 0:
                queue_data: Mapping[str, Tuple[Coroutine[Any, Any, AnalysisResponse], asyncio.Event]] = self._queue.pop(
                    0
                )
                key: str = list(queue_data.keys())[0]
                data: Tuple[Coroutine[Any, Any, AnalysisResponse], asyncio.Event] = queue_data[key]

                coro: Coroutine[Any, Any, AnalysisResponse] = data[0]
                event: asyncio.Event = data[1]

                resp = await coro
                self._values[key] = resp

                event.set()
                await asyncio.sleep(1 / self.qps)
            self._current_task = None

        except Exception as e:
            print(f"Ignoring error in perspective._iter_queue: {e}", file=sys.stderr)
            print(traceback.format_exc())

    async def _execute_ratelimited(self, coro: Coroutine[Any, Any, AnalysisResponse]) -> AnalysisResponse:
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
        languages: Union[List[str], str],
        requested_attributes: Union[List[Attribute], Attribute],
        *,
        session_id: Optional[str] = None,
        client_token: Optional[str] = None,
    ) -> AnalysisResponse:
        lang = [languages] if isinstance(languages, str) else languages
        attrib = [requested_attributes] if isinstance(requested_attributes, Attribute) else requested_attributes

        return await self._execute_ratelimited(
            self._make_request(text, lang, attrib, session_id=session_id, client_token=client_token)
        )

    async def _make_request(
        self,
        text: str,
        languages: List[str],
        requested_attributes: List[Attribute],
        *,
        session_id: Optional[str] = None,
        client_token: Optional[str] = None,
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
            url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={self.api_key}"
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    return AnalysisResponse(await resp.json())
                raise ConnectionError(
                    f"Connection to Perspective API failed:\nResponse code: {resp.status}\n\n{json.dumps(await resp.json(), indent=4)}"
                )
