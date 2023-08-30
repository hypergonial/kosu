# kosu

Asynchronous API wrapper for the [Perspective API](https://perspectiveapi.com/).

## Installation

```sh
pip install git+https://github.com/hypergonial/kosu.git
```

## Quick Start

```py
import asyncio
import kosu

attribs = [
    kosu.Attribute(kosu.AttributeName.TOXICITY),
    kosu.Attribute(kosu.AttributeName.SEVERE_TOXICITY),
    kosu.Attribute(kosu.AttributeName.INSULT),
]


async def main() -> None:
    client = kosu.Client("AIzaSyACns6TDjKBilALPCgOwRxzTnYF-VXBqVc")
    resp: kosu.AnalysisResponse = await client.analyze(
        "Shut up, you're an idiot!", attribs, languages=["en"]
    )

    for score in resp.attribute_scores:
        print(score)

    await client.close()


asyncio.run(main())

```

Outputs:

```py
AttributeScore(name=<AttributeName.INSULT: 'INSULT'>, summary=SummaryScore(value=0.9263389, type='PROBABILITY'), span=[SpanScore(value=0.9263389, type='PROBABILITY', begin=0, end=25)])

AttributeScore(name=<AttributeName.TOXICITY: 'TOXICITY'>, summary=SummaryScore(value=0.944597, type='PROBABILITY'), span=[SpanScore(value=0.944597, type='PROBABILITY', begin=0, end=25)])

AttributeScore(name=<AttributeName.SEVERE_TOXICITY: 'SEVERE_TOXICITY'>, summary=SummaryScore(value=0.25624833, type='PROBABILITY'), span=[SpanScore(value=0.25624833, type='PROBABILITY', begin=0, end=25)])
```
