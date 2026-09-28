"""Where a box can come from: a local weight, or a hosted vision model - never a committed answer.

Two providers, one protocol, and an order that is a cost decision rather than a preference.

**Local first, and it is the product's own code path.** `LocalWeights` builds
`app.inference.YoloDetector` with the geometry from the weight's record
(`app.models.requirement_for` -> `stretch` for the installed v1) and hands it the frame exactly as
`Pipeline` would. So a suggested box is a box the running app would itself draw: there is no second
implementation of the stretch rule to drift, no second idea of what `imgsz` means, and no
"the annotator sees it differently" class of bug. It is also free, offline and fast, which is why it
is asked first.

**Hosted second, for the two things local weights cannot do.** A general vision model can be *told*
the class list, so it is the only thing here that can help with a product no local weight knows
(local weights can only propose classes they were trained on - which is why a class with no v1 name
is unlabelable by them), and it can look at a frame where the local weight found nothing and say
whether there is an item there. It is asked **only** for those frames, behind an explicit setting,
because every call is money and store imagery leaving the machine.

Three rules that are not negotiable in this file:

- **A suggestion is never committed.** Nothing here writes to the store; `store.write` is a separate
  call made by a human pressing save.
- **A hosted model never marks a null.** "No boxes" from a hosted model is a weak signal about a
  cluttered frame, not the deliberate statement that there is no item - and a null is what teaches
  the head that the frame is background. Only a person can assert that.
- **Keys are read from the environment, never from `Settings`.** `Settings` is serialized wholesale
  to the renderer (`roboflow_api_key_present` exists for that reason), so an authoring tool's API key
  must not go near it: `credentials.env_value` is the reader, it never prints the value, and a
  missing key means the provider simply is not offered.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

from app.credentials import env_value

from .store import CLASS_NAMES, Box

# The generations a local weight can come from, in the order the app itself lists them. Read through
# `app.models` so the annotator's picker and the app's *Weights on disk* list are the same fact.
DEFAULT_IMGSZ = 640


class Suggestion(Protocol):
    """What a provider hands back: boxes, plus what produced them (recorded in provenance)."""


@dataclass(frozen=True)
class Proposed:
    boxes: list[Box]
    provider: str
    note: str = ""


def clamp_box(cls: int, cx: float, cy: float, w: float, h: float) -> Box:
    """A box pulled inside the frame, or dropped to a degenerate one.

    Model output is not trusted geometry: a hosted vision model in particular returns coordinates
    that are *approximately* normalized, and one at `-0.02` would be written to the label file and
    then silently clipped by the trainer - a frame that reads as labeled while contributing a box
    nobody can see. Clamping is the honest repair (the box is real, the edge estimate is not), and
    anything that clamps to nothing is dropped by `Box.valid()` before it reaches disk.
    """
    left, top = cx - w / 2, cy - h / 2
    right, bottom = cx + w / 2, cy + h / 2
    left, top = max(0.0, left), max(0.0, top)
    right, bottom = min(1.0, right), min(1.0, bottom)
    return Box(
        cls=int(cls),
        cx=(left + right) / 2,
        cy=(top + bottom) / 2,
        w=max(0.0, right - left),
        h=max(0.0, bottom - top),
    )


def parse_class_name(raw: object) -> int | None:
    """Map a hosted model's label onto a class index, or None when it is not one of ours.

    Deliberately strict - exact name, slug, or a case-insensitive match - and it never invents an
    index: an answer attributed to the wrong product is worse than no answer, and the local weight
    is the one that cannot make this mistake (it predicts an index directly).
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    for index, name in enumerate(CLASS_NAMES):
        if text == name or text.lower() == name.lower():
            return index
    from .store import CLASS_SLUGS  # local import: keeps the module's public surface small

    for index, slug in enumerate(CLASS_SLUGS):
        if text.lower() == slug.lower():
            return index
    return None


# --------------------------------------------------------------------------
# Local weights
# --------------------------------------------------------------------------


class LocalWeights:
    """`app.inference.YoloDetector`, driven the way the capture path drives it.

    The detector is built through a factory so tests run without ultralytics, a GPU or a weight file
    - the same injection point the sidecar's own tests use (`detector_factory`), which is why this
    class is worth having rather than calling `YOLO()` directly.

    The **roster guard** is here and it fails closed. A weight may propose boxes only if its own
    recorded class list matches the dataset's class list by name; otherwise the annotator would
    offer boxes under classes this dataset does not have (the 21-output failure, arriving as
    suggestions) and the frames would then be saved with indices that mean something else. It is
    the same check the probe shows the operator, applied where it can prevent work rather than
    explain it afterwards.
    """

    def __init__(
        self,
        weights: str,
        resize_mode: str = "stretch",
        conf: float = 0.25,
        imgsz: int = DEFAULT_IMGSZ,
        device: str = "cpu",
        detector_factory: Callable[..., object] | None = None,
        class_names: Iterable[str] | None = None,
    ):
        self.weights = weights
        self.resize_mode = resize_mode
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self._factory = detector_factory
        self._detector = None
        self.names = list(class_names) if class_names is not None else list(CLASS_NAMES)
        mismatch = self.roster_mismatch()
        if mismatch:
            raise ValueError(
                f"{weights} predicts a different class list than this dataset labels: {mismatch}. "
                "Suggesting from it would put boxes under classes the labels do not have - pick a "
                "weight whose recorded class_names match, or relabel with the matching dataset."
            )

    def roster_mismatch(self) -> str:
        have, want = set(self.names), set(CLASS_NAMES)
        missing, extra = sorted(want - have), sorted(have - want)
        if not missing and not extra:
            return ""
        parts = []
        if missing:
            parts.append("missing " + ", ".join(repr(n) for n in missing))
        if extra:
            parts.append("unexpected " + ", ".join(repr(n) for n in extra))
        return "; ".join(parts)

    def _build(self):
        if self._detector is None:
            factory = self._factory
            if factory is None:  # pragma: no cover - exercised by the real tool, not the suite
                from app.inference import YoloDetector

                factory = YoloDetector
            self._detector = factory(
                self.weights,
                self.device,
                self.conf,
                self.imgsz,
                resize_mode=self.resize_mode,
            )
        return self._detector

    def suggest(self, frame) -> Proposed:
        """Boxes for one BGR frame, normalized, clamped, and carrying their class index.

        `Detection.cls` is the class **name** and `Detection.box` is already normalized (that is
        `app.inference.normalize_detections`'s contract), so this translates a name to the
        dataset's index rather than copying a number across - which is what makes a weight whose
        own class order differs usable instead of silently wrong.
        """
        detector = self._build()
        boxes: list[Box] = []
        unknown: list[str] = []
        for det in detector.infer(frame):
            index = parse_class_name(getattr(det, "cls", None))
            if index is None:
                name = getattr(det, "cls", None)
                if name:
                    unknown.append(str(name))
                continue
            x1, y1, x2, y2 = (float(v) for v in det.box)
            box = clamp_box(index, (x1 + x2) / 2, (y1 + y2) / 2, abs(x2 - x1), abs(y2 - y1))
            if box.valid():
                boxes.append(box)
        note = f"{len(boxes)} proposed at conf {self.conf}"
        if unknown:
            note += f"; ignored labels outside this roster: {', '.join(sorted(set(unknown))[:3])}"
        return Proposed(boxes=boxes, provider=f"local:{_stem(self.weights)}", note=note)

    @property
    def configured(self) -> bool:
        return True


def _stem(path: str) -> str:
    return str(path).replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]


# --------------------------------------------------------------------------
# Hosted vision models
# --------------------------------------------------------------------------


# The three services, as one protocol. The key names are the ones the vendors document, and they
# are read from the environment or `sidecar/.env` (`credentials.env_value`) - never from `Settings`.
HOSTED_PROVIDERS: dict[str, dict] = {
    "openai": {
        "env": "OPENAI_API_KEY",
        "url": "https://api.openai.com/v1/chat/completions",
        "model_env": "OPENAI_VISION_MODEL",
        "model": "gpt-4o",
    },
    "gemini": {
        "env": "GEMINI_API_KEY",
        "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "model_env": "GEMINI_VISION_MODEL",
        "model": "gemini-2.0-flash",
    },
    "anthropic": {
        "env": "ANTHROPIC_API_KEY",
        "url": "https://api.anthropic.com/v1/messages",
        "model_env": "ANTHROPIC_VISION_MODEL",
        "model": "claude-3-5-sonnet-latest",
    },
}

# One prompt for all three, because the *contract* is what matters and not the vendor: name the
# classes, ask for normalized boxes, demand JSON and nothing else. The class list is injected and
# spelled exactly as the labels are, which is the whole reason a hosted model can help with a class
# no local weight knows.
PROMPT = """You are labelling grocery items for an object detector.

The classes are:
{classes}

Look at the image and return every instance of those items, as JSON on one line:
{{"objects": [{{"class": "<one of the class names above>", "x": <left>, "y": <top>, "w": <width>, "h": <height>}}]}}

x, y, w and h are fractions of the image width and height (0 to 1), with x and y at the box's
top-left corner. Use the exact class name. If there is no instance of any class, return
{{"objects": []}}. Return only the JSON, with no commentary."""


def http_post(url: str, headers: dict, payload: dict, timeout: float) -> tuple[int, dict]:
    """The real transport: `httpx.post`, as `(status, body)`.

    `httpx`, not a vendor SDK, for the reason `app/roboflow.py` gives - it is already a dependency,
    and three SDKs would be three more things to keep importable for a call the tool does not need
    to work. This is also the default `post` below, which it has to be: the annotator is constructed
    by `annotate/run.py`, and a seam that is only ever filled by a test is a provider that cannot
    run in the tool at all.
    """
    import httpx

    response = httpx.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        body = response.json()
    except ValueError:
        # A non-JSON body is not an answer from any of these models, but it is worth keeping the
        # first of it: `suggest` reports the status, and a sentence of the body is the difference
        # between "HTTP 502" and "HTTP 502 from a proxy".
        body = {"raw": response.text[:300]}
    return response.status_code, body


class HostedVision:
    """A vision model asked with the class list in the prompt, for frames the local weight missed.

    `post` is injectable (a callable taking `(url, headers, payload, timeout)` and returning
    `(status, body)`) so this whole class is testable with no network and no key - which is the
    property that makes a provider contract worth having rather than a pile of vendor SDKs. The
    default is `http_post`, the live one: injection is for tests, not for production wiring.
    """

    def __init__(
        self,
        name: str,
        api_key: str,
        post: Callable | None = None,
        model: str = "",
        timeout: float = 60.0,
    ):
        if name not in HOSTED_PROVIDERS:
            raise ValueError(f"unknown hosted provider {name!r}")
        if not api_key:
            raise ValueError(f"{name} needs an API key ({HOSTED_PROVIDERS[name]['env']})")
        self.name = name
        self.api_key = api_key
        self._post = post or http_post
        self.model = model or HOSTED_PROVIDERS[name]["model"]
        self.timeout = timeout

    def roster_mismatch(self) -> str:
        """Always empty: this provider is *told* the roster, so it cannot disagree about it."""
        return ""

    @property
    def configured(self) -> bool:
        return True

    def suggest(self, frame_bytes: bytes, media_type: str = "image/jpeg") -> Proposed:
        """Boxes from one encoded image. Raises on transport failure, with the key kept out."""
        payload = self._payload(frame_bytes, media_type)
        url, headers = self._request(frame_bytes, media_type, payload)
        try:
            status, body = self._post(url, headers, payload, self.timeout)
        except Exception as exc:
            # A transport failure is reported as a sentence rather than an exception whose text may
            # carry the key: Gemini's is a query parameter, so its own `ConnectError` would print it
            # into a log. Same rule as `_redact` below, applied to the path that has no status code
            # to attach it to.
            raise RuntimeError(
                f"{self.name} could not be reached: {_redact(str(exc), self.api_key)}"
            ) from None
        if status != 200:
            raise RuntimeError(
                f"{self.name} returned HTTP {status}: {_redact(str(body)[:300], self.api_key)}"
            )
        return self._parse(body)

    # -- per-vendor shapes -----------------------------------------------------
    #
    # One prompt, three envelopes. Written out rather than shared behind a base class because the
    # differences are not cosmetic: the image is a data URL to OpenAI, an inline base64 *part* to
    # Gemini, and a typed content block to Anthropic, and the answer comes back in as many shapes.

    def _payload(self, frame_bytes: bytes, media_type: str) -> dict:
        b64 = base64.b64encode(frame_bytes).decode("ascii")
        prompt = PROMPT.format(classes="\n".join(f"- {n}" for n in CLASS_NAMES))
        if self.name == "openai":
            return {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{media_type};base64,{b64}"},
                            },
                        ],
                    }
                ],
                "temperature": 0,
            }
        if self.name == "gemini":
            return {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": media_type, "data": b64}},
                        ]
                    }
                ],
                "generationConfig": {"temperature": 0},
            }
        return {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": b64},
                        },
                    ],
                }
            ],
        }

    def _request(self, frame_bytes: bytes, media_type: str, payload: dict) -> tuple[str, dict]:
        if self.name == "gemini":
            url = HOSTED_PROVIDERS["gemini"]["url"].format(model=self.model)
            return (
                f"{url}?key={self.api_key}",
                {"Content-Type": "application/json"},
            )
        if self.name == "anthropic":
            return (
                HOSTED_PROVIDERS["anthropic"]["url"],
                {
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                },
            )
        return (
            HOSTED_PROVIDERS["openai"]["url"],
            {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )

    def _text_of(self, body: object) -> str:
        """The model's answer, whatever envelope it arrived in. Also the one place a bad response
        turns into a sentence rather than an `AttributeError` three frames up."""
        if not isinstance(body, dict):
            return ""
        if self.name == "openai":
            choices = body.get("choices") or []
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message") or {}
                if isinstance(message, dict):
                    return str(message.get("content") or "")
            return ""
        if self.name == "gemini":
            candidates = body.get("candidates") or []
            if candidates and isinstance(candidates[0], dict):
                content = candidates[0].get("content") or {}
                parts = content.get("parts") if isinstance(content, dict) else None
                if parts and isinstance(parts[0], dict):
                    return str(parts[0].get("text") or "")
            return ""
        blocks = body.get("content") or []
        if blocks and isinstance(blocks[0], dict):
            return str(blocks[0].get("text") or "")
        return ""

    def _parse(self, body: object) -> Proposed:
        text = self._text_of(body).strip()
        objects = _json_objects(text)
        boxes: list[Box] = []
        unknown: list[str] = []
        for item in objects:
            if not isinstance(item, dict):
                continue
            index = parse_class_name(item.get("class") or item.get("name"))
            if index is None:
                label = item.get("class") or item.get("name")
                if label:
                    unknown.append(str(label))
                continue
            try:
                x, y = float(item.get("x", 0)), float(item.get("y", 0))
                w, h = float(item.get("w", 0)), float(item.get("h", 0))
            except (TypeError, ValueError):
                continue
            box = clamp_box(index, x + w / 2, y + h / 2, w, h)
            if box.valid():
                boxes.append(box)
        note = f"{len(boxes)} proposed by {self.name}:{self.model}"
        if unknown:
            # Named rather than dropped in silence: a model answering with its own vocabulary is
            # the ordinary failure here, and "it found something, under a name we do not have" is
            # a different fact from "it found nothing".
            note += f"; ignored labels outside this roster: {', '.join(sorted(set(unknown))[:3])}"
        return Proposed(boxes=boxes, provider=f"hosted:{self.name}", note=note)


def _json_objects(text: str) -> list:
    """The `objects` list out of a model's answer, tolerating a code fence around it.

    Structural rather than a regex over the whole response, for the same reason
    `app/roboflow.find_predictions` is: the wrapper is the part that varies, and a model that
    prefixes its JSON with a sentence is answering correctly.
    """
    candidates = [text]
    if "```" in text:
        for chunk in text.split("```"):
            stripped = chunk.strip()
            if stripped.startswith("{"):
                candidates.append(stripped)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            body = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(body, dict):
            objects = body.get("objects")
            if isinstance(objects, list):
                return objects
        if isinstance(body, list):
            return body
    return []


def _redact(text: str, key: str) -> str:
    """Never let a key reach a log line, a response body or a traceback - the rule
    `credentials.py` and `roboflow.py` both keep."""
    return text.replace(key, "***") if key else text


def hosted_available(getter: Callable[[str], str] = env_value) -> list[str]:
    """Which hosted providers this machine actually has a key for.

    The annotator runs with **no key at all** - local weights are enough to label 491 frames of
    classes they know - so this drives only which second opinion is offered, never whether the tool
    works.
    """
    return [name for name, spec in HOSTED_PROVIDERS.items() if getter(spec["env"])]


def build_hosted(
    name: str,
    post: Callable | None = None,
    getter: Callable[[str], str] = env_value,
    model: str = "",
) -> HostedVision | None:
    """A hosted provider for `name`, or None when this machine has no key for it.

    `post` is the network seam and defaults to the live transport (`http_post`) rather than to
    `None`: the annotator's own construction site passes nothing, so a default of `None` here would
    make every hosted suggestion a `TypeError` inside the request - a second opinion that is offered
    by `--hosted`, printed in the startup line and counted against `--max-hosted-calls`, and then
    fails on the one frame it is asked about.
    """
    spec = HOSTED_PROVIDERS.get(name)
    if not spec:
        return None
    key = getter(spec["env"])
    if not key:
        return None
    return HostedVision(name, key, post, model=model or getter(spec["model_env"]))


# --------------------------------------------------------------------------
# Ordering
# --------------------------------------------------------------------------


def suggest_for(
    frame_array,
    local: LocalWeights | None,
    hosted: HostedVision | None,
    encoded: bytes | None = None,
) -> Proposed:
    """Local first, hosted only where local found nothing. Pure: it proposes, it never records.

    The order is the cost rule from the module docstring, in one place so a caller cannot invert it
    by accident: a hosted call is money and egress, so it is spent only on the frames a free
    offline model could not help with - which is exactly where a second opinion is worth something.

    A hosted provider asked on its own (no local weight is configured or usable) is the one case
    where the rule has nothing to consult, and it is handled rather than refused: the alternative
    is an annotator that cannot suggest at all on a machine with no installed weight.
    """
    if local is None:
        if hosted is None:
            return Proposed(boxes=[], provider="none", note="no provider is configured")
        if encoded is None:
            raise ValueError("the hosted provider needs the encoded image")
        return hosted.suggest(encoded)

    proposed = local.suggest(frame_array)
    if proposed.boxes or hosted is None:
        if not proposed.boxes:
            proposed = Proposed(
                boxes=[],
                provider=proposed.provider,
                note=f"{proposed.note}; nothing found, and no hosted provider is configured",
            )
        return proposed
    if encoded is None:
        raise ValueError("the hosted provider needs the encoded image")
    second = hosted.suggest(encoded)
    return Proposed(
        boxes=second.boxes,
        provider=second.provider,
        note=f"local found nothing; {second.note}",
    )
