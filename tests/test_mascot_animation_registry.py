"""Validator for the mascot bird animation registry.

The animation system in templates/partials/mascot_bird.html keeps every
animation spec, particle kind, and trigger call site loose-coupled by
string name. There is no build step that would catch a typo or a
missing entry, so a stale fire('Excitd') would just silently no-op at
runtime.

This test parses the inline JS and enforces three contracts:

1. ANIMS schema — every entry has name, durationMs, weight, fn.
2. Particle integrity — every `particles: 'foo'` references a key that
   exists in PARTICLE_KINDS.
3. Call-site integrity — every `wmbMascot.fire('Name')` in the repo
   points at an animation that exists in the registry.

If anyone refactors the mascot file to use ES modules or `let ANIMS`,
the regex extractor breaks; the test will fail loudly with a clear
"could not locate ANIMS literal" message — that is the intended signal
to update this validator alongside the structural change.
"""

from __future__ import annotations

import re
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _read(relative_path: str) -> str:
    return (_project_root() / relative_path).read_text(encoding="utf-8")


def _extract_anims_body(mascot_html: str) -> str:
    """Return the JS source between the outermost `[` and `]` of the
    `var ANIMS = [...]` literal.

    A bracket counter is required (not a non-greedy regex) because
    animation bodies contain inner JS arrays — e.g. `palette = [...]`,
    `wedgeColors = [...]`, `layers = [...]` — whose closing `];` sits
    on its own line and would terminate a non-greedy match early.
    """
    head = re.search(r"var\s+ANIMS\s*=\s*\[", mascot_html)
    assert head, "could not locate ANIMS literal in mascot_bird.html"
    start = head.end()
    depth = 1
    i = start
    n = len(mascot_html)
    while i < n and depth > 0:
        c = mascot_html[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
        i += 1
    assert depth == 0, "ANIMS literal is unbalanced — JS syntax broken"
    return mascot_html[start : i - 1]


def _extract_anim_names(mascot_html: str) -> list[str]:
    return re.findall(r"name:\s*'([^']+)'", _extract_anims_body(mascot_html))


def _extract_anim_specs(mascot_html: str) -> list[dict[str, str]]:
    # Split entries on the top-level `{` … `}` separators. A simple
    # brace-counter is enough because the JS we maintain here has no
    # template literals or regex literals that would confuse it.
    src = _extract_anims_body(mascot_html)
    specs: list[dict[str, str]] = []
    depth = 0
    start = -1
    for i, ch in enumerate(src):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                entry = src[start : i + 1]
                spec: dict[str, str] = {}
                name = re.search(r"name:\s*'([^']+)'", entry)
                if name:
                    spec["name"] = name.group(1)
                duration = re.search(r"durationMs:\s*(\d+)", entry)
                if duration:
                    spec["durationMs"] = duration.group(1)
                weight = re.search(r"weight:\s*([0-9.]+)", entry)
                if weight:
                    spec["weight"] = weight.group(1)
                if re.search(r"\bfn:\s*function", entry):
                    spec["fn"] = "yes"
                # `particles` may be a single quoted string OR an array
                # of quoted strings (for combo effects like Sweating's
                # `['sweat', 'sun']`). Extract every quoted token that
                # appears in the particles slot — for the integrity test
                # we don't care about ordering, only that each name
                # resolves to a known kind.
                particles_array = re.search(
                    r"particles:\s*\[([^\]]*)\]", entry
                )
                if particles_array:
                    names = re.findall(r"'([^']+)'", particles_array.group(1))
                    if names:
                        spec["particles"] = ",".join(names)
                else:
                    particles = re.search(r"particles:\s*'([^']+)'", entry)
                    if particles:
                        spec["particles"] = particles.group(1)
                priority = re.search(r"priority:\s*'([^']+)'", entry)
                if priority:
                    spec["priority"] = priority.group(1)
                specs.append(spec)
                start = -1
    return specs


def _extract_particle_kinds(mascot_html: str) -> set[str]:
    # The closing anchor is `\n    };` with exactly 4 spaces — that
    # matches the IIFE-level indent of the object literal's close brace.
    # Inner `};` patterns inside emit() / draw() bodies sit at 16-space
    # indent, so they don't satisfy the anchor and the non-greedy `.*?`
    # walks past them correctly.
    block = re.search(
        r"var\s+PARTICLE_KINDS\s*=\s*\{(.*?)\n {4}\};",
        mascot_html,
        re.DOTALL,
    )
    assert block, "could not locate PARTICLE_KINDS literal in mascot_bird.html"
    # Kind keys are bare identifiers followed by `: {` at 8-space indent.
    return set(re.findall(r"^ {8}(\w+):\s*\{", block.group(1), re.MULTILINE))


def _find_fire_call_sites() -> list[tuple[Path, str]]:
    root = _project_root()
    hits: list[tuple[Path, str]] = []
    for sub in ("templates", "assets", "web"):
        base = root / sub
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in {".html", ".js", ".css"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for match in re.finditer(
                r"wmbMascot\.fire\(\s*['\"]([^'\"]+)['\"]\s*\)", text
            ):
                hits.append((path, match.group(1)))
    return hits


def test_every_anim_has_required_schema_fields():
    html = _read("templates/partials/mascot_bird.html")
    specs = _extract_anim_specs(html)
    assert specs, "no ANIMS entries parsed"
    for spec in specs:
        missing = {"name", "durationMs", "weight", "fn"} - spec.keys()
        assert not missing, (
            f"animation {spec.get('name', '<unnamed>')!r} is missing "
            f"required fields: {sorted(missing)}"
        )


def test_anim_names_are_unique():
    html = _read("templates/partials/mascot_bird.html")
    names = _extract_anim_names(html)
    assert names, "no animation names parsed from ANIMS"
    duplicates = {n for n in names if names.count(n) > 1}
    assert not duplicates, f"duplicate animation names in ANIMS: {duplicates}"


def test_particle_references_resolve():
    html = _read("templates/partials/mascot_bird.html")
    specs = _extract_anim_specs(html)
    kinds = _extract_particle_kinds(html)
    for spec in specs:
        if "particles" not in spec:
            continue
        # `particles` may carry one name or a comma-joined list (the
        # extractor normalises array-form to comma-separated). Check
        # every individual name against PARTICLE_KINDS.
        for name in spec["particles"].split(","):
            assert name in kinds, (
                f"animation {spec['name']!r} references particle kind "
                f"{name!r} which is not in PARTICLE_KINDS "
                f"(known: {sorted(kinds)})"
            )


def test_every_fire_call_site_targets_a_real_animation():
    html = _read("templates/partials/mascot_bird.html")
    names = set(_extract_anim_names(html))
    call_sites = _find_fire_call_sites()
    bad: list[str] = []
    for path, name in call_sites:
        if name in names:
            continue
        bad.append(f"{path.relative_to(_project_root())}: fire({name!r})")
    assert not bad, (
        "wmbMascot.fire() call sites reference unknown animations:\n  "
        + "\n  ".join(bad)
        + f"\nKnown animations: {sorted(names)}"
    )


def test_excited_event_animation_is_registered():
    """The Excited animation is the contract surface for the
    detection-event hook in led_ticker.html. If it gets renamed or
    deleted, the live-bird-detected feedback silently breaks."""
    html = _read("templates/partials/mascot_bird.html")
    specs = _extract_anim_specs(html)
    excited = next((s for s in specs if s.get("name") == "Excited"), None)
    assert excited is not None, (
        "'Excited' animation missing from ANIMS — the SSE detection "
        "hook in led_ticker.html fires it on every new detection."
    )
    assert excited.get("priority") == "event", (
        "'Excited' must be priority:'event' so the ambient scheduler "
        "does not pull it randomly — it is push-only by design."
    )
