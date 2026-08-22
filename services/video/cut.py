"""
The cut planner — turning a beat map plus a pile of clips into an edit.

This is the part that has an opinion. `beats.py` says where the music is;
everything here is editorial, and every rule below comes from how music
editors actually cut, not from what is convenient to compute:

**Length is chosen in beats, never in seconds.** A cut is 1, 2, 4, 8 or 16
beats long. That ladder is why the result feels intentional rather than merely
synchronised — the eye reads a 4-beat and an 8-beat shot as related, and reads
3.7 seconds as an accident. Two cuts in a piece are allowed off the ladder: a
short pickup that re-syncs the grid to the next downbeat, and the last one,
which ends where the music does.

**Energy picks the rung.** Loud bars cut fast, quiet bars hold. Each style
declares which rung it uses in the song's quietest bar and which in its
loudest, and `BeatMap.bar_energy` — a *rank*, not a level, for the reasons set
out in beats.py — interpolates between them. This is the single thing that
separates an automatic edit from a metronome: cutting every beat all the way
through is exhausting, and holding all the way through is inert.

**Long shots start on the "1".** Anything a bar or longer that would begin
mid-bar is shortened to land on the next downbeat instead, which re-syncs the
grid rather than dragging it. Section boundaries — where `beats.py` says the
music itself changed — always force a cut.

**The edit covers the whole song, not the whole grid.** A tracked beat grid
starts at the first downbeat it can justify and stops at the last, which on a
4/4 track at 120 BPM can leave a second and a half of music at each end with no
picture over it. So an edit that begins at bar 0 opens with a *pickup* — one
shot from 0.0 up to the first downbeat, exactly the way an editor opens on the
music rather than on the first bar — and the closing shot is held out to the
final sample. Only a span the user deliberately started later carries an offset,
and then the song is muxed offset to match.

**Sources are dealt from a bag.** All of them are shuffled, dealt out, then
reshuffled: every clip appears before any clip appears twice, and the same clip
never lands twice in a row. The user picked those clips because they want to
see all of them.

**Slow motion is not a special effect here, it is a requirement.** The clips
this tool generates are 2-4 seconds long and a 16-beat hold at 90 BPM is
10.7 s. Without stretching, the slow styles could not exist on this material,
so a shot that cannot fill its slot is slowed to fit — up to `max_stretch`,
past which the cut is shortened instead. Rhythm outranks any single shot.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

# The rungs, longest first. Powers of two because musical phrases are: a
# 3-beat shot inside a 4/4 bar reads as a mistake, not as a choice.
LADDER: tuple[int, ...] = (16, 8, 4, 2, 1)

# Below this a cut is a frame or two and reads as a glitch rather than an edit.
MIN_CUT_SECONDS = 0.18

# How far a shot may be slowed to fill its slot. 2.0 = half speed, which still
# looks deliberate; past it the motion turns to syrup and the clip's own
# interpolation artefacts start to show.
MAX_STRETCH_DEFAULT = 2.0
MAX_STRETCH_CEILING = 4.0

# Fade-in on the first frames of a shot that opens a new musical section. Not a
# cross-dissolve — the segment keeps its exact length, so the beat grid is
# untouched — just a two-frame lift out of black that marks the change.
SECTION_FADE_SECONDS = 0.10
OPENING_FADE_SECONDS = 0.60
CLOSING_FADE_SECONDS = 1.20


@dataclass(frozen=True)
class Style:
    """One way of reading a song.

    `quiet`/`loud` are indices into LADDER: the rung used in the least and the
    most energetic bar of the piece. Equal values mean a fixed pace; a wide
    span means the edit breathes with the music.
    """
    key: str
    label: str
    hint: str
    quiet: int
    loud: int
    jitter: float = 0.0      # chance per cut of stepping one rung off-plan


STYLES: tuple[Style, ...] = (
    Style("dramaturgie", "Dramaturgie",
          "Liest die Energie des Songs: hält in den leisen Takten, treibt in den lauten. "
          "Die größte Spannweite — der Standard.",
          quiet=0, loud=3, jitter=0.15),
    Style("atem", "Atem",
          "Sehr lange Einstellungen, 16 und 8 Schläge. Für Ambient und Drone — "
          "der Schnitt tritt zurück.",
          quiet=0, loud=1, jitter=0.10),
    Style("welle", "Welle",
          "8 und 4 Schläge, ruhig fließend. Der klassische Musikvideo-Grundschlag.",
          quiet=1, loud=2, jitter=0.15),
    Style("puls", "Puls",
          "4 und 2 Schläge, treibend. Jeder Takt bringt eine neue Einstellung.",
          quiet=2, loud=3, jitter=0.15),
    Style("stakkato", "Stakkato",
          "2 Schläge und einzelne Schläge. Sehr schnell — für kurze Stücke "
          "und viel Material.",
          quiet=3, loud=4, jitter=0.20),
)

STYLE_BY_KEY = {s.key: s for s in STYLES}
DEFAULT_STYLE = STYLES[0].key


def style_options() -> list[dict]:
    """The style list, for the frontend. Read from here rather than hardcoded
    on the client so a retuned style never has to be declared twice."""
    return [
        {"key": s.key, "label": s.label, "hint": s.hint,
         "fastest": LADDER[s.loud], "slowest": LADDER[s.quiet]}
        for s in STYLES
    ]


@dataclass(frozen=True)
class Source:
    """One thing that can appear in the edit."""
    key: str                 # "clip:<uuid>" | "video:<uuid>"
    duration: float          # playable seconds
    label: str = ""


@dataclass
class Cut:
    """One shot in the finished edit."""
    source: int              # index into the plan's source list
    src_in: float            # seconds into the source
    src_out: float           # seconds into the source (src_out - src_in = what is read)
    start: float             # seconds on the song's timeline
    end: float
    beats: int               # length in beats
    beat: int                # index into BeatMap.beats where it starts
    bar: int
    speed: float             # 1.0 = as shot, 0.5 = half speed
    downbeat: bool
    section: bool            # opens a new musical section
    fade_in: float = 0.0     # seconds, 0 = hard cut

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class EditPlan:
    cuts: list[Cut]
    style: str
    seed: int
    bpm: float
    beats_per_bar: int
    song_start: float
    song_end: float
    anticipation: float
    max_stretch: float
    warnings: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.song_end - self.song_start

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "EditPlan":
        payload = dict(data)
        payload["cuts"] = [Cut(**c) for c in payload.get("cuts", [])]
        payload.setdefault("warnings", [])
        return cls(**payload)


# ── The bag ──────────────────────────────────────────────────────────────────

class _Bag:
    """Deals every source once, reshuffles, deals again.

    Plain random choice clumps: with five clips it will show the same one twice
    running about one time in five, and leave another out entirely. Dealing
    from a bag guarantees balance, and the "not the one just used" rule at the
    reshuffle seam is the only place clumping could still get in.
    """

    def __init__(self, count: int, rng: random.Random) -> None:
        self._count = count
        self._rng = rng
        self._pending: list[int] = []
        self._last = -1

    def _refill(self) -> None:
        order = list(range(self._count))
        self._rng.shuffle(order)
        if self._count > 1 and order[0] == self._last:
            order[0], order[-1] = order[-1], order[0]
        self._pending = order

    def take(self, accepts=None) -> int:
        """Next source, preferring one `accepts(index)` approves of.

        A rejected source stays in the bag rather than being burned, so
        rejecting a short clip for a long slot does not cost it its turn.
        """
        for _ in range(2):
            if not self._pending:
                self._refill()
            if accepts is not None:
                for pos, idx in enumerate(self._pending):
                    if idx != self._last and accepts(idx):
                        self._pending.pop(pos)
                        self._last = idx
                        return idx
                self._pending = []          # nothing here fits; try a fresh deal
                continue
            break
        if not self._pending:
            self._refill()
        for pos, idx in enumerate(self._pending):
            if idx != self._last or len(self._pending) == 1:
                self._pending.pop(pos)
                self._last = idx
                return idx
        idx = self._pending.pop(0)
        self._last = idx
        return idx


# ── Length choice ────────────────────────────────────────────────────────────

def rung_for_energy(style: Style, energy: float, rng: random.Random) -> int:
    """Which LADDER index this bar's loudness calls for.

    Linear between the style's two endpoints, then jittered by at most one rung
    so a long stretch of even energy does not come out perfectly regular. An
    edit that is exactly periodic reads as a slideshow.
    """
    span = style.loud - style.quiet
    index = style.quiet + span * max(0.0, min(1.0, energy))
    rung = int(round(index))
    if style.jitter and rng.random() < style.jitter:
        rung += rng.choice((-1, 1))
    return max(0, min(len(LADDER) - 1, rung))


def _fit_to_bar(beats_wanted: int, offset_in_bar: int, beats_per_bar: int) -> int:
    """Shorten a bar-or-longer cut that would start mid-bar.

    Landing the *next* cut on the downbeat is worth more than this one's full
    length: it re-syncs the grid instead of carrying the offset forward through
    the whole piece.
    """
    if beats_wanted < beats_per_bar or offset_in_bar == 0:
        return beats_wanted
    return beats_per_bar - offset_in_bar


# ── Planning ─────────────────────────────────────────────────────────────────

def plan_cut(
    beatmap,
    sources: list[Source],
    *,
    style: str = DEFAULT_STYLE,
    seed: int | None = None,
    start_bar: int = 0,
    end_bar: int | None = None,
    max_stretch: float = MAX_STRETCH_DEFAULT,
    anticipation: float = 0.0,
) -> EditPlan:
    """Lay `sources` along `beatmap` and return the resulting edit.

    Deterministic in `seed`, so the preview the user approved is the edit that
    gets rendered, and "roll again" is a new seed rather than a new algorithm.
    """
    if not sources:
        raise ValueError("An edit needs at least one source")
    chosen = STYLE_BY_KEY.get(style) or STYLE_BY_KEY[DEFAULT_STYLE]
    # Resolved up front, not left implicit: the plan records the seed it was
    # built with, so the preview the user approved can be rebuilt exactly.
    seed = random.randrange(1 << 30) if seed is None else int(seed)
    rng = random.Random(seed)
    max_stretch = max(1.0, min(float(max_stretch), MAX_STRETCH_CEILING))

    beats = beatmap.beats
    bpb = beatmap.beats_per_bar
    bar_starts = beatmap.bar_starts()
    if len(beats) < 2 or not bar_starts:
        raise ValueError("The beat map has no usable grid")

    start_bar = max(0, min(start_bar, len(bar_starts) - 1))
    end_bar = len(bar_starts) if end_bar is None else max(start_bar + 1, min(end_bar, len(bar_starts)))
    first_beat = bar_starts[start_bar]
    last_beat = bar_starts[end_bar] if end_bar < len(bar_starts) else len(beats) - 1
    if last_beat <= first_beat:
        raise ValueError("The selected span is shorter than one cut")

    song_start = beats[first_beat]
    song_end = beats[last_beat]
    section_beats = {
        bar_starts[bar] for bar in beatmap.sections if bar < len(bar_starts)
    }

    warnings: list[str] = []
    ranks = beatmap.bar_energy
    bag = _Bag(len(sources), rng)
    cursors = [0.0] * len(sources)
    longest = max(range(len(sources)), key=lambda i: sources[i].duration)

    cuts: list[Cut] = []
    beat = first_beat
    while beat < last_beat:
        bar_index = max(0, (beat - beatmap.downbeat_phase) // bpb)
        energy = ranks[bar_index] if bar_index < len(ranks) else 0.5
        wanted = LADDER[rung_for_energy(chosen, energy, rng)]
        wanted = _fit_to_bar(wanted, (beat - beatmap.downbeat_phase) % bpb, bpb)
        wanted = min(wanted, last_beat - beat)

        # The music changing outranks the pace: never hold a shot across it.
        for boundary in sorted(section_beats):
            if beat < boundary < beat + wanted:
                wanted = boundary - beat
                break
        wanted = max(1, wanted)

        span = beats[beat + wanted] - beats[beat]
        if span < MIN_CUT_SECONDS and beat + wanted < last_beat:
            # Too short to read at this tempo — take the next rung up instead.
            while span < MIN_CUT_SECONDS and beat + wanted < last_beat:
                wanted += 1
                span = beats[beat + wanted] - beats[beat]

        index = bag.take(lambda i: sources[i].duration * max_stretch >= span - 1e-6)
        if sources[index].duration * max_stretch < span - 1e-6:
            index = longest
        source = sources[index]

        # What the shot can actually give, and how much it has to be slowed.
        available = min(source.duration, span)
        if cursors[index] + available > source.duration + 1e-6:
            cursors[index] = 0.0            # wrap rather than run off the end
        src_in = cursors[index]
        src_out = min(source.duration, src_in + available)
        read = max(1e-3, src_out - src_in)
        speed = read / span
        if speed < 1.0 / max_stretch:
            # Even at full stretch it cannot fill the slot: shorten the cut and
            # give the beats back to the next one.
            fillable = read * max_stretch
            while wanted > 1 and beats[beat + wanted] - beats[beat] > fillable + 1e-6:
                wanted -= 1
            span = beats[beat + wanted] - beats[beat]
            src_out = min(source.duration, src_in + span)
            read = max(1e-3, src_out - src_in)
            speed = read / span
        cursors[index] = src_out

        is_section = beat in section_beats
        cuts.append(Cut(
            source=index,
            src_in=round(src_in, 4),
            src_out=round(src_out, 4),
            start=round(beats[beat], 4),
            end=round(beats[beat + wanted], 4),
            beats=wanted,
            beat=beat,
            bar=bar_index,
            speed=round(speed, 5),
            downbeat=(beat - beatmap.downbeat_phase) % bpb == 0,
            section=is_section,
            fade_in=SECTION_FADE_SECONDS if is_section and cuts else 0.0,
        ))
        beat += wanted

    if not cuts:
        raise ValueError("The selected span is shorter than one cut")

    opens_at_start = start_bar == 0
    ends_at_finish = end_bar >= len(bar_starts)
    if opens_at_start and song_start > 0.02:
        cuts.insert(0, _pickup(cuts[0], song_start, first_beat, sources, max_stretch))
        song_start = 0.0
    if ends_at_finish and beatmap.duration > song_end + 0.02:
        song_end = round(beatmap.duration, 4)
        cuts[-1].end = song_end
        _refit(cuts[-1], sources, max_stretch)

    _break_repeats(cuts, sources, max_stretch)
    if anticipation > 0 and len(cuts) > 1:
        _apply_anticipation(cuts, anticipation)

    used = {c.source for c in cuts}
    if len(used) < len(sources):
        missing = [sources[i].label or sources[i].key for i in range(len(sources)) if i not in used]
        warnings.append(
            f"{len(missing)} Clip(s) kamen nicht vor — der Song ist zu kurz für die Auswahl: "
            + ", ".join(missing[:4])
        )
    slowest = min((c.speed for c in cuts), default=1.0)
    if slowest < 0.999:
        warnings.append(
            f"Langsamste Einstellung läuft auf {slowest * 100:.0f} % Geschwindigkeit "
            "(Zeitlupe, damit die Einstellung ihren Takt füllt)."
        )

    return EditPlan(
        cuts=cuts,
        style=chosen.key,
        seed=seed,
        bpm=beatmap.bpm,
        beats_per_bar=bpb,
        song_start=round(song_start, 4),
        song_end=round(song_end, 4),
        anticipation=round(anticipation, 4),
        max_stretch=max_stretch,
        warnings=warnings,
    )


def _refit(cut: Cut, sources: list[Source], max_stretch: float) -> None:
    """Re-read a cut whose slot changed length, keeping it as close to real
    speed as the source allows.

    Called only for the two shots the edge rule stretches. A shot that cannot
    fill its slot even at full stretch is handed the longest source instead —
    an opening or closing shot running at a crawl is worse than a different one
    running honestly.
    """
    span = max(1e-3, cut.end - cut.start)
    source = sources[cut.source]
    if source.duration * max_stretch < span:
        longest = max(range(len(sources)), key=lambda i: sources[i].duration)
        cut.source = longest
        cut.src_in = 0.0
        source = sources[longest]
    cut.src_in = min(cut.src_in, max(0.0, source.duration - min(span, source.duration)))
    cut.src_out = round(min(source.duration, cut.src_in + span), 4)
    cut.src_in = round(cut.src_in, 4)
    cut.speed = round(max(1e-3, cut.src_out - cut.src_in) / span, 5)


def _pickup(first: Cut, until: float, beats_before: int,
            sources: list[Source], max_stretch: float) -> Cut:
    """The opening shot, covering the music before the first downbeat.

    Never the shot the first planned cut already holds — the pickup runs
    straight into it, and the same picture on both sides of that boundary is
    not an edit, it is a stumble.
    """
    span = max(1e-3, until)
    pool = [i for i in range(len(sources)) if i != first.source] or [first.source]
    chosen = max(pool, key=lambda i: min(sources[i].duration, span * max_stretch))
    pickup = Cut(
        source=chosen, src_in=0.0, src_out=0.0,
        start=0.0, end=round(until, 4),
        beats=max(1, beats_before), beat=0, bar=0,
        speed=1.0, downbeat=False, section=True, fade_in=0.0,
    )
    _refit(pickup, sources, max_stretch)
    return pickup


def _break_repeats(cuts: list[Cut], sources: list[Source], max_stretch: float) -> None:
    """Last guard against the same shot landing on both sides of a cut.

    The bag prevents this while it is dealing, but `_refit` can reassign an
    edge shot to the longest source and land on its neighbour. One pass at the
    end costs nothing and makes the guarantee unconditional.
    """
    if len(sources) < 2:
        return
    for i in range(1, len(cuts)):
        if cuts[i].source != cuts[i - 1].source:
            continue
        span = max(1e-3, cuts[i].end - cuts[i].start)
        after = cuts[i + 1].source if i + 1 < len(cuts) else -1
        pool = [j for j in range(len(sources))
                if j != cuts[i - 1].source and j != after
                and sources[j].duration * max_stretch >= span]
        if not pool:
            pool = [j for j in range(len(sources)) if j != cuts[i - 1].source]
        if not pool:
            continue
        cuts[i].source = max(pool, key=lambda j: sources[j].duration)
        cuts[i].src_in = 0.0
        _refit(cuts[i], sources, max_stretch)


def _apply_anticipation(cuts: list[Cut], anticipation: float) -> None:
    """Pull every cut point a hair earlier than its beat.

    A hard cut that lands a sixteenth *before* the downbeat reads as more
    deliberate than one exactly on it — the picture arrives, then the music
    confirms it. The first cut keeps its start (there is nothing before it to
    borrow from) and simply gets shorter; the last keeps its end and gets
    longer. Every cut between them moves whole, so only those two change
    duration — and both have their speed refreshed, since the amount of source
    they read did not move with them.
    """
    for i in range(1, len(cuts)):
        shifted = max(cuts[i - 1].start + MIN_CUT_SECONDS, cuts[i].start - anticipation)
        cuts[i - 1].end = round(shifted, 4)
        cuts[i].start = round(shifted, 4)
    for cut in (cuts[0], cuts[-1]):
        span = cut.end - cut.start
        if span > 1e-6:
            cut.speed = round(max(1e-3, cut.src_out - cut.src_in) / span, 5)
