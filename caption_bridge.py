"""
caption_bridge.py — LoRa-Daddy Toolkit
========================================
Translates user instruction slang into the exact vocabulary your LoRA
was trained on, injected as a CAPTION VOCABULARY block in _build_message().

Your dataset caption formula (reverse-engineered from 26,926 captions):

  ACT TAG, [Shot type] of [woman desc]. [Body desc]. [Position mechanics].
  [Penetration/act description using: penetrating her pussy/anus, shaft
  partially visible, wet pink inner lips / tight pink opening].

This module detects scene type from raw user instruction and injects
the matching vocabulary guide so Gemma writes in caption dialect.

Scene buckets:
  - penetration_vaginal
  - penetration_anal
  - blowjob
  - pussy_licking
  - reveal_undress
  - handjob
  - fingering
  - solo_spread          (woman displaying herself, no act)
  - multiple_buckets     (e.g. blowjob + anal simultaneously)
"""

import re

# ══════════════════════════════════════════════════════════════════════════
#  SLANG → CANONICAL MAPS
# ══════════════════════════════════════════════════════════════════════════

# Hole mapping — what the user means vs what the caption says
HOLE_SLANG = {
    # pussy / vaginal
    "pussy":        "pussy",
    "cunt":         "pussy",
    "hole":         "pussy",
    "vag":          "pussy",
    "vagina":       "pussy",
    "her hole":     "pussy",
    "her cunt":     "pussy",
    "her pussy":    "pussy",
    "wet pussy":    "pussy",
    # anal
    "ass":          "anus",
    "butt":         "anus",
    "asshole":      "anus",
    "butthole":     "anus",
    "back door":    "anus",
    "backdoor":     "anus",
    "her ass":      "anus",
    "her butt":     "anus",
    "her asshole":  "anus",
    "her anus":     "anus",
    "anal":         "anus",
    "in the ass":   "anus",
    "up the ass":   "anus",
    "in her ass":   "anus",
}

# Act slang → canonical act
ACT_SLANG = {
    # penetration
    "fucking":          "penetration",
    "fucked":           "penetration",
    "getting fucked":   "penetration",
    "getting railed":   "penetration",
    "railing":          "penetration",
    "railed":           "penetration",
    "pounding":         "penetration",
    "getting pounded":  "penetration",
    "banging":          "penetration",
    "getting banged":   "penetration",
    "dicking":          "penetration",
    "getting dicked":   "penetration",
    "riding":           "penetration",
    "getting bred":     "penetration",
    "getting stuffed":  "penetration",
    "sex":              "penetration",
    "having sex":       "penetration",
    "intercourse":      "penetration",
    "penetrating":      "penetration",
    "penetrated":       "penetration",
    # blowjob
    "blowjob":          "blowjob",
    "bj":               "blowjob",
    "sucking cock":     "blowjob",
    "sucking dick":     "blowjob",
    "sucking him":      "blowjob",
    "giving head":      "blowjob",
    "getting head":     "blowjob",
    "head":             "blowjob",
    "going down on him":"blowjob",
    "deepthroat":       "blowjob",
    "deepthroating":    "blowjob",
    "throat":           "blowjob",
    "facefuck":         "blowjob",
    "face fuck":        "blowjob",
    "mouth":            "blowjob",
    # pussy licking
    "eating out":       "pussy_licking",
    "eating her out":   "pussy_licking",
    "eats her out":     "pussy_licking",
    "eats out":         "pussy_licking",
    "eating her pussy": "pussy_licking",
    "eats her pussy":   "pussy_licking",
    "cunnilingus":      "pussy_licking",
    "going down on her":"pussy_licking",
    "goes down on her": "pussy_licking",
    "licking her":      "pussy_licking",
    "licks her":        "pussy_licking",
    "licking her pussy":"pussy_licking",
    "licks her pussy":  "pussy_licking",
    "tongue":           "pussy_licking",
    "pussy licking":    "pussy_licking",
    # handjob
    "handjob":          "handjob",
    "hand job":         "handjob",
    "jerking":          "handjob",
    "jerking him":      "handjob",
    "stroking":         "handjob",
    "stroking him":     "handjob",
    "stroking his cock":"handjob",
    "wanking":          "handjob",
    # fingering
    "fingering":        "fingering",
    "fingered":         "fingering",
    "fingering her":    "fingering",
    "finger":           "fingering",
    "fingers her":      "fingering",
    "fingerfuck":       "fingering",
}

# Position slang → caption mechanics language
POSITION_SLANG = {
    "doggy":            "doggy",
    "doggy style":      "doggy",
    "doggystyle":       "doggy",
    "from behind":      "doggy",
    "hitting it from behind": "doggy",
    "bent over":        "doggy",
    "missionary":       "missionary",
    "on her back":      "missionary",
    "lying on her back":"missionary",
    "cowgirl":          "cowgirl",
    "on top":           "cowgirl",
    "riding him":       "cowgirl",
    "sitting on him":   "cowgirl",
    "reverse cowgirl":  "reverse_cowgirl",
    "facing away":      "reverse_cowgirl",
    "reverse":          "reverse_cowgirl",
    "standing":         "standing",
    "against the wall": "standing",
    "wall":             "standing",
    "spooning":         "spooning",
    "on her side":      "spooning",
    "side":             "spooning",
    "on all fours":     "doggy",
    "all fours":        "doggy",
}

# Garment slang → caption mechanics language
GARMENT_SLANG = {
    "panties":      "panties",
    "underwear":    "panties",
    "knickers":     "panties",
    "thong":        "thong",
    "g-string":     "thong",
    "g string":     "thong",
    "shorts":       "shorts",
    "leggings":     "leggings",
    "pants":        "pants",
    "jeans":        "jeans",
    "skirt":        "skirt",
    "dress":        "dress",
    "bra":          "bra",
    "top":          "top",
    "shirt":        "shirt",
    "blouse":       "top",
}

# Cock slang → caption term
COCK_SLANG = {
    "dick":     "cock",
    "penis":    "cock",
    "cock":     "cock",
    "shaft":    "cock",
    "rod":      "cock",
    "member":   "cock",
    "dong":     "cock",
    "schlong":  "cock",
    "package":  "cock",
}


# ══════════════════════════════════════════════════════════════════════════
#  POSITION MECHANICS — exact caption language per position
# ══════════════════════════════════════════════════════════════════════════

POSITION_MECHANICS = {
    "doggy": (
        "positioned on all fours, face down with ass raised toward the camera, "
        "weight resting on her elbows or hands flat on the surface. "
        "The penetrating partner kneels behind her, hands gripping her hips firmly."
    ),
    "missionary": (
        "lying on her back with her legs spread wide, hips tilted slightly upward. "
        "The penetrating partner is positioned above her, weight supported on arms or knees."
    ),
    "cowgirl": (
        "seated upright on top of her partner, facing him, knees spread wide and "
        "thighs straddling his hips. Her hips move rhythmically as she controls the motion. "
        "The partner lies on his back beneath her, hands resting on her hips or thighs."
    ),
    "reverse_cowgirl": (
        "seated on top of her partner in reverse cowgirl, facing away from him, "
        "her back arched and torso upright. The camera angle is from behind, "
        "focusing on her ass and the point of penetration."
    ),
    "standing": (
        "standing, leaning forward with hands braced against a wall or surface, "
        "body angled forward with her ass pushed back toward the partner who stands behind her."
    ),
    "spooning": (
        "lying on her side, body curved into her partner who lies behind her, "
        "her upper leg raised slightly to allow penetration from behind. "
        "Both bodies face the same direction."
    ),
}

# ══════════════════════════════════════════════════════════════════════════
#  PENETRATION ANATOMY DETAIL — exact caption formula per hole
# ══════════════════════════════════════════════════════════════════════════

PENETRATION_DETAIL = {
    "pussy": (
        "A [thick/large], [skin-tone] cock is actively penetrating her pussy, "
        "with the shaft partially visible as it enters the wet, pink inner lips. "
        "Her outer lips are slightly parted from the friction of penetration."
    ),
    "anus": (
        "A [thick/large], [skin-tone] cock is actively penetrating her anus, "
        "with the shaft partially visible as it enters the tight, pink opening. "
        "The skin around her anus is slightly stretched from the penetration."
    ),
}

# ══════════════════════════════════════════════════════════════════════════
#  ACT DETAIL — blowjob/oral/fingering formula
# ══════════════════════════════════════════════════════════════════════════

ACT_DETAIL = {
    "blowjob": (
        "She is performing a standard blowjob. Her lips are parted around "
        "the [tip/mid-shaft] of the cock. Her [hand/hands] grip the mid-shaft. "
        "Visible [saliva/cum] coats the shaft and her lips. "
        "The receiver's [skin tone] skin is visible [above/behind] her."
    ),
    "pussy_licking": (
        "The performer's face is pressed against the receiver's groin, "
        "mouth making direct contact with her wet, pink inner lips and clit. "
        "The performer's tongue is actively licking the wet, pink inner lips. "
        "The receiver lies on her back with her legs spread wide, "
        "thighs framing the performer's head."
    ),
    "handjob": (
        "Her hand grips the mid-shaft of the cock firmly, "
        "moving in a steady rhythm. The cock is [skin-tone] and [thick/large], "
        "with visible veins along the shaft. Her [manicured/bare] fingers "
        "wrap around the shaft, thumb on top."
    ),
    "fingering": (
        "Her [one/two] finger[s] are inserted into her pussy, "
        "with the knuckles partially visible at the wet, pink inner lips. "
        "Her other hand [rests on her thigh / pulls her outer lips apart]. "
        "Visible wetness coats her fingers and inner lips."
    ),
}

# ══════════════════════════════════════════════════════════════════════════
#  REVEAL/UNDRESS MECHANICS — garment-specific caption formula
# ══════════════════════════════════════════════════════════════════════════

REVEAL_MECHANICS = {
    "panties": (
        "She hooks both thumbs into the waistband of her [colour/fabric] panties, "
        "fingers curling over the elastic at each hip. She draws the fabric slowly "
        "downward over her hips and thighs, the material bunching as it descends. "
        "As the panties clear her hips, her pussy comes into view — "
        "[pubic hair description], puffy outer lips, small tucked inner lips "
        "that appear slightly wet. The panties are pulled down to [mid-thigh / ankles / "
        "bunched at her knees]."
    ),
    "thong": (
        "She hooks one thumb into the side strap of her thong, "
        "pulling the thin fabric away from her skin. She slides the thong "
        "down over her hips, the narrow strip of fabric peeling away from "
        "her skin as it descends. Her pussy is revealed as the fabric clears — "
        "thin outer lips, small inner lips, and a visible clit."
    ),
    "shorts": (
        "She grips the waistband of her [fabric/colour] shorts with both hands, "
        "thumbs pressing into the material at her hips. She pushes them down "
        "over her hips and thighs, the fabric sliding over her skin. "
        "As the shorts descend, her bare skin is revealed — "
        "her large, round ass comes into view, followed by her pussy visible "
        "between her thighs as she bends slightly forward."
    ),
    "leggings": (
        "She grips the waistband of her [fabric/colour] leggings, "
        "rolling the material down from her hips. The tight fabric peels "
        "away from her skin as she pushes it down over her thighs, "
        "revealing her bare skin. Her round ass and pussy become visible "
        "as the leggings descend to her mid-thigh."
    ),
    "skirt": (
        "She grips the hem of her [fabric/colour] skirt with both hands "
        "and lifts it upward, the material rising over her thighs and hips. "
        "As the skirt lifts clear, her bare skin is revealed — "
        "[panties if present, or directly:] her pussy visible between her thighs, "
        "outer lips prominent, inner lips slightly parted."
    ),
    "bra": (
        "She reaches behind her back, fingers finding the clasp of her bra. "
        "The clasp releases and the band slackens. She draws the straps "
        "forward off her shoulders one by one, the cups falling away "
        "from her chest. Her [size]-[shape] tits come into full view, "
        "[nipple colour] nipples and [areola size] areolas visible."
    ),
    "top": (
        "She grips the hem of her [fabric/colour] top with both hands "
        "and pulls it upward, the material rising over her stomach and chest. "
        "She lifts it over her head and off, her [bra-covered / bare] chest "
        "coming into view. Her [size]-[shape] tits [sit upright / hang naturally] "
        "on her chest, [nipple colour] nipples [covered by bra / exposed]."
    ),
    "dress": (
        "She reaches behind her and draws the zipper of her [fabric/colour] dress "
        "downward, the fabric parting along her back. She lets the dress "
        "fall from her shoulders, sliding down over her body and pooling "
        "at her feet. Her bare skin is revealed from shoulders to hips — "
        "[bra and panties if present, or fully bare]."
    ),
}

# ══════════════════════════════════════════════════════════════════════════
#  ANATOMY REVEAL DETAIL — what appears when anatomy is first exposed
# ══════════════════════════════════════════════════════════════════════════

ANATOMY_REVEAL_DETAIL = {
    "pussy": (
        "pussy visible: [pubic hair — trimmed landing strip / full bush / bare / light dusting], "
        "puffy outer lips, small tucked inner lips that appear slightly wet, "
        "small pink clit visible."
    ),
    "ass": (
        "large, round [skin tone] ass visible, smooth skin texture across the buttocks, "
        "anus visible as a small, [pink / pinkish-brown] puckered opening "
        "at the centre of the spread cheeks."
    ),
    "tits": (
        "[size]-[shape] tits [sit upright / hang naturally / hang heavily forward], "
        "[nipple colour] nipples [erect / soft], [areola size] [colour] areolas."
    ),
}


# ══════════════════════════════════════════════════════════════════════════
#  DETECTION ENGINE
# ══════════════════════════════════════════════════════════════════════════

def _normalise(text: str) -> str:
    return text.lower().strip()


def detect_acts(instruction: str) -> list:
    """Return list of detected canonical acts from instruction."""
    instr = _normalise(instruction)
    found = set()

    # Check multi-word slang first (longest match wins)
    for slang in sorted(ACT_SLANG.keys(), key=len, reverse=True):
        if slang in instr:
            found.add(ACT_SLANG[slang])

    return list(found)


def detect_holes(instruction: str) -> list:
    """Return list of detected holes (pussy / anus) from instruction."""
    instr = _normalise(instruction)
    found = set()

    for slang in sorted(HOLE_SLANG.keys(), key=len, reverse=True):
        if slang in instr:
            found.add(HOLE_SLANG[slang])

    # Fallback: if penetration detected but no explicit hole, default to pussy
    if not found and any(a in detect_acts(instruction) for a in ["penetration"]):
        found.add("pussy")

    return list(found)


def detect_position(instruction: str) -> str | None:
    """Return canonical position key or None."""
    instr = _normalise(instruction)

    for slang in sorted(POSITION_SLANG.keys(), key=len, reverse=True):
        if slang in instr:
            return POSITION_SLANG[slang]

    return None


def detect_garments(instruction: str) -> list:
    """Return list of garments mentioned in instruction."""
    instr = _normalise(instruction)
    found = []

    for slang in sorted(GARMENT_SLANG.keys(), key=len, reverse=True):
        if slang in instr and GARMENT_SLANG[slang] not in found:
            found.append(GARMENT_SLANG[slang])

    return found


def detect_reveal(instruction: str) -> bool:
    """True if instruction describes an undress/reveal action."""
    instr = _normalise(instruction)
    reveal_triggers = [
        "takes off", "takes her", "pulls down", "pulls off", "pulls up",
        "slides off", "slides down", "slips off", "removes", "undresses",
        "strips", "unzips", "unbuttons", "unhooks", "lifts her", "lifts up",
        "shows her", "reveals", "exposes", "pulls her", "takes down",
        "peels off", "shrugs off", "drops her", "lets fall",
    ]
    return any(t in instr for t in reveal_triggers)


def detect_cock_slang(instruction: str) -> str | None:
    """Return 'cock' if any cock synonym detected, else None."""
    instr = _normalise(instruction)
    for slang in COCK_SLANG:
        if slang in instr:
            return "cock"
    return None


# ══════════════════════════════════════════════════════════════════════════
#  INJECTION BUILDER — main export
# ══════════════════════════════════════════════════════════════════════════

def build_caption_bridge_injection(instruction: str) -> str:
    """
    Analyse instruction and return a CAPTION VOCABULARY block to inject
    into _build_message() before the SCENE TO WRITE A PROMPT FOR line.

    Returns empty string if no scene-specific translation needed.
    """
    if not instruction or not instruction.strip():
        return ""

    acts      = detect_acts(instruction)
    holes     = detect_holes(instruction)
    position  = detect_position(instruction)
    garments  = detect_garments(instruction)
    is_reveal = detect_reveal(instruction)
    has_cock  = detect_cock_slang(instruction)

    # Nothing actionable detected
    if not acts and not is_reveal and not garments and not holes:
        return ""

    parts = ["CAPTION VOCABULARY — USE THESE EXACT TERMS AND FORMULAS:\n"]

    # ── Cock terminology — only for acts that involve a cock ─────────────
    cock_acts = {"penetration", "blowjob", "handjob"}
    if has_cock or any(a in cock_acts for a in acts):
        parts.append(
            "COCK TERM: Always write 'cock' — never 'dick', 'penis', 'member', 'rod', or 'shaft' alone. "
            "Describe as: 'a [thick/large], [skin-tone] cock'.\n"
        )

    # ── Act translation ───────────────────────────────────────────────────
    if "penetration" in acts:
        hole_list = holes if holes else ["pussy"]
        for hole in hole_list:
            parts.append(f"ACT — PENETRATION INTO {hole.upper()}:")
            parts.append(f"  Caption verb: 'actively penetrating her {hole}'")
            parts.append(f"  NOT: 'fucking', 'banging', 'railing', 'pounding'")
            parts.append(f"  Formula: {PENETRATION_DETAIL[hole]}")
            parts.append("")

    if "blowjob" in acts:
        parts.append("ACT — BLOWJOB:")
        parts.append("  Lead tag: start the caption with 'blowjob,'")
        parts.append(f"  Formula: {ACT_DETAIL['blowjob']}")
        parts.append("  Wetness term: 'visible saliva coats the shaft' OR 'visible cum coats the shaft'")
        parts.append("  NOT: 'giving oral', 'performing oral sex', 'pleasuring him'")
        parts.append("")

    if "pussy_licking" in acts:
        parts.append("ACT — PUSSY LICKING:")
        parts.append("  Lead tag: start the caption with 'pussy licking,'")
        parts.append(f"  Formula: {ACT_DETAIL['pussy_licking']}")
        parts.append("  Anatomy terms: 'wet, pink inner lips', 'clit', 'outer lips'")
        parts.append("  NOT: 'going down on her', 'eating her out', 'oral on her'")
        parts.append("")

    if "handjob" in acts:
        parts.append("ACT — HANDJOB:")
        parts.append(f"  Formula: {ACT_DETAIL['handjob']}")
        parts.append("")

    if "fingering" in acts:
        parts.append("ACT — FINGERING:")
        parts.append(f"  Formula: {ACT_DETAIL['fingering']}")
        parts.append("")

    # ── Position mechanics ────────────────────────────────────────────────
    if position:
        parts.append(f"POSITION — {position.upper().replace('_', ' ')}:")
        parts.append(f"  Lead tag: start the caption with '{position.replace('_', ' ')},'")
        parts.append(f"  Body mechanics: {POSITION_MECHANICS[position]}")
        parts.append("")

    # ── Anatomy terminology ───────────────────────────────────────────────
    if holes or "penetration" in acts or is_reveal:
        parts.append("ANATOMY TERMINOLOGY (always use these exact terms):")
        if "pussy" in holes or (not holes and "penetration" in acts) or is_reveal:
            parts.append("  Pussy anatomy: 'outer lips', 'inner lips', 'clit', 'wet pink inner lips'")
            parts.append("  Wetness: 'appears slightly wet', 'visible wetness', 'wet, pink inner lips'")
            parts.append("  NOT: 'inner folds', 'soft folds', 'womanhood', 'center', 'entrance', 'moist interior'")
        if "anus" in holes:
            parts.append("  Anal anatomy: 'anus', 'tight pink opening', 'puckered opening'")
            parts.append("  NOT: 'back door', 'rear', 'backdoor', 'anal opening'")
        parts.append("")

    # ── Reveal / undress sequence ─────────────────────────────────────────
    if is_reveal and garments:
        parts.append("UNDRESS / REVEAL SEQUENCE — write the mechanics of removal, not just the result:")
        for garment in garments:
            if garment in REVEAL_MECHANICS:
                parts.append(f"\n  {garment.upper()} REMOVAL FORMULA:")
                # Indent each line of the formula
                for line in REVEAL_MECHANICS[garment].strip().split("\n"):
                    parts.append(f"    {line.strip()}")

        parts.append("\n  ANATOMY ON REVEAL — when anatomy first appears, describe it fully:")
        if "pussy" in holes or any(g in ["panties", "thong", "shorts", "leggings", "skirt", "dress"] for g in garments):
            parts.append(f"    {ANATOMY_REVEAL_DETAIL['pussy']}")
        if any(g in ["bra", "top", "dress"] for g in garments):
            parts.append(f"    {ANATOMY_REVEAL_DETAIL['tits']}")
        parts.append(
            "\n  KEY RULE: describe the MOTION (grip, pull, fabric movement, material response) "
            "BEFORE describing what is revealed. The reveal is earned by the mechanics."
        )
        parts.append("")

    elif is_reveal and not garments:
        # Reveal detected but no specific garment — give general reveal guidance
        parts.append(
            "UNDRESS / REVEAL SEQUENCE DETECTED — no specific garment identified.\n"
            "General rule: describe the grip and motion of removal first, then the anatomy revealed.\n"
            "Anatomy on reveal: outer lips, inner lips, clit, anus — named explicitly on first appearance.\n"
        )

    # ── Shot type reminder ────────────────────────────────────────────────
    parts.append(
        "SHOT TYPE: Begin the caption with a shot type — "
        "'Medium shot', 'Close-up', or 'Close-up shot'. "
        "Medium shot is most common in your training data. "
        "If a position tag is present, it goes BEFORE the shot type: "
        "'doggy, Medium shot of...' not 'Medium shot, doggy...'.\n"
    )

    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════
#  DEBUG / TEST
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_cases = [
        "a woman gets fucked doggy style in her pussy",
        "she's getting railed in the ass missionary",
        "she pulls down her panties and shows her pussy",
        "she takes off her bra and top while he watches",
        "she's riding him cowgirl, bouncing on his cock",
        "she gives him a blowjob on her knees",
        "he eats her out while she lies on the bed",
        "she's getting fingered from behind",
        "she strips off her leggings and bends over",
        "a woman in lingerie shows off her ass",
        "she pulls her skirt up and slides her panties down revealing her pussy",
        "reverse cowgirl anal",
    ]

    for tc in test_cases:
        print(f"\n{'='*60}")
        print(f"INPUT: {tc}")
        print(f"{'='*60}")
        acts     = detect_acts(tc)
        holes    = detect_holes(tc)
        position = detect_position(tc)
        garments = detect_garments(tc)
        reveal   = detect_reveal(tc)
        print(f"acts={acts} holes={holes} position={position} garments={garments} reveal={reveal}")
        print("--- INJECTION ---")
        out = build_caption_bridge_injection(tc)
        print(out if out else "[no injection]")
