"""Generate emotion_circumplex.csv mapping 200 emotions to Russell's circumplex model.

Circumplex axes:
  0° = Pleasure, 90° = Arousal, 180° = Displeasure, 270° = Sleepiness
  valence = intensity * cos(angle)
  arousal = intensity * sin(angle)
"""

import csv
import math
from pathlib import Path

# (emotion, angle_degrees, intensity)
# Ordered to match data/texts/emotions200.txt
EMOTIONS = [
    # --- Group 1: Pleasure → Excitement (entries 1–25) ---
    ("happy", 2, 0.85),
    ("pleased", 355, 0.65),
    ("glad", 0, 0.70),
    ("delighted", 15, 0.85),
    ("joyful", 8, 0.90),
    ("elated", 25, 0.90),
    ("cheerful", 5, 0.70),
    ("thrilled", 30, 0.90),
    ("overjoyed", 15, 0.95),
    ("ecstatic", 42, 1.00),
    ("blissful", 355, 0.95),
    ("jubilant", 20, 0.90),
    ("euphoric", 35, 1.00),
    ("radiant", 10, 0.80),
    ("gleeful", 18, 0.80),
    ("merry", 8, 0.70),
    ("jovial", 5, 0.70),
    ("exuberant", 30, 0.90),
    ("rapturous", 38, 0.95),
    ("enchanted", 20, 0.75),
    ("gratified", 350, 0.65),
    ("fulfilled", 348, 0.70),
    ("blessed", 345, 0.70),
    ("lighthearted", 12, 0.60),
    ("upbeat", 25, 0.65),
    # --- Group 2: Excitement → Arousal (entries 26–50) ---
    ("excited", 48, 0.85),
    ("astonished", 65, 0.85),
    ("aroused", 80, 0.80),
    ("stimulated", 72, 0.70),
    ("energized", 55, 0.75),
    ("exhilarated", 50, 0.90),
    ("animated", 52, 0.70),
    ("invigorated", 55, 0.75),
    ("electrified", 60, 0.90),
    ("fired-up", 65, 0.85),
    ("hyperactive", 80, 0.85),
    ("wired", 82, 0.80),
    ("amped", 75, 0.85),
    ("keyed-up", 78, 0.75),
    ("activated", 70, 0.65),
    ("alert", 85, 0.60),
    ("awake", 88, 0.50),
    ("attentive", 82, 0.55),
    ("vigilant", 85, 0.65),
    ("eager", 48, 0.75),
    ("enthusiastic", 50, 0.80),
    ("passionate", 55, 0.90),
    ("intense", 75, 0.85),
    ("vivacious", 45, 0.80),
    ("spirited", 48, 0.75),
    # --- Group 3: Arousal → Distress (entries 51–75) ---
    ("tense", 100, 0.80),
    ("alarmed", 95, 0.85),
    ("angry", 120, 0.90),
    ("afraid", 108, 0.85),
    ("nervous", 98, 0.70),
    ("anxious", 102, 0.75),
    ("stressed", 105, 0.80),
    ("frightened", 108, 0.85),
    ("panicked", 95, 0.95),
    ("terrified", 112, 0.95),
    ("agitated", 98, 0.80),
    ("furious", 125, 0.95),
    ("enraged", 128, 1.00),
    ("irate", 125, 0.85),
    ("livid", 130, 0.90),
    ("incensed", 128, 0.85),
    ("outraged", 130, 0.90),
    ("hostile", 132, 0.85),
    ("aggressive", 125, 0.85),
    ("rattled", 100, 0.70),
    ("frantic", 95, 0.90),
    ("hysterical", 92, 0.95),
    ("overwrought", 105, 0.80),
    ("apprehensive", 110, 0.60),
    ("uneasy", 108, 0.55),
    # --- Group 4: Distress → Displeasure (entries 76–100) ---
    ("annoyed", 138, 0.60),
    ("distressed", 140, 0.80),
    ("frustrated", 142, 0.75),
    ("resentful", 148, 0.70),
    ("bitter", 150, 0.75),
    ("irritated", 138, 0.65),
    ("aggravated", 140, 0.70),
    ("exasperated", 145, 0.75),
    ("displeased", 160, 0.65),
    ("dissatisfied", 162, 0.60),
    ("discontent", 165, 0.60),
    ("troubled", 148, 0.70),
    ("upset", 145, 0.75),
    ("bothered", 140, 0.55),
    ("perturbed", 142, 0.60),
    ("vexed", 145, 0.65),
    ("irked", 140, 0.55),
    ("peeved", 142, 0.55),
    ("miffed", 140, 0.50),
    ("offended", 155, 0.65),
    ("hurt", 158, 0.75),
    ("wounded", 160, 0.80),
    ("anguished", 155, 0.90),
    ("tormented", 152, 0.95),
    ("afflicted", 168, 0.80),
    # --- Group 5: Displeasure → Depression (entries 101–125) ---
    ("miserable", 185, 0.90),
    ("sad", 195, 0.80),
    ("gloomy", 200, 0.75),
    ("depressed", 218, 0.85),
    ("unhappy", 188, 0.70),
    ("sorrowful", 195, 0.85),
    ("mournful", 200, 0.85),
    ("grief-stricken", 198, 0.95),
    ("heartbroken", 192, 0.95),
    ("despondent", 215, 0.85),
    ("dejected", 210, 0.75),
    ("melancholy", 205, 0.70),
    ("forlorn", 208, 0.80),
    ("woeful", 198, 0.80),
    ("dismal", 210, 0.75),
    ("bleak", 215, 0.70),
    ("hopeless", 220, 0.90),
    ("despairing", 218, 0.95),
    ("wretched", 190, 0.90),
    ("desolate", 215, 0.90),
    ("crestfallen", 205, 0.65),
    ("downcast", 208, 0.65),
    ("glum", 210, 0.60),
    ("morose", 218, 0.70),
    ("sullen", 222, 0.65),
    # --- Group 6: Depression → Sleepiness (entries 126–150) ---
    ("bored", 230, 0.55),
    ("droopy", 245, 0.55),
    ("tired", 252, 0.65),
    ("sleepy", 262, 0.65),
    ("fatigued", 250, 0.70),
    ("exhausted", 248, 0.85),
    ("weary", 245, 0.70),
    ("drained", 240, 0.80),
    ("lethargic", 255, 0.70),
    ("sluggish", 252, 0.60),
    ("listless", 248, 0.60),
    ("apathetic", 235, 0.65),
    ("indifferent", 232, 0.50),
    ("numb", 238, 0.70),
    ("empty", 235, 0.75),
    ("hollow", 235, 0.70),
    ("vacant", 240, 0.55),
    ("disengaged", 235, 0.50),
    ("uninspired", 232, 0.50),
    ("unmotivated", 238, 0.55),
    ("passive", 255, 0.45),
    ("inactive", 258, 0.45),
    ("idle", 255, 0.40),
    ("drowsy", 265, 0.65),
    ("somnolent", 270, 0.65),
    # --- Group 7: Sleepiness → Relaxation (entries 151–175) ---
    ("calm", 295, 0.70),
    ("relaxed", 300, 0.70),
    ("serene", 300, 0.80),
    ("tranquil", 298, 0.75),
    ("peaceful", 295, 0.80),
    ("placid", 292, 0.65),
    ("still", 280, 0.55),
    ("quiet", 282, 0.50),
    ("restful", 288, 0.65),
    ("soothed", 298, 0.70),
    ("mellow", 305, 0.60),
    ("laid-back", 308, 0.55),
    ("easygoing", 312, 0.55),
    ("untroubled", 310, 0.55),
    ("composed", 295, 0.55),
    ("collected", 292, 0.50),
    ("centered", 290, 0.60),
    ("balanced", 290, 0.55),
    ("grounded", 288, 0.60),
    ("settled", 285, 0.55),
    ("hushed", 278, 0.45),
    ("subdued", 275, 0.50),
    ("gentle", 305, 0.50),
    ("soft", 302, 0.45),
    ("leisurely", 310, 0.50),
    # --- Group 8: Relaxation → Pleasure (entries 176–200) ---
    ("content", 340, 0.75),
    ("satisfied", 342, 0.70),
    ("at ease", 325, 0.65),
    ("comfortable", 330, 0.65),
    ("cozy", 328, 0.60),
    ("snug", 325, 0.55),
    ("secure", 332, 0.65),
    ("safe", 330, 0.60),
    ("relieved", 335, 0.70),
    ("thankful", 345, 0.70),
    ("grateful", 348, 0.75),
    ("appreciative", 348, 0.65),
    ("warm", 340, 0.65),
    ("tender", 338, 0.60),
    ("affectionate", 342, 0.70),
    ("loving", 345, 0.80),
    ("fond", 342, 0.60),
    ("caring", 340, 0.65),
    ("compassionate", 345, 0.70),
    ("sympathetic", 348, 0.60),
    ("kind", 345, 0.60),
    ("gentle", 338, 0.55),
    ("sentimental", 335, 0.60),
    ("nostalgic", 330, 0.60),
    ("hopeful", 355, 0.75),
]

assert len(EMOTIONS) == 200, f"Expected 200 emotions, got {len(EMOTIONS)}"

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "constants" / "emotion_circumplex.csv"


def main():
    rows = []
    for emotion, angle_deg, intensity in EMOTIONS:
        angle_rad = math.radians(angle_deg)
        valence = intensity * math.cos(angle_rad)
        arousal = intensity * math.sin(angle_rad)
        rows.append({
            "emotion": emotion,
            "angle": angle_deg,
            "intensity": round(intensity, 2),
            "valence": round(valence, 4),
            "arousal": round(arousal, 4),
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["emotion", "angle", "intensity", "valence", "arousal"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} emotions to {OUT_PATH}")

    # Spot checks
    by_name = {r["emotion"]: r for r in rows}
    checks = [
        ("happy",     lambda r: r["valence"] > 0.8 and abs(r["arousal"]) < 0.1),
        ("terrified", lambda r: r["valence"] < 0 and r["arousal"] > 0.8),
        ("depressed", lambda r: r["valence"] < 0 and r["arousal"] < 0),
        ("calm",      lambda r: r["valence"] > 0 and r["arousal"] < 0),
    ]
    for name, check in checks:
        r = by_name[name]
        status = "PASS" if check(r) else "FAIL"
        print(f"  {status}: {name} -> valence={r['valence']:.3f}, arousal={r['arousal']:.3f}")


if __name__ == "__main__":
    main()
