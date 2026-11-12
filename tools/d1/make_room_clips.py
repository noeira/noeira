#!/usr/bin/env python3
"""The room clips of d1-Omni's step 3 (docs/NN_MODEL_PORTS_PLAN.md §12.6): spoken
requests to the robot, each with the answer d1 should give, for comparing d1-Omni
HEARING a clip with Parakeet -> d1 READING its transcript
(`examples/d1/heard_vs_transcribed.mojo`).

    python3 tools/d1/make_room_clips.py [--out ~/.cache/noeira/d1/room_clips] \
        [--originals ~/.cache/noeira/parakeet/clips]

macOS only: the voices are the system's (`say`), converted to 16 kHz mono 16-bit
WAV by `afconvert`. Writes `<out>/<name>.wav` and `<out>/manifest.tsv`:

    name  lang  voice  text  addressed  destination

`addressed` is yes / no / ? (? = either answer is defensible: "j'ai soif",
"bonjour, comment vas-tu ?"; left out of the addressed score). `destination` is
fridge / door / table / sofa / nowhere: where the robot should walk, with the
robot standing AWAY from every place (the step-3 state). A clip that is not a
request has destination nowhere.

The 18 clips of the Parakeet gate (`fr_01..fr_12`, `en_13..en_18`) are copied
in unchanged, so the first rows are the same audio as steps 1 and 2. The new
clips are mostly French (step 1 found hearing weakest there) and spread over
the system's French voices.

⚠ Synthetic speech: clean, no room, no microphone. A ranking on these clips is
a ranking on TTS; it says nothing about far-field audio from the robot's head.
"""

import argparse
import os
import shutil
import subprocess
import tempfile

ORIGINAL_LABELS = {
    "fr_01.wav": ("yes", "fridge"), "fr_02.wav": ("?", "fridge"), "fr_03.wav": ("yes", "door"),
    "fr_04.wav": ("yes", "table"), "fr_05.wav": ("yes", "sofa"), "fr_06.wav": ("yes", "sofa"),
    "fr_07.wav": ("yes", "nowhere"), "fr_08.wav": ("?", "nowhere"), "fr_09.wav": ("yes", "nowhere"),
    "fr_10.wav": ("yes", "door"), "fr_11.wav": ("yes", "nowhere"), "fr_12.wav": ("no", "nowhere"),
    "en_13.wav": ("yes", "fridge"), "en_14.wav": ("yes", "fridge"), "en_15.wav": ("yes", "door"),
    "en_16.wav": ("yes", "nowhere"), "en_17.wav": ("yes", "nowhere"), "en_18.wav": ("no", "nowhere"),
}

FR = ["Thomas (Français (France))", "Eddy (Français (France))", "Flo (Français (France))", "Jacques",
      "Sandy (Français (France))", "Shelley (Français (France))", "Rocko (Français (France))",
      "Grandma (Français (France))", "Grandpa (Français (France))", "Amélie (Français (Canada))"]
EN = ["Samantha (Anglais (É.-U.))", "Albert", "Kathy", "Ralph", "Reed (Anglais (É.-U.))",
      "Shelley (Anglais (É.-U.))", "Fred", "Sandy (Anglais (É.-U.))", "Grandpa (Anglais (É.-U.))"]

NEW = [
    # French: walk requests, direct and indirect
    ("fr", "va jusqu'au réfrigérateur", "yes", "fridge"),
    ("fr", "apporte-moi quelque chose de frais du frigo", "yes", "fridge"),
    ("fr", "va chercher une bouteille d'eau dans le frigo", "yes", "fridge"),
    ("fr", "va à la porte", "yes", "door"),
    ("fr", "ouvre la porte, s'il te plaît", "yes", "door"),
    ("fr", "on sonne, va ouvrir", "yes", "door"),
    ("fr", "dirige-toi vers la sortie", "yes", "door"),
    ("fr", "marche jusqu'à la porte d'entrée", "yes", "door"),
    ("fr", "va à la table", "yes", "table"),
    ("fr", "approche-toi de la table", "yes", "table"),
    ("fr", "le dîner est servi, va à table", "yes", "table"),
    ("fr", "mets-toi à côté de la table à manger", "yes", "table"),
    ("fr", "va t'asseoir sur le canapé", "yes", "sofa"),
    ("fr", "va vers le sofa", "yes", "sofa"),
    ("fr", "va te reposer sur le canapé", "yes", "sofa"),
    ("fr", "rejoins-moi près du canapé", "yes", "sofa"),
    # French: a negation, alone or with the place to go instead
    ("fr", "ne va pas vers le frigo", "yes", "nowhere"),
    ("fr", "surtout ne t'approche pas de la table", "yes", "nowhere"),
    ("fr", "va au frigo, pas à la porte", "yes", "fridge"),
    ("fr", "ne va pas au canapé, va à la table", "yes", "table"),
    # French: requests that are not a walk
    ("fr", "ne bouge pas", "yes", "nowhere"),
    ("fr", "reste ici", "yes", "nowhere"),
    ("fr", "arrête-toi", "yes", "nowhere"),
    ("fr", "lève le bras droit", "yes", "nowhere"),
    ("fr", "salue-moi de la main", "yes", "nowhere"),
    # French: not a request
    ("fr", "la table est en bois", "no", "nowhere"),
    ("fr", "le canapé est confortable", "no", "nowhere"),
    ("fr", "il fait beau aujourd'hui", "no", "nowhere"),
    ("fr", "la porte est fermée", "no", "nowhere"),
    ("fr", "j'ai rangé le lait dans le frigo", "no", "nowhere"),
    # English
    ("en", "go to the table", "yes", "table"),
    ("en", "please head over to the sofa", "yes", "sofa"),
    ("en", "go sit on the couch", "yes", "sofa"),
    ("en", "open the front door", "yes", "door"),
    ("en", "someone is at the door, go and see", "yes", "door"),
    ("en", "get me a drink from the fridge", "yes", "fridge"),
    ("en", "go where we eat", "yes", "table"),
    ("en", "don't go to the fridge, go to the table", "yes", "table"),
    ("en", "stay where you are", "yes", "nowhere"),
    ("en", "don't move", "yes", "nowhere"),
    ("en", "wave at me", "yes", "nowhere"),
    ("en", "the sofa is blue", "no", "nowhere"),
]


def speak(voice, text, wav):
    with tempfile.TemporaryDirectory() as tmp:
        aiff = os.path.join(tmp, "x.aiff")
        subprocess.run(["say", "-v", voice, "-o", aiff, text], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", aiff, wav], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/noeira/d1/room_clips"))
    ap.add_argument("--originals", default=os.path.expanduser("~/.cache/noeira/parakeet/clips"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    with open(os.path.join(args.originals, "manifest.tsv")) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            shutil.copyfile(os.path.join(args.originals, p[0]), os.path.join(args.out, p[0]))
            addressed, dest = ORIGINAL_LABELS[p[0]]
            rows.append([p[0], p[1], p[2], p[3], addressed, dest])
    n = {"fr": 0, "en": 0}
    for i, (lang, text, addressed, dest) in enumerate(NEW):
        voices = FR if lang == "fr" else EN
        voice = voices[n[lang] % len(voices)]
        n[lang] += 1
        name = f"{lang}_{19 + i:02d}.wav"
        speak(voice, text, os.path.join(args.out, name))
        rows.append([name, lang, voice, text, addressed, dest])
    with open(os.path.join(args.out, "manifest.tsv"), "w") as f:
        for r in rows:
            f.write("\t".join(r) + "\n")
    print(f"{len(rows)} clips in {args.out} ({sum(r[1] == 'fr' for r in rows)} French)")


if __name__ == "__main__":
    main()
