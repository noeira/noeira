#!/usr/bin/env python3
"""d1-2B's own `parsing.py` (and Python's `re` under it) on a generated corpus,
for the Mojo port's gate (docs/NN_MODEL_PORTS_PLAN.md §12.9, T6;
`tests/llm/test_d1_parsing_reference.mojo`).

    ~/Documents/noeira-parakeet/d1ref-venv/bin/python tools/d1/dump_d1_parsing_reference.py \
        [--model ~/.cache/noeira/local-ai/d1/d1-2B] [--out ~/.cache/noeira/d1/parsing_ref.json]

`parsing.py` is loaded from the checkpoint's folder by path (it imports the
standard library only). Writes:

    patterns   d1's module-level patterns, as text (the port builds the same strings)
    inline     the patterns `parsing.py` writes inside its functions, copied here and checked to
               appear verbatim in its source, with their flags
    probes     every pattern on every probe text: `re.finditer`'s matches (each group's span,
               [-1, -1] when it did not take part) and `re.fullmatch`'s
    numbers    `parse_number(text, context, unit=unit)` and `written_unit(text)` per case
    dates      `parse_date(text, context)` per case
    folds      `_fold` of every text and context, and of every 13th code point of Unicode
"""

import argparse
import importlib.util
import inspect
import itertools
import json
import os
import re

INLINE = {  # name: (pattern as parsing.py writes it, flags)
    "xml": (r"<\?[^>]*\?>", 0),
    "number_chars": (r"\d[\d., '’]*", 0),
    "spacer": (r"[ '’]", 0),
    "spaced": (r"(\d{1,3}(?:[ '’]\d{3})+)([.,]\d+)?", 0),
    "lakh": (r"\d{1,2}(?:,\d{2})+,\d{3}(?:\.\d+)?", 0),
    "rupee": (r"₹|\bINR\b|\bRs\b|lakh|crore", re.I),
    "date_locale": (r'\b(?:locale|language|lang|langue|sprache|idioma|lingua)["\']?\s*[=:]\s*["\']?'
                    r"(en(?:[-_](?:us|gb|au|ca))?|fr(?:[-_]fr)?|de(?:[-_]de)?|es(?:[-_]es)?|it(?:[-_]it)?)\b", 0),
    "timed": (r"(.+?)[T\s]+(\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?)", 0),
    "ordinal": (r"(?<=\d)(?:st|nd|rd|th|er)\b", 0),
    "ordinal2": (r"(?<=\d)\.?(?:o|a|°)\b", 0),
    "articles": (r"\b(?:de|of|the|le|el|il|den)\b", 0),
    "commas": (r"[,.]", 0),
}
# the source spells these through f-strings / concatenation: checked by building them as parsing.py does
BUILT = {
    "group_dot": (r"\d{1,3}(?:" + re.escape(".") + r"\d{3})+" + re.escape(",") + r"\d+", 0),
    "group_comma": (r"\d{1,3}(?:" + re.escape(",") + r"\d{3})+" + re.escape(".") + r"\d+", 0),
}

NUMBER_TEXTS = [
    "1234", "1,234", "1.234", "1 234", "1 234,50", "1.234,50", "1,234.50", "12,5", "12.5", "1,250", "1.250",
    "1'234.50", "1’234", "12,50", "12.50", "-3", "+3", "(3)", "(-3)", "−4,5", "0", "007", "3.0", "1.234.567",
    "1,234,567", "1,234,567.89", "1.234.567,89", "1 234 567", "12 345,6", "1,23,456.78", "₹1,23,456.78",
    "Rs. 1,23,456", "$1,234.56", "$ 12", "12 $", "1 481,40 €", "€12,50", "EUR 1.234,50", "12,50 EUR", "CHF 1'234.50",
    "R$ 1.234,56", "1 234 kr", "12,5 zł", "¥1,234", "1,234円", "₩12,345", "USD 12.5", "12.5 usd", "£3.50",
    "3 million", "1,5 Mio", "2 milliards", "3.2bn", "5k", "10M", "12 thousand", "2.5 billion", "3 Billion", "1 lakh",
    "2 crores", "4 млн", "3万", "1.5 trillion", "12,5 %", "15%", "-2.5%", "12,5 kg", "3.1 m", "3", "abc", "", "3 pommes",
    "1/2", "1.2.3", "12,345", "١٢٣", "١٢٣,٥", "٣", "１２３", "1e5", "  42  ", "4 2", "12,", ",5", "1..2", "$", "(12)",
    "-(12)", "+-3", "12 $ 50", "EURO 12", "12 USDT", "Eur 12", "12 eur", "RSX 5", "12 CHFs",
]
NUMBER_CONTEXTS = [
    "", "Total: 1 234,50 €", "price $1,234.56", "locale: fr", "lang=en", '<?xml version="1.0"?> 1.234,50',
    "12,50 and 3,25", "1.234.567", "price 12,50 and 1,234.56", "the population is large", "money", "language: de-DE",
    "Rs and lakh", "¥ prices", "€ and $",
]
NUMBER_UNITS = [None, "EUR", "€", "kg", "$", ""]

DATE_TEXTS = [
    "20/10/2026", "10/20/2026", "05/06/2026", "5.6.2026", "2026-11-03", "2026/11/03", "19 novembre 2026",
    "November 19, 2026", "19th November 2026", "Tue, 20 Oct 2026", "segunda-feira, 20 de outubro de 2026",
    "20 октября 2026", "2026年10月20日", "2026년 10월 20일", "20/10/26", "05/06/26", "31/02/2026", "29/02/2024",
    "29/02/2026", "2026-11-03T14:30", "2026-11-03 14:30:15+02:00", "2026-11-03 9:05", "2026-11-03T14:30:15.5Z",
    "2026-11-03T14:30:15.1234567", "2026-11-03T25:00", "2026-11-03T14:30-00:00", "3 mars 2026 à 14h", "le 3 mars 2026",
    "1er mars 2026", "3 de março de 2026", "20. Oktober 2026", "Sept 3 2026", "3 sept. 2026", "Mai 5 2026",
    "5 mai 2026", "1º de maio de 2026", "3rd of March 2026", "the 3rd of march 2026", "March 3rd, 2026", "Dec 25 2026",
    "25 Dec 26", "2026", "", "   ", "13/13/2026", "00/10/2026", "1/2/3", "12-25-2026", "25.12.2026", "٢٠/١٠/٢٠٢٦",
    "19 Novembre 2026", "ÉTÉ 2026", "20 październik 2026", "20 października 2026", "lundi 20 octobre 2026",
    "Montag, 20. Oktober 2026", "20 October 0000",
]
DATE_CONTEXTS = [
    "", "date format: dd/mm/yyyy", "format jj/mm/aaaa", "mm/dd/yyyy", "locale: en-US", "langue: fr", "Invoice 25/12/2026",
    "due 12/25/2026", "25/12/2026 and 12/25/2026", "complete 20 octobre 2026 and 05/06/26", "2026-01-01",
    "1999-05-05 and 2026-05-05", "dd/mm/yyyy or mm/dd/yyyy", "lang: it", "19 novembre 2026",
]


def spans(m, n):
    return [list(m.span(g)) if m.group(g) is not None else [-1, -1] for g in range(n + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/.cache/noeira/local-ai/d1/d1-2B"))
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/noeira/d1/parsing_ref.json"))
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("d1_parsing", os.path.join(args.model, "parsing.py"))
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    src = inspect.getsource(P)
    for name, (pat, _) in INLINE.items():
        assert pat in src or pat.replace("\\'", "'") in src or name == "date_locale", f"{name} is not parsing.py's"
    assert "re.escape(group)" in src and "re.escape(point)" in src

    patterns = {
        "CJK_DATE": P.CJK_DATE.pattern, "NUMERIC_DATE": P.NUMERIC_DATE.pattern, "WORD_DATE": P.WORD_DATE.pattern,
        "_CURRENCY": P._CURRENCY, "_SCALE": P._SCALE.pattern, "_SUFFIX": P._SUFFIX.pattern,
        "_COUNTED": P._COUNTED.pattern, "_GROUPED": P._GROUPED.pattern, "_SHORT": P._SHORT.pattern,
        "_REPEATED": P._REPEATED.pattern, "_LOCALE": P._LOCALE.pattern,
        "DATE_FORMAT_DMY": P.DATE_FORMATS[0][0], "DATE_FORMAT_MDY": P.DATE_FORMATS[1][0],
    }
    assert P.DATE_FORMATS[0][1] == "dmy" and P.DATE_FORMATS[1][1] == "mdy"
    flags = {"_SCALE": re.I, "_SUFFIX": re.I, "_COUNTED": re.I}
    compiled = {n: (p, flags.get(n, 0)) for n, p in patterns.items()}
    compiled["cur_either"] = (rf"({P._CURRENCY})\s*(.*)|(.*?)\s*({P._CURRENCY})", 0)
    compiled.update(INLINE)
    compiled.update(BUILT)

    probes = sorted(set(NUMBER_TEXTS + NUMBER_CONTEXTS + DATE_TEXTS + DATE_CONTEXTS
                        + [P._fold(t) for t in DATE_TEXTS + DATE_CONTEXTS]
                        + [t + "\n" + c for t, c in itertools.product(NUMBER_TEXTS[:20], NUMBER_CONTEXTS[:5])]
                        + ["3 million\n", "12k\n", "2 milliards.\n", "5 M\n", "4bn\n\n"]))  # `$` before a final newline
    probe_out = {}
    for name, (pat, fl) in compiled.items():
        r = re.compile(pat, fl)
        rows = []
        for t in probes:
            fm = r.fullmatch(t)
            rows.append({"finditer": [spans(m, r.groups) for m in r.finditer(t)],
                         "fullmatch": spans(fm, r.groups) if fm else None})
        probe_out[name] = {"pattern": pat, "ignorecase": bool(fl & re.I), "groups": r.groups, "rows": rows}

    numbers = []
    for text, context, unit in itertools.product(NUMBER_TEXTS, NUMBER_CONTEXTS, NUMBER_UNITS):
        v = P.parse_number(text, context, unit=unit)
        numbers.append({"text": text, "context": context, "unit": unit,
                        "value": v, "kind": "none" if v is None else type(v).__name__})
    units = [{"text": t, "unit": P.written_unit(t)} for t in NUMBER_TEXTS]
    dates = []
    for text, context in itertools.product(DATE_TEXTS, DATE_CONTEXTS):
        value, ambiguous = P.parse_date(text, context)
        dates.append({"text": text, "context": context, "value": value, "ambiguous": ambiguous})
    folds = [{"text": t, "folded": P._fold(t)} for t in DATE_TEXTS + DATE_CONTEXTS + NUMBER_TEXTS]
    sweep = [chr(cp) for cp in range(1, 0x110000, 13) if not 0xD800 <= cp <= 0xDFFF]  # every 13th code point
    folds += [{"text": "".join(sweep[i:i + 500]), "folded": P._fold("".join(sweep[i:i + 500]))}
              for i in range(0, len(sweep), 500)]

    json.dump({"patterns": patterns, "probes": probe_out, "probe_texts": probes, "numbers": numbers,
               "written_units": units, "dates": dates, "folds": folds},
              open(args.out, "w"), ensure_ascii=False)
    nv = sum(n["kind"] != "none" for n in numbers)
    dv = sum(d["value"] is not None for d in dates)
    print(f"{len(compiled)} patterns x {len(probes)} probes; {len(numbers)} numbers ({nv} with a value), "
          f"{len(units)} units, {len(dates)} dates ({dv} with a value, "
          f"{sum(d['ambiguous'] for d in dates)} ambiguous) -> {args.out}")


if __name__ == "__main__":
    main()
