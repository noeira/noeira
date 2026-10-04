# +--------------------------------------------------------------------------+ #
# | API keys — environment first, then `.env`
# +--------------------------------------------------------------------------+ #
"""Where an AI client finds its credential.

    var key = api_key(["ANTHROPIC_API_KEY"])
    var jev = api_key(["JEV_API_KEY", "TYPESAFE_API_KEY"])

Order, first hit wins, per name in the list:

1. the process environment (`export OPENAI_API_KEY=...`);
2. `./.env` — the working directory, which is the repo root for every
   `pixi run mojo run -I .`;
3. `$PIXI_PROJECT_ROOT/.env` — the same file reached from anywhere else.

⚠ AN EMPTY KEY IS REFUSED HERE, not at the server. `Authorization: Bearer `
with nothing after it comes back as a 401 that reads exactly like a wrong
key; naming the variables that were tried is the diagnosis.
"""

from std.os import getenv
from std.pathlib import Path

from noeira.core.dotenv import load_dotenv


def _dotenv_paths() -> List[String]:
    var out = List[String]()
    out.append(String(".env"))
    var root = getenv("PIXI_PROJECT_ROOT")
    if root.byte_length() > 0:
        out.append(root + "/.env")
    return out^


def find_api_key(names: List[String]) raises -> String:
    """The first non-empty value among `names`, or "" when none is set."""
    for i in range(len(names)):
        var v = getenv(names[i])
        if v.byte_length() > 0:
            return v^
    var paths = _dotenv_paths()
    for p in range(len(paths)):
        if not Path(paths[p]).exists():
            continue
        var env = load_dotenv(paths[p])
        for i in range(len(names)):
            if names[i] in env:
                var v = env[names[i]]
                if v.byte_length() > 0:
                    return v^
    return String("")


def api_key(names: List[String]) raises -> String:
    """Like `find_api_key`, but raises with the names tried when none is set."""
    var v = find_api_key(names)
    if v.byte_length() > 0:
        return v^
    var tried = String("")
    for i in range(len(names)):
        if i > 0:
            tried += " / "
        tried += names[i]
    raise Error(
        "no API key: set " + tried + " in the environment or in .env"
        " (the repo root's .env is git-ignored)"
    )
