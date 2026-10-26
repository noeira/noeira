"""Find the first op of a MIL program that the ANE refuses.

`ANECCompile() FAILED` names nothing (the system log redacts it), so this
compiles every PREFIX of the function body, each returning the value of its
last fp16 op, and prints the first prefix that fails, i.e. the op that
breaks the compile (or the whole program's result if every prefix passes).

    pixi run -e apple mojo run -I . noeira/experimental/ane/probe_mil.mojo <file.mil>
    ... probe_mil.mojo a.mil b.mil ...   # several: only whether each compiles
"""

from std.sys import argv
from noeira.experimental.ane.runtime import ANE, Program


def compiles(a: ANE, text: String) -> Bool:
    var pool = a.o.pool_push()
    var ok = True
    try:
        var p = Program(a, text)
        p.run(a)
        p.free(a)
    except:
        ok = False
    a.o.pool_pop(pool)
    return ok


def prefix(head: String, body: List[String], end: Int) raises -> String:
    """The program up to body line `end`, returning that line's value."""
    var t = head
    for i in range(end + 1):
        t += body[i] + "\n"
    var ln = String(body[end].strip())
    var rest = String(ln[byte = ln.find("> ") + 2 :])
    t += String("    } -> (", rest[byte = 0 : rest.find(" ")], ");\n}\n")
    return t^


def main() raises:
    var a = ANE()
    var args = argv()
    if len(args) > 2:  # several files: just say which compile
        for i in range(1, len(args)):
            var t: String
            with open(String(args[i]), "r") as f:
                t = f.read()
            print("  ", args[i], ":", "OK" if compiles(a, t) else "FAIL")
        return
    var path = String(args[1])
    var text: String
    with open(path, "r") as f:
        text = f.read()
    var lines = text.split("\n")
    var head = String()
    var body = List[String]()
    var in_body = False
    for ln in lines:
        var s = String(ln)
        if not in_body:
            head += s + "\n"
            if s.find("func main") >= 0:
                in_body = True
        elif s.find("} -> (") >= 0:
            break
        else:
            body.append(s)
    if compiles(a, text):
        print("the whole program compiles")
        return
    # binary search over prefixes that end on an fp16 tensor op
    var ends = List[Int]()
    for i in range(len(body)):
        if body[i].strip().startswith("tensor<fp16"):
            ends.append(i)
    var lo = 0
    var hi = len(ends) - 1  # invariant: prefix ends[hi] (or the program) fails
    var first_fail = -1

    while lo <= hi:
        var mid = (lo + hi) // 2
        if compiles(a, prefix(head, body, ends[mid])):
            lo = mid + 1
        else:
            first_fail = mid
            hi = mid - 1
    if first_fail < 0:
        print("every prefix compiles; the failure is in the outputs or their combination")
        return
    print("first failing op (line", ends[first_fail], "of the body):")
    print(body[ends[first_fail]])
    if first_fail > 0:
        print("last passing op:")
        print(body[ends[first_fail - 1]])
