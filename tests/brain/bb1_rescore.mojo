"""BB-1 scored again from a file of replies, without the model.

    pixi run mojo run -I . tests/brain/bb1_rescore.mojo runs/brain/bb1_replies.tsv [tests/brain]

`benchmarks/bench_brain_grounded_gpu.mojo` writes every reply to a TSV. When a
scoring rule changes (`tools/brain/make_bb1.py`, then regenerate), this prints
the bench's table for the same replies under the rules as they are now: the
model is not asked again, so only the rules moved. Items are matched by id;
the file's own pass columns are ignored.
"""

from std.sys import argv

from noeira.brain.bb1 import BB1Set, reply_passes, reply_in_language
from noeira.core.bytes import string_from_bytes
from noeira.io.fileio import read_file_bytes


comptime CATS = "where_dir,where_dist,nearest,doing,before,can_yes,can_no,unknown,self,chat,task_next,task_state,held_can_no,held_can_yes,held_unknown,where_mut"


def main() raises:
    var args = argv()
    if len(args) < 2:
        raise Error("usage: bb1_rescore <replies.tsv> [tests/brain]")
    var set = BB1Set.load(String(args[2]) if len(args) > 2 else String("tests/brain"))
    var cats = List[String]()
    for c in String(CATS).split(","):
        cats.append(String(c))
    var arms = List[String]()
    var n_pass = List[List[Int]]()
    var n_all = List[List[Int]]()
    var alt = List[Int]()
    var in_lang = List[Int]()
    var txt = string_from_bytes(read_file_bytes(String(args[1])))
    var first = True
    for l in txt.split("\n"):
        var line = String(l)
        if first or line.byte_length() == 0:
            first = False
            continue
        var c = List[String]()
        for x in line.split("\t"):
            c.append(String(x))
        if len(c) < 12:
            raise Error("a row with " + String(len(c)) + " columns")
        var a = -1
        for k in range(len(arms)):
            if arms[k] == c[0]:
                a = k
        if a < 0:
            a = len(arms)
            arms.append(c[0].copy())
            n_pass.append(List[Int](length=len(cats), fill=0))
            n_all.append(List[Int](length=len(cats), fill=0))
            alt.append(0)
            in_lang.append(0)
        var it = -1
        for k in range(len(set.items)):
            if set.items[k].id == c[1]:
                it = k
        if it < 0 or set.items[it].question != c[10]:
            raise Error("row " + c[1] + " is not this set's item (regenerated with other questions?)")
        ref item = set.items[it]
        var ci = 0
        for k in range(len(cats)):
            if cats[k] == item.cat:
                ci = k
        n_all[a][ci] += 1
        if reply_passes(c[11], item.must, item.forbid):
            n_pass[a][ci] += 1
        if item.cat == "where_mut" and reply_passes(c[11], item.alt_must, item.alt_forbid):
            alt[a] += 1
        if reply_in_language(c[11], item.lang):
            in_lang[a] += 1
    for a in range(len(arms)):
        var grounded = 0
        var grounded_all = 0
        var held = 0
        var held_all = 0
        var tasks = 0
        var tasks_all = 0
        var total = 0
        var line = String("")
        for k in range(len(cats)):
            line += "  " + cats[k] + " " + String(n_pass[a][k]) + "/" + String(n_all[a][k])
            total += n_all[a][k]
            if cats[k].startswith("held_"):
                held += n_pass[a][k]
                held_all += n_all[a][k]
            elif cats[k].startswith("task_"):
                tasks += n_pass[a][k]
                tasks_all += n_all[a][k]
            elif cats[k] != "where_mut":
                grounded += n_pass[a][k]
                grounded_all += n_all[a][k]
        var mi = len(cats) - 1
        print(arms[a] + ":", grounded, "/", grounded_all, "; tasks", tasks, "/", tasks_all, "; held out", held, "/", held_all, "|" + line)
        print("   mutant rows: right for the swapped room", n_pass[a][mi], "/", n_all[a][mi],
              "; still right for the room before the swap", alt[a], "/", n_all[a][mi])
        print("   in the question's language", in_lang[a], "/", total)
