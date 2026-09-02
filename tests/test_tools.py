"""Checks on the tools themselves, below the graph and the UI."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmtui.tools.mcp import WHITELISTED_TOOLS
from llmtui.tools.search_books import book_pattern

results = {}


def check(label, conditions):
    ok = True
    print(f"\n{'=' * 66}\n{label}\n{'=' * 66}")
    for desc, passed in conditions:
        print(f"    {'PASS' if passed else 'FAIL'}  {desc}")
        ok = ok and passed
    results[label] = ok


# the model names a book either way, and the filenames are underscored. the old
# filter stripped underscores out of the argument, so a real filename could
# never match its own file and the tool reported the book as missing
UNDERSCORED = "against_the_cult_of_the_reptile_god"
SPACED = "against the cult of the reptile god"

check("1  book filter matches a title however it is spelled", [
    ("underscores become wildcards",
     book_pattern(UNDERSCORED) == "%against%the%cult%of%the%reptile%god%"),
    ("spaces reach the same pattern", book_pattern(SPACED) == book_pattern(UNDERSCORED)),
    ("hyphens too", book_pattern("against-the-cult") == "%against%the%cult%"),
    ("a one word title still works", book_pattern("dnd") == "%dnd%"),
    ("a leading number is kept",
     book_pattern("n1_against_the_cult") == "%n1%against%the%cult%"),
    # a quote would close the expression string, a backslash escapes whatever
    # follows it, and % is the wildcard we are inserting ourselves
    ("quotes, backslashes and stray wildcards are dropped",
     book_pattern('bad" \\ arg%%') == "%bad%arg%"),
    ("surrounding whitespace does not leave an empty segment",
     book_pattern("  dnd  ") == "%dnd%"),
])

check("2  only the tools we chose reach the model", [
    ("the ones the prompt names are all loadable",
     {"search_simple", "vault_list", "vault_get_document_map", "vault_read"}
     <= WHITELISTED_TOOLS),
    ("running arbitrary obsidian commands is not on the menu",
     not {"command_list", "command_execute"} & WHITELISTED_TOOLS),
])

print("\n" + "=" * 66)
print("SUMMARY")
print("=" * 66)
for label, ok in results.items():
    print(f"  {'PASS' if ok else 'FAIL'}   {label}")
print(f"\n{sum(results.values())}/{len(results)} cases passed")
