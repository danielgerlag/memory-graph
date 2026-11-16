"""Interpret positional arguments independently of the product CLI."""

import re

from controlled_core import command_arguments

VALUE_OPTIONS = {
    "find": {"--tag", "--kind", "--scope", "--limit"},
    "show": {"--scope", "--commit"},
    "neighbors": {"--direction", "--label", "--scope", "--depth", "--limit"},
    "review": {"--scope", "--limit"},
}
FLAG_OPTIONS = {
    "find": {"--include-unscoped"},
    "show": {"--include-unscoped", "--include-inactive"},
    "neighbors": {"--include-unscoped"},
    "review": {"--include-unscoped"},
}


def canonical_option(command, option):
    allowed = VALUE_OPTIONS.get(command, set()) | FLAG_OPTIONS.get(command, set())
    if not allowed or option in allowed:
        return option
    matches = [candidate for candidate in allowed if candidate.startswith(option)]
    return matches[0] if len(matches) == 1 else None


def has_flag(args, name):
    args = command_arguments(args)
    if not args:
        return False
    for value in args[1:]:
        if value == "--":
            break
        if value.startswith("--") and "=" not in value and canonical_option(args[0], value) == name:
            return True
    return False


def positional_argument(args):
    """Return the sole positional argument, regardless of option placement."""
    args = command_arguments(args)
    if not args or args[0] not in VALUE_OPTIONS:
        return None
    positionals = []
    index = 1
    options_ended = False
    while index < len(args):
        value = args[index]
        if not options_ended and value == "--":
            options_ended = True
        elif not options_ended and value.startswith("--"):
            option, separator, _ = value.partition("=")
            option = canonical_option(args[0], option)
            if option in VALUE_OPTIONS[args[0]]:
                if not separator:
                    index += 1
                    if index >= len(args) or args[index].startswith("--"):
                        return None
            elif option not in FLAG_OPTIONS[args[0]] or separator:
                return None
        elif (
            not options_ended
            and value.startswith("-")
            and value != "-"
            and re.fullmatch(r"-\d+|-\d*\.\d+", value) is None
        ):
            return None
        else:
            positionals.append(value)
        index += 1
    return positionals[0] if len(positionals) == 1 else None
