#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import sys
import unicodedata
import six
from six.moves import range  # pylint: disable=redefined-builtin

# Conversion between Unicode and UTF-8, if required (on Python2)
_native_to_unicode = (lambda s: s.decode("utf-8")) if six.PY2 else (lambda s: s)

# This set contains all letter and number characters.
_ALPHANUMERIC_CHAR_SET = set(
    six.unichr(i) for i in range(sys.maxunicode)
    if (unicodedata.category(six.unichr(i)).startswith("L") or
        unicodedata.category(six.unichr(i)).startswith("N")))


_SUFFIX = {
    'st', 'nd', 'rd', 'th',
    'mm', 'm',
    'ns', 'ms', 's', 'hr', 'h'
}


def split_number(token):
    x = [token[0]]
    x.extend('##' + c for c in token[1:])
    return x


def split_number_suffix(token, suffix):
    x = [token[0]]
    x.extend('##' + c for c in token[1:suffix])
    x.append('##' + token[suffix:])
    return x


def encode(text):
    """Encode a unicode string as a list of tokens.
    Args:
      text: a unicode string
    Returns:
      a list of tokens as Unicode strings
    """
    if not text:
        return []
    ret = []
    token_start = 0
    # Classify each character in the input string
    is_alnum = [c in _ALPHANUMERIC_CHAR_SET for c in text]
    for pos in range(1, len(text)):
        if is_alnum[pos] != is_alnum[pos - 1]:
            token = text[token_start:pos]
            if token != u" " or token_start == 0:
                ret.extend(split_token(token))
            token_start = pos
    final_token = text[token_start:]
    ret.extend(split_token(final_token))
    return ret


def split_token(t):
    if t[0] not in _ALPHANUMERIC_CHAR_SET:
        return list(t)

    if t[0].isdigit():
        # Numbers w/ two-letter suffixes 1st, 2nd, etc.
        if len(t) >= 3 and t[-2:] in _SUFFIX and t[:-2].isdigit():
            return split_number_suffix(t, -2)

        # Numbers w/ one-letter suffixes, e.g. 1930s, 500m
        if len(t) >= 2 and t[-1:] in _SUFFIX and t[:-1].isdigit():
            return split_number_suffix(t, -1)

        # Numbers
        if t.isdigit():
            r = split_number(t)
            return r

    ret = []
    start = 0
    cjk = [0x4e00 <= ord(c) <= 0x9fff for c in t]
    for i, c in enumerate(t):
        if cjk[i]:
            if start != i:
                ret.extend(split_token(t[start:i]))
            start = i + 1
            ret.append(c)
            continue
    if start < len(t):
        if start == 0:
            ret.append(t)
        else:
            ret.extend(split_token(t[start:]))

    return ret


def main():
    for line in sys.stdin:
        tokens = encode(line.strip())
        print(' '.join(tokens))


if __name__ == '__main__':
    main()
