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
        ret.append(token)
      token_start = pos
  final_token = text[token_start:]
  ret.append(final_token)
  return ret


def main():
    for line in sys.stdin:
        tokens = encode(line.strip())
        print(' '.join(tokens))


if __name__ == '__main__':
    main()
