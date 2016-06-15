#!/usr/bin/env python3
"""Rename alpha results to the current format."""

import re
from pathlib import Path

DRY_RUN = True


values = {2**x: str(x) for x in range(-12, 6)}
decimal_pattern = r'[-+]?\d*\.\d+|\d+'


def main():
    for path in Path('../word2vec_data').glob('**/*_alpha_*'):
        match = re.findall(decimal_pattern, path.name.split('_')[-1])
        if not match:
            print("ERROR1")
            return
        val = match[0]
        fval = float(val)
        if fval not in values:
            print("ERROR2:", path.name, val)
            return
        name = path.name.replace(val, values[fval])
        newpath = path.parent / name
        print("{} -> {}".format(path, newpath))
        if not DRY_RUN:
            path.rename(newpath)


if __name__ == '__main__':
    main()
