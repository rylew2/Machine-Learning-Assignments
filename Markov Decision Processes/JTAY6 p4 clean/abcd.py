#!/usr/bin/env python
"""
convert dos linefeeds (crlf) to unix (lf)
usage: dos2unix.py
"""
original = "./Solution/QL Q-Learning L0.1 q0.0 E0.1 Hard Iter 641 Policy Map.pkl"
destination = "./Solution/QL Q-Learning L0.1 q0.0 E0.1 Hard Iter 641 Policy Map New.pkl"

content = ''
outsize = 0
with open(original, 'rb') as infile:
    content = infile.read()
with open(destination, 'wb') as output:
    for line in content.splitlines():
        outsize += len(line) + 1
        output.write(line + str.encode('\n'))

print("Done. Saved %s bytes." % (len(content)-outsize))