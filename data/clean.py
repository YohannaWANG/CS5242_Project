"""
Simple script to get rid of starting comma in
boxes.txt and boxes-tiny.txt
"""
import pandas as pd

with open("boxes.txt", 'r') as fp:
    lines = fp.readlines()

outlines = []
for line in lines:
    line = line.replace(" ", "").strip("\n")
    line = line.split(",")
    
    if len(line) > 6:
        start = len(line) - 6
        line = line[start:]

    outlines.append(line)

df = pd.DataFrame(outlines, columns=["Label", "Left", "Top", "Right", "Bottom", "Filename"])

df.to_csv("boxes-clean.csv", index=False)
