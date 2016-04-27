from __future__ import print_function
import csv

source = 'movies.csv'
target = 'movies_questions.txt'

with open(source, 'rU') as f:
    reader = csv.reader(f)
    rows = [row for row in reader]

with open(target, 'w+') as outfile:
    print(": Brand-Origin", file=outfile)
    for row in rows:
        for otherRow in rows:
            if not row[0] == otherRow[0]:
                print(row[0]+' '+row[1]+' '+otherRow[0]+' '+otherRow[1], file=outfile)
