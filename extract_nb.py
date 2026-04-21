import json
import sys

filename = sys.argv[1]
outfile = sys.argv[2]
with open(filename, encoding='utf-8') as f:
    n = json.load(f)

with open(outfile, 'w', encoding='utf-8') as fw:
    for c in n['cells']:
        if c['cell_type'] == 'code':
            fw.write(''.join(c['source']))
            fw.write('\n\n--- \n\n')
