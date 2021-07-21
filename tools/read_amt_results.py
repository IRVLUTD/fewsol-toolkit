#!/usr/bin/env python3
import json

filename = '../data/AMT/synthetic_objects/results_batch-000.txt'
with open(filename, 'r') as f:
    lines = f.readlines()

data = []
for l in lines:
    data.append(json.loads(l))

num = len(data)
for i in range(num):
    result = data[i]
    print(result['worker_id'])
