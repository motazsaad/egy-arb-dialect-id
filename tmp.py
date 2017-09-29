
import json
import pandas as pd

with open('test.json') as f:
    data = pd.DataFrame(json.loads(line) for line in f)
    print(data)
    print('info: \n{}'.format(data.info()))