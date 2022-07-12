import os

import requests

TARGET_DIR = './data//kodak24'

if __name__ == '__main__':
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    for img_name in [f'kodim{i.zfille(2)}.png' for i in range(1, 25)]:
        with requests.get(f'http://www.cs.albany.edu/~xypan/research/img/Kodak/{img_name}', stream=True) as r:
            with open(os.path.join(TARGET_DIR, img_name), 'wb') as f:
                f.write(r.content)
