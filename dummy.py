#make 700 copies of the image data/content/08.jpg and save them in data/content1/

import shutil
import os

for i in range(700):
    shutil.copy2('data/content/08.jpg', 'data/content1/08_'+str(i)+'.jpg')
    print('data/content/08.jpg copied to data/content1/08_'+str(i)+'.jpg')

    