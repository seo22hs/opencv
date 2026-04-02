import urllib.request
import os
import numpy as np 
import cv2 as cv 

def get_sample(filename, repo='insightbook'):
    if not os.path.exists(filename):
        if repo == 'insightbook':
            url = f"https://raw.githubusercontent.com/dltpdn/insightbook.opencv_project_python/master/img/{filename}"
        else:
            url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, filename)
    return filename

img = cv.imread(get_sample('messi5.jpg', repo='opencv'))

# 스케일링
res = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)

# 결과 확인 (둘 다 비교 추천 👍)
cv.imshow("Original", img)
cv.imshow("Scaling", res)

cv.waitKey(0)
cv.destroyAllWindows()