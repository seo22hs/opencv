import urllib.request
import os
import cv2 as cv

def get_sample(filename, repo='insightbook'):
    if not os.path.exists(filename):
        if repo == 'insightbook':
            url = f"https://raw.githubusercontent.com/dltpdn/insightbook.opencv_project_python/master/img/{filename}"
        else:
            url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, filename)
    return filename

# 실제 실행 코드
img_path = get_sample('morphological.png', repo='insightbook')
img = cv.imread(img_path)

cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()