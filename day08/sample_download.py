import urllib.request
import os

def get_sample(filename):
    if os.path.exists(filename):
        return filename

    # 1순위: OpenCV 공식 저장소 시도
    opencv_url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
    # 2순위: 인사이트북 저장소 시도
    insight_url = f"https://raw.githubusercontent.com/dltpdn/insightbook.opencv_project_python/master/img/{filename}"

    for url in [opencv_url, insight_url]:
        try:
            print(f"Trying to download from: {url}")
            urllib.request.urlretrieve(url, filename)
            print(f"Successfully downloaded {filename}")
            return filename
        except Exception:
            continue # 실패하면 다음 URL 시도

    print(f"Error: {filename}을 어느 저장소에서도 찾을 수 없습니다.")
    return filename