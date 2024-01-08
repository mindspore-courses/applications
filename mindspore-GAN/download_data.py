from download import download

def download_MNIST():
      url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
            "notebook/datasets/MNIST_Data.zip"
      path = download(url, "./", kind="zip", replace=True)

if __name__ == "__main__":
      download_MNIST()