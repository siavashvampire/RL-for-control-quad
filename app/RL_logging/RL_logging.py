from selenium import webdriver
import subprocess
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.webdriver import WebDriver

chrome_options = Options()
chrome_options.add_experimental_option("detach", True)


def open_log_web() -> WebDriver:
    subprocess.Popen("tensorboard --logdir=logs --host=0.0.0.0 --port=8585", shell=True)

    driver: WebDriver = webdriver.Chrome(options=chrome_options)
    try:
        driver.get("http://localhost:8585/")
    except:
        pass
    return driver


def close_tensorboard():
    subprocess.call("TASKKILL /F /IM tensorboard.exe", shell=True)
