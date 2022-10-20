from selenium import webdriver
import subprocess
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.webdriver import WebDriver

chrome_options = Options()
chrome_options.add_experimental_option("detach", True)


def open_log_web() -> WebDriver:
    subprocess.Popen("tensorboard --logdir=logs", shell=True)

    driver: WebDriver = webdriver.Chrome(options=chrome_options)
    try:
        driver.get("http://localhost:6006/")
    except:
        pass
    return driver
