import os
import urllib.request

from chromedriver_py import binary_path
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

caps = DesiredCapabilities().CHROME
caps["pageLoadStrategy"] = "eager"
browser = webdriver.Chrome(executable_path=binary_path, desired_capabilities=caps)
browser.get("https://malinkakat.ru/utro/list-1.php")


try:
    os.mkdir("gm_rus")
except FileExistsError:
    pass

count = 0

#/html/body/div[1]/div[3]/div[2]/ul/li[4]/a/img
#/html/body/div[1]/div[3]/div[2]/ul/li[1]/a
def searchAndLoadImages(count=count, page_num = 0):
    for i in range(1, 18):
        browser.get("https://malinkakat.ru/utro/list-" + str(page) + ".php")
        # time.sleep(3)
        link = browser.find_element_by_xpath("/html/body/div[1]/div[3]/div[2]/ul/li[" + str(i) + "]/a").get_attribute("href")
        browser.get(link)
        elem = browser.find_element_by_xpath("/html/body/div[1]/div[3]/div[2]/div[4]/img")
        src = elem.get_attribute("src")
        print(src)
        count += 1
        urllib.request.urlretrieve(src,
                                   os.path.join("gm_rus", 'utro' + str(count) + '.jpg'))


    return count

page = 1
for i in range(62):
    count = searchAndLoadImages(count, i+1)
