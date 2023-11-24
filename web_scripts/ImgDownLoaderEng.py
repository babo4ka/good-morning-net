import os
import time
import urllib.request

from chromedriver_py import binary_path
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

caps = DesiredCapabilities().CHROME
caps["pageLoadStrategy"] = "eager"
browser = webdriver.Chrome(executable_path=binary_path, desired_capabilities=caps)
browser.get("https://onlymyenglish.com/good-morning-images/")

try:
    os.mkdir("gm_eng")
except FileExistsError:
    pass

count = 312

# for i in range(1, 85):
#     elem = browser.find_element_by_xpath("//*[@id=\"post-4377\"]/div/div/figure[" + str(i) + "]/img")
#     src = elem.get_attribute("src")
#     print(src)
#     count += 1
#     urllib.request.urlretrieve(src,
#                                os.path.join("gm_eng", 'mrng' + str(count) + '.jpg'))


#//*[@id="arya-content-main"]/div[2]/div[1]/figure/img
#//*[@id="arya-content-main"]/div[2]/div[7]/figure/img
#//*[@id="arya-content-main"]/div[2]/div[81]/figure/img
#//*[@id="post-26837"]/div[2]/div[132]/figure/img


for i in range(1, 132):
    print("//*[@id=\"post-26837\"]/div[2]/div[" + str(i) + "]/figure/img")
    try:
        elem = browser.find_element_by_xpath("//*[@id=\"post-26837\"]/div[2]/div[" + str(i) + "]/figure/img")
        src = elem.get_attribute("src")
        print(src)
        count += 1
        urllib.request.urlretrieve(src, os.path.join("gm_eng", 'mrng' + str(count) + '.jpg'))
    except:
        pass