from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import time
import json

driver = webdriver.Firefox()
driver.get('https://www.powerball.com/previous-results?gc=powerball&sd=1992-04-22&ed=2023-05-08')


def scroll_down(delay):
    """A method for scrolling the page."""

    # Get scroll height.
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:

        # Scroll down to the bottom.
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load the page.
        time.sleep(delay)

        # Calculate new scroll height and compare with last scroll height.
        new_height = driver.execute_script("return document.body.scrollHeight")

        if new_height == last_height:

            break

        last_height = new_height

res=driver.find_element(By.ID,"searchNumbersResults")
count=0
while True:
    count+=1
    if count > 200: delay=1
    else: delay=0.5
    scroll_down(delay)
    try:
        res.find_element(By.ID,'loadMore').click()
    except Exception as e:
        print(e)
        break

line=res.find_elements(By.CLASS_NAME,"card")
print(f'Number of Dates: {len(line)}')
powerball_numbers={}
for l in line:
    try:
        white_balls=[]
        date=l.find_elements(By.CLASS_NAME,"card-title")[0].text
        for ball in l.find_elements(By.CLASS_NAME,"white-balls"):
            white_balls.append(ball.text)
        powerball=l.find_element(By.CLASS_NAME,"powerball").text
        powerball_numbers[date]=(white_balls,powerball)
    except Exception as e:
        print(e)

jdump=json.dumps(powerball_numbers)
with open("powerballnumbers.json",'w') as f:
    f.write(jdump)
    f.close()
driver.close()