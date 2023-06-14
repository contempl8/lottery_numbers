from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import time
import json
import datetime
today=datetime.datetime.today()
date_str=f'{today.year}-{today.month}-{today.day}'
powerball_versions = dict(first_version=('04-22-1992','11-01-1997'),
                          second_version=('11-05-1997','10-05-2002'),
                          third_version=('10-09-2002','08-27-2005'),
                          fourth_version=('08-31-2005','01-03-2009'),
                          fifth_version=('01-07-2009','01-14-2012'),
                          sixth_version=('01-18-2012','10-03-2015'),
                          seventh_version=('10-07-2015',date_str))


for version, date in powerball_versions.items():
    driver = webdriver.Firefox()
    driver.get(f'https://www.powerball.com/previous-results?gc=powerball&sd={date[0]}&ed={date[1]}')


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
    filename=f'{version}.json'
    try:
        with open(filename,'x') as f:
            f.write(json.dumps({}))
            f.close()
    except FileExistsError:
        pass
    with open(filename,'r') as f:
        data=json.loads(f.read())

    res=driver.find_element(By.ID,"searchNumbersResults")
    if len(data):
        pass
    else:
        data={}
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
    data.update(powerball_numbers)
    jdump=json.dumps(data)
    with open(filename,'w') as f:
        f.write(jdump)
        f.close()
    driver.close()