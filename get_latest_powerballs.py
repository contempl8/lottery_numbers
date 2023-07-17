from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import time
import json
import datetime
import dateparser

def get_todays_date():
    """
    Get today's date as a formatted string.

    Returns:
        A string representing the last date in the data, formatted as 'YYYY-MM-DD'.
    """
    today=datetime.datetime.today()
    return f'{today.year}-{today.month}-{today.day}'

def get_last_date_in_data(data):
    """
    This function takes in a dictionary of data and returns the last date in the data as a string.

    Parameters:
        - data (dict): A dictionary containing dates as keys.

    Returns:
        - l_date_str (str): A string representing the last date in the data, formatted as 'YYYY-MM-DD'.
    """
    all_dates=[]
    for date in data.keys():
        all_dates.append(dateparser.parse(date))
    last_date=sorted(all_dates,reverse=True)[0]
    return f'{last_date.year}-{last_date.month}-{last_date.day}'

file="seventh_version.json"
with open('number_data/'+file,'r') as f:
    data=json.loads(f.read())
    f.close()

date_str=get_todays_date()
l_date_str=get_last_date_in_data(data)

powerball_versions = dict(seventh_version=(l_date_str,date_str))


for version, date in powerball_versions.items():
    driver = webdriver.Chrome()
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

    res=driver.find_element(By.ID,"searchNumbersResults")
    while True:
        delay=0.5
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
    with open('number_data/'+filename,'w') as f:
        f.write(jdump)
        f.close()
    driver.close()