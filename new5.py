import os
import time
import img2pdf
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image

# Set up Selenium WebDriver
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Run without opening a browser (remove for debugging)
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")  # Ensure a large enough viewport

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Use the base URL (without a specific slide parameter)
url = ("https://docs.google.com/presentation/d/e/2PACX-1vRYBdlK2uqwAkxS4iHH1efpehBlBTKaKSbwiSRRVWHZLYZiHuxSFW7KnIslfQv_Aw/pub?start=false&loop=false&delayms=60000")
driver.get(url)
time.sleep(5)  # Wait for the presentation to load

# Set the number of slides (adjust if needed)
total_slides = 18  # Change to the actual number of slides in your presentation

os.makedirs("slides", exist_ok=True)
image_files = []

for i in range(total_slides):
    print(f"Capturing slide {i+1}...")
    slide_path = f"slides/slide_{i+1}.png"
    
    # Take a screenshot of the full page
    driver.save_screenshot(slide_path)
    image_files.append(slide_path)
    
    # Navigate to the next slide using the RIGHT arrow key
    driver.find_element(By.TAG_NAME, "body").send_keys(Keys.RIGHT)
    time.sleep(2)  # Wait for the slide transition

driver.quit()

# Convert all slide images to a single PDF
if image_files:
    pdf_path = "presentation3.pdf"
    with open(pdf_path, "wb") as f:
        f.write(img2pdf.convert(image_files))
    print(f"PDF saved as {pdf_path}")
else:
    print("No slides captured.")