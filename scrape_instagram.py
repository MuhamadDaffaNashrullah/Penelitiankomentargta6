from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import csv

# Setup ChromeDriver
service = Service('chromedriver.exe')
options = Options()
options.add_argument("--start-maximized")
options.add_argument("--disable-notifications")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--disable-extensions")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--enable-unsafe-swiftshader")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)

try:
    driver = webdriver.Chrome(service=service, options=options)
    driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
        'source': '''
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            })
        '''
    })
    
    print("Membuka Instagram...")
    driver.get("https://www.instagram.com/p/DHb3m83Rl_v/?igsh=MXF6ZGNmNXBpNzczZQ==&tag=chatgpt")

    # Login manual
    input("Login IG dulu, lalu tekan Enter di terminal...")

    print("Menunggu konten dimuat...")
    time.sleep(15)

    print("Mencari bagian komentar...")
    
    # Scroll ke bagian komentar beberapa kali untuk memuat lebih banyak komentar
    for _ in range(5):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)

    print("Mencoba mengambil komentar...")
    
    comments = []
    try:
        # Coba beberapa selector yang berbeda
        selectors = [
            "span._aacl._aaco._aacu._aacx._aad7._aade",  # Selector lama
            "div._a9zs span",  # Selector alternatif 1
            "div[data-testid='post-comment-root'] span",  # Selector alternatif 2
            "ul._a9z6 span"  # Selector alternatif 3
        ]
        
        for selector in selectors:
            comment_elements = driver.find_elements(By.CSS_SELECTOR, selector)
            if comment_elements:
                print(f"Berhasil dengan selector: {selector}")
                break
        
        print(f"Menemukan {len(comment_elements)} komentar")
        
        # Ekstrak teks komentar
        for element in comment_elements:
            try:
                # Tunggu sebentar untuk memastikan elemen dapat diakses
                WebDriverWait(driver, 5).until(
                    EC.visibility_of(element)
                )
                comment_text = element.text.strip()
                if comment_text and not comment_text.isdigit():
                    comments.append(comment_text)
                    print(f"Komentar ditemukan: {comment_text[:50]}...")
            except:
                continue
                
    except Exception as e:
        print(f"Error saat mengambil komentar: {str(e)}")

    if not comments:
        print("Tidak ditemukan komentar")
    else:
        print(f"Total komentar yang ditemukan: {len(comments)}")
        # Simpan ke CSV
        with open("komentar_grandtheftauto6.csv", "w", newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["komentar"])  # header
            for comment in comments:
                writer.writerow([comment])
            print(f"Berhasil menyimpan {len(comments)} komentar ke komentar_grandtheftauto6.csv")

except Exception as e:
    print(f"Terjadi error: {str(e)}")

finally:
    try:
        driver.quit()
    except:
        pass
