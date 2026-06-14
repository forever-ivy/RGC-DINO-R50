#!/usr/bin/env python3
"""
Automated submission uploader for AIC2026 competition.
Uploads prediction ZIP files to the competition platform.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
except ImportError:
    print("ERROR: selenium not installed. Run: pip install selenium", file=sys.stderr)
    sys.exit(1)


SUBMISSION_URL = "https://reg.aicomp.cn/app/JSGLPT/639980063d903c241eb85102"
LOCAL_STORAGE_ORIGIN = "https://reg.aicomp.cn"


def setup_chrome_driver(
    headless: bool = True,
    user_data_dir: Optional[Path] = None,
    chrome_binary: Optional[str] = None,
) -> webdriver.Chrome:
    """
    Setup Chrome WebDriver with appropriate options.

    Args:
        headless: Run in headless mode
        user_data_dir: Chrome user data directory for persistent session
        chrome_binary: Path to Chrome/Chromium binary

    Returns:
        Chrome WebDriver instance
    """
    chrome_options = Options()

    if headless:
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')

    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)

    # Use existing user session if provided
    if user_data_dir:
        chrome_options.add_argument(f'--user-data-dir={user_data_dir}')

    if chrome_binary:
        chrome_options.binary_location = chrome_binary

    driver = webdriver.Chrome(options=chrome_options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    return driver


def load_cookies_to_driver(driver: webdriver.Chrome, cookies_file: Path):
    """Load cookies from file into WebDriver."""
    if not cookies_file.exists():
        print(f"Warning: Cookie file {cookies_file} not found", file=sys.stderr)
        return

    with open(cookies_file, 'r', encoding='utf-8') as f:
        cookies = json.load(f)

    for name, value in cookies.items():
        driver.add_cookie({
            'name': name,
            'value': value,
            'domain': '.aicomp.cn',
        })


def _local_storage_value(value) -> Optional[str]:
    """Convert JSON values to strings accepted by window.localStorage."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def load_local_storage_to_driver(
    driver: webdriver.Chrome,
    storage_file: Path,
    origin_url: str = LOCAL_STORAGE_ORIGIN,
) -> int:
    """Load localStorage values from a JSON file into WebDriver."""
    if not storage_file.exists():
        print(f"Warning: localStorage file {storage_file} not found", file=sys.stderr)
        return 0

    with open(storage_file, 'r', encoding='utf-8') as f:
        storage = json.load(f)

    if isinstance(storage, dict) and isinstance(storage.get('localStorage'), dict):
        storage = storage['localStorage']

    if not isinstance(storage, dict):
        raise ValueError(f"localStorage file must contain a JSON object: {storage_file}")

    if 'currentRoleId' not in storage and storage.get('createdRoleId') is not None:
        storage = dict(storage)
        storage['currentRoleId'] = storage['createdRoleId']

    driver.get(origin_url)

    loaded = 0
    for key, value in storage.items():
        storage_value = _local_storage_value(value)
        if storage_value is None:
            continue
        driver.execute_script(
            "window.localStorage.setItem(arguments[0], arguments[1]);",
            str(key),
            storage_value,
        )
        loaded += 1

    return loaded


def normalize_text(text: str) -> str:
    """Normalize UI text so Chinese button labels with spaces match reliably."""
    return ''.join((text or '').split())


def choose_zip_file_input(file_inputs: Sequence):
    """Choose the file input intended for ZIP submissions."""
    zip_inputs = []
    for element in file_inputs:
        accept = (element.get_attribute('accept') or '').lower()
        if '.zip' in accept:
            zip_inputs.append(element)

    for element in zip_inputs:
        accept = (element.get_attribute('accept') or '').lower()
        if '.rar' in accept:
            return element

    return zip_inputs[0] if zip_inputs else None


def click_submission_entry(driver: webdriver.Chrome, wait_time: int):
    """Open the submission modal from the competition upload list."""
    def find_element(current_driver):
        for tag in ('a', 'button'):
            for element in current_driver.find_elements(By.TAG_NAME, tag):
                if normalize_text(element.text) == '提交作品' and element.is_displayed() and element.is_enabled():
                    return element
        return False

    element = WebDriverWait(driver, wait_time).until(find_element)
    driver.execute_script("arguments[0].scrollIntoView();", element)
    time.sleep(0.5)
    driver.execute_script("arguments[0].click();", element)
    return element


def find_zip_file_input(driver: webdriver.Chrome, wait_time: int):
    """Wait for the modal upload control and return its ZIP file input."""
    def find_element(current_driver):
        inputs = current_driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
        return choose_zip_file_input(inputs) or False

    return WebDriverWait(driver, wait_time).until(find_element)


def find_final_submit_button(driver: webdriver.Chrome, wait_time: int):
    """Find the modal's final submit button."""
    def find_element(current_driver):
        for element in current_driver.find_elements(By.TAG_NAME, 'button'):
            if normalize_text(element.text) == '提交' and element.is_displayed() and element.is_enabled():
                return element
        return False

    return WebDriverWait(driver, wait_time).until(find_element)


def submit_prediction(
    driver: webdriver.Chrome,
    zip_path: Path,
    wait_time: int = 30,
    dry_run: bool = False,
) -> Dict:
    """
    Submit prediction ZIP file to competition platform.

    Args:
        driver: Chrome WebDriver
        zip_path: Path to submission ZIP file
        wait_time: Maximum wait time for elements (seconds)

    Returns:
        Dict with submission result
    """
    result = {
        'success': False,
        'dry_run': dry_run,
        'timestamp': datetime.now().isoformat(),
        'zip_path': str(zip_path),
    }

    try:
        # Navigate to submission page
        print(f"Opening submission page...")
        driver.get(SUBMISSION_URL)
        WebDriverWait(driver, wait_time).until(
            lambda current_driver: '提交作品' in current_driver.find_element(By.TAG_NAME, 'body').text
        )

        try:
            print("Opening submission modal...")
            click_submission_entry(driver, wait_time)

            file_input = find_zip_file_input(driver, wait_time)
            print("Found ZIP upload input")

            if dry_run:
                result['success'] = True
                result['message'] = 'Dry run reached ZIP upload input; no file was uploaded'
                print("✓ Dry run reached ZIP upload input")
                return result

            # Upload file
            print(f"Uploading {zip_path.name}...")
            driver.execute_script(
                "arguments[0].style.display = 'block';"
                "arguments[0].style.visibility = 'visible';"
                "arguments[0].style.opacity = 1;",
                file_input,
            )
            file_input.send_keys(str(zip_path.absolute()))
            time.sleep(8)

            submit_button = find_final_submit_button(driver, wait_time)
            print("Clicking submit button...")
            driver.execute_script("arguments[0].click();", submit_button)
            time.sleep(8)

            # Check for success or obvious validation errors.
            success_texts = ['成功', 'success', '提交成功', 'uploaded', '保存成功']
            error_texts = ['失败', '错误', '请选择', '不能为空', '上传失败']
            page_text = driver.page_source.lower()

            if any(text in page_text for text in error_texts):
                result['message'] = 'Submit clicked, but page shows an error or validation message'
                print("⚠ Submit clicked, but page shows an error or validation message")
            elif any(text in page_text for text in success_texts):
                result['success'] = True
                result['message'] = 'Submission successful'
                print("✓ Submission successful!")
            else:
                result['success'] = True
                result['message'] = 'Submit button clicked; no explicit success confirmation found'
                print("⚠ Submit button clicked, please verify leaderboard later")


        except Exception as e:
            result['message'] = f'Error during submission: {str(e)}'
            result['error'] = str(e)
            print(f"✗ Error: {e}", file=sys.stderr)

            # Save screenshot for debugging
            screenshot_path = Path('outputs/submission_error.png')
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            driver.save_screenshot(str(screenshot_path))
            result['screenshot'] = str(screenshot_path)
            print(f"Saved screenshot to {screenshot_path}")

    except Exception as e:
        result['message'] = f'Fatal error: {str(e)}'
        result['error'] = str(e)
        print(f"✗ Fatal error: {e}", file=sys.stderr)

    return result


def save_submission_log(result: Dict, log_file: Path):
    """Append submission result to log file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Upload prediction ZIP to AIC2026 competition platform'
    )
    parser.add_argument(
        'zip_path',
        type=Path,
        help='Path to submission ZIP file'
    )
    parser.add_argument(
        '--cookies',
        type=Path,
        help='Path to cookies JSON file'
    )
    parser.add_argument(
        '--user-data-dir',
        type=Path,
        help='Chrome user data directory (for persistent logged-in session)'
    )
    parser.add_argument(
        '--local-storage',
        type=Path,
        help='JSON file with localStorage auth values exported from logged-in browser'
    )
    parser.add_argument(
        '--chrome-binary',
        help='Path to Chrome/Chromium binary'
    )
    parser.add_argument(
        '--no-headless',
        action='store_true',
        help='Show browser window (for debugging)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Open the submission modal and find the ZIP upload input without uploading'
    )
    parser.add_argument(
        '--log',
        type=Path,
        default=Path('outputs/submission_log.jsonl'),
        help='Submission log file'
    )
    parser.add_argument(
        '--wait',
        type=int,
        default=30,
        help='Maximum wait time for elements (seconds)'
    )

    args = parser.parse_args()

    # Validate ZIP file
    if not args.zip_path.exists():
        print(f"ERROR: ZIP file not found: {args.zip_path}", file=sys.stderr)
        sys.exit(1)

    if not args.zip_path.suffix.lower() == '.zip':
        print(f"WARNING: File does not have .zip extension: {args.zip_path}", file=sys.stderr)

    print(f"\n{'='*60}")
    print(f"AIC2026 Automated Submission")
    print(f"{'='*60}")
    print(f"ZIP file: {args.zip_path}")
    print(f"Size: {args.zip_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # Setup driver
    try:
        driver = setup_chrome_driver(
            headless=not args.no_headless,
            user_data_dir=args.user_data_dir,
            chrome_binary=args.chrome_binary,
        )
        print("✓ Chrome driver initialized")
    except Exception as e:
        print(f"ERROR: Failed to initialize Chrome driver: {e}", file=sys.stderr)
        print("\nTroubleshooting:")
        print("  1. Install Chrome/Chromium: sudo apt install chromium-browser")
        print("  2. Install ChromeDriver: sudo apt install chromium-chromedriver")
        print("  3. Or install via pip: pip install webdriver-manager")
        sys.exit(1)

    try:
        # Load cookies if provided
        if args.cookies:
            driver.get("https://reg.aicomp.cn")
            time.sleep(2)
            load_cookies_to_driver(driver, args.cookies)
            print(f"✓ Loaded cookies from {args.cookies}")

        if args.local_storage:
            loaded = load_local_storage_to_driver(driver, args.local_storage)
            print(f"✓ Loaded {loaded} localStorage values from {args.local_storage}")

        # Submit prediction
        result = submit_prediction(driver, args.zip_path, wait_time=args.wait, dry_run=args.dry_run)

        # Save log
        save_submission_log(result, args.log)
        print(f"\n✓ Submission log saved to {args.log}")

        # Exit with appropriate code
        sys.exit(0 if result['success'] else 1)

    finally:
        driver.quit()


if __name__ == '__main__':
    main()
