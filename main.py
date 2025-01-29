# Run `python main.py` in the terminal

# Note: Python is lazy loaded so the first run will take a moment,
# But after cached, subsequent loads are super fast! ⚡️

import platform
print(f"Hello Python v{platform.python_version()}!")

import cProfile
import logging
import os
import pickle
import random
import re
import shutil
import threading
import time
from builtins import FileNotFoundError, range
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Lock
from typing import List, Optional

import psutil
import undetected_chromedriver as uc
from circuitbreaker import CircuitBreaker, CircuitBreakerError
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.style import Style as RichStyle
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich.traceback import install as install_rich_traceback
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    ElementNotInteractableException,
    StaleElementReferenceException,
    TimeoutException,
)
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# -------------------------------------------------------------------
#                        GLOBAL CONFIGURATION
# -------------------------------------------------------------------
TEST_MODE = True  # True => 'Agent' message is replaced with junk for testing
MAX_TIP_THRESHOLD = 3.0  # Orders above this tip are processed
MAX_ORDERS_PER_BATCH = 30
HEADLESS_MODE = False  # Set True to run browser in background without GUI
RECONNECT_TIMEOUT = 90

# Update constants
WINDOW_WIDTH = 360  # Mobile viewport width
WINDOW_HEIGHT = 640  # Mobile viewport height
STARTUP_DELAY = 5  # Seconds to wait after driver creation

MIN_AGENT_WAIT = 5
MAX_SCROLL_ATTEMPTS = 50
PARALLEL_TIMEOUT = 6
CHUNK_SIZE = 6
MAX_WORKERS = 6
CHAT_BATCH_SIZE = 5
MESSAGE_DEDUPE_DELAY = 1.0
MAX_CHAT_RETRIES = 3
CHAT_RETRY_DELAY = 2.0
TAB_POOL_SIZE = 20
TAB_WAIT = 0.3

# Define consistent colors
MAIN_COLOR = "#D91400"
ACCENT_COLOR = "#FFD700"

# Setup custom theme
custom_theme = Theme(
    {
        "info": "bold white",
        "warning": "yellow",
        "error": "bold red",
        "success": "green",
        "border": MAIN_COLOR,
        "accent": ACCENT_COLOR,
    }
)
console = Console(theme=custom_theme)

# Initialize rich console and traceback
install_rich_traceback(show_locals=False, max_frames=3)

# Status Styles
SUCCESS_STYLE = RichStyle(color="green", bold=True)
ERROR_STYLE = RichStyle(color="red", bold=True)
INFO_STYLE = RichStyle(color="blue", bold=True)
WARNING_STYLE = RichStyle(color="yellow", bold=True)

# Standard Timeouts and Waits
IMPLICIT_WAIT = 4
PAGE_LOAD_TIMEOUT = 5
BUTTON_WAIT = 0.3

# Operational Waits
SCROLL_WAIT = 0.4
SHORT_ACTION_WAIT = 0.3
BETWEEN_ELEMENT_RETRIES_WAIT = 0.3
RETRY_OPERATION_WAIT = 0.3

# Fast Mode Timeouts (Used when fast=True)
FAST_MESSAGE_WAIT = 0.2
FAST_BUTTON_WAIT = 0.3
FAST_ACTION_WAIT = 0.5

# Normal Mode Timeouts (Used when fast=False)
NORMAL_MESSAGE_WAIT = 0.2
NORMAL_BUTTON_WAIT = 0.5
NORMAL_ACTION_WAIT = 0.6

# Retry Configuration
MAX_RETRIES = 3
RETRY_DELAY = 4.0

# Initialize global variables
customer_name: Optional[str] = None
customer_email: Optional[str] = None

# Setup logging
logging.basicConfig(
    filename="app.log", filemode="a", format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


def log_debug(message: str, transient: bool = True) -> None:  # New debug log level
    """Log debug messages when detailed output is needed."""
    logging.debug(message)
    if transient:
        console.print(f"🐞 [dim blue] {message}[/]", end="\r")
    else:
        console.print(f"🐞 [dim blue] {message}[/]")


def log_success(message: str, transient: bool = True) -> None:
    """Log success messages with proper spacing."""
    logging.info(message)
    if transient:
        console.print(f"✅ [green bold] {message}[/]", end="\r", style="success")  # Apply style here
    else:
        console.print(f"✅ [green bold] {message}[/]", style="success")  # Apply style here
        console.print()  # Keep newline for non-transient


def log_error(message: str) -> None:
    """Log error messages with rich formatting and file logging."""
    logging.error(message)
    console.print(f"❌ [red bold] {message}[/]", style="error")  # Apply style here
    console.print()


def log_info(message: str, transient: bool = True) -> None:
    """Log info messages with rich formatting and file logging."""
    logging.info(message)
    if transient:
        console.print(f"ℹ️ [blue bold] {message}[/]", end="\r", style="info")  # Apply style here
    else:
        console.print(f"ℹ️ [blue bold] {message}[/]", style="info")  # Apply style here


def log_warning(message: str) -> None:
    """Log warning messages with proper spacing."""
    logging.warning(message)
    console.print(f"⚠️ [yellow bold] {message}[/]", style="warning")  # Apply style here
    console.print()


# --- Performance Monitoring ---
_start_time = None
_memory_snapshots = []


def start_timer():
    """Start a global timer."""
    global _start_time
    _start_time = time.perf_counter()


def stop_timer(operation_name="Operation"):
    """Stop the timer and log the duration."""
    if _start_time is None:
        return 0.0
    elapsed_time = time.perf_counter() - _start_time
    log_info(f"{operation_name} completed in {elapsed_time:.4f} seconds.", transient=False)
    return elapsed_time


def snapshot_memory(label=""):
    """Take a snapshot of memory usage."""
    memory_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # in MB
    _memory_snapshots.append((label, memory_usage, time.time()))
    log_info(f"Memory Snapshot - {label}: {memory_usage:.2f} MB", transient=False)


def track_memory_delta(func):
    """Decorator to track memory usage before and after function execution."""

    def wrapper(*args, **kwargs):
        start_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        result = func(*args, **kwargs)
        end_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        delta = end_memory - start_memory
        log_info(f"Memory Delta - {func.__name__}: {delta:.2f} MB", transient=False)
        return result

    return wrapper


class TimingContext:
    """Context manager for hierarchical timing."""

    def __init__(self, operation_name):
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        log_info(f"Starting: {self.operation_name}...", transient=False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.perf_counter() - self.start_time
        log_info(f"Finished: {self.operation_name} in {elapsed_time:.4f} seconds.", transient=False)


# --- Circuit Breaker ---
driver_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
support_chat_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
send_message_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)


def shorten_order_id(order_id: str) -> str:
    """Shorten the order ID to the last 6 characters."""
    return order_id[-6:]


def print_order_status(order, current, total):
    """Print properly formatted order status."""
    status_text = Text()
    status_text.append("✓", style="success")  # Use style name instead of hardcoding
    status_text.append(f" Found eligible order ({current}/{total}): ", style="info")  # Use style name
    status_text.append(f"Order #{shorten_order_id(order.id)}", style="cyan")
    status_text.append(" | ", style="white")
    status_text.append("💰", style="accent")  # Use style name
    status_text.append(f" ${order.amount:.2f}", style="green")
    status_text.append(" | Status: ", style="white")
    status_text.append(
        "✅" if not order.cancelled else "❌", style="success" if not order.cancelled else "error"
    )  # Use styles

    console.print(status_text)


# Create undetectable chrome driver using Brave Browser
@driver_breaker
def create_driver():
    path = "chromedriver.exe"
    custom_user_agent = "Mozilla/5.0 (iPhone; CPU iPhone OS 16_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Mobile/15E148 Safari/604.1"
    options = uc.ChromeOptions()
    options.binary_location = "C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe"
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-infobars")
    options.add_argument("--mute-audio")
    options.add_argument("--user-agent=" + custom_user_agent)
    cookies_ext_path = os.path.join(os.path.dirname(__file__), "cookie_editor")
    options.add_argument(f"--load-extension={cookies_ext_path}")
    options.add_argument("--disable-brave-memory-saver")  # Disable Brave Memory Saver Popup
    options.add_argument("--suppress-message-center-popups")  # Try to suppress general popups
    options.add_argument("--disable-promotional-tabs")  # Disable promotional tabs on startup
    options.add_argument("--disable-features=Brave Shields,BraveNews")  # Try disabling Brave Shields and News features
    options.add_argument("--disable-blink-features=AutomationControlled")  # attempt to further reduce detection
    driver = uc.Chrome(executable_path=path, options=options)
    driver.set_window_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    time.sleep(STARTUP_DELAY)  # Wait for browser to stabilize after resizing
    return driver


def get_element_from_text(
    parent: uc.WebElement, tag_name: str, text: str, exact: bool = True, timeout: int = PAGE_LOAD_TIMEOUT
) -> Optional[uc.WebElement]:
    """Find element by tag and text, with a timeout and error handling."""
    try:
        wait = WebDriverWait(parent, timeout)
        all_tag_elements = wait.until(lambda driver: driver.find_elements(By.TAG_NAME, tag_name))
        for element in all_tag_elements:
            try:
                if (exact and element.text == text) or (not exact and text in element.text):
                    return element
            except StaleElementReferenceException:
                continue
        return None
    except Exception as e:
        log_warning(f"Error while trying to get element with text: {e}")
        return None


def get_element_from_attribute(
    parent: uc.WebElement, tag_name: str, attribute: str, value: str, timeout: int = PAGE_LOAD_TIMEOUT
) -> Optional[uc.WebElement]:
    """Find element by tag, attribute, and value, with error handling."""
    try:
        wait = WebDriverWait(parent, timeout)
        all_tag_elements = wait.until(lambda driver: driver.find_elements(By.TAG_NAME, tag_name))
        for element in all_tag_elements:
            try:
                attr_value = element.get_attribute(attribute)
                if attr_value and attr_value == value:
                    return element
            except StaleElementReferenceException:
                continue
        return None
    except Exception as e:
        log_warning(f"Error while trying to get element with attribute: {e}")
        return None


class Order:
    __slots__ = ["id", "receipt_url", "amount", "cancelled", "url"]

    def __init__(self, order_element: uc.WebElement):
        """Initialize Order with defaults and extract order details."""
        self.id = "unknown"
        self.receipt_url = None
        self.amount = 0.0
        self.cancelled = False
        self.url = None

        try:
            for attempt in range(MAX_RETRIES):
                try:
                    links = order_element.find_elements(By.TAG_NAME, "a")
                    if not links:
                        raise ValueError("No links found in order element")

                    self.receipt_url = links[-1].get_attribute("href")
                    if not self.receipt_url:
                        raise ValueError("Empty receipt URL")

                    self.id = self.receipt_url.split("/orders/")[-1].replace("/receipt/", "").split("?")[0]

                    amount_element = get_element_from_text(order_element, "span", " item", exact=False)
                    if not amount_element:
                        raise ValueError("Amount element not found")

                    amount_text = amount_element.text.split(" • ")[1].replace("$", "")
                    self.amount = float(amount_text)

                    self.cancelled = any(
                        [
                            bool(get_element_from_text(order_element, "span", text, exact=False))
                            for text in ["Order Cancelled", "Refund"]
                        ]
                    )

                    self.url = f"https://doordash.com/orders/{self.id}/help/"
                    break
                except (StaleElementReferenceException, ValueError, IndexError) as e:
                    if attempt == MAX_RETRIES - 1:
                        raise
                    log_warning(f"Retry {attempt + 1}/{MAX_RETRIES} for order {self.id}: {str(e)}")
                    time.sleep(RETRY_OPERATION_WAIT)
        except Exception as e:
            log_error(f"Failed to initialize order {self.id}: {str(e)}")

    def has_tip(self) -> bool:
        """Check if order has tip above threshold."""
        return bool(self.amount and self.amount > MAX_TIP_THRESHOLD)

    def __str__(self) -> str:
        """Enhanced string representation for Order class."""
        cancelled_status = "❌" if self.cancelled else "✅"
        return f"Order #{shorten_order_id(self.id)} | 💰 ${self.amount:.2f} | Status: {cancelled_status}"

    def get_remove_tip_message(self) -> str:
        """Generate the tip removal message for support."""
        global customer_name, customer_email  # Ensure these are correctly globally scoped

        options = [
            "Please remove the dasher tip to $0",
            "Hey, pls remove the tip and adjust it to $0",
            "Hi, i want you to remove whole dasher tip and make it $0",
            "Hey, remove full dasher tip and make it $0 pls. Application is glitching and it charged my card twice for the tip idk what is happening",
            "hey remove dasher's tip and adjust to $0",
        ]
        message = random.choice(options)

        if customer_name and customer_email:  # Check if customer info is available
            return f"{customer_name}\n\n{customer_email}\n\n{message}"
        else:
            log_warning("Customer name or email not available when creating message.")  # Log if data is missing
            return f"Could not retrieve customer info.\n\n{message}"  # Fallback message

    def remove_tip(
        self, driver: uc.Chrome, index: int, total: int, fast: bool = True, test_mode: bool = TEST_MODE
    ) -> None:
        """Open a new tab and initiate the tip removal process"""
        try:
            original_window = driver.current_window_handle
            driver.switch_to.new_window("tab")
            WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(lambda d: len(d.window_handles) > 1)

            self.open_support_chat(driver, fast=fast)

            time.sleep(SHORT_ACTION_WAIT)

            # First send the random tip removal message
            message = self.get_remove_tip_message()
            self.send_message_to_support(message, driver, fast=fast)
            time.sleep(FAST_ACTION_WAIT if fast else NORMAL_ACTION_WAIT)

            # Then send the test/agent message
            agent_message = "assadasfsfbafascascadae" if test_mode else "Agent"
            self.send_message_to_support(agent_message, driver, fast=fast)
            time.sleep(FAST_ACTION_WAIT if fast else NORMAL_ACTION_WAIT)

            driver.switch_to.window(original_window)
            # Use non-transient print for order completion message
            console.print(f"[yellow]Successfully processed order {shorten_order_id(self.id)} ({index} / {total})[/]")
            console.print()  # Add newline after order completion

        except CircuitBreakerError:
            log_error(f"Circuit breaker open for remove_tip on order {shorten_order_id(self.id)}")
            raise
        except Exception as e:
            log_error(f"Error during remove_tip for order {shorten_order_id(self.id)}: {str(e)}")
            raise

    @send_message_breaker
    def send_message_to_support(self, message: str, driver: uc.Chrome, fast: bool = True) -> None:
        """Send a message to DoorDash support with proper formatting and retries."""
        message_wait = FAST_MESSAGE_WAIT if fast else NORMAL_MESSAGE_WAIT

        def try_send_message():
            text_input_element = WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(
                EC.presence_of_element_located((By.TAG_NAME, "textarea"))
            )
            WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(EC.element_to_be_clickable((By.TAG_NAME, "textarea")))

            # Ensure element is in view and interactable
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", text_input_element)
            time.sleep(SHORT_ACTION_WAIT)

            text_input_element.clear()
            time.sleep(SHORT_ACTION_WAIT)

            lines = message.split("\n")
            for i, line in enumerate(lines):
                try:
                    text_input_element.send_keys(line)
                except ElementNotInteractableException:
                    driver.execute_script("arguments[0].value = arguments[1];", text_input_element, line)
                if i < len(lines) - 1:
                    ActionChains(driver).key_down(Keys.SHIFT).send_keys(Keys.ENTER).key_up(Keys.SHIFT).perform()
                time.sleep(SHORT_ACTION_WAIT)

            try:
                text_input_element.send_keys(Keys.RETURN)
            except ElementNotInteractableException:
                driver.execute_script(
                    "arguments[0].dispatchEvent(new KeyboardEvent('keypress', {'key': 'Enter'}))", text_input_element
                )

        for attempt in range(MAX_RETRIES):
            try:
                try_send_message()
                time.sleep(message_wait)
                return
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise Exception(f"Failed to send message after {MAX_RETRIES} attempts: {str(e)}")
                time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
                continue

    @support_chat_breaker
    def open_support_chat(self, driver: uc.Chrome, fast: bool = True) -> None:
        """Open the support chat using robust element finding and clicks."""
        button_wait = FAST_BUTTON_WAIT if fast else NORMAL_BUTTON_WAIT

        try:
            driver.get(self.url)
            log_info(f"Opened {self.url}")

            log_info(f"Getting user info for order: {shorten_order_id(self.id)}")
            time.sleep(1.5)

            something_else_xpath = '//button[@aria-label="It\'s something else"]'
            contact_support_xpath = "//button[@aria-label='Contact support']"

            for attempt in range(MAX_RETRIES):
                try:
                    something_else_element = WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(
                        EC.presence_of_element_located((By.XPATH, something_else_xpath))
                    )
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", something_else_element)
                    time.sleep(SHORT_ACTION_WAIT)

                    try:
                        something_else_element.click()
                    except (ElementClickInterceptedException, ElementNotInteractableException):
                        driver.execute_script("arguments[0].click();", something_else_element)

                    log_debug(
                        f"Clicked 'It's something else' for order: {shorten_order_id(self.id)}", transient=True
                    )  # Reduced logging verbosity
                    break
                except (TimeoutException, StaleElementReferenceException) as e:
                    log_warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(button_wait)
                        continue
                    log_error("Failed to click first button after all retries")
                    return

            for attempt in range(MAX_RETRIES):
                try:
                    contact_support_element = WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(
                        EC.presence_of_element_located((By.XPATH, contact_support_xpath))
                    )
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", contact_support_element)
                    time.sleep(SHORT_ACTION_WAIT)

                    try:
                        contact_support_element.click()
                    except (ElementClickInterceptedException, ElementNotInteractableException):
                        driver.execute_script("arguments[0].click();", contact_support_element)

                    log_debug(
                        f"Clicked 'Contact support' for order: {shorten_order_id(self.id)}", transient=True
                    )  # Reduced logging
                    break
                except (TimeoutException, StaleElementReferenceException) as e:
                    log_warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(button_wait)
                        continue
                    log_error("Failed to click second button after all retries")
                    return
        except CircuitBreakerError:
            log_error(f"Circuit breaker open for open_support_chat on order {shorten_order_id(self.id)}")
            raise
        except Exception as e:
            log_error(f"Error in open_support_chat for order {shorten_order_id(self.id)}: {str(e)}")


@track_memory_delta
def process_single_order(driver: uc.Chrome, order: Order) -> bool:
    """Process a single order and return success status."""
    try:
        order.open_support_chat(driver, fast=True)
        time.sleep(SHORT_ACTION_WAIT)

        message = order.get_remove_tip_message()
        order.send_message_to_support(message, driver, fast=True)
        time.sleep(FAST_ACTION_WAIT)

        agent_message = "assadasfsfbafascascadae" if TEST_MODE else "Agent"
        order.send_message_to_support(agent_message, driver, fast=True)
        return True
    except CircuitBreakerError:
        log_error(f"Circuit breaker open in process_single_order for order {order.id}")
        return False
    except Exception as e:
        log_error(f"Error processing order {order.id}: {str(e)}")
        return False


@track_memory_delta
def get_orders(driver: uc.Chrome, max_orders: int = MAX_ORDERS_PER_BATCH) -> List[Order]:
    """Retrieve orders with improved loading and error handling."""
    console.print("")
    log_info("Starting order collection...")
    snapshot_memory("Before getting orders")

    try:
        driver.get(url="https://www.doordash.com/orders")
        time.sleep(SCROLL_WAIT)

        scroll_count = 0
        last_height = driver.execute_script("return document.body.scrollHeight")
        no_change_count = 0
        MAX_NO_CHANGE = 3  # Maximum times we allow height to stay the same

        with Progress() as progress:
            scroll_task = progress.add_task("Loading orders...", total=None)

            while scroll_count < MAX_SCROLL_ATTEMPTS:
                try:
                    # Scroll to bottom
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight); ")
                    time.sleep(SCROLL_WAIT)

                    # Try to find and click "Load More"
                    try:
                        load_more = WebDriverWait(driver, 2).until(
                            EC.presence_of_element_located((By.XPATH, "//span[contains(text(), 'Load More')]"))
                        )
                        driver.execute_script("arguments[0].scrollIntoView({ block: 'center' }); ", load_more)
                        load_more.click()
                        scroll_count += 1
                        no_change_count = 0  # Reset counter on successful click
                        progress.update(scroll_task, description=f"Loading orders (page {scroll_count})...")
                        continue
                    except TimeoutException:
                        # Check if we've reached the end
                        new_height = driver.execute_script("return document.body.scrollHeight")
                        if new_height == last_height:
                            no_change_count += 1
                            if no_change_count >= MAX_NO_CHANGE:
                                log_info("Reached end of order list")
                                break
                        else:
                            no_change_count = 0
                            last_height = new_height
                            continue

                except Exception as e:
                    log_warning(f"Scroll error: {str(e)}")
                    break

        # Verify orders were loaded
        try:
            wait = WebDriverWait(driver, PARALLEL_TIMEOUT)
            completed_span = wait.until(EC.presence_of_element_located((By.XPATH, "//span[text()='Completed']")))

            orders_container = completed_span.find_element(By.XPATH, "..")
            orders_div = orders_container.find_element(By.XPATH, "./div[last()]")
            all_order_elements = orders_div.find_elements(By.XPATH, "./*")

            if not all_order_elements:
                log_warning("No order elements found - retrying page load")
                driver.refresh()
                time.sleep(2)
                return get_orders(driver, max_orders)  # Recursive retry

            orders = process_orders_in_parallel(driver, all_order_elements)
            snapshot_memory("After processing orders in parallel")
            return orders[:max_orders]

        except Exception as e:
            log_error(f"Error getting order elements: {str(e)}")
            return []

    except Exception as e:
        log_error(f"Error in order collection: {str(e)}")
        return []


def process_orders_in_parallel(driver: uc.Chrome, elements: List[uc.WebElement]) -> List[Order]:
    """Process order elements in parallel and return eligible orders."""
    processor = OrderProcessor()
    order_stats = OrderStats()

    with processor.progress:
        # Clear line after progress bar
        task = processor.progress.add_task("[bold]Initializing orders in parallel...", total=len(elements))

        eligible_orders = []
        for element in elements:
            order = Order(element)
            if order and order.has_tip() and not order.cancelled:
                eligible_orders.append(order)
            processor.progress.update(task, advance=1)

        # Add padding between progress and summary
        console.print("\n")

        order_stats.total_orders = len(elements)
        order_stats.eligible_orders = len(eligible_orders)
        order_stats.print_summary()

        for idx, order in enumerate(eligible_orders, 1):
            console.print(
                f"✅ Found eligible order ({idx} / {len(eligible_orders)}): "
                f"Order #{order.id} | 💰 ${order.amount:.2f} | Status: ✅"
            )

    # Clear final progress message
    console.print("\n", end="")
    return eligible_orders


def countdown_timer(seconds: int, message: str) -> None:
    """Display a countdown timer with a message."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task(message, total=seconds)
        for _ in range(seconds):
            time.sleep(1)
            progress.update(task, advance=1)


class SupportAgent:
    __slots__ = [
        "AGENT_CHECK_INTERVAL",
        "MAX_AGENT_CHECKS",
        "AGENT_NAME_PATTERN",
        "AGENT_MESSAGE_TEMPLATE",
        "MAX_SEND_RETRIES",
        "SEND_RETRY_DELAY",
    ]
    """Handle interactions with support agents."""

    def __init__(self):
        self.AGENT_CHECK_INTERVAL = 2
        self.MAX_AGENT_CHECKS = 15
        self.AGENT_NAME_PATTERN = r"You are now connected to our support agent[:\s]+([A-Za-z]+)"
        self.AGENT_MESSAGE_TEMPLATE = (
            "Hi {agent_name},\n" "Yes please remove entire dasher tip to $0.\n" "Thank you, {agent_name}"
        )
        self.MAX_SEND_RETRIES = 3
        self.SEND_RETRY_DELAY = 1

    def _extract_agent_name(self, text: str) -> Optional[str]:
        """Extract agent name with improved error handling."""
        if not text:
            return None
        try:
            match = re.search(self.AGENT_NAME_PATTERN, text)
            if match:
                name = match.group(1).strip()
                return name if name else None
            return None
        except Exception as e:
            log_warning(f"Error extracting agent name: {str(e)}")
            return None

    def _send_agent_message(self, driver: uc.Chrome, message: str) -> bool:
        """Enhanced message sending with retries and validation."""
        for attempt in range(self.MAX_SEND_RETRIES):
            try:
                text_area = WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(
                    EC.presence_of_element_located((By.TAG_NAME, "textarea"))
                )
                WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(EC.element_to_be_clickable((By.TAG_NAME, "textarea")))

                text_area.clear()
                time.sleep(SHORT_ACTION_WAIT)

                lines = message.split("\n")
                for i, line in enumerate(lines):
                    text_area.send_keys(line)
                    if i < len(lines) - 1:
                        ActionChains(driver).key_down(Keys.SHIFT).send_keys(Keys.ENTER).key_up(Keys.SHIFT).perform()
                    time.sleep(SHORT_ACTION_WAIT)

                text_area.send_keys(Keys.RETURN)
                return True
            except Exception as e:
                log_warning(f"Send attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.MAX_SEND_RETRIES - 1:
                    time.sleep(self.SEND_RETRY_DELAY)
                    continue
                return False
        return False

    def _handle_agent_interaction(self, driver: uc.Chrome, order_id: str) -> bool:
        """Enhanced agent interaction with improved detection and messaging."""
        for check in range(self.MAX_AGENT_CHECKS):
            try:
                reconnect_button = get_element_from_text(driver, "span", "Chat with an agent", exact=False)
                if reconnect_button:
                    reconnect_button.click()
                    log_success(f"Clicked reconnect in {shorten_order_id(order_id)}", transient=True)
                    time.sleep(SHORT_ACTION_WAIT)
                    continue

                agent_spans = driver.find_elements(
                    By.XPATH, "//span[contains(text(), 'You are now connected to our support agent')]"
                )
                for span in agent_spans:
                    agent_name = self._extract_agent_name(span.text)
                    if agent_name:
                        message = self.AGENT_MESSAGE_TEMPLATE.format(agent_name=agent_name)
                        self._send_agent_message(driver, message)
                        log_success(f"Sent followup to agent {agent_name} in {shorten_order_id(order_id)}")
                        return True

                time.sleep(self.AGENT_CHECK_INTERVAL)
            except Exception as e:
                log_warning(f"Agent interaction error in {shorten_order_id(order_id)}: {str(e)}")
                time.sleep(self.AGENT_CHECK_INTERVAL)

        return False

    def process_batch(self, driver: uc.Chrome, orders: List[Order]) -> None:
        """Process a batch of orders with comprehensive agent interaction handling."""
        original_handle = driver.current_window_handle
        order_handles = {}

        try:
            for order in orders:
                driver.switch_to.new_window("tab")
                current_handle = driver.current_window_handle
                order_handles[current_handle] = order.id
                order.open_support_chat(driver, fast=True)

            countdown_timer(RECONNECT_TIMEOUT, "Establishing initial connections")

            reconnection_status = {}
            for handle, order_id in order_handles.items():
                try:
                    driver.switch_to.window(handle)
                    success = self._handle_agent_interaction(driver, order_id)
                    reconnection_status[order_id] = success
                    if not success:
                        log_warning(f"Could not complete agent interaction for {shorten_order_id(order_id)}")
                except Exception as e:
                    reconnection_status[order_id] = False
                    log_error(f"Error processing {shorten_order_id(order_id)}: {str(e)}")

            log_info("Checking agent connections and sending follow-ups...")
            for handle, order_id in order_handles.items():
                if not reconnection_status.get(order_id, False):
                    try:
                        driver.switch_to.window(handle)
                        if self.check_agent_presence(driver):  # Replaced with AgentReconnectionManager check
                            self.send_followup(driver, order_id)  # Replaced with AgentReconnectionManager followup
                            reconnection_status[order_id] = True
                            log_success(f"Successfully reconnected with agent for {shorten_order_id(order_id)}")
                    except Exception as e:
                        log_warning(f"Reconnection failed for {shorten_order_id(order_id)}: {str(e)}")

            countdown_timer(RECONNECT_TIMEOUT, "Finalizing agent interactions")

            successful_reconnects = sum(1 for success in reconnection_status.values() if success)
            log_info(f"Successfully reconnected with {successful_reconnects}/{len(order_handles)} agents")

            reconnection_manager = AgentReconnectionManager()
            all_reconnected = reconnection_manager.process_reconnections(driver, order_handles)

            if all_reconnected:
                for handle in list(order_handles.keys()):
                    driver.switch_to.window(handle)
                    driver.close()
            else:
                console.print("[yellow]Some reconnections pending - keeping tabs open[/]")

            driver.switch_to.window(original_handle)
            log_success("Batch processing completed")

            with Progress() as progress:
                progress.add_task("[yellow]Finalizing before cleanup...", total=MIN_AGENT_WAIT, start=True)
                for _ in range(MIN_AGENT_WAIT):
                    time.sleep(1)
        except CircuitBreakerError:
            log_error("Circuit breaker open in SupportAgent.process_batch")
            driver.switch_to.window(original_handle)
        except Exception as e:
            log_error(f"Critical error in batch processing: {str(e)}")
            driver.switch_to.window(original_handle)


class AgentReconnectionManager:
    __slots__ = ["reconnection_status", "pending_followups"]
    """Manage agent reconnections and follow-up messages."""

    def __init__(self):
        self.reconnection_status = {}
        self.pending_followups = set()

    def check_agent_presence(self, driver: uc.Chrome) -> bool:
        """Check if an agent is present in the chat."""
        try:
            agent_element = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".agent-name"))
            )
            return bool(agent_element)
        except TimeoutException:
            return False

    def send_followup(self, driver: uc.Chrome, order_id: str) -> bool:
        """Send a follow-up message to the agent."""
        try:
            chat_input = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "textarea.chat-input"))
            )
            chat_input.send_keys("Thank you for your help with this request!")
            chat_input.send_keys(Keys.RETURN)
            return True
        except Exception:
            return False

    def process_reconnections(self, driver: uc.Chrome, order_handles: dict) -> bool:
        """Process agent reconnections and send follow-ups."""
        console.print("\n[bold cyan]━━━ Reconnecting Agents ━━━[/bold cyan]")
        original_handle = driver.current_window_handle

        with Progress() as progress:
            reconnect_task = progress.add_task("[cyan]Checking agent connections...", total=len(order_handles))

            for handle, order_id in order_handles.items():
                try:
                    driver.switch_to.window(handle)
                    if self.check_agent_presence(driver):
                        self.send_followup(driver, order_id)
                        self.reconnection_status[order_id] = True
                        console.print(f"[green]✓ Agent reconnected for Order #{shorten_order_id(order_id)}[/]")
                    else:
                        self.pending_followups.add(order_id)
                        console.print(f"[yellow]! Pending reconnect for Order #{shorten_order_id(order_id)}[/]")
                except Exception as e:
                    log_error(f"Reconnection error for {order_id}: {str(e)}")
                    progress.update(reconnect_task, advance=1)

        successful = len([x for x in self.reconnection_status.values() if x])
        console.print("\n[bold]Reconnection Summary[/bold]")
        console.print(f"✓ Successfully reconnected: {successful}/{len(order_handles)}")
        if self.pending_followups:
            console.print(f"! Pending followups: {len(self.pending_followups)}")

        driver.switch_to.window(original_handle)
        return successful == len(order_handles)


def clean_string(s: str) -> str:
    """Clean string by removing special characters and spaces."""
    return re.sub(r"[^a-zA-Z0-9]", "", s)


def is_valid_cookie_file(file_path: str) -> bool:
    """Check if cookie file exists and is not empty."""
    try:
        path = Path(file_path)
        return path.exists() and path.stat().st_size > 0
    except Exception as e:
        logging.error(f"Error checking cookie file: {str(e)}")
        return False


def create_manual_cookies_backup() -> bool:
    """Create backup of cookies file with timestamp for manual login."""
    try:
        if not is_valid_cookie_file("cookies.pkl"):
            logging.info("No valid cookies file to backup")
            return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"manual_login_{timestamp}.pkl"
        backup_path = Path("cookiesBAK") / backup_name

        shutil.copy2("cookies.pkl", backup_path)
        os.remove("cookies.pkl")
        logging.info(f"Created manual login cookies backup: {backup_name}")
        return True
    except Exception as e:
        logging.error(f"Failed to create manual login cookies backup: {str(e)}")
        return False


def create_cookies_backup(first_name: str, last_name: str, email: str) -> bool:
    """Create backup of cookies file with customer info and timestamp."""
    try:
        if not is_valid_cookie_file("cookies.pkl"):
            logging.info("No valid cookies file to backup")
            return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_fname = clean_string(first_name)
        clean_lname = clean_string(last_name)
        clean_email = clean_string(email)

        backup_name = f"ficookies_{clean_fname}_{clean_lname}_{clean_email}_{timestamp}.pkl"
        backup_path = Path("cookiesBAK") / backup_name

        shutil.copy2("cookies.pkl", backup_path)
        logging.info(f"Created cookies backup: {backup_name}")
        return True
    except Exception as e:
        logging.error(f"Failed to create cookies backup: {str(e)}")
        return False


def create_cookies_directory() -> None:
    """Create cookies backup directory if it doesn't exist."""
    Path("cookiesBAK").mkdir(exist_ok=True)


def save_cookies_with_backup(cookies_obj, first_name: str, last_name: str, email: str) -> None:
    """Save cookies to file and create backup."""
    if cookies_obj:
        with open("cookies.pkl", "wb") as f:
            pickle.dump(cookies_obj, f)

    if is_valid_cookie_file("cookies.pkl"):
        create_cookies_backup(first_name, last_name, email)
    else:
        logging.warning("No cookie data to save")


def backup_and_wipe_cookies(first_name: str, last_name: str, email: str) -> None:
    """Create backup of cookies before wiping them."""
    if is_valid_cookie_file("cookies.pkl"):
        create_cookies_backup(first_name, last_name, email)
        os.remove("cookies.pkl")


def wait_for_profile_page(driver: uc.Chrome, max_attempts=9999) -> bool:
    """Wait for user to reach profile page or home page after manual login."""
    profile_url = "https://www.doordash.com/consumer/edit_profile/"
    attempts = 0

    while attempts < max_attempts:
        url = driver.current_url

        if (url == profile_url) or ("doordash.com" in url and "/home" in url):
            return True

        elif "doordash.com" in url and "action=Login" in url and "/home/" not in url:
            try:
                sign_in = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//span[contains(text(), 'Sign In')]"))
                )
                sign_in.click()
            except (TimeoutException, ElementClickInterceptedException, ElementNotInteractableException) as e:
                logging.debug(f"Sign in element not found or not clickable: {str(e)}")
                pass

        time.sleep(0.25)
        attempts += 1

    return False


# Add message tracking
_last_message = None
_last_message_time = 0


def notify_cookie_save_status(success: bool, backup_success: bool, manual: bool = False) -> None:
    """Display cookie save status notifications with duplicate prevention."""
    global _last_message, _last_message_time

    login_type = "Manual" if manual else "Automated"
    current_time = time.time()

    if success and backup_success:
        message = "Cookies saved successfully"
    elif success:
        message = "Cookies saved to cookies.pkl"
    else:
        message = f"{login_type} Login: Failed to save cookies"

    if message != _last_message or (current_time - _last_message_time) > 2:
        if success and backup_success:
            log_success(message, transient=False)
        elif success:
            log_success(message, transient=True)
            log_error("Backup creation failed")
        else:
            log_error(message)

        _last_message = message
        _last_message_time = current_time


def save_cookies_after_login(driver: uc.Chrome, manual: bool = False, customer_info: dict = None) -> None:
    """Save cookies after successful login with single notification."""
    create_cookies_directory()

    try:
        cookies = driver.get_cookies()
        success = False
        backup_success = False

        if cookies:
            with open("cookies.pkl", "wb") as f:
                pickle.dump(cookies, f)
            success = True

        if manual:
            backup_success = create_manual_cookies_backup()
        elif customer_info:
            backup_success = create_cookies_backup(
                customer_info["first_name"], customer_info["last_name"], customer_info["email"]
            )

        notify_cookie_save_status(success, backup_success, manual)

    except Exception as e:
        logging.error(f"Cookie save error: {str(e)}")
        notify_cookie_save_status(False, False, manual)


@track_memory_delta
def monitor_orders_auto(driver: uc.Chrome, check_interval: int = RECONNECT_TIMEOUT) -> None:
    """Monitor orders with delayed tab cleanup."""
    try:
        console.print("\n[bold cyan]━━━ Order Monitoring Active ━━━[/]")
        console.print("[dim]Press Ctrl+C to stop monitoring[/dim]\n")
        snapshot_memory("Start monitoring")

        session.active = True
        while session.active:
            with TimingContext("Monitoring Cycle"):
                snapshot_memory("Start cycle - Before tab cleanup")
                tab_manager.cleanup_tabs(driver)  # Cleanup tabs from previous cycle
                snapshot_memory("After tab cleanup")

                with TimingContext("Get Orders"):
                    orders = get_orders(driver)
                    snapshot_memory("After get_orders")

                if orders:
                    eligible_orders = [o for o in orders if o.has_tip() and not o.cancelled]
                    if eligible_orders:
                        console.print("\n[bold green]━━━ Processing Eligible Orders ━━━[/]")
                        with TimingContext("Process Orders Batch"):
                            if process_orders_batch(driver, eligible_orders):
                                console.print("\n[bold green]Orders processed successfully![/]")
                                console.print("\n[yellow]Returning to monitoring mode...[/]")
                                snapshot_memory("After process_orders_batch")
                            else:
                                console.print("\n[yellow]No eligible orders in current batch[/]")
                    else:
                        console.print("\n[yellow]No eligible orders found[/]")

                with session.display_lock:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                        TextColumn("\n"),  # Add newline before countdown
                        TextColumn(f"[{MAIN_COLOR}]{{task.fields[time_left]}}s remaining[/{MAIN_COLOR}]"),
                    ) as progress:
                        countdown = progress.add_task(
                            "[yellow]Next check in", total=check_interval, time_left=check_interval
                        )
                        for remaining in range(check_interval, 0, -1):
                            if not session.active:
                                break
                            progress.update(countdown, advance=1, time_left=remaining - 1)
                            time.sleep(1)
                snapshot_memory("End monitoring cycle")

    except KeyboardInterrupt:
        session.cleanup()
        tab_manager.cleanup_tabs(driver)  # Final cleanup on exit
        console.print("\n[bold red]Monitoring stopped by user (Ctrl+C)[/bold red]\n")
    except CircuitBreakerError:
        session.cleanup()
        tab_manager.cleanup_tabs(driver)  # Final cleanup on CircuitBreaker error
        log_error("Circuit breaker open in monitor_orders_auto, monitoring stopped.")
    except Exception as e:
        session.cleanup()
        tab_manager.cleanup_tabs(driver)  # Final cleanup on error
        log_error(f"Error in monitoring loop: {e}")
        raise


def process_orders_batch(driver: uc.Chrome, orders: List[Order]) -> bool:
    """
    Processes a batch of orders.

    Args:
    - driver (uc.Chrome): The Chrome driver instance.
    - orders (List[Order]): A list of Order objects.

    Returns:
    - bool: True if the batch was processed successfully, False otherwise.
    """
    if not orders:
        return False

    processed = False
    main_window = driver.current_window_handle
    tab_pool = []

    try:
        progress_manager.start(len(orders), "Processing orders...")

        for _ in range(min(TAB_POOL_SIZE, len(orders))):
            driver.switch_to.new_window("tab")
            tab_pool.append(driver.current_window_handle)

        for i, order in enumerate(orders):
            tab = tab_pool[i % len(tab_pool)]
            driver.switch_to.window(tab)

            try:
                success = process_single_order(driver, order)
                if success:
                    processed = True
                    progress_manager.update()
            except CircuitBreakerError:
                log_error(f"Circuit breaker open while processing order {order.id}")
                continue
            except Exception as e:
                log_error(f"An error occurred while processing order {order.id}: {str(e)}")
                continue

            time.sleep(TAB_WAIT)

        # Set tabs for cleanup in next cycle
        tab_manager.set_pending_tabs(tab_pool, main_window)
        driver.switch_to.window(main_window)
        progress_manager.stop()

        return processed

    except CircuitBreakerError:
        log_error("Circuit breaker open in process_orders_batch")
        try:
            driver.switch_to.window(main_window)
        except Exception:
            pass
        progress_manager.stop()
        return False
    except Exception as e:
        log_error(f"An error occurred while processing batch: {str(e)}")
        try:
            driver.switch_to.window(main_window)
        except Exception:
            pass
        progress_manager.stop()
        return False


def send_chat_with_retry(driver: uc.Chrome, order: Order, message: str) -> bool:
    """Send chat message with retries and deduplication."""
    message_key = f"{order.id}:{message}"
    if message_key in session.sent_messages:
        return True

    original_window = driver.current_window_handle
    for attempt in range(MAX_CHAT_RETRIES):
        try:
            # Verify window still exists
            try:
                driver.current_window_handle
            except BaseException:
                log_warning(f"Window lost for order {order.id}, switching to original")
                driver.switch_to.window(original_window)
                return False

            order.open_support_chat(driver, fast=True)
            time.sleep(CHAT_RETRY_DELAY)
            order.send_message_to_support(message, driver, fast=True)
            session.sent_messages.add(message_key)
            return True
        except CircuitBreakerError:
            log_error(f"Circuit breaker open in send_chat_with_retry for order {order.id}")
            return False
        except Exception as e:
            if attempt == MAX_CHAT_RETRIES - 1:
                log_error(f"Failed to send message to {order.id}: {str(e)}")
                return False
            time.sleep(RETRY_DELAY)
            try:
                driver.switch_to.window(original_window)
            except BaseException:
                pass
            continue
    return False


def process_agent_messages(driver: uc.Chrome, orders: List[Order], message: str) -> None:
    """Process agent messages in parallel."""
    error_map = {
        "TimeoutException": "Connection timeout - retrying",
        "WebDriverException": "Browser error - reconnecting",
        "ElementNotInteractableException": "Element not ready - waiting",
        "StaleElementReferenceException": "Page changed - reloading",
    }

    def handle_message_error(order_id: str, error: Exception, attempt: int) -> None:
        error_type = error.__class__.__name__
        error_msg = error_map.get(error_type, "Operation failed")

        if attempt < MAX_CHAT_RETRIES:
            log_info(f"Retrying order {order_id}: {error_msg}")
        else:
            log_error(f"Failed to process order {order_id} after {MAX_CHAT_RETRIES} attempts")

        logging.error(
            f"""
            Order: {order_id}
            Error: {error_type}
            Details: {str(error)}
            Attempt: {attempt + 1}/{MAX_CHAT_RETRIES}
            Timestamp: {datetime.now()}
        """.strip()
        )

    def send_message_safe(order: Order) -> bool:
        for attempt in range(MAX_CHAT_RETRIES):
            try:
                message_key = f"{order.id}:{message}"
                if message_key in session.sent_messages:
                    return True

                order.open_support_chat(driver, fast=True)
                time.sleep(CHAT_RETRY_DELAY)
                order.send_message_to_support(message, driver, fast=True)
                session.sent_messages.add(message_key)
                return True

            except CircuitBreakerError:
                log_error(f"Circuit breaker open in send_message_safe for order {order.id}")
                return False
            except Exception as e:
                handle_message_error(order.id, e, attempt)
                if attempt == MAX_CHAT_RETRIES - 1:
                    return False
                time.sleep(RETRY_DELAY)
                continue
        return False

    try:
        with ThreadPoolExecutor(max_workers=CHAT_BATCH_SIZE) as executor:
            futures = [executor.submit(send_message_safe, order) for order in orders]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Thread pool error: {str(e)}")

    except Exception as e:
        logging.error(f"Critical error in message processor: {str(e)}")
        log_error("Message processing failed - please try again")


def process_orders(elements: List[uc.WebElement]) -> List[Order]:
    """Process order elements and return valid orders.

    Args:
        elements: List of WebElements to process

    Returns:
        List[Order]: List of processed valid orders
    """
    total = len(elements)
    orders = []

    with Progress(...) as progress:
        task = progress.add_task("Processing orders...", total=total)

        for elem in elements:
            try:
                order = Order(elem)
                if order.has_tip() and not order.cancelled:
                    orders.append(order)
                    console.print(f"[green]✓ Found eligible order: {str(order)}[/green]")
            except Exception as e:
                logging.debug(f"Failed to process order: {str(e)}")
            progress.advance(task)

        console.print("\n[bold cyan]Order Discovery Summary[/bold cyan]")
        console.print(f"[cyan]{'─' * 30}[/cyan]")
        log_info(f"Total orders scanned: {total}", transient=False)

    return orders  # Add this return statement to use the collected orders


def get_progress_bar_style() -> tuple:
    """Returns a tuple of style configurations for progress bars."""
    return (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style=MAIN_COLOR, finished_style=ACCENT_COLOR),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    )


class OrderProcessor:
    __slots__ = ["progress", "_lock", "console"]
    """Class to handle order processing with progress tracking."""

    def __init__(self):
        self.progress = Progress(*get_progress_bar_style(), refresh_per_second=10, transient=False)
        self._lock = Lock()
        self.console = Console()

    def print_status(self, message: str, style: str = "default") -> None:
        """Print status message above progress bar."""
        with self._lock:
            print("\033[s", end="")
            print("\033[2A", end="")
            print(f"\033[K{message}")
            print("\033[u", end="")

    def process_orders(self, orders: List[Order]) -> None:
        """Process orders with progress display."""
        with self.progress:
            task = self.progress.add_task("Processing orders...", total=len(orders))

            for order in orders:
                self.print_status(f"[green]✓ Processing: {str(order)}[/green]")

                try:
                    # Process order here
                    pass  # Placeholder for actual order processing
                except Exception as e:
                    self.print_status(f"[red]✗ Error processing {str(order)}: {str(e)}[/red]")
                self.progress.update(task, advance=1)
                time.sleep(0.1)

    def process_batch(self, orders: List[Order], parallel: bool = True) -> None:
        """Process a batch of orders with proper progress display."""
        with self.progress:
            task = self.progress.add_task(
                f"{'Parallel' if parallel else 'Sequential'} processing...", total=len(orders)
            )

            if parallel:
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = {executor.submit(self.process_single_order, order): order for order in orders}

                    for future in as_completed(futures):
                        order = futures[future]
                        try:
                            result = future.result()
                            status = "[green]✓[/green]" if result else "[red]✗[/red]"
                            self.print_status(f"{status} {str(order)}")
                        except CircuitBreakerError:
                            self.print_status(f"[yellow]! Circuit Breaker Open for {str(order)}[/yellow]")
                        except Exception as e:
                            self.print_status(f"[red]✗ Error: {str(e)}[/red]")
                        finally:
                            self.progress.update(task, advance=1)
            else:
                for order in orders:
                    self.process_single_order(order)
                    self.progress.update(task, advance=1)

    def process_single_order(self, driver: uc.Chrome, order: Order) -> bool:
        """Process a single order and return success status."""
        try:
            order.open_support_chat(driver, fast=True)
            time.sleep(SHORT_ACTION_WAIT)

            message = order.get_remove_tip_message()
            order.send_message_to_support(message, driver, fast=True)
            time.sleep(FAST_ACTION_WAIT)

            agent_message = "assadasfsfbafascascadae" if TEST_MODE else "Agent"
            order.send_message_to_support(agent_message, driver, fast=True)
            return True
        except CircuitBreakerError:
            log_error(f"Circuit breaker open in OrderProcessor.process_single_order for order {order.id}")
            return False
        except Exception as e:
            log_error(f"Error processing order {order.id}: {str(e)}")
            return False


def send_messages_in_parallel(
    driver: uc.Chrome, orders: List[Order], message_template: str, fast: bool = True
) -> List[bool]:
    """Send messages to multiple orders in parallel with deduplication."""
    results = []
    sent_messages = set()

    def send_single_message(order):
        for attempt in range(MAX_RETRIES):
            try:
                message_key = f"{order.id}:{message_template}"
                if message_key in sent_messages:
                    time.sleep(MESSAGE_DEDUPE_DELAY)

                order.open_support_chat(driver, fast=fast)
                time.sleep(SHORT_ACTION_WAIT)

                if message_key not in sent_messages:
                    order.send_message_to_support(message_template, driver, fast=fast)
                    sent_messages.add(message_key)
                if fast:
                    time.sleep(FAST_ACTION_WAIT)
                else:
                    time.sleep(NORMAL_ACTION_WAIT)
                return True
            except CircuitBreakerError:
                log_warning(f"Circuit breaker open in send_messages_in_parallel for order {order.id}", transient=True)
                return False
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    log_warning(f"Retrying message send for {order.id}: {str(e)}", transient=True)
                    time.sleep(RETRY_DELAY)
                    continue
                log_error(f"Failed to send message to {order.id}")
                return False
        return False

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        task = progress.add_task("[cyan]Sending messages...", total=len(orders))

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_order = {executor.submit(send_single_message, order): order for order in orders}

        for future in as_completed(future_to_order):
            order = future_to_order[future]
            try:
                success = future.result()
                results.append(success)
            except Exception as e:
                log_error(f"Error processing {order.id}: {str(e)}")
                results.append(False)
            finally:
                progress.advance(task)

    return results


class SessionState:
    __slots__ = ["display_lock", "chat_queue", "active", "sent_messages"]
    """Manage global session state and resources."""

    def __init__(self):
        self.display_lock = Lock()
        self.chat_queue = Queue()
        self.active = True
        self.sent_messages = set()

    def cleanup(self):
        """Clean up session resources."""
        self.active = False
        with self.display_lock:
            self.chat_queue = Queue()
            self.sent_messages.clear()


session = SessionState()


class ProgressManager:
    __slots__ = ["progress", "task_id", "_lock"]
    """Manage global progress display."""

    def __init__(self):
        self.progress = None
        self.task_id = None
        self._lock = threading.Lock()

    def start(self, total: int, description: str = "Processing orders...") -> None:
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=50),  # Set a fixed width for the bar
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
            transient=True,
        )
        self.progress.start()
        self.task_id = self.progress.add_task(description, total=total)

    def update(self, advance: int = 1) -> None:
        with self._lock:
            if self.progress:
                self.progress.update(self.task_id, advance=advance)

    def stop(self) -> None:
        if self.progress:
            self.progress.stop()


progress_manager = ProgressManager()


def print_progress(progress_text: str, percentage: float):
    """Print progress bar at bottom of terminal."""
    print(f"\033[K{progress_text}", end="\r")


class OrderStats:
    __slots__ = ["total_orders", "eligible_orders"]
    """Class to track and display order statistics."""

    def __init__(self):
        self.total_orders = 0
        self.eligible_orders = 0

    def print_summary(self):
        """Print summary of orders found and eligible for processing."""
        summary_table = Table(
            title="[bold]Order Summary[/bold]",
            title_style=f"bold {MAIN_COLOR}",
            border_style=MAIN_COLOR,
            padding=(1, 2),
            show_header=True,  # Make sure headers are shown
            header_style="bold white",  # Style for headers
            row_styles=["none", "dim"],  # Alternating row styles for better readability
            expand=False,  # Tables to not expand to full console width
        )
        summary_table.add_column("Metric", style="white", no_wrap=True)
        summary_table.add_column("Value", style="cyan", justify="right")
        summary_table.add_row("Orders Found", str(self.total_orders))
        summary_table.add_row("Eligible Orders", str(self.eligible_orders))
        console.print(summary_table, justify="center")  # Center the table in console
        console.print()


def is_eligible_order(element: uc.WebElement) -> bool:
    """Check if an order element is eligible for processing."""
    try:
        order = Order(element)
        return order.has_tip() and not order.cancelled
    except Exception:
        return False


def process_orders_with_stats(elements, processor):
    """Process orders and print statistics with progress."""
    stats = OrderStats()
    stats.total_orders = len(elements)
    orders = []

    for elem in elements:
        if is_eligible_order(elem):
            orders.append(elem)

    stats.eligible_orders = len(orders)
    stats.print_summary()

    with Progress(*get_progress_bar_style()) as progress:
        task = progress.add_task("Processing orders...", total=len(orders))

        for idx, order in enumerate(orders, 1):
            console.print(
                f"✓ Found eligible order ({idx}/{len(orders)}): " f"{str(order)} | 💰 ${order.amount:.2f} | Status: ✅"
            )
            progress.update(task, advance=1)
            time.sleep(0.1)

    return orders


class TabManager:
    __slots__ = ["pending_tabs", "main_window", "_lock"]
    """Manage browser tabs across monitoring cycles."""

    def __init__(self):
        self.pending_tabs = []
        self.main_window = None
        self._lock = Lock()

    def set_pending_tabs(self, tabs: List[str], main_window: str) -> None:
        """Set tabs to be cleaned up in next cycle."""
        with self._lock:
            self.pending_tabs = tabs
            self.main_window = main_window

    def cleanup_tabs(self, driver: uc.Chrome) -> None:
        """Clean up tabs from previous cycle."""
        with self._lock:
            if not self.pending_tabs:
                return

            original_window = driver.current_window_handle
            for tab in self.pending_tabs:
                try:
                    driver.switch_to.window(tab)
                    driver.close()
                except Exception:
                    continue

            try:
                driver.switch_to.window(self.main_window or original_window)
            except Exception:
                driver.switch_to.window(original_window)

            self.pending_tabs = []
            self.main_window = None


class DriverContextManager:
    """Context manager for WebDriver with improved error handling."""

    def __init__(self):
        self.driver = None
        self.start_time = None

    def __enter__(self):
        log_info("Setting up WebDriver...", transient=False)
        snapshot_memory("Before driver creation")

        try:
            self.driver = create_driver()
            self.driver.implicitly_wait(IMPLICIT_WAIT)
            self.start_time = time.perf_counter()  # Only set start time on successful creation
            snapshot_memory("After driver creation")
            return self.driver

        except Exception as e:
            log_error(f"Failed to initialize WebDriver: {e}")
            # Don't set start_time since driver creation failed
            return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.start_time:  # Only calculate time if driver was created successfully
                elapsed_session_time = time.perf_counter() - self.start_time
                log_info(f"Total session time: {elapsed_session_time:.4f} seconds.", transient=False)

            snapshot_memory("Before driver quit")
            log_info("Tearing down WebDriver...", transient=False)

            if self.driver:
                try:
                    self.driver.quit()
                    log_success("WebDriver closed successfully.", transient=False)
                except Exception as e:
                    log_error(f"Error closing WebDriver: {e}")
            else:
                log_warning("WebDriver was not initialized properly. Skipping quit operation.")

            snapshot_memory("After driver quit")

            # Print memory snapshots at the end of the session
            if _memory_snapshots:
                memory_table = Table(
                    title="[bold]Memory Usage Snapshots[/bold]",
                    title_style=f"bold {MAIN_COLOR}",
                    border_style=MAIN_COLOR,
                    padding=(1, 2),
                    show_header=True,
                    header_style="bold white",
                    row_styles=["none", "dim"],
                    expand=False,
                )
                memory_table.add_column("Label", style="white", no_wrap=True)
                memory_table.add_column("Memory (MB)", style="cyan", justify="right")
                memory_table.add_column("Time (seconds)", style="magenta", justify="right")

                # Use session start time if snapshots exist and driver was created successfully
                session_start_time = (
                    _memory_snapshots[0][2] if _memory_snapshots else (self.start_time or time.perf_counter())
                )

                for label, memory_mb, snapshot_time in _memory_snapshots:
                    time_from_start = snapshot_time - session_start_time
                    memory_table.add_row(label, f"{memory_mb:.2f}", f"{time_from_start:.2f}")
                console.print(memory_table, justify="center")

            _memory_snapshots.clear()

        except Exception as e:
            log_error(f"Error in driver cleanup: {e}")


# Initialize tab manager globally
tab_manager = TabManager()


def main():
    global customer_name, customer_email  # Ensure global scope for customer info
    logging.info("Starting new session")
    try:
        with open("cookies.pkl", "rb") as file:
            cookies = pickle.load(file)
    except (FileNotFoundError, pickle.PickleError, EOFError):
        cookies = None

    console.clear()

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cookies_status = "[green]Found[/]" if cookies else "[red]Not Found[/]"
    test_mode_status = "[yellow]ON[/]" if TEST_MODE else "[green]OFF[/]"

    menu_content = (
        "\n"
        f"[bold {MAIN_COLOR}]1.[/] [white]Sign into existing customer[/white]\n"
        f"[bold {MAIN_COLOR}]2.[/] [white]Sign in with saved cookies[/white]\n"
        f"\nTest Mode:         {test_mode_status}"
        f"\nHeadless Mode:     [{'green' if HEADLESS_MODE else 'yellow'}]{'ON' if HEADLESS_MODE else 'OFF'}[/]"
        f"\nCookies:           {cookies_status}"
        f"\nBatch Size:        [magenta]{MAX_ORDERS_PER_BATCH}[/magenta]"
        f"\nMin Tip:           [red]${MAX_TIP_THRESHOLD}[/red]"
    )

    menu_panel = Panel(
        menu_content,
        title="[#D91400 bold]RUNS! RUNS! RUNS!!![/#D91400 bold]",
        title_align="left",
        subtitle=f"[dim]{current_time}[/dim]",
        subtitle_align="right",
        border_style=MAIN_COLOR,
        padding=(0, 2),
        width=60,
    )

    console.print(menu_panel)

    choice = console.input(f"\n[bold {MAIN_COLOR}]>[/] ")

    with DriverContextManager() as driver:  # Use context manager for driver
        if choice == "1":
            create_manual_cookies_backup()
            driver.get(url="https://www.doordash.com/consumer/login/")

            if wait_for_profile_page(driver):
                save_cookies_after_login(driver, manual=True)
            else:
                log_error("Manual login timed out")
                return  # Exit main function gracefully

        if choice == "2" and cookies is not None:
            if is_valid_cookie_file("cookies.pkl"):
                try:
                    driver.get("https://www.doordash.com/")

                    for cookie in cookies:
                        if cookie.get("domain", "").startswith(".www."):
                            cookie["domain"] = cookie["domain"].replace(".www.", ".")
                        driver.add_cookie(cookie)

                    driver.get("https://www.doordash.com/home")
                except Exception as e:
                    log_error(f"Error during cookie sign in: {str(e)}")
                    return  # Exit main function gracefully
            else:
                log_error("Invalid or empty cookies file")
                return  # Exit main function gracefully

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            login_task = progress.add_task("Waiting for login...", transient=True, total=None)
            while True:
                url = driver.current_url

                if "doordash.com" in url and "/home" in url:
                    progress.stop_task(login_task)
                    log_success("Login successful!")
                    break

                elif "doordash.com" in url and "action=Login" in url and "/home" not in url:
                    time.sleep(SHORT_ACTION_WAIT)

                else:
                    time.sleep(SHORT_ACTION_WAIT)

        driver.get(url="https://www.doordash.com/consumer/edit_profile/")
        time.sleep(RETRY_OPERATION_WAIT)

        email_address_element = get_element_from_attribute(driver, "input", "type", "email")
        first_name_element = get_element_from_attribute(driver, "input", "data-testid", "givenName_input")
        last_name_element = get_element_from_attribute(driver, "input", "data-testid", "familyName_input")

        customer_info = {
            "first_name": first_name_element.get_attribute("value"),
            "last_name": last_name_element.get_attribute("value"),
            "email": email_address_element.get_attribute("value"),
        }
        save_cookies_after_login(driver, manual=False, customer_info=customer_info)

        global customer_name, customer_email  # Make sure global variables are updated
        customer_email = customer_info["email"] if customer_info["email"] else "could not get email"
        customer_name = (
            f"{customer_info['first_name']} {customer_info['last_name']}"
            if customer_info["first_name"] and customer_info["last_name"]
            else "could not get name"
        )

        customer_info_panel = Panel(
            f"\n[white]{customer_name}[/white]\n[dim]{customer_email}[/dim]\n",
            title=f"[bold {MAIN_COLOR}]Customer Information[/]",
            title_align="left",
            border_style=MAIN_COLOR,
            padding=(0, 2),
            width=60,
        )
        console.print(customer_info_panel)

        monitor_orders_auto(driver)

    logging.info("End of session")


if __name__ == "__main__":
    with cProfile.Profile() as profiler:
        main()
    # profiler.print_stats(sort="cumtime") # Keep profiler output commented out unless needed
