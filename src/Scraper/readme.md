# WebDriver Downloader Script

This Python script automates the process of downloading and setting up WebDriver for browsers such as Edge, Chrome, and Firefox on multiple operating systems (Windows, Linux, macOS). The script prompts the user for their operating system, browser choice, and WebDriver version, then proceeds to download and extract the appropriate WebDriver for browser automation tasks.

## Features
- **Cross-Platform Support**: Works on Windows, Linux, and macOS.
- **Browser Compatibility**: Supports Edge, Chrome, and Firefox WebDrivers.
- **Automated WebDriver Setup**: Downloads and extracts the WebDriver based on user inputs.
- **User-Friendly Prompts**: Allows users to choose their OS, browser, and WebDriver version.

## Prerequisites
- Python 3.x installed
- Required libraries: `requests`, `zipfile`, `os`, `platform`, and `subprocess`
- Web driver of the specific wanted broweser such as Chrome, Firefox, Edge.

To install the necessary libraries, run:
```bash
pip install requests
```

## Usage
### Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/yourusername/WebDriverDownloader.git
```
then you shall run each code.
the code afterwards will ask you questions regarding which OS you are, which Browser you are goign to use. 

### The script will:

- Download the Chrome WebDriver version 114.0 for Windows.
- Extract the WebDriver and make it executable.
- Start the WebDriver process to enable browser automation.

### Notes
- Make sure the WebDriver version matches your browser version to avoid compatibility issues.
- For macOS and Linux, you may need to give execute permissions to the WebDriver:
```bash
chmod +x chromedriver
```

## License
This project is licensed under the MIT License.

## Acknowledgments
WebDriver download links are sourced from the official websites of Microsoft Edge WebDriver, ChromeDriver, and GeckoDriver for Firefox.








