
import requests
from bs4 import BeautifulSoup
from states import ResumeState


def webscrap(state: ResumeState):
    url = state['jd_link']
    headers = {"User-Agent" : "Mozzilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        for element in soup(["script", "style", "header", "footer", "nav"]):
            element.decompose()

        all_text = soup.get_text(separator = '\n', strip=True)
        return {'raw_website_text' : all_text}
    else:
        return {'error_code': 1}