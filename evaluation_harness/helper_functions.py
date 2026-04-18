"""Implements helper functions to assist evaluation cases where other evaluators are not suitable."""
import json
import os
from typing import Any
from urllib.parse import urlparse

import requests
from playwright.sync_api import CDPSession, Page

from browser_env.env_config import (
    ACCOUNTS,
    GITLAB,
    MAP,
    REDDIT,
    SHOPPING,
    SHOPPING_ADMIN,
    WIKIPEDIA,
)
from llms.providers.openai_utils import (
    generate_from_openai_chat_completion,
)


def shopping_get_auth_token() -> str:
    response = requests.post(
        url=f"{SHOPPING}/rest/default/V1/integration/admin/token",
        headers={"content-type": "application/json"},
        data=json.dumps(
            {
                "username": ACCOUNTS["shopping_site_admin"]["username"],
                "password": ACCOUNTS["shopping_site_admin"]["password"],
            }
        ),
    )
    token: str = response.json()
    return token


def shopping_get_latest_order_url() -> str:
    """Get the latest order url from the shopping website."""

    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }

    params = {
        "searchCriteria[sortOrders][0][field]": "created_at",
        "searchCriteria[sortOrders][0][direction]": "DESC",
        "searchCriteria[pageSize]": "1",
    }

    response = requests.get(
        f"{SHOPPING}/rest/V1/orders", params=params, headers=header
    )
    assert response.status_code == 200
    response_obj = response.json()["items"][0]
    order_id = int(response_obj["increment_id"])
    order_url = f"{SHOPPING}/sales/order/view/order_id/{order_id}/"
    return order_url


def shopping_get_sku_latest_review_author(sku: str) -> str:
    """Get the latest review for shopping admin."""
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(
        f"{SHOPPING}/rest/V1/products/{sku}/reviews", headers=header
    )
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    author: str = response_obj[-1]["nickname"]
    return author


def shopping_get_sku_latest_review_rating(sku: str) -> str:
    """Get the latest review for shopping admin."""
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(
        f"{SHOPPING}/rest/V1/products/{sku}/reviews", headers=header
    )
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    assert response_obj[0]["ratings"][0]["rating_name"] == "Rating"
    rating: str = str(response_obj[-1]["ratings"][0]["percent"])
    return rating


def reddit_get_post_url(url: str) -> str:
    """Get the post url"""
    # Url is http://domain/f/subreddit/post_id/...
    # get domain, subreddit, post_id
    domain = urlparse(url).netloc
    tok_url = urlparse(url).path.split("/")
    # not a valid post/comment url, return the url as is
    if len(tok_url) < 4:
        return url
    if tok_url[1] != "f":
        return url
    subreddit = urlparse(url).path.split("/")[2]
    post_id = urlparse(url).path.split("/")[3]
    scheme = urlparse(url).scheme
    post_url = f"{scheme}://{domain}/f/{subreddit}/{post_id}/"
    return post_url


def gitlab_get_project_memeber_role(page: Page, account_name: str) -> str:
    # get the account index
    try:
        account_idx = page.evaluate(
            f"""(() => {{
                const elements = document.querySelectorAll("td[data-label='Account'] span.gl-avatar-labeled-sublabel");
                let index = -1;  // Default value if not found

                for(let i = 0; i < elements.length; i++) {{
                    if(elements[i].outerText === '@{account_name}') {{
                        index = i;
                        break;
                    }}
                }}

                return index;
            }})()"""
        )

        # get the role
        role: str = page.evaluate(
            f"""(() => {{
                return document.querySelectorAll("td.col-max-role span")[{account_idx}].outerText;
            }})()"""
        )
    except Exception:
        role = ""

    return role


def llm_fuzzy_match(pred: str, reference: str, question: str) -> float:
    """Check whether the prediction matches the reference with an LLM judge.

    The model used is controlled by the EVAL_LLM_MODEL environment variable
    (default: gpt-4-1106-preview). Set OPENAI_API_BASE to redirect to a
    local vLLM endpoint.
    """
    messages: list[dict[str, Any]] = []
    # construct the question to ask
    message = "You are grading a student's answer against a reference answer. Determine whether the student's answer is semantically equivalent to the reference answer, allowing for different phrasing or wording.\n"
    message += f"Question: {question}\n"
    message += f"Reference answer: {reference}\n"
    message += "Note: the string 'N/A' means 'not achievable'.\n"
    message += f"Student answer: {pred}\n"
    message += "\nReply with exactly one word on its own line: correct, incorrect, or partially correct."
    messages = [
        {"role": "system", "content": "You are a grading assistant. Reply with exactly one word: correct, incorrect, or partially correct."},
        {"role": "user", "content": message},
    ]

    eval_model = os.environ.get("EVAL_LLM_MODEL", "gpt-4-1106-preview")
    raw = generate_from_openai_chat_completion(
        model=eval_model,
        messages=messages,
        temperature=0,
        max_tokens=128,
        top_p=1.0,
        context_length=0,
    )
    if not raw:
        return 0.0
    response = raw.lower()
    if "partially correct" in response or "incorrect" in response:
        return 0.0
    elif "correct" in response:
        return 1.0
    else:
        return 0.0


def llm_ua_match(pred: str, reference: str, question: str) -> float:
    """Check whether the prediction matches the reference with an LLM judge.

    The model used is controlled by the EVAL_LLM_MODEL environment variable
    (default: gpt-4-1106-preview). Set OPENAI_API_BASE to redirect to a
    local vLLM endpoint.
    """
    messages: list[dict[str, Any]] = []
    # construct the question to ask
    message = ""
    message += f"task: {question}\n"
    message += f"actual unachievable reason: {reference}\n"
    message += f"reported unachievable reason: {pred}\n"
    message += (
        "The task described above is inherently unachievable due to the reason specified under 'actual unachievable reason'. "
        "An individual previously attempted this task and was unable to complete it. They provided a reason for their failure, "
        "which is listed under 'reported unachievable reason'. Your role is to review both the actual and reported reasons. "
        "Determine if the reported reason aligns with the actual reason, even if implicitly. "
        "Reply with exactly one word on its own line: same or different."
    )
    messages = [
        {"role": "system", "content": "You are a grading assistant. Reply with exactly one word: same or different."},
        {"role": "user", "content": message},
    ]

    eval_model = os.environ.get("EVAL_LLM_MODEL", "gpt-4-1106-preview")
    raw = generate_from_openai_chat_completion(
        model=eval_model,
        messages=messages,
        temperature=0,
        max_tokens=128,
        top_p=1.0,
        context_length=0,
    )
    if not raw:
        return 0.0
    response = raw.lower()
    if "different" in response:
        return 0.0
    elif "same" in response:
        return 1.0
    else:
        return 0.0


class PseudoPage:
    def __init__(self, original_page: Page, url: str):
        self.url = url
        self.original_page = original_page

    def __getattr__(self, attr: str) -> Any:
        # Delegate attribute access to the original page object
        if attr not in ["url"]:
            return getattr(self.original_page, attr)
        else:
            return getattr(self, attr)
