"""
Rewrite abstracts for out-of-domain researchers using LLMs.

Usage:
    python main.py --n_papers=25 --model=text-davinci-003
"""

import argparse
import json
import os
from typing import Dict, List

import jsonlines
import openai
import requests
from tqdm import tqdm

S2_API_URL_BASE = "https://partner.semanticscholar.org/graph/v1"


def load_secrets(secrets_fp):
    with open(secrets_fp, "r") as f:
        return json.load(f)


def get_paper_details(id: str, fields: List[str], api_key: str) -> Dict:
    url = f"{S2_API_URL_BASE}/paper/{id}"
    params = {"fields": ",".join(fields)}
    headers = {"x-api-key": api_key}
    res = requests.get(url, params, headers=headers).json()
    return res


def generate_personalized_summary(
    summary: str,
    source: str,
    target: str,
    model: str = "text-davinci-003",
    temperature: float = 0.7,
    max_tokens: int = 500,
):
    if model in ["gpt-4", "gpt-3.5-turbo"]:
        completion = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"""You are a scientific research assistant supporting interdisciplinary research.

                    Rewrite the following abstract from a paper in {source}, for a researcher with a background in {target}. Include details that might be relevant to researchers in {target}, contextualize the writing to their field and research values, and use explanations of concepts that draw on their background.

                    Abstract: {summary}

                    Rewritten abstract:

                    """,
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content.strip()
    else:
        response = openai.Completion.create(
            model=model,
            prompt=f"""You are a scientific research assistant supporting interdisciplinary research.

            Rewrite the following abstract from a paper in {source}, for a researcher with a background in {target}. Include details that might be relevant to researchers in {target}, contextualize the writing to their field and research values, and use explanations of concepts that draw on their background.

            Abstract: {summary}

            Rewritten abstract:

            """,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return response.choices[0].text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--secrets_file", default="secrets.json")
    parser.add_argument("-n", "--n_papers", type=int)
    parser.add_argument(
        "--model",
        choices=["text-davinci-003", "gpt-3.5-turbo", "gpt-4"],
    )
    args = parser.parse_args()

    secrets = load_secrets(args.secrets_file)
    openai.api_key = secrets["openai_api_key"]

    paper_shas_file = "trending_ids_04_11_2023.txt"
    with open(paper_shas_file, "r") as f:
        paper_shas = [line.strip() for line in f]
    paper_shas = paper_shas[: args.n_papers]

    output_fp = f"nlp_neuroscience_{args.model}.json"
    processed_shas = []
    if os.path.exists(output_fp):
        with jsonlines.open(output_fp) as reader:
            processed_shas = [line["paper_sha"] for line in reader]

    with jsonlines.open(output_fp, mode="a") as writer:
        for paper_sha in tqdm(paper_shas):
            if paper_sha in processed_shas:
                continue
            details = get_paper_details(
                paper_sha, fields=["abstract"], api_key=secrets["s2_api_key"]
            )
            abstract = details["abstract"]
            response = None
            if abstract:
                response = generate_personalized_summary(
                    summary=abstract,
                    source="natural language processing",
                    target="neuroscience",
                    model=args.model,
                    temperature=0.5,
                )
            result = {
                "paper_sha": paper_sha,
                "model": args.model,
                "original": abstract,
                "rewritten": response,
            }
            writer.write(result)


if __name__ == "__main__":
    main()
