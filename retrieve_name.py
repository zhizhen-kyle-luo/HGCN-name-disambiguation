
import requests
import argparse
import sys


def get_author_info(openalex_id):
    if not openalex_id.startswith("https://api.openalex.org/people/"):
        openalex_id = f"https://api.openalex.org/people/{openalex_id}"

    response = requests.get(openalex_id)

    if response.status_code == 200:
        try:
            data = response.json()

            # Author name
            author_name = data.get("display_name", "Unknown")

            # Summary stats
            summary = data.get("summary_stats", {})
            h_index = summary.get("h_index", "N/A")
            i10_index = summary.get("i10_index", "N/A")
            citedness = summary.get("2yr_mean_citedness", "N/A")

            # Affiliation info
            affiliations = data.get("affiliations", [])
            institutions = []
            for aff in affiliations:
                inst = aff.get("institution", {})
                inst_name = inst.get("display_name", "Unknown")
                inst_id = inst.get("id", "N/A")
                country = inst.get("country_code", "N/A")
                institutions.append({
                    "name": inst_name,
                    "id": inst_id,
                    "country": country
                })

            return {
                "author_name": author_name,
                "summary_stats": {
                    "h_index": h_index,
                    "i10_index": i10_index,
                    "2yr_mean_citedness": citedness
                },
                "institutions": institutions
            }

        except requests.exceptions.JSONDecodeError:
            return {"error": "JSON parsing failed"}
    else:
        return {"error": f"Request failed with status {response.status_code}"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch author info from OpenAlex')
    parser.add_argument('--id', type=str, required=True, help='OpenAlex Author ID (e.g., A5029006010)')
    args = parser.parse_args()

    info = get_author_info(args.id)

    if "error" in info:
        print("Error:", info["error"])
    else:
        print("Author:", info["author_name"])
        print("Summary Stats:")
        for k, v in info["summary_stats"].items():
            print(f"  {k}: {v}")
        print("Affiliations:")
        for inst in info["institutions"]:
            print(f"  - {inst['name']} ({inst['country']}) | ID: {inst['id']}")


    author_name = get_author_info(args.id)
    print(f"Author name for id {args.id}:")
    print(f"{author_name}")