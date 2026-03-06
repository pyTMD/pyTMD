import re
import pathlib
import urllib.request

# documentation directory
directory = pathlib.Path(__file__).parent
# fetch the bibtex entry for the JOSS paper
doi = "10.21105/joss.08566"
url = f"https://doi.org/{doi}"
# post to doi.org with the accept header for bibtex
headers = {"accept": "application/x-bibtex"}
request = urllib.request.Request(url, headers=headers)
response = urllib.request.urlopen(request, timeout=3600)
# read the bibtex entry from the response
bibtex_contents = response.read().decode("utf8").strip()
# valid bibtex entry types
entry_regex = r"[?<=\@](article)[\s]?\{(.*?)[\s]?,[\s]?"
R1 = re.compile(entry_regex, flags=re.IGNORECASE)
# bibtex fields to be printed in the output file
bibtex_field_types = r"|".join(
    [
        "author",
        "doi",
        "issn",
        "journal",
        "month",
        "number",
        "pages",
        "publisher",
        "title",
        "url",
        "volume",
        "year",
    ]
)
field_regex = r"[\s]?(" + bibtex_field_types + r")\=[\{](.*?)[\}][\,]"
R2 = re.compile(field_regex, flags=re.IGNORECASE)
# extract bibtex entry type and bibtex cite key
bibtype, bibkey = R1.findall(bibtex_contents).pop()
bibtex_field_entries = R2.findall(bibtex_contents)
# write bibtex entry to file
bibkey_formatted = bibkey.replace("_", "-")
bibtex_file = directory.joinpath("_assets", f"{bibkey_formatted}.bib")
with bibtex_file.open(mode="w", encoding="utf8") as fid:
    # print the bibtex citation
    print(f"@{bibtype}{{{bibkey},", file=fid)
    # for each field within the entry
    for k, v in bibtex_field_entries:
        # prep the field key and value for printing
        k, v = k.lower(), v.strip()
        if k == "month":
            print(f"{k} = {v.lower()},", file=fid)
        elif k == "title":
            print(f"{k} = {{{{{v}}}}},", file=fid)
        else:
            print(f"{k} = {{{v}}},", file=fid)
    print("}", file=fid)
