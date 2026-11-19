from pathlib import Path
import unicodedata
import pandas as pd



def keep_combined_seasons(dataframe):
    kept_rows = []

    for key, group in dataframe.groupby(["player_id", "season"]):
        if len(group) == 1:
            kept_rows.append(group)
        else:
            combined = group[group["team"].str.fullmatch(r"TOT|[2-9][0-9]*TM")]

            if len(combined) != 1:
                raise ValueError("Expected one combined record for " + str(key))

            kept_rows.append(combined)

    return pd.concat(kept_rows, ignore_index=True)


def normalize_name(name):
    name = name.casefold()
    name = name.replace("ı", "i")
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")

    cleaned_name = ""

    for character in name:
        if character.isalnum():
            cleaned_name = cleaned_name + character

    return cleaned_name


name_aliases = {
    "Isaac Austin": "Ike Austin",
    "Adama-Alpha Bal": "Adama Bal",
    "Walter Clayton": "Walter Clayton Jr.",
    "Egor Dёmin": "Egor Dëmin",
    "Isaac Fontaine": "Ike Fontaine",
    "Kiwane Lemorris Garris": "Kiwane Garris",
    "DaRon Holmes": "DaRon Holmes II",
    "Eugene Jeter": "Pooh Jeter",
    "Ibo Kutluay": "Ibrahim Kutluay",
    "Rich Manning": "Richard Manning",
    "Stanislav Medvedenko": "Slava Medvedenko",
    "Nathan Mensah": "Nate Mensah",
    "Ronald Murray": "Flip Murray",
    "Norm Richardson": "Norman Richardson",
    "Danny Schayes": "Dan Schayes",
    "Ha Seung-Jin": "Ha Ha",
    "Mike Sweetney": "Michael Sweetney",
    "Clarence Weatherspoon": "Clar. Weatherspoon",
    "Sun Yue": "Sun Sun",
    "Cui Yongxi": "Cui Cui",
    "Wade Baldwin": "Wade Baldwin IV",
    "Brandon Boston Jr.": "Brandon Boston",
    "Brian Bowen": "Brian Bowen II",
    "Reggie Bullock": "Reggie Bullock Jr.",
    "Jimmy Butler": "Jimmy Butler III",
    "John Butler": "John Butler Jr.",
    "Mitch Creek": "Mitchell Creek",
    "Jeff Dowtin": "Jeff Dowtin Jr.",
    "Vince Edwards": "Vincent Edwards",
    "Vítor Luiz Faverani": "Vitor Faverani",
    "Melvin Frazier": "Melvin Frazier Jr.",
    "Michael Frazier": "Michael Frazier II",
    "Harry Giles": "Harry Giles III",
    "Nigel Hayes": "Nigel Hayes-Davis",
    "Ron Holland": "Ronald Holland II",
    "Vince Hunter": "Vincent Hunter",
    "Matthew Hurt": "Matt Hurt",
    "GG Jackson II": "GG Jackson",
    "Trey Jemison": "Trey Jemison III",
    "Perry Jones": "Perry Jones III",
    "Kevin Knox": "Kevin Knox II",
    "Roger Mason": "Roger Mason Jr.",
    "Marcus Morris": "Marcus Morris Sr.",
    "Zach Norvell": "Zach Norvell Jr.",
    "Johnny O'Bryant": "Johnny O'Bryant III",
    "Bobby Portis": "Bobby Portis Jr.",
    "Cameron Reynolds": "Cam Reynolds",
    "Glen Rice Jr.": "Glen Rice",
    "Jermaine Samuels": "Jermaine Samuels Jr.",
    "Tre Scott": "Trevon Scott",
    "Jeff Taylor": "Jeffery Taylor",
    "Xavier Tillman Sr.": "Xavier Tillman",
    "Derrick Walton": "Derrick Walton Jr.",
    "Andrew White": "Andrew White III",
    "Jeenathan Williams": "Nate Williams",
    "Matt Williams": "Matt Williams Jr.",
    "Robert Williams": "Robert Williams III"
}

# Names alone cannot distinguish these players. IDs and season totals do.
explicit_ids = {
    "mitchto02": "203502",
    "mitchto03": "203183",
    "johnsch03": "202419",
    "johnsch04": "203187",
    "willima03": "200766",
    "willima04": "201173",
    "jonesch01": "279",
    "jonesch03": "1869",
    "smithst01": "120"
}


def name_key(name):
    name = name_aliases.get(name, name)
    return normalize_name(name)

