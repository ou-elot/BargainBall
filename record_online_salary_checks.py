"""Record salary rows confirmed by the 2025-26 online source review."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
PAIRS = ROOT / "data" / "processed" / "salary_cap_normalized_pairs.csv"
OVERRIDES = ROOT / "data" / "raw" / "salaries" / "verified_salary_overrides.csv"
LEDGER = ROOT / "data" / "processed" / "online_salary_verifications.csv"

SOURCES = {
    "MarJon Beauchamp": (
        "https://www.aol.com/articles/2026-nba-free-agent-rankings-143040000.html",
        "AOL/USA Today salary-history listing reports 2025-26 earnings of $347,451.",
    ),
    "Jamaree Bouyea": (
        "https://m.aiscore.com/salary/basketball/player-jamaree-bouyea-o17p8s082vajkjw",
        "AiScore salary history reports 2025-26 annual salary of $878,375.",
    ),
    "Colin Castleton": (
        "https://www.hoopsrumors.com/2025/10/magic-converting-colin-castleton-to-two-way-deal.html",
        "Hoops Rumors reports the 2025-26 two-way salary as $636,435.",
    ),
    "Isaiah Crawford": (
        "https://nbaanalysis.net/houston-rockets-2025-26-payroll-revealed-kevin-durant-joins-team-and-adds-54-7-million-salary-to-the-books/",
        "NBAAnalysis payroll list reports $636,435 for Crawford.",
    ),
    "Trentyn Flowers": (
        "https://m.aiscore.com/salary/basketball/player-trentyn-flowers-zrknxso1oma0qle",
        "AiScore and EuroBasket salary listings report 2025-26 salary of $250,937.",
    ),
    "Enrique Freeman": (
        "https://www.aol.com/articles/2026-nba-free-agent-rankings-143040000.html",
        "AOL/USA Today salary-history listing reports 2025-26 earnings of $636,435.",
    ),
    "Harrison Ingram": (
        "https://www.expressnews.com/sports/spurs/article/2025-26-spurs-get-season-s-roster-breakdown-21097182.php",
        "Express-News Spurs roster breakdown lists 2025-26 salary as $636,435.",
    ),
    "Trey Jemison": (
        "https://www.hoopsrumors.com/2025/12/the-knicks-players-who-benefited-most-from-nba-cup-prize-money.html",
        "Hoops Rumors lists Jemison’s 2025-26 two-way base salary as $636,435.",
    ),
    "Colby Jones": (
        "https://www.spotrac.com/nba/player/_/id/82320/colby-jones",
        "Spotrac salary/cash table reports $85,300 for 2025-26.",
    ),
    "Johnny Juzang": (
        "https://www.spotrac.com/nba/player/_/id/78583/johnnyjuzang",
        "Spotrac reports a 2025-26 two-way contract value of $636,435.",
    ),
    "Chris Livingston": (
        "https://www.aiscore.com/salary/basketball/player-chris-livingston-jek3pse8nxu2qo2",
        "AiScore lists $2,296,274 with Milwaukee plus $527,611 with Cleveland; their sum is $2,823,885.",
    ),
    "Kevin McCullar Jr.": (
        "https://nbasalaries.com/players/kevin-mccullar-jr/",
        "NBA Salaries lists the 2025-26 two-way salary as $636,435.",
    ),
    "Antonio Reeves": (
        "https://nbaanalysis.net/charlotte-hornets-2025-26-payroll-revealed-lamelo-ball-must-earn-his-37-million-salary-with-strong-performances/",
        "NBAAnalysis payroll list reports $636,435 for Reeves.",
    ),
    "Orlando Robinson": (
        "https://www.hoopsrumors.com/2025/07/magic-sign-orlando-robinson-to-two-way-contract.html",
        "Hoops Rumors reports 2025-26 salary of $636,435.",
    ),
    "David Roddy": (
        "https://www.aol.com/articles/2026-nba-free-agent-rankings-143040000.html",
        "AOL/USA Today salary-history listing reports 2025-26 earnings of $128,686.",
    ),
    "Drew Peterson": (
        "https://m.aiscore.com/basketball/player-drew-peterson/oj7x6srg61ig7g3",
        "AiScore reports 2025-26 salary of $209,114.",
    ),
    "Daeqwon Plowden": (
        "https://m.aiscore.com/salary/basketball/player-daeqwon-plowden-o17p8sryp6fjkjw",
        "AiScore reports 2025-26 salary of $636,435.",
    ),
    "Tristen Newton": (
        "https://sports.yahoo.com/articles/uconns-tristen-newton-waived-houston-210927982.html",
        "Yahoo/CT Insider reports Newton earned $321,714 on his 2025-26 two-way deal.",
    ),
    "Drew Timme": (
        "https://www.aol.com/articles/2026-nba-free-agent-rankings-143040000.html",
        "AOL/USA Today reports 2025-26 earnings of $447,182; AiScore gives the same figure.",
    ),
    "Luke Travers": (
        "https://www.espn.com/nba/story/_/id/45445233/nba-offseason-2025-key-dates-deadlines-contracts",
        "ESPN’s 2025 offseason salary list reports $636,435.",
    ),
    "Stanley Umude": (
        "https://m.aiscore.com/salary/basketball/player-stanley-umude-g676jsvmdga9kry",
        "AiScore and EuroBasket report 2025-26 salary of $363,537.",
    ),
}


def main():
    pairs = pd.read_csv(PAIRS)
    existing = pd.read_csv(OVERRIDES)
    rows = []

    for player, (url, note) in SOURCES.items():
        match = pairs[(pairs["player"].eq(player)) & pairs["target_season"].eq(2026)]
        if len(match) != 1:
            raise ValueError(f"Expected one salary row for {player} 2025-26; found {len(match)}.")
        row = match.iloc[0]
        rows.append({
            "player": player,
            "player_id": row["player_id"],
            "target_season": 2026,
            "target_salary": row["target_salary"],
            "verification_source_url": url,
            "verification_note": note,
        })

    ledger = pd.DataFrame(rows)
    ledger.to_csv(LEDGER, index=False)
    overrides = ledger[["player_id", "target_season", "target_salary", "verification_source_url"]].rename(
        columns={"verification_source_url": "salary_source_url"}
    )
    overrides["salary_needs_source_review"] = False
    overrides["salary_method"] = "verified_online_source_check"
    combined = pd.concat([existing, overrides], ignore_index=True)
    combined = combined.drop_duplicates(["player_id", "target_season"], keep="last")
    combined.to_csv(OVERRIDES, index=False)
    print(f"Recorded {len(ledger)} online salary checks.")
    print(f"Total persisted verified overrides: {len(combined)}.")


if __name__ == "__main__":
    main()
