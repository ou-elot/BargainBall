"""Definition of the forecast profile used for PCA-based player similarity."""

SKILL_TARGETS = [
    "pie",
    "ws_48",
    "bpm",
    "ts_percent",
    "usg_percent",
    "trb_percent",
    "ast_percent",
    "stl_percent",
    "blk_percent",
    "tov_percent",
    "x3p_ar",
    "f_tr",
    "e_fg_percent",
    "x3p_percent",
    "ft_percent",
]

# 3P% and FT% remain prediction targets, but are omitted from the similarity
# PCA because these rates are undefined for players with no attempts. Their
# attempt rates and overall efficiency measures remain in the PCA profile.
PROFILE_TARGETS = SKILL_TARGETS + ["minutes"]
PCA_TARGETS = [
    target for target in SKILL_TARGETS
    if target not in {"x3p_percent", "ft_percent"}
]
NEXT_SEASON_COLUMNS = {
    target: ("next_season_minutes" if target == "minutes" else "next_season_" + target)
    for target in PROFILE_TARGETS
}
BASELINE_COLUMNS = {
    target: ("mp" if target == "minutes" else target)
    for target in PROFILE_TARGETS
}
