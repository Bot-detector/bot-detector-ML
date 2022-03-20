import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

skills = [
    "attack",
    "defence",
    "strength",
    "hitpoints",
    "ranged",
    "prayer",
    "magic",
    "cooking",
    "woodcutting",
    "fletching",
    "fishing",
    "firemaking",
    "crafting",
    "smithing",
    "mining",
    "herblore",
    "agility",
    "thieving",
    "slayer",
    "farming",
    "runecraft",
    "hunter",
    "construction",
]
minigames = [
    "league",
    "bounty_hunter_hunter",
    "bounty_hunter_rogue",
    "cs_all",
    "cs_beginner",
    "cs_easy",
    "cs_medium",
    "cs_hard",
    "cs_elite",
    "cs_master",
    "lms_rank",
    "soul_wars_zeal",
]


class hiscoreData:
    """
    This class is responsible for cleaning data & creating features
    """

    def __init__(self, data: List[dict]) -> None:
        self.df = pd.DataFrame(data)
        self.df_clean = self.df.copy()

        self.skills = skills
        self.minigames = minigames

        self.__clean()
        self.__skill_ratio()
        self.__boss_ratio()

    def __clean(self) -> None:
        # cleanup
        self.df_clean.drop(columns=["id", "timestamp", "ts_date"], inplace=True)
        # unique index
        self.df_clean.set_index(["Player_id"], inplace=True)

        # if not on the hiscores it shows -1
        self.df_clean = self.df_clean.replace(-1, 0)

        # bosses
        self.bosses = [
            c for c in self.df_clean.columns if c not in ["total"] + skills + minigames
        ]
        # total is not always on hiscores
        self.df_clean["total"] = self.df_clean[self.skills].sum(axis=1)
        self.df_clean["boss_total"] = self.df_clean[self.bosses].sum(axis=1)

        # fillna
        self.df_clean.fillna(0, inplace=True)

        # get low lvl players
        mask = self.df_clean["total"] < 1_000_000
        self.df_low = self.df_clean[mask].copy()

    def __skill_ratio(self):
        self.skill_ratio = pd.DataFrame(index=self.df_clean.index)

        total = self.df_clean["total"]
        for skill in self.skills:
            self.skill_ratio[f"{skill}/total"] = self.df_clean[skill] / total

        self.skill_ratio.fillna(0, inplace=True)

    def __boss_ratio(self):
        self.boss_ratio = pd.DataFrame(index=self.df_clean.index)

        total = self.df_clean["boss_total"]
        for boss in self.bosses:
            self.boss_ratio[f"{boss}/total"] = self.df_clean[boss] / total

        self.boss_ratio.fillna(0, inplace=True)

    def features(
        self, base: bool = True, skill_ratio: bool = True, boss_ratio: bool = True
    ):
        features = pd.DataFrame(index=self.df_clean.index)
        if base:
            features = features.merge(self.df_clean, left_index=True, right_index=True)
        if skill_ratio:
            features = features.merge(
                self.skill_ratio, left_index=True, right_index=True
            )
        if boss_ratio:
            features = features.merge(
                self.boss_ratio, left_index=True, right_index=True
            )
        return features


class playerData:
    def __init__(self, player_data: List[dict], label_data: List[dict]) -> None:
        self.df_players = pd.DataFrame(player_data)
        self.df_labels = pd.DataFrame(label_data)

        self.__clean()

    def __clean(self):
        # clean players
        self.df_players.set_index("id", inplace=True)

        # clean labels
        self.df_labels.set_index("id", inplace=True)

        # merge
        self.df_players = self.df_players.merge(
            self.df_labels, left_on="label_id", right_index=True
        )
        self.df_players.drop(columns=["label_id"], inplace=True)

        # binary label, 1 = bot, 0 = not bot
        self.df_players["binary_label"] = np.where(
            self.df_players["label_jagex"] == 2, 1, 0
        )

    def get(self, binary: bool = False):

        if binary:
            out = self.df_players.loc[:, ["binary_label"]]
            out.rename(columns={"binary_label": "target"}, inplace=True)
        else:
            out = self.df_players.loc[:, ["label"]]
            out.rename(columns={"label": "target"}, inplace=True)

        return out
