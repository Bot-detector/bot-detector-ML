from pydantic import BaseModel

from datetime import date, datetime
from typing import Optional


class Skill(BaseModel):
    skill_name: str
    skill_value: int


class Activity(BaseModel):
    activity_name: str
    activity_value: int


class ScraperDataV3(BaseModel):
    created_at: datetime
    record_date: date
    scraper_id: int
    player_id: int
    skills: list[Skill]
    activities: list[Activity]


class HighscoreData(BaseModel):
    Player_id: int
    attack: Optional[int] = 0
    defence: Optional[int] = 0
    strength: Optional[int] = 0
    hitpoints: Optional[int] = 0
    ranged: Optional[int] = 0
    prayer: Optional[int] = 0
    magic: Optional[int] = 0
    cooking: Optional[int] = 0
    woodcutting: Optional[int] = 0
    fletching: Optional[int] = 0
    fishing: Optional[int] = 0
    firemaking: Optional[int] = 0
    crafting: Optional[int] = 0
    smithing: Optional[int] = 0
    mining: Optional[int] = 0
    herblore: Optional[int] = 0
    agility: Optional[int] = 0
    thieving: Optional[int] = 0
    slayer: Optional[int] = 0
    farming: Optional[int] = 0
    runecraft: Optional[int] = 0
    hunter: Optional[int] = 0
    construction: Optional[int] = 0
    league: Optional[int] = 0
    bounty_hunter_hunter: Optional[int] = 0
    bounty_hunter_rogue: Optional[int] = 0
    cs_all: Optional[int] = 0
    cs_beginner: Optional[int] = 0
    cs_easy: Optional[int] = 0
    cs_medium: Optional[int] = 0
    cs_hard: Optional[int] = 0
    cs_elite: Optional[int] = 0
    cs_master: Optional[int] = 0
    lms_rank: Optional[int] = 0
    soul_wars_zeal: Optional[int] = 0
    abyssal_sire: Optional[int] = 0
    alchemical_hydra: Optional[int] = 0
    barrows_chests: Optional[int] = 0
    bryophyta: Optional[int] = 0
    callisto: Optional[int] = 0
    cerberus: Optional[int] = 0
    chambers_of_xeric: Optional[int] = 0
    chambers_of_xeric_challenge_mode: Optional[int] = 0
    chaos_elemental: Optional[int] = 0
    chaos_fanatic: Optional[int] = 0
    commander_zilyana: Optional[int] = 0
    corporeal_beast: Optional[int] = 0
    crazy_archaeologist: Optional[int] = 0
    dagannoth_prime: Optional[int] = 0
    dagannoth_rex: Optional[int] = 0
    dagannoth_supreme: Optional[int] = 0
    deranged_archaeologist: Optional[int] = 0
    general_graardor: Optional[int] = 0
    giant_mole: Optional[int] = 0
    grotesque_guardians: Optional[int] = 0
    hespori: Optional[int] = 0
    kalphite_queen: Optional[int] = 0
    king_black_dragon: Optional[int] = 0
    kraken: Optional[int] = 0
    kreearra: Optional[int] = 0
    kril_tsutsaroth: Optional[int] = 0
    mimic: Optional[int] = 0
    nightmare: Optional[int] = 0
    nex: Optional[int] = 0
    phosanis_nightmare: Optional[int] = 0
    obor: Optional[int] = 0
    phantom_muspah: Optional[int] = 0
    sarachnis: Optional[int] = 0
    scorpia: Optional[int] = 0
    skotizo: Optional[int] = 0
    tempoross: Optional[int] = 0
    the_gauntlet: Optional[int] = 0
    the_corrupted_gauntlet: Optional[int] = 0
    theatre_of_blood: Optional[int] = 0
    theatre_of_blood_hard: Optional[int] = 0
    thermonuclear_smoke_devil: Optional[int] = 0
    tombs_of_amascut: Optional[int] = 0
    tombs_of_amascut_expert: Optional[int] = 0
    tzkal_zuk: Optional[int] = 0
    tztok_jad: Optional[int] = 0
    venenatis: Optional[int] = 0
    vetion: Optional[int] = 0
    vorkath: Optional[int] = 0
    wintertodt: Optional[int] = 0
    zalcano: Optional[int] = 0
    zulrah: Optional[int] = 0
    rifts_closed: Optional[int] = 0
    artio: Optional[int] = 0
    calvarion: Optional[int] = 0
    duke_sucellus: Optional[int] = 0
    spindel: Optional[int] = 0
    the_leviathan: Optional[int] = 0
    the_whisperer: Optional[int] = 0
    vardorvis: Optional[int] = 0
