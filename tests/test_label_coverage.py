"""A vision label only affects clip selection if it is scored (scoring.DEVOTIONAL / highlights.SPECIAL); a role-group label only
does anything if the vision model can actually emit it. Either gap makes a label silently dead."""
from aikyam_video.creative.roles import CFG
from aikyam_video.highlights import SPECIAL
from aikyam_video.scoring import DEVOTIONAL
from aikyam_video.vision import LABEL_PROMPTS


def test_every_vision_label_is_scored():
    assert set(LABEL_PROMPTS) <= set(DEVOTIONAL) | set(SPECIAL)


def test_every_role_group_label_can_be_detected():
    assert {l for g in CFG["labels"].values() for l in g} <= set(LABEL_PROMPTS)
