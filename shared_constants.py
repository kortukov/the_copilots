EVALUATION_SEEDS = list(range(2283220, 2283220 + 50))
PVP_EVALUATION_SEEDS = list(range(69420, 69420+1000))
VIDEO_SEEDS = [2283220, 2283221]

DEFAULT_TD3_PARAMS = {
    "eps": 0.0,
    "learning_rate_actor": 0.0001,
    "update_target_every": 100,
    "polyak": 0.995,
    "prioritize": False,
    "is_hockey_env": True,
}