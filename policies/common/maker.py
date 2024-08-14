
def make_policy(config, stage=None):  # TODO: remove this function and use the config file
    policy_maker = config["policy_maker"]
    policy = policy_maker(config, stage)
    assert policy is not None, "Please use the make_policy function in the config file"
    return policy
