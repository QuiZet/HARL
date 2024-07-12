"""Train an algorithm."""
import sys
sys.path.append('.')
import argparse
import json
from harl.utils.configs_tools import get_defaults_yaml_args, update_args
from harl.utils.envs_tools import make_train_env

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="happo",
        choices=[
            "happo",
            "hatrpo",
            "chatrpo",
            "haa2c",
            "haddpg",
            "hatd3",
            "hasac",
            "had3qn",
            "maddpg",
            "matd3",
            "mappo",
            "embd",
        ],
        help="Algorithm name. Choose from: happo, hatrpo, chatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, matd3, mappo, embd.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="pettingzoo_mpe",
        choices=[
            "smac",
            "mamujoco",
            "pettingzoo_mpe",
            "gym",
            "football",
            "dexhands",
            "smacv2",
            "lag",
        ],
        help="Environment name. Choose from: smac, mamujoco, pettingzoo_mpe, gym, football, dexhands, smacv2, lag.",
    )
    parser.add_argument(
        "--exp_name", type=str, default="installtest", help="Experiment name."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    parser.add_argument(
        "--modified_env",
        type=bool,
        default=False,
    )
    parser.add_argument(
        '--class_ac',
        type=bool,
        default=False,
        help='Whether to use class-based actor and critic'
    )
    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict
    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:  # load config from corresponding yaml file
        algo_args, env_args = get_defaults_yaml_args(args['algo'], args['env'])

    update_args(unparsed_dict, algo_args, env_args)

    if args["env"] == "dexhands":
        import isaacgym  # isaacgym has to be imported before PyTorch

    # note: isaac gym does not support multiple instances, thus cannot eval separately
    if args["env"] == "dexhands":
        algo_args["eval"]["use_eval"] = False
        algo_args["train"]["episode_length"] = env_args["hands_episode_length"]

    # Initialize the environment
    envs = make_train_env(
        args["env"],
        algo_args["seed"]["seed"],
        algo_args["train"]["n_rollout_threads"],
        env_args,
    )

    # Modified part to initialize agent classes and critics
    if args['class_ac']:
        from harl.utils.envs_tools import get_num_agents

        num_agents = get_num_agents(args['env'], env_args, envs)
        agent_classes = {}  # Dictionary to hold class-based agents
        for agent_id in range(num_agents):
            obs_space = envs.observation_space[agent_id]
            act_space = envs.action_space[agent_id]
            obs_act_tuple = (tuple(obs_space.shape), obs_space.dtype.name, tuple(act_space.shape), act_space.dtype.name)
            obs_act_str = str(obs_act_tuple)  # Convert tuple to string
            if obs_act_str not in agent_classes:
                agent_classes[obs_act_str] = len(agent_classes) + 1
            class_id = agent_classes[obs_act_str]
            if class_id not in agent_classes:
                agent_classes[class_id] = []
            agent_classes[class_id].append(agent_id)
        algo_args['agent_classes'] = agent_classes

    # start training
    from harl.runners import RUNNER_REGISTRY

    if args['class_ac']:
        print(f'agent_classes in train_copy.py: {agent_classes}')
        runner = RUNNER_REGISTRY[args['algo']](args, algo_args, env_args, agent_classes)
    else:
        runner = RUNNER_REGISTRY[args['algo']](args, algo_args, env_args)
    runner.run()
    runner.close()


if __name__ == "__main__":
    main()