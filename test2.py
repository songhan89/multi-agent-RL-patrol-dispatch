import ray
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gym
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.examples.policy.random_policy import RandomPolicy


class BasicMultiAgentMultiSpaces(MultiAgentEnv):
    def __init__(self):
        self.agents = {"agent0", "agent1"}
        self.dones = set()
        # Here i'm replacing the env spaces ie: self.observation_space = gym.spaces.Box(.....) to a multiAgentDict space
        self.observation_space = gym.spaces.Dict(
            {"agent0": gym.spaces.Box(low="-1", high=1, shape=(10,)),
             "agent1": gym.spaces.Box(low="-1", high=1, shape=(20,))})
        self.action_space = gym.spaces.Dict(
            {"agent0": gym.spaces.Discrete(2), "agent1": gym.spaces.Discrete(3)})
        self._agent_ids = set(self.agents)

        self._spaces_in_preferred_format = True
        super().__init__()

    def reset(self):
        self.dones = set()
        return {i: self.observation_space[i].sample() for i in self.agents}

    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}
        for i, action in action_dict.items():
            obs[i], rew[i], done[i], info[i] = self.observation_space[
                                                   i].sample(), 0.0, False, {}
            if done[i]:
                self.dones.add(i)
        done["__all__"] = len(self.dones) == len(self.agents)
        print("step")
        return obs, rew, done, info


def main():
    tune.register_env(
        "ExampleEnv",
        lambda c: BasicMultiAgentMultiSpaces()
    )

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        # agent0 -> main0
        # agent1 -> main1
        return f"main{agent_id[-1]}"

    ray.init(local_mode=True)
    tune.run(
        "PPO",
        stop={"episode_reward_mean": 200},
        config={
            "env": "ExampleEnv",
            "num_gpus": 0,
            "num_workers": 1,
            "multiagent": {
                "policies": {
                    "main0": PolicySpec(observation_space=gym.spaces.Box(low="-1", high=1, shape=(10,)), action_space=gym.spaces.Discrete(2)),
                    "main1": PolicySpec(observation_space=gym.spaces.Box(low="-1", high=1, shape=(20,)), action_space=gym.spaces.Discrete(3)),
                    "random": PolicySpec(policy_class=RandomPolicy),
                },
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": ["main0"]
            },
            "framework": "torch"
        }
    )


if __name__ == "__main__":
    main()
