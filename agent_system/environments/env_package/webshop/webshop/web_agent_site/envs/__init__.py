from gym.envs.registration import register

from web_agent_site.envs.web_agent_site_env import WebAgentSiteEnv
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

# disable_env_checker / order_enforce: this env was written for gym 0.24 and does
# not declare action_space/observation_space. Newer gym (0.26+) wraps make() with
# PassiveEnvChecker which hard-asserts those exist. Downstream (Ray worker + env_manager)
# drives the env via text actions and never samples the spaces, so we skip the checker.
register(
  id='WebAgentSiteEnv-v0',
  entry_point='web_agent_site.envs:WebAgentSiteEnv',
  disable_env_checker=True,
  order_enforce=False,
)

register(
  id='WebAgentTextEnv-v0',
  entry_point='web_agent_site.envs:WebAgentTextEnv',
  disable_env_checker=True,
  order_enforce=False,
)