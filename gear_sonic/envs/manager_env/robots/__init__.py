# Re-export robot configs for backward compatibility.
# Import the corresponding robot module directly for new code.
from gear_sonic.envs.manager_env.robots.g1 import *  # noqa: F401,F403
from gear_sonic.envs.manager_env.robots.h2 import *  # noqa: F401,F403
from gear_sonic.envs.manager_env.robots.elf3 import *  # noqa: F401,F403
