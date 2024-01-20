import numpy as np
import pygame
import sys

import gymnasium as gym
from gymnasium import spaces

class Bridge(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None, size=8):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                
            }
        )

        self.grid_types = {
            "water": 0,
            "land": 1
        }

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        self.grid_list = None
        self.grid_array=None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

        #for including the landscape in the observation space if we want to train the agent in similar environments:
        #return {"agent": self._agent_location, "target": self._target_location, "landscape": self.grid_list}

    def _get_info(self):
        return {
        "distance": np.linalg.norm(
            self._agent_location - self._target_location, ord=1
        )
    }

    def get_grid_type(self, location):
        return self.grid_list[location[1]][location[0]]

    def reset(self, seed=None, options=None):
    # We need the following line to seed self.np_random
        super().reset(seed=seed)

        vertical_land_line = [[self.grid_types["land"] for _ in range(self.size)]]

        self.grid_list = vertical_land_line + vertical_land_line + [
            [self.grid_types["land"] if (i+1) % self.size == self.size/2 or (i+1) % self.size == self.size/2+1 else self.grid_types["water"] for i in range(self.size)]
            for _ in range(self.size-4) #range needs to be self.size- "number of vertical land lines"
        ] + vertical_land_line + vertical_land_line

        self.grid_array = np.array(self.grid_list)

        # Choose the agent's location at uniformly at random and ensure that it doesn't spawn in the water
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        agent_grid_type =  self.get_grid_type(self._agent_location)
        while agent_grid_type == self.grid_types["water"]:
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            agent_grid_type = self.get_grid_type(self._agent_location)

         # Choose the target's location uniformly at random and ensure that it doen't spawn in the water or on the agent
        self._target_location = self._agent_location
        target_grid_type = self.get_grid_type(self._target_location)
        while np.array_equal(self._target_location, self._agent_location) or target_grid_type == self.grid_types["water"]:
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            target_grid_type = self.get_grid_type(self._target_location)

        #alternatively: set the target location to a fixed point
        #self._target_location = np.array([self.size-1, self.size-1])

        observation = self._get_obs()
        info = self._get_info()
        self.render()

        return observation, info
    
    def step(self, action):
        terminated = False
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        #give a negative reward for eacch step taken
        reward = -1  

        # An episode is done iff the agent has reached the target or falls into the water
        if np.array_equal(self._agent_location, self._target_location):
            reward =100
            terminated = True

        if self.get_grid_type(self._agent_location)==self.grid_types["water"]:
            reward = -100
            terminated = True

        observation = self._get_obs()
        info = self._get_info()

        self.render()

        return observation, reward, terminated, False, info
    
    def render(self):
        if self.render_mode == "human":
         return self._render_frame()

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None:
            self.clock = pygame.time.Clock()


        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 0, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        y_coordinate=0
        for row in self.grid_array:
            x_coordinate=0
            for grid in row:
                if self.grid_types["water"] == grid:
                    pygame.draw.rect(
                        canvas,
                        (0,0,255),
                        pygame.Rect(
                        (x_coordinate*pix_square_size, y_coordinate*pix_square_size),
                        (pix_square_size, pix_square_size),
                        ),
                    )
                    
                elif self.grid_types["land"] == grid:
                    pygame.draw.rect(
                        canvas,
                         (139, 69, 19),
                        pygame.Rect(
                        (x_coordinate*pix_square_size, y_coordinate*pix_square_size),
                        (pix_square_size, pix_square_size),
                        ),
                    )
                x_coordinate+=1
            y_coordinate+=1

        #draw the target
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (255, 255, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(canvas, canvas.get_rect())

        #check if the window has been closed manually before handling additional events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                pygame.quit()
                sys.exit()

        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

#registering the environment: note that the module (file) name, in this case bridge_env needs to be written without the filetype .py

gym.register(
    id='bridge-v0',
    entry_point="src.environments.bridge:Bridge",  # Replace with your actual module and class name
    max_episode_steps=300,
    kwargs={'render_mode': None}
)