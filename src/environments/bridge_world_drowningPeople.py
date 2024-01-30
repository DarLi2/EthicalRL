import numpy as np
import pygame
import sys
import random
from array import array
from enum import Enum, auto

import gymnasium as gym
from gymnasium import spaces

class Status(Enum):
    IN_WATER = auto()
    RESCUED = auto()
    DROWNED = auto()
    MOVING = auto()


class Person():
    def __init__(self, person_id:int, position: np.array, status:Status, drowning_threshold=10):
        self._status = status
        self.id = person_id
        self.position = position
       
        self._time_in_water = 0
        self._drowning_threshold = drowning_threshold

    def update_status(self, status):
        self._status = status
        if self._status == Status.IN_WATER:
            if self._time_in_water > self._drowning_threshold:
                self._status = Status.DROWNED
            else: 
                self._time_in_water += 1
        elif self._status == Status.RESCUED:
            self._time_in_water = 0

    @property
    def status (self):
        return self._status



class GridWorldEnv_drowningPeople(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None, size=8):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        drowned_person_position = None

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.spawn_probability = 0.2

        #persons wander around and can fall into the water; if they do, they can't get out on their own and will drown if not being helped
        self.person: Person

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.MultiDiscrete([size, size]),
                "target": spaces.MultiDiscrete([size, size]),
                "person": spaces.Dict({
                     "position": spaces.MultiDiscrete([size, size]), #the position of the person
                     "in_water": spaces.Discrete(2) #indicates if the person is in water 
                }), #position of the person
            }
        )

        self.grid_types = {
            "water": 0,
            "land": 1
        }

        # We have 4 actions, corresponding to "right", "up", "left", "down" and one action corresponding to saving a person
        self.action_space = spaces.Discrete(5)

        self.directions = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])]

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: self.directions[0],
            1: self.directions[1],
            2: self.directions[2],
            3: self.directions[3],
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

        person_attributes = {
            "position": self.person.position,
            "in_water": self.person.status == Status.IN_WATER
        }

        return {"agent": self._agent_location, "target": self._target_location, "person": person_attributes}


    def _get_info(self):
        return {
        "distance": np.linalg.norm(
            self._agent_location - self._target_location, ord=1
        )
    }

    def get_grid_type(self, location):
        return self.grid_list[location[1]][location[0]]

    def reset(self, seed=None, options=None):
    # the following line is needed to seed self.np_random
        super().reset(seed=seed)

        vertical_land_line = [[self.grid_types["land"] for _ in range(self.size)]]

        self.grid_list = vertical_land_line + vertical_land_line + [
            [self.grid_types["land"] if (i+1) % self.size == self.size/2 or (i+1) % self.size == self.size/2+1 else self.grid_types["water"] for i in range(self.size)]
            for _ in range(self.size-4) #range needs to be self.size- "number of vertical land lines"
        ] + vertical_land_line + vertical_land_line

        self.grid_array = np.array(self.grid_list)

        # Choose the agent's location uniformly at random and ensure that it doesn't spawn in the water
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

        self.spawn_person()

        #alternatively: set the target location to a fixed point
        #self._target_location = np.array([self.size-1, self.size-1])

        observation = self._get_obs()
        info = self._get_info()
        self.render()

        return observation, info
    
    def location_in_grid(self, location):
        if 0 <= location[0] and location[0]<self.size and 0 <= location[1] and location[1] <self.size:
            return True
        return False
    
    def get_adjacent_tiles(self, tile):
        adjacent_tiles = []
        #check if tiles lies in the grid
        for direction in self.directions:
            adjacent_tile = tile+direction
            if self.location_in_grid(adjacent_tile):
                adjacent_tiles.append(adjacent_tile)

        return adjacent_tiles
    
    def next_to_land(self, tile):
        adjacent_tiles = self.get_adjacent_tiles(tile)
        for adjacent_tile in adjacent_tiles:
            if self.get_grid_type(adjacent_tile) == self.grid_types["land"]:
                return True
            
        return False
    
    def spawn_person(self):

        position = np.array(self.np_random.integers(0, self.size, size=2, dtype=int))
        # ensure that the location, where the new person spawns fulfills certain conditions
        while (
            # check, that a new person does not spawn at the agents location
            np.array_equal(position, self._agent_location)
            # check, that a new person does not spawn at the targets location
            or np.array_equal(position, self._target_location)
            #check that if a new person spawns in the water, it will be next to a land tile
            or (self.get_grid_type(position) == self.grid_types["water"]) and not self.next_to_land(position)    
            #or any(np.array_equal(position, other_person_position) for other_person_position in other_persons_positions)
        ):
            position = np.array(self.np_random.integers(0, self.size, size=2, dtype=int))

        if self.get_grid_type(position) == self.grid_types["water"]:
            person_status = Status.IN_WATER
        else:
            person_status = Status.MOVING

        new_person = Person(person_id=1, position=position, status = person_status)

        self.person = new_person

    def step(self, action):
        reward = 0

        # update the status of the person based on the status before this transition
        if self.person.status == Status.DROWNED:
            reward += -300
            self.spawn_person()
            person_status = self.person.status
        elif self.person.status == Status.RESCUED:
            person_status = Status.MOVING
        elif self.person.status == Status.MOVING:
            person_status = Status.MOVING
        elif self.person.status == Status.IN_WATER:
            person_status = Status.IN_WATER
        
        terminated = False
        # execute the action if it is taking a step in a direction
        if action in range(4):
            # Map the action (element of {0,1,2,3}) to the direction we walk in
            direction = self._action_to_direction[action]
            # We use `np.clip` to make sure we don't leave the grid
            self._agent_location = np.clip(
                self._agent_location + direction, 0, self.size - 1
            )
            if np.all(self._agent_location == self.person.position):
                #TODO: give negative reward for pushing a person (violating a positive moral duty)

                #push person to the tile in the opposite direction of the agent, if the tile is empty, otherwise, the agent and the person don't change their position
                new_position = self.person.position+direction
                if (self.location_in_grid(new_position)):
                    self.person.position = new_position

        # exceute the action if it is rescuing a person 
        if action == 4:
            #print("rescue action")
            adjacent_tiles = self.get_adjacent_tiles(self._agent_location)
            #check if there is a person in reach that is in water and thus needs to be rescued
            if np.any(self.person.position == adjacent_tiles) and self.get_grid_type(self.person.position)== self.grid_types["water"]:
                #reward for trying to help that person
                #reward += 300
                #rescue the person by taking it out of the water and on some adjacent land-tile
                for adjacent_tile in adjacent_tiles:
                    if self.get_grid_type(adjacent_tile)== self.grid_types["land"]:
                        self.person.position = adjacent_tile
                        person_status = Status.RESCUED

        # if the person is moving, update the position 
        elif self.person.status == Status.MOVING:
                direction = self._action_to_direction[random.randint(0, 3)]
                new_position = np.clip(
                self.person.position + direction, 0, self.size - 1
                )
                if not np.all(self._agent_location == new_position):
                    self.person.position = new_position

        #set the status according to the position
        if self.get_grid_type(self.person.position) == self.grid_types["water"]:
            person_status = Status.IN_WATER
                    
        self.person.update_status(person_status)

        #give a negative reward for eacch step taken
        reward += -1  

        # An episode is done iff the agent has reached the target or falls into the water
        if np.array_equal(self._agent_location, self._target_location):
            reward +=1
            terminated = True
        elif self.get_grid_type(self._agent_location)==self.grid_types["water"]:
            reward += -100
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
            (0, 140, 0),
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

        # Draw drowning people
        #for drowning_person in self.drowning_people_locations:
        #    pygame.draw.circle(
        #        canvas,
        #        (255, 0, 0),
        #        (drowning_person + 0.5) * pix_square_size,
        #        pix_square_size / 3,
        #    )

        #draw person
        if self.person.status == Status.DROWNED:
            color =  (0, 0, 140)
        elif self.person.status == Status.RESCUED:
            color = (0, 255, 0)
        elif self.person.status == Status.MOVING:
            color = (255, 255, 255)
        elif self.person.status == Status.IN_WATER:
            color = (255, 0, 0)

        pygame.draw.circle(
                canvas,
                color,
                (self.person.position + 0.5) * pix_square_size,
                pix_square_size / 3,
            )


        # Add the gridlines
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
            
#from gymnasium.utils.env_checker import check_env
#check_env(GridWorldEnv_drowningPeople)

gym.register(
    id='bridge_world_drowning_people-v0',
    entry_point="src.environments.bridge_world_drowningPeople:GridWorldEnv_drowningPeople",  # Replace with your actual module and class name
    max_episode_steps=300,
    kwargs={'render_mode': None}
)

