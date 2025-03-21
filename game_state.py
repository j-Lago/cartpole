from enum import Enum

class GAMESTATE(Enum):
    PRE_INIT = -1
    RUN = 1
    PAUSED = 0
    TIMEOUT = -2
    GAME_OVER = -3