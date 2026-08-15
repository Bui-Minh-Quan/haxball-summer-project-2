from abc import ABC, abstractmethod
from typing import Any
import pygame


class GameContext:
    """Service container holding shared dependencies injected into every state."""

    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.clock = pygame.time.Clock()
        self.state_manager: "StateManager | None" = None
        self.running = True



class GameState(ABC):
    """Abstract Base Class defining the lifecycle interface for scenes."""

    def __init__(self, context: GameContext):
        self.context = context

    def enter(self, **kwargs: Any) -> None:
        """Called when this state becomes active or is entered."""
        pass

    def exit(self) -> None:
        """Called when this state is destroyed or replaced."""
        pass

    @abstractmethod
    def handle_event(self, event: pygame.event.Event) -> None:
        """Processes OS inputs directed at this active scene."""
        pass

    @abstractmethod
    def update(self, dt: float) -> None:
        """Updates physics, timers, and game logic."""
        pass

    @abstractmethod
    def draw(self, surface: pygame.Surface) -> None:
        """Renders scene graphics onto the target surface."""
        pass


class StateManager:
    """Stack-based Finite State Machine supporting transitions and overlays."""

    def __init__(self, context: GameContext):
        self.context = context
        self.context.state_manager = self
        self._stack: list[GameState] = []

    @property
    def current_state(self) -> GameState | None:
        """Returns the top-most active state."""
        return self._stack[-1] if self._stack else None

    def change_state(self, new_state: GameState, **kwargs: Any) -> None:
        """Clears the stack and transitions to a fresh state (e.g. Menu -> Play)."""
        while self._stack:
            popped = self._stack.pop()
            popped.exit()

        self._stack.append(new_state)
        new_state.enter(**kwargs)

    def push_state(self, new_state: GameState, **kwargs: Any) -> None:
        """Pushes an overlay state on top without destroying underlying scenes (e.g. Pause)."""
        self._stack.append(new_state)
        new_state.enter(**kwargs)

    def pop_state(self) -> None:
        """Removes the active overlay state and resumes the state underneath."""
        if len(self._stack) > 1:
            popped = self._stack.pop()
            popped.exit()

    def handle_event(self, event: pygame.event.Event) -> None:
        """Dispatches input events strictly to the top-most active state."""
        if self.current_state:
            self.current_state.handle_event(event)

    def update(self, dt: float) -> None:
        """Updates simulation strictly on the top-most active state."""
        if self.current_state:
            self.current_state.update(dt)

    def draw(self, surface: pygame.Surface) -> None:
        """Draws all states in stack order so translucent overlays render over previous scenes."""
        for state in self._stack:
            state.draw(surface)