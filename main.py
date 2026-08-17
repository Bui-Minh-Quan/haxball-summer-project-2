import pygame
from config.physics_config import PhysicsConfig
from src.game.app import App

if __name__ == "__main__":
    # Initialize display to read hardware screen size
    pygame.display.init()
    info = pygame.display.Info()
    
    # Leave a 100px margin so the window title bar and taskbar remain accessible
    max_w = info.current_w - 100
    max_h = info.current_h - 100
    
    # Calculate desired size based on pitch, but clamp it to the screen's max
    desired_w = PhysicsConfig.DEFAULT_PITCH_WIDTH + 200
    desired_h = PhysicsConfig.DEFAULT_PITCH_HEIGHT + 200
    
    window_w = min(desired_w, max_w)
    window_h = min(desired_h, max_h)
    
    app = App(width=int(window_w), height=int(window_h))
    app.run()