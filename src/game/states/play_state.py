import pygame
import pygame.gfxdraw
from src.bots.heuristic_bot import HeuristicBot
from src.engine.simulation import Simulation
from src.engine.vector import Vec2
from src.game.state_manager import GameState
from src.game.states.pause_state import PauseState
from src.game.ui.button import Button
from src.game.camera import Camera


class PlayState(GameState):

    def __init__(self, context):
        super().__init__(context)
        self.sim = Simulation(
            center_x=context.screen_width / 2,
            center_y=context.screen_height / 2 + 20, 
            team_size=1,
        )
        self.bot = HeuristicBot(team="blue")

        # Initialize Camera
        self.camera = Camera(context.screen_width, context.screen_height)

        self.font_score = pygame.font.SysFont("Arial", 28, bold=True)
        self.font_hud = pygame.font.SysFont("Arial", 16)
        self.font_big = pygame.font.SysFont("Arial", 60, bold=True)

        self.btn_pause = Button(
            pygame.Rect(context.screen_width - 100, 12, 80, 32),
            "Pause",
            self.font_hud,
            self._pause_game,
        )

    def _pause_game(self):
        self.context.state_manager.push_state(PauseState(self.context))

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN and (event.key == pygame.K_ESCAPE or event.key == pygame.K_p):
            self._pause_game()
        self.btn_pause.handle_event(event)

    def update(self, dt: float):
        keys = pygame.key.get_pressed()
        
        move_red = Vec2(0, 0)
        if keys[pygame.K_w]: move_red.y -= 1
        if keys[pygame.K_s]: move_red.y += 1
        if keys[pygame.K_a]: move_red.x -= 1
        if keys[pygame.K_d]: move_red.x += 1


        kick_red = keys[pygame.K_SPACE]

        move_blue, kick_blue = self.bot.get_action(self.sim.blue_team[0], self.sim)
    
        self.sim.step(
            red_inputs=[(move_red, kick_red)],
            blue_inputs=[(move_blue, kick_blue)],
            dt=dt,
        )

        # Update Camera to follow the ball
        p = self.sim.pitch
        world_bounds = pygame.Rect(
            p.outer_left, p.outer_top, 
            p.outer_right - p.outer_left, p.outer_bottom - p.outer_top
        )
        self.camera.update(self.sim.ball.pos, world_bounds)

    def _draw_aa_entity(self, surface, pos, radius, fill_color, outline_color, outline_thickness=3):
        """Helper to draw smooth thick anti-aliased circles."""
        x, y = pos
        # Draw Outline (Outer Circle)
        pygame.gfxdraw.filled_circle(surface, x, y, radius, outline_color)
        pygame.gfxdraw.aacircle(surface, x, y, radius, outline_color)
        
        # Draw Inner Fill (Slightly smaller circle)
        inner_radius = radius - outline_thickness
        if inner_radius > 0:
            pygame.gfxdraw.filled_circle(surface, x, y, inner_radius, fill_color)
            pygame.gfxdraw.aacircle(surface, x, y, inner_radius, fill_color)

    def draw(self, surface: pygame.Surface):
        surface.fill((28, 30, 38))
        p = self.sim.pitch
        cam = self.camera

        # 1. Draw Pitch Geometry
        pitch_rect = cam.apply_rect(pygame.Rect(p.left, p.top, p.cfg.PITCH_WIDTH, p.cfg.PITCH_HEIGHT))
        pygame.draw.rect(surface, (45, 125, 60), pitch_rect)
        pygame.draw.rect(surface, (240, 240, 240), pitch_rect, width=4)

        # Center line & Anti-Aliased Center Circle
        pygame.draw.line(surface, (240, 240, 240), cam.apply((p.center.x, p.top)), cam.apply((p.center.x, p.bottom)), 2)
        cx, cy = cam.apply((p.center.x, p.center.y))
        
        # Draw 2 pixels of AA circle for a slightly thicker center ring
        c_rad = int(p.cfg.CENTER_CIRCLE_RADIUS)
        pygame.gfxdraw.aacircle(surface, cx, cy, c_rad, (240, 240, 240))
        pygame.gfxdraw.aacircle(surface, cx, cy, c_rad - 1, (240, 240, 240))

        # Goal Nets
        left_goal = cam.apply_rect(pygame.Rect(p.left - p.cfg.GOAL_DEPTH, p.goal_top, p.cfg.GOAL_DEPTH, p.cfg.GOAL_HEIGHT))
        right_goal = cam.apply_rect(pygame.Rect(p.right, p.goal_top, p.cfg.GOAL_DEPTH, p.cfg.GOAL_HEIGHT))
        pygame.draw.rect(surface, (35, 95, 45), left_goal)
        pygame.draw.rect(surface, (240, 240, 240), left_goal, width=3)
        pygame.draw.rect(surface, (35, 95, 45), right_goal)
        pygame.draw.rect(surface, (240, 240, 240), right_goal, width=3)

        # Goal Posts (Anti-Aliased)
        for post in p.posts:
            self._draw_aa_entity(surface, cam.apply(post.pos), int(post.radius), (255, 255, 255), (40, 40, 40), 1)

        # ---------------------------------------------------------
        # Z-INDEX FIX: Draw Ball FIRST so players render ON TOP
        # ---------------------------------------------------------
        ball = self.sim.ball
        self._draw_aa_entity(surface, cam.apply(ball.pos), int(ball.radius), (255, 255, 255), (20, 20, 20), 2)

        # Red Team
        for player in self.sim.red_team:
            pos = cam.apply(player.pos)
            fill_c = (225, 55, 55)
            # Outline is Black normally, White when kicking (using the visual timer!)
            out_c = (255, 255, 255) if player.kick_visual_timer > 0 else (20, 20, 20)
            thickness = 4 if player.kick_visual_timer > 0 else 3
            self._draw_aa_entity(surface, pos, int(player.radius), fill_c, out_c, thickness)

        # Blue Team
        for player in self.sim.blue_team:
            pos = cam.apply(player.pos)
            fill_c = (50, 110, 235)
            # Outline is Black normally, White when kicking
            out_c = (255, 255, 255) if player.kick_visual_timer > 0 else (20, 20, 20)
            thickness = 4 if player.kick_visual_timer > 0 else 3
            self._draw_aa_entity(surface, pos, int(player.radius), fill_c, out_c, thickness)
            
        # Ball
        ball = self.sim.ball
        self._draw_aa_entity(surface, cam.apply(ball.pos), int(ball.radius), (255, 255, 255), (20, 20, 20), 2)

        # 3. GOAL! Celebration Text
        if self.sim.state == "GOAL_SCORED":
            goal_txt = self.font_big.render("GOAL!", True, (255, 220, 50))
            surface.blit(goal_txt, goal_txt.get_rect(center=(self.context.screen_width // 2, self.context.screen_height // 2 - 100)))

        # 4. Draw HUD (Fixed on screen, bypasses camera)
        hud_bar = pygame.Rect(0, 0, self.context.screen_width, 50)
        pygame.draw.rect(surface, (20, 22, 30), hud_bar)
        pygame.draw.line(surface, (50, 55, 70), (0, 50), (self.context.screen_width, 50), 2)

        # Multi-colored Scoreboard
        red_text = self.font_score.render(f"RED {self.sim.score_red}", True, (225, 55, 55))
        dash_text = self.font_score.render("  -  ", True, (245, 245, 245))
        blue_text = self.font_score.render(f"{self.sim.score_blue} BLUE", True, (50, 110, 235))
        
        # Center the combined scoreboard
        total_w = red_text.get_width() + dash_text.get_width() + blue_text.get_width()
        start_x = (self.context.screen_width - total_w) // 2
        
        surface.blit(red_text, (start_x, 10))
        surface.blit(dash_text, (start_x + red_text.get_width(), 10))
        surface.blit(blue_text, (start_x + red_text.get_width() + dash_text.get_width(), 10))

        controls_lbl = self.font_hud.render("WASD: Move | SPACE: Kick", True, (150, 160, 180))
        surface.blit(controls_lbl, (20, 16))

        self.btn_pause.draw(surface)