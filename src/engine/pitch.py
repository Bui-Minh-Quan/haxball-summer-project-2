from config.physics_config import PhysicsConfig
from src.engine.entities import GoalPost
from src.engine.vector import Vec2


class Pitch:

    def __init__(
        self,
        center: Vec2,
        cfg: PhysicsConfig,
        width: float | None = None,
        height: float | None = None,
    ):
        self.cfg = cfg
        self.center = center
        self.width = width or cfg.DEFAULT_PITCH_WIDTH
        self.height = height or cfg.DEFAULT_PITCH_HEIGHT

        # Inner Pitch Bounds (Ball bounces here)
        self.left = center.x - self.width / 2
        self.right = center.x + self.width / 2
        self.top = center.y - self.height / 2
        self.bottom = center.y + self.height / 2

        # Outer Fence Bounds (Players can run outside lines up to here)
        self.outer_left = self.left - cfg.WIDTH_MARGIN
        self.outer_right = self.right + cfg.WIDTH_MARGIN
        self.outer_top = self.top - cfg.HEIGHT_MARGIN
        self.outer_bottom = self.bottom + cfg.HEIGHT_MARGIN

        # Goal Y-interval
        self.goal_top = center.y - cfg.GOAL_HEIGHT / 2
        self.goal_bottom = center.y + cfg.GOAL_HEIGHT / 2

        # Goal Posts
        self.posts = [
            GoalPost(Vec2(self.left, self.goal_top), cfg),
            GoalPost(Vec2(self.left, self.goal_bottom), cfg),
            GoalPost(Vec2(self.right, self.goal_top), cfg),
            GoalPost(Vec2(self.right, self.goal_bottom), cfg),
        ]