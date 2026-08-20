from src.engine.entities import Player
from src.engine.simulation import Simulation
from src.engine.vector import Vec2

class TeamHeuristicCoordinator:
    """
    Advanced Multi-Agent Heuristic Coordinator featuring:
    - Dynamic Role Swapping & Angle-Weighted Chaser Selection
    - Relative Pitch Scaling (No magic numbers)
    - Smoothed Arc-Around & Drive-Through striking
    """

    def __init__(self, team: str = "blue"):
        self.team = team
        # Cache current tick to avoid redundant team-wide calculations per player
        self._last_chaser = None
        self._last_ball_pos = None

    def get_action(self, bot_player: Player, sim: Simulation) -> tuple[Vec2, bool]:
        my_team = sim.red_team if self.team == "red" else sim.blue_team
        opp_team = sim.blue_team if self.team == "red" else sim.red_team
        ball = sim.ball
        p = sim.pitch
        sign = 1.0 if self.team == "red" else -1.0

        # 1. Pitch-Relative Metrics (Scalable)
        pitch_width = p.bottom - p.top
        pitch_length = p.right - p.left
        
        own_goal_x = p.left if self.team == "red" else p.right
        opp_goal_x = p.right if self.team == "red" else p.left
        own_goal_pos = Vec2(own_goal_x, sim.center.y)
        opp_goal_pos = Vec2(opp_goal_x, sim.center.y)
        goal_dir = (opp_goal_pos - ball.pos).normalize()

        ball_in_own_half = (sign * (ball.pos.x - sim.center.x) < 0)
        dist_ball_to_own_goal = ball.pos.distance_to(own_goal_pos)

        # 2. Dynamic Team State & Chaser Selection
        # We only want to calculate the optimal chaser once per frame.
        if self._last_ball_pos != ball.pos:
            self._last_ball_pos = ball.pos
            self._last_chaser = self._select_best_chaser(my_team, ball, goal_dir, own_goal_pos, pitch_length)

        is_chaser = (bot_player == self._last_chaser)
        
        # 3. Dynamic Role Identification
        is_gk = False
        if not is_chaser:
            # Assign GK to the player closest to own goal who isn't the chaser
            candidates_gk = [pl for pl in my_team if pl != self._last_chaser]
            if candidates_gk:
                gk_player = min(candidates_gk, key=lambda pl: pl.pos.distance_to(own_goal_pos))
                is_gk = (bot_player == gk_player)

        # 4. Target Position Calculation
        target = bot_player.pos

        if is_chaser:
            if is_gk:
                # Emergency GK Clearance
                target = Vec2(ball.pos.x + (sign * pitch_length * 0.05), ball.pos.y)
            else:
                # --- SMOOTHED ARC & STRIKE LOGIC ---
                bot_to_ball = ball.pos - bot_player.pos
                dist_to_ball = bot_to_ball.length()
                
                dot_align = (bot_to_ball.x * goal_dir.x + bot_to_ball.y * goal_dir.y) / max(dist_to_ball, 1e-5)

                if dot_align > 0.4:
                    # DRIVE THROUGH: Good alignment, push through the ball
                    lead_ball = ball.pos + (ball.vel * 0.15)
                    target = lead_ball + (goal_dir * (pitch_length * 0.08))
                else:
                    # ARC AROUND: Blend target smoothly based on distance
                    prep_behind = ball.pos - (goal_dir * (pitch_length * 0.06))
                    side = 1.0 if (bot_player.pos.y >= ball.pos.y) else -1.0
                    perp = Vec2(-goal_dir.y * side, goal_dir.x * side)
                    
                    # The closer we are, the wider we swing to avoid hitting the ball backwards
                    swing_weight = max(0, 1.0 - (dist_to_ball / (pitch_length * 0.15)))
                    swing_offset = perp * (pitch_length * 0.08 * swing_weight)
                    
                    target = prep_behind + swing_offset
        else:
            # --- PASSIVE / FORMATION LOGIC ---
            if is_gk:
                # Angle Cutting Goalkeeper
                to_ball = (ball.pos - own_goal_pos)
                if to_ball.length_sq() > 0:
                    to_ball = to_ball.normalize()
                
                # Step out further if ball is close, stay back if far
                step_out = min(pitch_length * 0.08, dist_ball_to_own_goal * 0.2)
                target = own_goal_pos + (to_ball * step_out)
                
                # Clamp GK to goal mouth
                goal_half_width = pitch_width * 0.15
                target.y = max(sim.center.y - goal_half_width, min(sim.center.y + goal_half_width, target.y))

            else:
                # Support Players (Dynamic Formation)
                # Sort remaining players by y-position to dynamically assign Left/Center/Right
                supports = sorted([pl for pl in my_team if pl not in (self._last_chaser, gk_player)], key=lambda pl: pl.pos.y)
                
                if len(supports) > 0:
                    my_rank = supports.index(bot_player)
                    total_supports = len(supports)
                    
                    # Spread evenly across the y-axis
                    y_spacing = pitch_width / (total_supports + 1)
                    target_y = p.top + (y_spacing * (my_rank + 1))
                    
                    # Push forward/back relative to ball
                    if ball_in_own_half:
                        # Defensive line, staggered slightly ahead of goal
                        target_x = own_goal_x + (sign * pitch_length * 0.25)
                    else:
                        # Offensive support, trailing behind ball
                        target_x = ball.pos.x - (sign * pitch_length * 0.15)
                        
                    target = Vec2(target_x, target_y)

        # Boundary Clamping
        padding = bot_player.radius + 8.0
        target.x = max(p.outer_left + padding, min(p.outer_right - padding, target.x))
        target.y = max(p.outer_top + padding, min(p.outer_bottom - padding, target.y))

        # 5. Steering & Smart Kicking
        to_target = target - bot_player.pos
        dist_to_target = to_target.length()
        move_dir = to_target.normalize() if dist_to_target > 10.0 else Vec2(0, 0)

        kick = False
        kick_reach = bot_player.radius + ball.radius + bot_player.stats.kick_margin + 6.0

        if bot_player.pos.distance_to(ball.pos) <= kick_reach:
            bot_to_ball_dir = (ball.pos - bot_player.pos).normalize()
            shot_alignment = bot_to_ball_dir.x * goal_dir.x + bot_to_ball_dir.y * goal_dir.y

            if is_gk:
                # GK clears anywhere away from own goal
                if (bot_to_ball_dir.x * sign) > -0.1:
                    kick = True
            else:
                # Outfield players shoot if aligned, or if deep in enemy territory
                in_attacking_third = (sign * (ball.pos.x - sim.center.x) > (pitch_length * 0.15))
                if shot_alignment > 0.25 or (in_attacking_third and (bot_to_ball_dir.x * sign) > 0.15):
                    kick = True

        return move_dir, kick

    def _select_best_chaser(self, team, ball, goal_dir, own_goal_pos, pitch_length):
        """Selects chaser based on distance AND positional alignment."""
        best_chaser = team[0]
        best_score = float('inf')
        
        # If ball is dangerously close to our goal, GK becomes priority chaser
        dist_ball_to_own_goal = ball.pos.distance_to(own_goal_pos)
        if dist_ball_to_own_goal < pitch_length * 0.15:
            # Assume GK is closest to own goal
            return min(team, key=lambda pl: pl.pos.distance_to(own_goal_pos))

        for player in team:
            to_ball = ball.pos - player.pos
            dist = to_ball.length()
            
            if dist > 0:
                dir_to_ball = to_ball.normalize()
                # Alignment: 1.0 = approaching from directly behind the ball (perfect)
                # -1.0 = approaching from opponent's side (terrible, requires looping around)
                alignment = dir_to_ball.x * goal_dir.x + dir_to_ball.y * goal_dir.y
                
                # Cost function: Heavily penalize bad angles. 
                # A player twice as far away but on the correct side might score better.
                angle_penalty = 2.0 - alignment 
                score = dist * angle_penalty
            else:
                score = 0
                
            if score < best_score:
                best_score = score
                best_chaser = player
                
        return best_chaser