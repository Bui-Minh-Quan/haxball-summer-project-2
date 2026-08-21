from src.engine.entities import Player
from src.engine.simulation import Simulation
from src.engine.vector import Vec2

class TeamHeuristicCoordinator:
    """
    Advanced Multi-Agent Heuristic Coordinator featuring:
    - Aggressive forward kicking & self-passing
    - Cross-product Arc-Around pathing (prevents backward clipping)
    - Ray-Casted Shot suppression (only holds ball if blocked)
    """

    def __init__(self, team: str = "blue"):
        self.team = team
        self._last_chaser = None
        self._last_ball_pos = None

    def _is_path_blocked(self, start: Vec2, end: Vec2, opponents: list[Player], radius_threshold: float = 35.0) -> bool:
        """Raycast check to see if an opponent blocks the shot trajectory."""
        ray = end - start
        ray_len = ray.length()
        if ray_len == 0:
            return False
        
        ray_dir = ray.normalize()
        
        for opp in opponents:
            to_opp = opp.pos - start
            proj_length = to_opp.x * ray_dir.x + to_opp.y * ray_dir.y
            
            # Check if opponent is between ball and target
            if 0 < proj_length < ray_len:
                perp_dist = abs(to_opp.x * (-ray_dir.y) + to_opp.y * ray_dir.x)
                if perp_dist < radius_threshold:
                    return True
        return False

    def get_action(self, bot_player: Player, sim: Simulation) -> tuple[Vec2, bool]:
        my_team = sim.red_team if self.team == "red" else sim.blue_team
        opp_team = sim.blue_team if self.team == "red" else sim.red_team
        ball = sim.ball
        p = sim.pitch
        sign = 1.0 if self.team == "red" else -1.0

        pitch_width = p.bottom - p.top
        pitch_length = p.right - p.left
        
        own_goal_x = p.left if self.team == "red" else p.right
        opp_goal_x = p.right if self.team == "red" else p.left
        own_goal_pos = Vec2(own_goal_x, sim.center.y)
        opp_goal_pos = Vec2(opp_goal_x, sim.center.y)
        
        # The vector pointing straight from the ball to the center of the opponent's net
        goal_dir = (opp_goal_pos - ball.pos).normalize()

        ball_in_own_half = (sign * (ball.pos.x - sim.center.x) < 0)
        dist_ball_to_own_goal = ball.pos.distance_to(own_goal_pos)

        # 1. Assign Active Chaser
        if self._last_ball_pos != ball.pos:
            self._last_ball_pos = ball.pos
            self._last_chaser = self._select_best_chaser(my_team, ball, goal_dir, own_goal_pos, pitch_length)

        is_chaser = (bot_player == self._last_chaser)
        
        # 2. Assign Goalkeeper
        is_gk = False
        gk_player = None
        if not is_chaser:
            candidates_gk = [pl for pl in my_team if pl != self._last_chaser]
            if candidates_gk:
                gk_player = min(candidates_gk, key=lambda pl: pl.pos.distance_to(own_goal_pos))
                is_gk = (bot_player == gk_player)

        # 3. Movement Target Calculation
        target = bot_player.pos

        if is_chaser:
            if is_gk:
                # GK clears ball aggressively forward
                target = Vec2(ball.pos.x + (sign * 50.0), ball.pos.y)
            else:
                bot_to_ball = ball.pos - bot_player.pos
                dist_to_ball = bot_to_ball.length()
                dir_to_ball = bot_to_ball.normalize() if dist_to_ball > 0 else goal_dir
                
                # Alignment: 1.0 is perfectly behind the ball.
                alignment = dir_to_ball.x * goal_dir.x + dir_to_ball.y * goal_dir.y

                if alignment > 0.25:
                    # PHASE 1: CHARGE (Aggressive Attack)
                    # Target a point slightly *through* the ball to maintain a straight power drive
                    target = ball.pos + (goal_dir * 20.0)
                else:
                    # PHASE 2: ARC AROUND (Repositioning)
                    behind_ball = ball.pos - (goal_dir * 55.0)
                    
                    if dist_to_ball < 120.0:
                        # Use cross-product to find the shortest evasion path (left or right)
                        cross = goal_dir.x * bot_to_ball.y - goal_dir.y * bot_to_ball.x
                        side = 1.0 if cross > 0 else -1.0
                        perp = Vec2(-goal_dir.y * side, goal_dir.x * side)
                        
                        # Step wider the closer the bot is to the ball to prevent clipping
                        evasion_weight = max(0.0, 1.0 - (dist_to_ball / 120.0))
                        target = behind_ball + (perp * 85.0 * evasion_weight)
                    else:
                        target = behind_ball
        else:
            if is_gk:
                to_ball = (ball.pos - own_goal_pos)
                if to_ball.length_sq() > 0:
                    to_ball = to_ball.normalize()
                
                step_out = min(pitch_length * 0.08, dist_ball_to_own_goal * 0.2)
                target = own_goal_pos + (to_ball * step_out)
                
                goal_half_width = pitch_width * 0.15
                target.y = max(sim.center.y - goal_half_width, min(sim.center.y + goal_half_width, target.y))
            else:
                supports = sorted([pl for pl in my_team if pl not in (self._last_chaser, gk_player)], key=lambda pl: pl.pos.y)
                if len(supports) > 0:
                    my_rank = supports.index(bot_player)
                    y_spacing = pitch_width / (len(supports) + 1)
                    target_y = p.top + (y_spacing * (my_rank + 1))
                    
                    if ball_in_own_half:
                        target_x = own_goal_x + (sign * pitch_length * 0.25)
                    else:
                        target_x = ball.pos.x - (sign * pitch_length * 0.15)
                    target = Vec2(target_x, target_y)

        # Boundary Clamping
        padding = bot_player.radius + 8.0
        target.x = max(p.outer_left + padding, min(p.outer_right - padding, target.x))
        target.y = max(p.outer_top + padding, min(p.outer_bottom - padding, target.y))

        to_target = target - bot_player.pos
        dist_to_target = to_target.length()
        move_dir = to_target.normalize() if dist_to_target > 5.0 else Vec2(0, 0)

        # 4. Smart Kicking Logic
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
                # Condition A: Deep in own half? Clear it forward immediately.
                if dist_ball_to_own_goal < (pitch_length * 0.3) and (bot_to_ball_dir.x * sign) > 0.1:
                    kick = True
                
                # Condition B: Midfield or Attacking (Aligned for a shot/pass)
                elif shot_alignment > 0.4:
                    blocked = self._is_path_blocked(ball.pos, opp_goal_pos, opp_team, radius_threshold=35.0)
                    
                    if not blocked:
                        kick = True
                    else:
                        # If blocked, but very close to the net, blast it anyway for a rebound
                        if ball.pos.distance_to(opp_goal_pos) < (pitch_length * 0.25):
                            kick = True
                        # Otherwise, let physics dribble the ball left/right around the block

        return move_dir, kick

    def _select_best_chaser(self, team, ball, goal_dir, own_goal_pos, pitch_length):
        best_chaser = team[0]
        best_score = float('inf')
        
        # Emergency Override: If ball is in our box, closest player takes it
        if ball.pos.distance_to(own_goal_pos) < pitch_length * 0.15:
            return min(team, key=lambda pl: pl.pos.distance_to(own_goal_pos))

        for player in team:
            to_ball = ball.pos - player.pos
            dist = to_ball.length()
            if dist > 0:
                dir_to_ball = to_ball.normalize()
                alignment = dir_to_ball.x * goal_dir.x + dir_to_ball.y * goal_dir.y
                # Less severe angle penalty, allowing faster interceptors to take charge
                angle_penalty = 1.5 - (alignment * 0.5)
                score = dist * angle_penalty
            else:
                score = 0
                
            if score < best_score:
                best_score = score
                best_chaser = player
        return best_chaser