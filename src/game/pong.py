import pygame
import random
import math
from src.vision.finger_flow_tracker import FingerFlowTracker

# --------------------
# Tracker (OpenCV/MediaPipe/Optical Flow)
# --------------------
tracker = FingerFlowTracker()
tracker.start()

# --------------------
# Game constants
# --------------------
W, H = 800, 600
FPS = 60

PADDLE_W, PADDLE_H = 16, 110
BALL_R = 10

SMASH_SPEED_THRESHOLD = 100.0   # px/s (à calibrer)
SMASH_MULT = 1.6
MAX_BALL_SPEED = 1400.0
SPIN_FACTOR = 0.25              # influence de finger_vy sur la balle (px/s -> delta vy)

# --------------------
# Pygame init
# --------------------
pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Pong Optical Flow Smash")
clock = pygame.time.Clock()
font = pygame.font.SysFont("comicsansms", 28)

def clamp(x, a, b):
    return max(a, min(b, x))

def reset_ball(direction=1):
    x, y = W // 2, H // 2
    speed = 420.0
    angle = random.uniform(-0.35, 0.35)
    vx = direction * speed
    vy = speed * math.tan(angle)
    return pygame.Vector2(x, y), pygame.Vector2(vx, vy)

# Player paddle (left)
paddle = pygame.Rect(40, H // 2 - PADDLE_H // 2, PADDLE_W, PADDLE_H)

# AI paddle (right)
ai = pygame.Rect(W - 40 - PADDLE_W, H // 2 - PADDLE_H // 2, PADDLE_W, PADDLE_H)
AI_SPEED = 300.0
AI_AIM_ERROR = 50  # pixels (30 easy, 50 normal, 80 très easy)

ball_pos, ball_vel = reset_ball(direction=random.choice([-1, 1]))
score_p, score_ai = 0, 0

# Smash feedback timer
smash_flash = 0.0

# Finger motion variables (from tracker)
finger_vy = 0.0
finger_speed = 0.0

running = True

try:
    while running:
        # dt first (game time)
        dt = clock.tick(FPS) / 1000.0

        # events
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                    
        # read tracker (optical flow-based velocity)
        state = tracker.read(with_vis=False)
        frame = state["frame"][:,:,::-1]
        frame = pygame.surfarray.make_surface(frame.swapaxes(0,1))
        frame = pygame.transform.scale(frame, (W, H))

        # ---- INPUT (finger tracker) ----
        if state["active"] and state["pos"] is not None:
            my = state["pos"][1]          # finger Y position
            finger_vy = state["vel"][1]   # finger velocity Y (px/s) from optical flow
            finger_speed = state["speed"] # speed magnitude (px/s)
        else:
            # fallback for debug if finger not detected
            my = pygame.mouse.get_pos()[1]
            finger_vy = 0.0
            finger_speed = 0.0

        # paddle follows finger (vertical only)
        paddle.centery = int(clamp(my, PADDLE_H // 2, H - PADDLE_H // 2))

        # ---- AI movement ----
        target_y = ball_pos.y + random.uniform(-AI_AIM_ERROR, AI_AIM_ERROR) # ajoute de l'erreur dans la visée 
        dy = target_y - ai.centery
        step = clamp(dy, -AI_SPEED * dt, AI_SPEED * dt)
        ai.centery = int(clamp(ai.centery + step, PADDLE_H // 2, H - PADDLE_H // 2))

        # ---- BALL PHYSICS ----
        ball_pos += ball_vel * dt

        # bounce top/bottom
        if ball_pos.y < BALL_R:
            ball_pos.y = BALL_R
            ball_vel.y *= -1
        if ball_pos.y > H - BALL_R:
            ball_pos.y = H - BALL_R
            ball_vel.y *= -1

        ball_rect = pygame.Rect(
            int(ball_pos.x - BALL_R),
            int(ball_pos.y - BALL_R),
            BALL_R * 2,
            BALL_R * 2,
        )

        # ---- COLLISIONS ----
        # Player hit
        if ball_rect.colliderect(paddle) and ball_vel.x < 0:
            ball_pos.x = paddle.right + BALL_R
            ball_vel.x *= -1

            smash = finger_speed > SMASH_SPEED_THRESHOLD
            if smash:
                ball_vel *= SMASH_MULT
                smash_flash = 0.15  # seconds

            # add "spin" based on finger vertical velocity
            ball_vel.y += SPIN_FACTOR * finger_vy

            # clamp speed
            sp = ball_vel.length()
            if sp > MAX_BALL_SPEED:
                ball_vel.scale_to_length(MAX_BALL_SPEED)

        # AI hit
        if ball_rect.colliderect(ai) and ball_vel.x > 0:
            ball_pos.x = ai.left - BALL_R
            ball_vel.x *= -1

            # small randomness
            ball_vel.y += random.uniform(-80, 80)

            sp = ball_vel.length()
            if sp > MAX_BALL_SPEED:
                ball_vel.scale_to_length(MAX_BALL_SPEED)

        # ---- SCORE ----
        if ball_pos.x < -50:
            score_ai += 1
            ball_pos, ball_vel = reset_ball(direction=1)
        if ball_pos.x > W + 50:
            score_p += 1
            ball_pos, ball_vel = reset_ball(direction=-1)

        # ---- RENDER ----
        overlay = pygame.Surface((W,H))
        overlay.fill((0,0,0))
        overlay.set_alpha(120)

        screen.blit(frame,(0,0))
        screen.blit(overlay,(0,0))
   
        # center line
        pygame.draw.line(screen, (200, 200, 200), (W // 2, 0), (W // 2, H), 2)

        # paddles
        pygame.draw.rect(screen, (0, 220, 255), paddle, border_radius=8)
        pygame.draw.rect(screen, (255, 80, 120), ai, border_radius=8)

        # ball
        if smash_flash > 0:
            pygame.draw.circle(
                screen, (255, 255, 255),
                (int(ball_pos.x), int(ball_pos.y)),
                BALL_R + 6
            )
            smash_flash -= dt

        pygame.draw.circle(
            screen, (255, 255, 255),
            (int(ball_pos.x), int(ball_pos.y)),
            BALL_R
        )

        # HUD
        hud = font.render(f"{score_p}   {score_ai}", True, (230, 230, 230))
        screen.blit(hud, (W // 2 - hud.get_width() // 2, 20))

        # Debug: show finger speed and smash threshold
        dbg = font.render(
            f"finger_speed={finger_speed:.0f} (smash>{SMASH_SPEED_THRESHOLD:.0f})",
            True,
            (150, 150, 150),
        )
        screen.blit(dbg, (20, H - 40))

        pygame.display.flip()

finally:
    tracker.stop()
    pygame.quit()