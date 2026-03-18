import cv2
import time
import numpy as np
import mediapipe as mp


def clamp_roi(x, y, r, w, h):
    """ROI carrée centrée sur (x,y) avec rayon r, clampée dans l'image."""
    x1 = max(0, x - r)
    x2 = min(w, x + r)
    y1 = max(0, y - r)
    y2 = min(h, y + r)
    return x1, y1, x2, y2


def draw_flow_roi(gray_roi, flow_roi, step=12):
    """Dessine le champ de vecteurs Farneback sur une ROI grayscale."""
    h, w = gray_roi.shape[:2]
    y, x = np.mgrid[step // 2 : h : step, step // 2 : w : step].reshape(2, -1).astype(np.int32)

    fx, fy = flow_roi[y, x].T
    lines = np.stack([x, y, x - fx, y - fy], axis=1).reshape(-1, 2, 2)
    lines = np.int32(np.round(lines))

    out = cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2BGR)
    cv2.polylines(out, lines, isClosed=False, color=(0, 255, 0), thickness=1)
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(out, (int(x1), int(y1)), 1, (0, 255, 0), -1)
    return out


def compute_tip_velocity_from_flow(flow_roi, tip_px, roi_top_left, dt, patch=9):
    """
    Estime la vitesse (vx, vy) du doigt avec optical flow autour du tip.
    - flow_roi: HxWx2 en px/frame (Farneback)
    - tip_px: (x,y) dans l'image complète
    - roi_top_left: (x1,y1)
    - dt: secondes entre frames
    - patch: taille impaire du patch autour du tip
    Retour: (vx, vy, speed) en px/s
    """
    if tip_px is None or dt <= 1e-9:
        return 0.0, 0.0, 0.0

    x1, y1 = roi_top_left
    tx, ty = tip_px

    cx = tx - x1  # tip en coords ROI
    cy = ty - y1

    h, w = flow_roi.shape[:2]
    r = patch // 2

    x0 = max(0, cx - r)
    x2 = min(w, cx + r + 1)
    y0 = max(0, cy - r)
    y2 = min(h, cy + r + 1)

    if x2 <= x0 or y2 <= y0:
        return 0.0, 0.0, 0.0

    patch_flow = flow_roi[y0:y2, x0:x2, :]  # px/frame

    # robuste: médiane
    fx = float(np.median(patch_flow[..., 0]))
    fy = float(np.median(patch_flow[..., 1]))

    # px/frame -> px/s
    vx = fx / dt
    vy = fy / dt
    speed = float(np.hypot(vx, vy))
    return vx, vy, speed


class FingerFlowTracker:
    """
    Tracker doigt:
    - MediaPipe : localise le tip (landmark index 8) => ROI
    - Optical Flow (Farneback) : vitesse du tip via flow local (vx, vy) en px/s
    - Sortie par frame : dict (pos, vel, speed, etc.)
    """

    def __init__(
        self,
        cam_index=0,
        width=800,
        height=600,
        model_path="hand_landmarker.task",
        roi_radius=80,
        flow_step=25,
        flow_patch=9,
        ema_alpha=0.25,  # lissage vitesse (optical flow peut jitter)
        show_flow=True,
    ):
        self.cam_index = cam_index
        self.width = int(width)
        self.height = int(height)
        self.model_path = model_path

        self.roi_radius = int(roi_radius)
        self.flow_step = int(flow_step)
        self.flow_patch = int(flow_patch)
        self.ema_alpha = float(ema_alpha)
        self.show_flow = bool(show_flow)

        self.cap = None
        self.landmarker = None

        self.prevgray_full = None
        self.last_frame_time = None
        self.t0 = None

        # vitesse lissée (optical flow only)
        self.vx_s = 0.0
        self.vy_s = 0.0

        # Farneback params
        self.fb_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15, #taille du patch 
            iterations=5, # le nb d'itérations à chaque lvl pyramide car on trouve pas la solution d'un coup 
            poly_n=5,
            poly_sigma=1.2, #eviter que le bruit casse le modèle 
            flags=0,
        )

    def start(self):
        # MediaPipe tasks setup
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la caméra (index {self.cam_index}).")

     
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

 
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.stop()
            raise RuntimeError("Impossible de lire la première frame.")

        frame = cv2.flip(frame, 1)  # miroir
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        self.prevgray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        self.last_frame_time = time.perf_counter()
        self.t0 = self.last_frame_time

      
        self.landmarker = HandLandmarker.create_from_options(options)


        real_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[FingerFlowTracker] Camera requested: {self.width}x{self.height} | reported: {real_w}x{real_h}")

    def stop(self):
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
            self.landmarker = None

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self.prevgray_full = None
        self.last_frame_time = None
        self.t0 = None
        self.vx_s = 0.0
        self.vy_s = 0.0

    def read(self, with_vis=True):
        """
        Lit une frame, calcule tip + vitesse (optical flow), renvoie un dict.
        Si with_vis=True: dict["vis"] contient l'image annotée (BGR).
        """
        if self.cap is None or self.landmarker is None:
            raise RuntimeError("Tracker non démarré. Appelle start() avant read().")

        ok, frame = self.cap.read()
        if not ok or frame is None:
            return {
                "ok": False,
                "active": False,
                "pos": None,
                "vel": (0.0, 0.0),
                "speed": 0.0,
                "dt": 0.0,
                "fps": 0.0,
                "roi": None,
                "vis": None if with_vis else None,
                "frame": frame
            }

        # timing
        now = time.perf_counter()
        dt = now - self.last_frame_time if self.last_frame_time is not None else 0.0
        self.last_frame_time = now
        fps_loop = (1.0 / dt) if dt > 1e-9 else 0.0

        # preprocess: mirror + resize to 800x600
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        h, w = frame.shape[:2]
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # MediaPipe detect_for_video wants timestamp in ms
        timestamp_ms = int((now - self.t0) * 1000.0)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        tip_px = None
        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            x = int(lm[8].x * w)
            y = int(lm[8].y * h)
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            tip_px = (x, y)

        vx, vy, speed = 0.0, 0.0, 0.0
        roi = None

        vis = frame.copy() if with_vis else None

        if tip_px is not None and self.prevgray_full is not None and dt > 1e-9:
            x, y = tip_px
            x1, y1, x2, y2 = clamp_roi(x, y, self.roi_radius, w, h)
            roi = (x1, y1, x2, y2)

            prev_roi = self.prevgray_full[y1:y2, x1:x2]
            gray_roi = gray_full[y1:y2, x1:x2]

            flow_roi = cv2.calcOpticalFlowFarneback(
                prev_roi, gray_roi, None, **self.fb_params
            )

            # vitesse (optical flow) localisée autour du tip
            vx_raw, vy_raw, speed_raw = compute_tip_velocity_from_flow(
                flow_roi=flow_roi,
                tip_px=tip_px,
                roi_top_left=(x1, y1),
                dt=dt,
                patch=self.flow_patch,
            )

            # lissage EMA (toujours uniquement basé sur optical flow)
            a = self.ema_alpha
            self.vx_s = (1 - a) * self.vx_s + a * vx_raw
            self.vy_s = (1 - a) * self.vy_s + a * vy_raw

            vx, vy = self.vx_s, self.vy_s
            speed = float(np.hypot(vx, vy))

            if with_vis:
                # Dessin champ de flow dans la ROI
                if self.show_flow:
                    flow_vis_roi = draw_flow_roi(gray_roi, flow_roi, step=self.flow_step)
                    vis[y1:y2, x1:x2] = flow_vis_roi

                cv2.circle(vis, tip_px, 5, (0, 0, 255), -1)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # update prev
        self.prevgray_full = gray_full

        if with_vis:
            cv2.putText(
                vis,
                f"v=({vx:+.0f},{vy:+.0f}) px/s |speed|={speed:.0f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                f"FPS: {fps_loop:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        state = {
            "ok": True,
            "active": tip_px is not None,
            "pos": tip_px,               # (x,y) en 800x600
            "vel": (vx, vy),             # (vx,vy) en px/s (optical flow)
            "speed": speed,              # norme en px/s
            "dt": dt,
            "fps": fps_loop,
            "roi": roi,                  # (x1,y1,x2,y2) ou None
            "timestamp_ms": timestamp_ms,
            "frame":frame
        }

        if with_vis:
            state["vis"] = vis

        return state

    def demo(self, window_name="Index ROI Optical Flow"):
        """Démo OpenCV pour valider visuellement."""
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.width, self.height)

        while True:
            st = self.read(with_vis=True)
            if not st["ok"]:
                break

            cv2.imshow(window_name, st["vis"])

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = FingerFlowTracker(
        width=800,
        height=600,
        roi_radius=80,
        flow_step=25,
        flow_patch=9,
        ema_alpha=0.25,
        show_flow=True,
    )
    try:
        tracker.start()
        tracker.demo()
    finally:
        tracker.stop()