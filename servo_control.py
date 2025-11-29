# servo_control.py
import RPi.GPIO as GPIO
import time

# ---------------- SERVO PINS ----------------
TILT_A_PIN = 17
TILT_B_PIN = 27
ROTATE_PIN = 22
# --------------------------------------------

GPIO.setmode(GPIO.BCM)

GPIO.setup(TILT_A_PIN, GPIO.OUT)
GPIO.setup(TILT_B_PIN, GPIO.OUT)
GPIO.setup(ROTATE_PIN, GPIO.OUT)

# 50 Hz PWM for all servos
pwm_a = GPIO.PWM(TILT_A_PIN, 50)
pwm_b = GPIO.PWM(TILT_B_PIN, 50)
pwm_rotate = GPIO.PWM(ROTATE_PIN, 50)

pwm_a.start(0)
pwm_b.start(0)
pwm_rotate.start(0)

# Convert angle to duty cycle (stable and smooth)
def angle_to_duty(angle):
    return 2 + (angle / 18)

# ---------- Move TWO servos simultaneously ----------
def safe_move_dual(pwm1, angle1, pwm2, angle2):
    duty1 = angle_to_duty(angle1)
    duty2 = angle_to_duty(angle2)

    # Apply both duty cycles at the SAME time
    pwm1.ChangeDutyCycle(duty1)
    pwm2.ChangeDutyCycle(duty2)

    time.sleep(0.5)

    # Stop jitter
    pwm1.ChangeDutyCycle(0)
    pwm2.ChangeDutyCycle(0)

# ---------- Move a SINGLE servo ----------
def safe_move_single(pwm, angle):
    duty = angle_to_duty(angle)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.5)
    pwm.ChangeDutyCycle(0)

# ---------------- ACTION FUNCTIONS ----------------

def neutral_position():
    print("→ Moving to neutral position (simultaneous tilt + rotate)")
    safe_move_dual(pwm_a, 83, pwm_b, 87)
    safe_move_single(pwm_rotate, 75)

def move_tilting(angle_a, angle_b):
    print(f"→ Tilting simultaneously A={angle_a}, B={angle_b}")
    safe_move_dual(pwm_a, angle_a, pwm_b, angle_b)

def rotate_waste(angle):
    print(f"→ Rotating waste servo to {angle}°")
    safe_move_single(pwm_rotate, angle)

def cleanup():
    print("→ Cleaning up servos...")
    pwm_a.stop()
    pwm_b.stop()
    pwm_rotate.stop()
    GPIO.cleanup()

