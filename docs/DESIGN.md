# Design

The task, the physical model, the learning recipe and the evaluation.

A Unitree G1 humanoid learns to ride an unpowered pogo stick in simulation, with
contact physics only: nothing attaches the hands or feet to the stick.

## 1. Task

The robot stands on the pedals with both hands on the handles. One neural policy,
seeing proprioception (joint encoders, pelvis gyro and accelerometer), an
estimated pelvis linear velocity (as legged RL does; on hardware a state
estimator), a 5-frame history and a command, must:

1. hop continuously from a standstill;
2. track hop-height commands within 3 cm; continued hopping is required even
   at zero commanded planar velocity;
3. track planar velocity and yaw-rate commands from a person (keyboard) or a
   path follower;
4. stay robust to pushes, latency and unknown pogo parameters.

Main question: can PPO learn controllable, continuous hopping
on this passive, contact-rich system, and what must a dense reward score for
such a periodic skill to be learned?

It is an underactuated hybrid system: the rider can only pump the passive spring
through the timing of its whole body.

## 2. Physical model (`pneumatic-v5`)

| Part | Model | Source |
|---|---|---|
| Robot | mjlab G1, 29 body joints and actuators; 33.8 kg with hands | mjlab 1.6.0 |
| Hands | BrainCo Revo2, 6 torque-limited servos per hand (0.5 / 1.1 / 2 N m); distal joints coupled to proximal ones by the linkage ratio | Unitree `g1_with_brainco_hand` URDF |
| Pogo | Small V4 Pro size: 4.54 kg (frame 3.64 + plunger 0.90), pedal top 0.4826 m above the tip | manufacturer drawings and FAQ (ASSETS.md); mass split is an estimate |
| Spring | gas law, 128 kPa gauge, 76.2 mm bore, 3.09 L, n = 1.3, 0.432 m stroke | [patent][pneumatic-patent], [operation guide][vurtego-guide] |
| Losses | 10 N s/m viscous damping, 12 N seal friction, passive end stops | engineering estimates |
| Contacts | explicit material pairs, elliptic cones; ground/pedal/grip mu 0.8/0.8/1.0; tip spin and rolling resistance | engineering estimates, fixture tests |
| Integration | 2 ms implicitfast, 10 physics steps per 50 Hz action; MuJoCo and MJWarp 3.15 | 2 ms vs 1 ms convergence |

- Spring: `F(x) = A [(p + p_atm) (V0 / (V0 - A x))^n - p_atm]`. Pressure follows
  the manufacturer rule (rider lb / 4 psi); preload 584 N is 1.55 times the
  system weight, so a standing rider does not compress the stick. It runs natively: the
  [preloaded cubic][native-spring] Taylor expansion of F on the slider joint,
  plus the integral Taylor remainder as a 12-node Gauss-Legendre sum of positive
  cubic [tendon springs][native-tendon] k_i max(0, x - t_i)^3 (no extra DOF or
  callback). Over the stroke the force error is < 0.003% (0.06 N of 3450 N)
  and stored energy matches to 1e-8 at 80-180 kPa and n = 1.0-1.4; every term
  is conservative and rewards read the exact potential. A linear spring would
  misstate full-stroke energy by about 30%. Chamber volume, mass split
  and loss coefficients are design values, not measurements of a real stick.
- Contacts: [pairs][contact semantics] stop MuJoCo's max-friction mixing from
  overriding a slippery surface. Elliptic cones keep friction isotropic; MuJoCo's pyramid is up to
  29% weaker along its diagonals, which matters for a friction-only grasp. Tip
  and end-stop time constants are 4 ms, the stiffest MuJoCo allows at 2 ms.
- The grasp is found physically: fingers close on a free bar of the handle
  radius (`robot.settled_grasp`). `scripts/grip_capacity.py` measures pull-out
  capacity per revision; it is not the product's pinch-force rating.
- Acceptance (`scripts/physics_checks.py`) covers every hop the stick can make:
  coupled rider drops from 0.15, 0.5, 1.0 and 1.5 m (1.5 m reaches the end
  stop), plus 1.0 m at 80 kPa/n=1.0 and 180 kPa/n=1.4, must lose energy
  monotonically and give the same rebound height and stance time at 2 ms and
  1 ms within 3%. Penetration
  (< 1 cm) and stop overtravel (< 5 mm) are tunnelling guards only: compliance
  is judged by the convergence test, and bottoming out on the stop is allowed.
  The mounted start begins with at least four finger bodies on each handle and
  under 1.5 mm penetration (servo-loaded contact compliance; a tunnelling
  guard); rollouts with the motors off and on stay finite.
- The evaluator refuses checkpoints from other physics revisions.

## 3. Learning recipe

One policy, trained from scratch with PPO, follows height, planar velocity and
yaw-rate commands together.

- **Actions**: 29 joint-position offsets around the mounted pose (G1 action
  scale, PD actuators). The finger servos keep a fixed closing command.
- **Observations**: actor as in section 1 (5-frame history). Without the
  velocity estimate the policy could not brake: horizontal velocity is
  constant and unsensed in flight, and after a forward command it kept going
  at ~0.4 m/s. The critic adds
  privileged pogo and contact state (compression, energy height H, tip force and
  clearance, pogo pose and velocity, system COM velocity, foot and grip errors).
- **Commands** `[h, vx, vy, yaw_rate]`, resampled every 4-8 s; 20% in place.
  vx -0.4..0.5, vy +/-0.4 m/s, yaw +/-0.6 rad/s, sampled independently of h.
  Height is uniform between 0.05 m and a ceiling that starts at 0.15 m and
  rises by 5 cm once its top 5 cm band is practised: over a window of 20000
  landings commanded in that band and made >= 3 s after the command last
  changed, the mean measured apex reaches >= 85% of the mean command, and
  falls stay <= 2% per landing under exploration noise (so about two thirds
  of 20 s top-band episodes run to the time limit). Cap 1.5 m; full-stroke
  spring energy bounds H at about 1.7 m. The gate protects practice, not
  deployed safety or the 3 cm accuracy: exploration noise (action std
  ~0.9-1.0) inflates falls 10-100x over the deterministic policy, and both
  safety and accuracy are judged by evaluation.
- **Reward**: every task objective is scored per hop cycle at its task
  tolerance, with a weak dense guide, and earns up to the same rate:
  - height: measured tip apex at each valid landing against the command
    latched at takeoff, `S(e)=1/(1+(e/s)^2)`, s = 3 cm, weight 4, multiplied
    by the time since the previous valid landing (cap 2 s) so cadence earns
    nothing by itself; dense guide on H = (z_com - z_rest) + v_z^2/2g +
    E_spring/(Mg), s = 8 cm, weight 1;
  - planar velocity and yaw rate: the landing-to-landing average against the
    command, same kernel and time weighting, s = 0.15 m/s and 0.15 rad/s
    (practical: 0.75 m drift over 5 s, 26 deg over 3 s, corrected by a human
    or planner in the loop), weight 2 each; dense Gaussian guides on the
    instantaneous values, width 0.5, weight 0.5 each; a squared penalty on
    the instantaneous yaw-rate error, -0.1, so the turning gait does not
    swing (the Gaussian guide saturates). With a looser kernel, ignoring a
    turn command kept half the score and turning was never learned.
  - pogo/torso upright, feet on pedals, alive; action rate, torque, joint
    acceleration and limit regularisers; -5 in the final 40 mm of travel;
    -200 on termination. The landing that ends a reset-made flight earns no
    credit.
- **Resets**: half from the mounted pose, half mid-hop (lifted, moving), so
  landings appear from the first rollout.
- **Terminations**: pogo tilt > 45 deg, torso tilt > 60 deg, a foot > 8 cm from
  its pedal, a hand > 5 cm from its handle, body or frame touching the ground.
- **PPO** (rsl_rl via mjlab): 4096 worlds x 24 steps, 5 epochs x 4 minibatches,
  adaptive learning rate 1e-3 (KL 0.01), gamma 0.99, lambda 0.95, clip 0.2,
  entropy 0.005; MLP 512-256-128 ELU for actor and critic. Left-right mirror
  augmentation on. Domain randomisation (`robust=True`: spring, pogo mass,
  friction, COM, masses, gains, 0-20 ms latency, pushes) is off by default.

## 4. Evaluation

`scripts/eval_suite.py`: static mounted start, 256 worlds x 30 s per condition,
final checkpoint only. Conditions: height and velocity commands, a 30 s
stop/forward/stop/turn/stop sequence, dt/2, pushes, actuator latency (seconds),
pogo parameters outside the training range, terrain. Metrics: success, hops,
measured apex error over settled hops (the last two thirds of the run: from rest
the policy climbs to 0.75 m over ~8 s), velocity and yaw error after 3 s (per
sample, and averaged over each landing-to-landing hop cycle), grip force and
moment percentiles, motor power, wrist saturation. Climb time after a command
change is a separate responsiveness metric. Failures are reported with successes; seeds separately.
Each condition also records the first-failure cause per world and the spring
stroke used. `scripts/capability.py` (command matrix) and
`scripts/push_phase.py` (push timing) run on the same evaluator; `scripts/watch_curves.py` gives learning curves.

Pneumatic conditions add pressure, thermal exponent, seal loss, materials and
terrain. `survival` reports the old no-failure fraction. `success` additionally requires a
valid first hop within 3 s and no subsequent gap longer than 2 s. A valid hop
still needs >=0.1 s of flight and >=2 cm peak clearance. This prevents standing
on the finite tip contact patch from passing a continuous-hopping milestone.
Missing measurements (e.g. no valid hops) are JSON null, not zero error. Report
names and metadata include checkpoint, seed, world count and horizon.

### Terrain and external commands

- Use mjlab's native plane or heightfields (waves, inverted pyramid slope,
  rough cells, raised steps), with 24 m patches. The slope has a 1 m flat valley spawn and rises outward. Native
  terrain origins plus a surface ray place the mounted reset on the actual
  surface. The actor stays proprioceptive; terrain rays are used by the critic,
  reward, metrics and failure detection.
- Tip clearance and the height surrogate use the local surface, not world z=0.
  Leaving the supported terrain is a failure. Ray misses cannot count as jumps.
- `HopCommand.set_command(apex, vx, vy, yaw_rate)` holds input across automatic
  command resampling and resets. Velocities are in the robot yaw frame. WASD
  and the path follower share this interface; they never drive
  roots, attach the pogo, or apply hidden stabilising forces. Runtime keyboard
  events are consumed on the simulation thread and enter the next observation.
- Viewer input is an interface, not a learned steering result. New physics needs
  new training and command/terrain evaluation before teleoperation is accepted.

[pneumatic-patent]: https://patents.google.com/patent/US10245467B2/en
[vurtego-guide]: https://web.archive.org/web/20200625032436/https://www.vurtegopogo.com/guide/
[native-spring]: https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint-stiffness
[contact semantics]: https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters
[native-tendon]: https://mujoco.readthedocs.io/en/stable/XMLreference.html#tendon-spatial-springlength
