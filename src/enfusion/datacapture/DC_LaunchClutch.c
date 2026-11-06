// ============================================================================
// DC_LaunchClutch - the Mode A launch clutch for the Reforger 1.9 powertrain (DD-14c, mode slip DD-14d)
// ============================================================================
//
// WHY. Mode A drives a driverless car by calling the VehicleWheeledSimulation setters every physics step (r2 3.2),
// and up to 1.8.0.13 it held the clutch at 1 from standstill, as vanilla Game/Cinematics/VehicleCinematicTrack.c does:
// its m_fClutch defaults to 1.0 (:16-17), and OnApply calls ForceStartEngine (:138-140) and SetClutch(m_fClutch) on
// every apply (:159). That file is byte-identical in 1.8.0.13 and 1.9.0.37.
//   - 1.8.0.13: the powertrain slipped at idle and could not stall. In E-01 of T0-ENG run 3 (launch
//     20261103T061122Z-wb-86bd, e01 play_01) the rpm held at 800 for about 30 steps while the S105 accelerated at
//     2 m/s2.
//   - 1.9.0.37: the powertrain stalls. The first 1.9 Workbench Play (launch 20261106T052424Z-wb-addc, e01 play_01)
//     settled in neutral with the handbrake on at 800 rpm. At step 246 it released the handbrake into first gear
//     (SetGear(2)) at throttle 0.5 and clutch 1. Step 247 read 528 rpm; step 248 read 18 rpm with EngineIsOn false.
//     The car never moved.
// Vanilla script offers nothing for this case. Evidence: the official script dump, tag v1.9.0.37 (and
// G19_0_STATIC_1.9.0.37.md section 1):
//   - the clutch semantics are unchanged: GetClutch / SetClutch, "0 .. clutch disengaged, 1 .. clutch fully engaged",
//     VehicleWheeledSimulation.c:38-44 (:23-29 on 1.8.0.13);
//   - script has no anti-stall, auto-clutch or stall member, event or callback. 1.9 added only the
//     Update*Configuration maps (:26-32), GetHandbrake (:47-48) and EngineGetRPMRedline (:78-79) to
//     VehicleWheeledSimulation;
//   - only a seated driver gets the controller's native automatic clutch (S105_base.et:1093-1094 of the installed
//     1.9.0.37 data: ClutchUncoupleTime 0.45, ClutchCoupleTime 0.85; the clutch itself is MaxClutchTorque 130 Nm,
//     :975). CharacterControllerComponent.SetManualClutch
//     (Game/generated/Components/CharacterControllerComponent.c:32-33) is a character input setting;
//   - FULL driving assistance ("automatic engine startup", EVehicleDrivingAssistanceMode.c:12-21) acts neither on
//     our direct setters nor on a stalled driverless engine: that Play ran with it and was never restarted.
// So Mode A needs its own launch clutch, and this file holds it. All three actuation paths run it, so they behave the
// same:
//   - DC_ProbeActuator (DC_ProbeSequences.c), for every probe that drives;
//   - llc_v1 (DC_LLCv1.c), on the scripted gearbox of a vehicle card;
//   - DC_ActuatorModeA.Start (DC_StreamRecorder.c), which does llc_v1's restart.
//
// WHAT IT DOES. It acts like a centrifugal clutch. Throttle at standstill revs the engine, and the clutch bites
// progressively. The clutch slips near idle instead of stalling when the car brakes to a stop. At cruise it is 1.
// Mode "off" passes the commanded clutch through unchanged: exactly today's behaviour, which is what 1.8 did. Mode
// "ramp", per Step, with the physics time dt the call covers (covered steps x step dt, the callers' convention):
//   - c (internal, 0 = open .. 1 = coupled) is 0 when not in a forward or reverse gear, while the handbrake is
//     commanded, during a dip, or while the engine is off;
//   - below stall_guard_rpm, c = 0 at once (a guard drop when c was > 0);
//   - at or below engage_rpm, c = 0 at once as well. Opening never stalls an engine; a rate-limited release
//     below the band kept loading the engine under a full brake (Codex round 1, the toy model);
//   - inside the band, target = clamp((rpm - engage_rpm) / (full_rpm - engage_rpm), 0, 1):
//       - when target > c, c rises toward it by at most dt / couple_time_s;
//       - otherwise c falls toward it by at most dt / uncouple_time_s;
//   - the applied clutch is min(commanded clutch, c). A dip (the caller's clutch_dip_s, or any commanded 0) forces 0
//     and resets c to 0, so the clutch re-ramps after it;
//   - rpm and engine_on are the caller's last readback (EngineGetRPM, EngineIsOn), the state after the previous step:
//     the probe's rpm / engine_on columns, llc_v1's simulation read at the start of its callback.
// Two deviations from the DD-14c sketch. First, the sketch keys the open clutch on rpm alone; here an engine that is
// off is never coupled either. Its rpm is the drag of the wheels, not a running engine, and the restart then re-ramps
// from c = 0. Second, the sketch rate-limits every release; here only releases inside the band are (above).
//
// BOUNDS. Check holds the ranges. CheckEngine adds the vehicle, at the caller's start (modes ramp and slip):
//   - engage_rpm must lie above the engine's idle (EngineGetRPMIdle), so idle never bites;
//   - full_rpm must lie below FULL_RPM_MAX_FRACTION of its max rpm (EngineGetRPMMax, VehicleWheeledSimulation.c:77 of
//     1.9.0.37, :60 of 1.8.0.13; the probe's vehicle.rpm_max), so the band is attainable.
// Shift rules are the caller's:
//   - the probe actuator refuses a full_rpm at or above its policy's downshift rpm;
//   - a vehicle card refuses a ramp on a scripted gearbox whose downshifts fall into the band. llc_v1 shifts by speed,
//     so the card carries each scripted gear's measured m/s per 1000 rpm (gearbox.scripted_mps_per_krpm). Each
//     downshift speed, in the gear it shifts down from, must be at least full_rpm + DOWNSHIFT_RPM_MARGIN
//     (DC_VehicleCard.CheckLaunchDownshifts; fit_steer_kappa.py mirrors it).
//
// STALL RESTART (modes "ramp" and "slip", restart_on_stall): when the engine is off, the handbrake is not commanded and
// restart_cooldown_s has passed since the last restart (the first restart has no cooldown), Step requests a restart.
// The cooldown is the physics time covered since the last restart's call, measured at this call's start, so calls
// covering 1 to 3 steps never bring two restarts closer than restart_cooldown_s:
//   - c resets to 0 and the restart is counted;
//   - ConsumeRestart then returns true once, and the caller restarts the engine the way Mode A starts it
//     (ForceStartEngine when IsEngineOn is false, VehicleCinematicTrack.c:138-140): DC_ProbeActuator.StartModeA, or
//     DC_ActuatorModeA.Start for llc_v1;
//   - restarts counts these requests.
//
// COUNTERS (every mode counts stalls; the rest only count in "ramp" and "slip"; lock-ups only in "slip"):
//   - stalls: EngineIsOn seen true, then false on the next Step;
//   - restarts;
//   - guard drops;
//   - max_slip_s: the longest span with 0 < c < 1 in gear, without a handbrake or dip, with the engine on.
// The probe kit writes them as phase notes and run header lines. The capture side logs them to the console only:
// dc-2.0 and features_v1 are frozen, so there is no stream column and no event type.
//
// DEFAULTS: engage 1100 rpm, full 1700 rpm, couple 0.85 s, uncouple 0.45 s, stall guard 600 rpm, restart after a
// stall with a 1 s cooldown.
//   - The S105 idles at 800 rpm (run headers vehicle.rpm_idle), so its idle never bites.
//   - full_rpm stays below the probe shift policy's downshift rpm (1800), so the clutch is 1 at every rpm the policy
//     drives in.
//   - The couple and uncouple times are the S105 controller's ClutchCoupleTime / ClutchUncoupleTime.
//   - tests/test_launch_clutch.py drives a Python mirror of Step through a TOY engine, clutch and vehicle model
//     (rpm_idle 800, peak power 4800, max 8000, step dt 1/30 s): with these defaults it launches at throttle 0.3, 0.5
//     and 1.0 and brakes to a stop in first gear without stalling, and "off" stalls as the 1.9 Play did. That model
//     catches control-loop bugs; it does not predict 1.9.
//
// UNVERIFIED on 1.9.0.37 (needs a Workbench Play): every in-engine behaviour of a partial clutch value (how
// SetClutch(c) for 0 < c < 1 maps to transmitted torque); the engine inertia that sets the loop gain (the toy's
// 0.12 kg m2 is an estimate from the 800 -> 528 rpm step); whether ForceStartEngine restarts a stalled engine in one
// step; automatic gearboxes (no caller ramps one: the probe kit and llc_v1 leave them as they are). DD-14d answered the
// first two from the first 1.9 run with the ramp (below).
//
// MODE SLIP (DD-14d). The first 1.9.0.37 Play with mode ramp (launch 20261106T194516Z-wb-ab6a, e01 play_01) launched
// without a stall but the clutch chattered: at throttle 0.5 the engine revved to about 1930 rpm, the ramp coupled at
// dt / couple_time_s while its target stayed above c, the rpm collapsed through engage_rpm, the clutch opened at once
// and the cycle repeated (2 openings at 3 and 6 km/h before c = 1 at 2.0 s; 4 at throttle 0.3, 3 in gear_scripted's
// launch at 0.4; none at 1.0). tools/ident/fit_launch_clutch.py fits that Play's launches (engine inertia, torque
// curve, clutch capacity, rolling load; tools/ident/launch_clutch_model.py):
//   - J 0.039 kg m2, clutch capacity 118 Nm at c = 1 (Coulomb: the rpm rate falls by a constant 38.5 rpm per step per
//     0.039 of c whatever the slip speed), gross torque 58 .. 79 Nm at throttle 1 (linear in throttle), drag
//     1.134 / s x (rpm - 800) above idle, none below it; SetClutch acts from the step it is set, SetThrottle and
//     SetBreak one step later, and the readback alternates between half a step and no step after the setters;
//   - the clutch moves the rpm by about 965 rpm per step at c = 1, so the ramp's 600 rpm band is a loop gain of 1.6 per
//     step: no rate limit or band below the shift policy's 1800 rpm downshift damps it (DD-14c's toy-model caveat:
//     below about 0.07 kg m2 it rings);
//   - replayed in closed loop on that model, the ramp gives the Play's openings (2 / 4-5 / 1 / 0 / 3 at throttle 0.5 /
//     0.3 / 0.6 / 1.0 / 0.4) and its c = 1 times (2.0 / 3.8 / 1.6 / 1.0 / 2.5 s).
// Mode "slip" is an rpm-hold slip controller with lock-up when the engine and the driveline turn together. The caller
// passes the throttle and brake it commands this call (drive = throttle > 0 and no brake), the forward gear index and
// the driveline rpm (speed / the gear's m/s per 1000 rpm, <= 0 unknown). Per Step (driving: in gear, no handbrake, no
// dip, engine on; otherwise c = 0, not locked):
//   - the rpm it uses is the mean of this readback and the last one (the readback timing alternates); its rate [rpm/s]
//     is the change of the mean over the two calls before, and the predicted rpm is rpm + rate (if falling) x
//     RELEASE_LOOKAHEAD_CALLS x this call's duration (the clutch holds for this call, then the next one decides);
//   - an upshift (a higher gear index than the last call's) restarts that history and holds the lookahead and the
//     drop guard off for SYNC_S: the synchronisation drop is no fall to predict. A downshift raises the rpm and keeps
//     the history, so a braking downshift keeps the braking lookahead;
//   - below stall_guard_rpm: c = 0 at once (a guard drop when c was > 0), not locked;
//   - coupled (locked): c = 1 until the rpm reaches its floor (engage_rpm, or under drive the lug rpm
//     stall_guard_rpm + LUG_MARGIN_RPM, at most engage_rpm: the engine pulls, so it may lug below engage_rpm, as after
//     gear_scripted's upshift into third at 1075 rpm), or, below RELEASE_LOOKAHEAD_ENGAGE_FACTOR x engage_rpm, the
//     predicted rpm reaches it, or, without drive, the rpm falls faster than MAX_DROP_RPM_PER_S (a full brake in first
//     gear falls about 8000 rpm/s; faster is wheels locking). Then c = 0 at once, not locked;
//   - slipping (a drive-off): c = 0 at once at or below engage_rpm, without drive (a brake or no throttle: a stop
//     before the lock-up) and when the predicted rpm reaches the lug rpm; otherwise c moves by dt x (slip_kp x rate +
//     slip_ki x (mean - full_rpm)): a PI that holds the engine at full_rpm, bounded by dt / couple_time_s up and
//     dt / uncouple_time_s down, within [0, 1]. The gains are per GAIN_STEP_S (the 1/30 s fixed time slice): a call
//     covering n steps divides slip_kp by n and slip_ki by n x n, n held at the longest recent call and decaying by
//     CALL_STEPS_DECAY per call, so 2- and 3-step callbacks (llc_v1) keep the loop's margin;
//   - lock-up: when the driveline rpm is within lock_tolerance_rpm of the rpm on LOCK_AGREE_CALLS calls in a row (under
//     drive: a brake's wheel slip makes body speed read above the shaft, the unsafe direction) or c reaches 1: c = 1 at
//     once, counted as a lock-up.
// DD-14d round 1 (Codex) found a stall while slipping (a brake at 5 km/h with 3-step calls: 219 rpm), a false lock-up
// under a brake with wheel slip (c 0.25 -> 1, 426 rpm), gear_scripted's upshifts opening the clutch and openings with
// 3-step calls; the rules above are the fix. On the fitted model (tests/test_launch_clutch.py): launches at throttle
// 0.3 / 0.5 / 0.6 / 1.0, flat and at +-5 % grade, over engine inertia and clutch capacity +-30 %, torque +-20 %,
// callbacks of 1, 2, 3 and mixed steps and both readback timings: no opening, no step-to-step oscillation of c, rpm
// never below the 800 idle; c = 1 at throttle 0.5 by 1.5 s (1-step calls; 2.1 s with 3-step calls); stops by a brake
// of 0.3 / 0.6 / 1.0 before the lock-up at 3 / 5 / 7 km/h and coupled from 15 / 30 / 40 km/h never below the idle;
// gear_scripted's 2, 3, 4, 3, 2 without an opening. full_rpm is the held rpm (1400 for the S105), so the shift rules
// are those of a ramp: below the policy's downshift rpm (probe kit), each card downshift at least full_rpm +
// DOWNSHIFT_RPM_MARGIN; mode slip adds that each upshift lands at or above the lookahead ceiling (the probe kit with
// its m/s per 1000 rpm attribute, the card with scripted_mps_per_krpm).
// Mode ramp and mode off are unchanged.
//
// BUILD-AGNOSTIC: every engine call of the callers exists with the same declaration in 1.8.0.13 and 1.9.0.37 (official
// script dump): SetClutch / GetClutch (VehicleWheeledSimulation.c:44 / :41; 1.8 :29 / :26), EngineGetRPM (:81; 1.8
// :62), EngineIsOn (:69; 1.8 :52), BaseVehicleControllerComponent.ForceStartEngine (:52) and IsEngineOn (:63) on
// both. This file itself calls no engine API.
// ============================================================================

//------------------------------------------------------------------------------------------------
//! The launch clutch parameters (DD-14c; mode slip DD-14d). Sources: the DC_ProbeSequences attributes (probe kit), and
//! the optional gearbox.launch_clutch object of a vehicle card for llc_v1 (absent = mode off). Check holds the ranges
//! both apply, mirrored by tools/ident/fit_steer_kappa.py (launch_clutch_violations)
class DC_LaunchClutchConfig
{
	static const string MODE_OFF = "off";
	static const string MODE_RAMP = "ramp";
	static const string MODE_SLIP = "slip";

	//! Defaults: DC_ProbeSequences' attributes, Ramp(), the S105 placeholder card (see the header)
	static const float DEFAULT_ENGAGE_RPM = 1100;
	static const float DEFAULT_FULL_RPM = 1700;
	static const float DEFAULT_COUPLE_TIME_S = 0.85;
	static const float DEFAULT_UNCOUPLE_TIME_S = 0.45;
	static const float DEFAULT_STALL_GUARD_RPM = 600;
	static const bool DEFAULT_RESTART_ON_STALL = true;
	static const float DEFAULT_RESTART_COOLDOWN_S = 1;

	//! Mode slip (DD-14d, Slip()): full_rpm is the held rpm; the PI gains and the lock-up tolerance (the header)
	static const float DEFAULT_SLIP_FULL_RPM = 1400;
	static const float DEFAULT_SLIP_KP = 0.0003;
	static const float DEFAULT_SLIP_KI = 0.001;
	static const float DEFAULT_LOCK_TOLERANCE_RPM = 100;
	//! Mode slip: the predicted rpm looks this many of the call's own durations ahead; coupled, below this factor x
	//! engage_rpm (the header)
	static const float RELEASE_LOOKAHEAD_CALLS = 1.5;
	static const float RELEASE_LOOKAHEAD_ENGAGE_FACTOR = 2;
	//! Mode slip: lock-up after this many agreeing calls in a row; after an upshift no lookahead or drop guard for SYNC_S
	static const int LOCK_AGREE_CALLS = 2;
	static const float SYNC_S = 0.3;
	//! Mode slip, coupled under drive: the floor is stall_guard_rpm + this, at most engage_rpm (GetLugRpm)
	static const float LUG_MARGIN_RPM = 300;
	//! Mode slip, coupled without drive: an rpm falling faster than this [rpm/s] opens (wheels locking)
	static const float MAX_DROP_RPM_PER_S = 12000;
	//! Mode slip: slip_kp / slip_ki are per call of GAIN_STEP_S [s]; the held call length decays by CALL_STEPS_DECAY
	static const float GAIN_STEP_S = 0.033333;
	static const float CALL_STEPS_DECAY = 0.05;

	//! Ranges (Check): 0 <= stall_guard_rpm < engage_rpm < full_rpm <= RPM_MAX; couple_time_s and uncouple_time_s in
	//! (0, TIME_MAX_S]; restart_cooldown_s in [0, TIME_MAX_S]
	static const float RPM_MAX = 10000;
	static const float TIME_MAX_S = 10;
	//! CheckEngine: full_rpm below this share of the engine's max rpm (S105: 0.5 x 8000 = 4000)
	static const float FULL_RPM_MAX_FRACTION = 0.5;
	//! Vehicle card, scripted gearbox, mode ramp or slip: each downshift's rpm at least full_rpm + this (BOUNDS in the
	//! header)
	static const float DOWNSHIFT_RPM_MARGIN = 50;
	//! Ranges (Check) of the mode slip fields, checked in every mode: slip_kp in [0, SLIP_KP_MAX] per rpm, slip_ki in
	//! (0, SLIP_KI_MAX] per rpm s, lock_tolerance_rpm in [0, LOCK_TOLERANCE_MAX_RPM]
	static const float SLIP_KP_MAX = 0.01;
	static const float SLIP_KI_MAX = 0.1;
	static const float LOCK_TOLERANCE_MAX_RPM = 1000;

	string m_sMode = MODE_OFF;
	float m_fEngageRpm = DEFAULT_ENGAGE_RPM;			//!< engage_rpm: the clutch starts to bite here
	float m_fFullRpm = DEFAULT_FULL_RPM;				//!< full_rpm: fully coupled at or above this
	float m_fCoupleTimeS = DEFAULT_COUPLE_TIME_S;		//!< couple_time_s: fastest 0 -> 1 [s]
	float m_fUncoupleTimeS = DEFAULT_UNCOUPLE_TIME_S;	//!< uncouple_time_s: fastest 1 -> 0 [s]
	float m_fStallGuardRpm = DEFAULT_STALL_GUARD_RPM;	//!< stall_guard_rpm: below this the clutch opens at once
	bool m_bRestartOnStall = DEFAULT_RESTART_ON_STALL;	//!< restart_on_stall
	float m_fRestartCooldownS = DEFAULT_RESTART_COOLDOWN_S;	//!< restart_cooldown_s: minimum time between restarts
	float m_fSlipKp = DEFAULT_SLIP_KP;					//!< slip_kp: coupling per rpm of the mean's change (mode slip, per GAIN_STEP_S)
	float m_fSlipKi = DEFAULT_SLIP_KI;					//!< slip_ki: coupling per rpm s above full_rpm (mode slip, per GAIN_STEP_S)
	float m_fLockToleranceRpm = DEFAULT_LOCK_TOLERANCE_RPM;	//!< lock_tolerance_rpm: engine vs driveline rpm (mode slip)

	//------------------------------------------------------------------------------------------------
	//! The defaults with mode "ramp": the DD-14c probe kit default (DD-14d moved it to Slip())
	static DC_LaunchClutchConfig Ramp()
	{
		DC_LaunchClutchConfig config = new DC_LaunchClutchConfig();
		config.m_sMode = MODE_RAMP;
		return config;
	}

	//------------------------------------------------------------------------------------------------
	//! The defaults with mode "slip" and full_rpm DEFAULT_SLIP_FULL_RPM (DD-14d): the probe kit's default
	//! (DC_ProbeActuator.Init, the DC_ProbeSequences attributes) and the 1.9.0.37 card's launch clutch
	static DC_LaunchClutchConfig Slip()
	{
		DC_LaunchClutchConfig config = new DC_LaunchClutchConfig();
		config.m_sMode = MODE_SLIP;
		config.m_fFullRpm = DEFAULT_SLIP_FULL_RPM;
		return config;
	}

	//------------------------------------------------------------------------------------------------
	bool IsRamp()
	{
		return m_sMode == MODE_RAMP;
	}

	//------------------------------------------------------------------------------------------------
	bool IsSlip()
	{
		return m_sMode == MODE_SLIP;
	}

	//------------------------------------------------------------------------------------------------
	//! The launch clutch acts (mode ramp or slip); mode off passes the commanded clutch
	bool IsActive()
	{
		return m_sMode == MODE_RAMP || m_sMode == MODE_SLIP;
	}

	//------------------------------------------------------------------------------------------------
	//! The release lookahead ceiling of mode slip [rpm]: RELEASE_LOOKAHEAD_ENGAGE_FACTOR x engage_rpm. The callers check
	//! that every upshift lands at or above it, so an upshift's synchronisation never releases a coupled clutch
	float GetLookaheadCeilingRpm()
	{
		return RELEASE_LOOKAHEAD_ENGAGE_FACTOR * m_fEngageRpm;
	}

	//------------------------------------------------------------------------------------------------
	//! Mode slip, coupled under drive: the lug rpm, stall_guard_rpm + LUG_MARGIN_RPM, at most engage_rpm
	float GetLugRpm()
	{
		return Math.Min(m_fStallGuardRpm + LUG_MARGIN_RPM, m_fEngageRpm);
	}

	//------------------------------------------------------------------------------------------------
	//! The value ranges (see the constants); checked in every mode. false (failure) names the first violation
	bool Check(out string failure)
	{
		if (m_sMode != MODE_OFF && m_sMode != MODE_RAMP && m_sMode != MODE_SLIP)
		{
			failure = "launch_clutch mode '" + m_sMode + "' is none of off, ramp and slip";
			return false;
		}

		if (m_fEngageRpm <= 0 || m_fFullRpm <= m_fEngageRpm || m_fFullRpm > RPM_MAX)
		{
			failure = string.Format("launch_clutch engage_rpm %1 and full_rpm %2 must satisfy 0 < engage_rpm < full_rpm <= %3", m_fEngageRpm, m_fFullRpm, RPM_MAX);
			return false;
		}

		if (m_fStallGuardRpm < 0 || m_fStallGuardRpm >= m_fEngageRpm)
		{
			failure = string.Format("launch_clutch stall_guard_rpm %1 is outside [0, engage_rpm %2)", m_fStallGuardRpm, m_fEngageRpm);
			return false;
		}

		if (m_fCoupleTimeS <= 0 || m_fCoupleTimeS > TIME_MAX_S || m_fUncoupleTimeS <= 0 || m_fUncoupleTimeS > TIME_MAX_S)
		{
			failure = string.Format("launch_clutch couple_time_s %1 and uncouple_time_s %2 must lie in (0, %3] s", m_fCoupleTimeS, m_fUncoupleTimeS, TIME_MAX_S);
			return false;
		}

		if (m_fRestartCooldownS < 0 || m_fRestartCooldownS > TIME_MAX_S)
		{
			failure = string.Format("launch_clutch restart_cooldown_s %1 is outside [0, %2] s", m_fRestartCooldownS, TIME_MAX_S);
			return false;
		}

		if (m_fSlipKp < 0 || m_fSlipKp > SLIP_KP_MAX || m_fSlipKi <= 0 || m_fSlipKi > SLIP_KI_MAX)
		{
			failure = string.Format("launch_clutch slip_kp %1 and slip_ki %2 must lie in [0, %3] and (0, %4]", m_fSlipKp, m_fSlipKi, SLIP_KP_MAX, SLIP_KI_MAX);
			return false;
		}

		if (m_fLockToleranceRpm < 0 || m_fLockToleranceRpm > LOCK_TOLERANCE_MAX_RPM)
		{
			failure = string.Format("launch_clutch lock_tolerance_rpm %1 is outside [0, %2]", m_fLockToleranceRpm, LOCK_TOLERANCE_MAX_RPM);
			return false;
		}

		failure = string.Empty;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! The band against the vehicle (mode ramp or slip; see BOUNDS in the header): rpmIdle and rpmMax are
	//! VehicleWheeledSimulation.EngineGetRPMIdle and EngineGetRPMMax. false (failure) when engage_rpm is not above the
	//! idle, or the max rpm is unknown (<= 0), or full_rpm is not below FULL_RPM_MAX_FRACTION of it
	bool CheckEngine(float rpmIdle, float rpmMax, out string failure)
	{
		if (!IsActive())
		{
			failure = string.Empty;
			return true;
		}

		if (m_fEngageRpm <= rpmIdle)
		{
			failure = string.Format("launch_clutch engage_rpm %1 is not above the engine's idle %2 rpm (the idle would bite)", m_fEngageRpm, rpmIdle);
			return false;
		}

		float limit = FULL_RPM_MAX_FRACTION * rpmMax;
		if (rpmMax <= 0 || m_fFullRpm >= limit)
		{
			failure = string.Format("launch_clutch full_rpm %1 is not below %2 x the engine's max %3 rpm (the band must be attainable)", m_fFullRpm, FULL_RPM_MAX_FRACTION, rpmMax);
			return false;
		}

		failure = string.Empty;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! "<prefix>.<field>=<value>" lines, the card's field names (run header block launch_clutch.*)
	void AppendHeaderLines(notnull array<string> lines, string prefix)
	{
		string restart = "0";
		if (m_bRestartOnStall)
			restart = "1";

		lines.Insert(prefix + ".mode=" + m_sMode);
		lines.Insert(prefix + ".engage_rpm=" + m_fEngageRpm.ToString(-1, 1));
		lines.Insert(prefix + ".full_rpm=" + m_fFullRpm.ToString(-1, 1));
		lines.Insert(prefix + ".couple_time_s=" + m_fCoupleTimeS.ToString(-1, 3));
		lines.Insert(prefix + ".uncouple_time_s=" + m_fUncoupleTimeS.ToString(-1, 3));
		lines.Insert(prefix + ".stall_guard_rpm=" + m_fStallGuardRpm.ToString(-1, 1));
		lines.Insert(prefix + ".restart_on_stall=" + restart);
		lines.Insert(prefix + ".restart_cooldown_s=" + m_fRestartCooldownS.ToString(-1, 3));
		lines.Insert(prefix + ".slip_kp=" + m_fSlipKp.ToString(-1, 6));
		lines.Insert(prefix + ".slip_ki=" + m_fSlipKi.ToString(-1, 6));
		lines.Insert(prefix + ".lock_tolerance_rpm=" + m_fLockToleranceRpm.ToString(-1, 1));
	}

	//------------------------------------------------------------------------------------------------
	//! One log fragment: "off (clutch as commanded)" or the ramp or slip parameters
	string Describe()
	{
		if (!IsActive())
			return "off (clutch as commanded, the 1.8 sequence)";

		string restart = "no restart";
		if (m_bRestartOnStall)
			restart = string.Format("restart after a stall, >= %1 s apart", m_fRestartCooldownS.ToString(-1, 2));

		if (IsSlip())
			return string.Format("slip: open at or below %1 rpm, hold %2 rpm while slipping (kp %3 per rpm, ki %4 per rpm s, couple %5 s, uncouple %6 s), coupled within %7 rpm of the driveline, open below %8 rpm, %9", m_fEngageRpm.ToString(-1, 0), m_fFullRpm.ToString(-1, 0), m_fSlipKp.ToString(-1, 6), m_fSlipKi.ToString(-1, 6), m_fCoupleTimeS.ToString(-1, 2), m_fUncoupleTimeS.ToString(-1, 2), m_fLockToleranceRpm.ToString(-1, 0), m_fStallGuardRpm.ToString(-1, 0), restart);

		return string.Format("ramp: bite %1 -> full %2 rpm, couple %3 s, uncouple %4 s, open below %5 rpm, %6", m_fEngageRpm.ToString(-1, 0), m_fFullRpm.ToString(-1, 0), m_fCoupleTimeS.ToString(-1, 2), m_fUncoupleTimeS.ToString(-1, 2), m_fStallGuardRpm.ToString(-1, 0), restart);
	}
}

//------------------------------------------------------------------------------------------------
//! The launch clutch of one actuation path (see the header): Configure once, Step every physics step (or every
//! callback over the steps it covers), ConsumeRestart right after each Step. Calls no engine API itself
class DC_LaunchClutch
{
	static const string SOURCE_FILE = "DC_LaunchClutch.c";

	protected ref DC_LaunchClutchConfig m_Config;
	protected float m_fCoupling;				//!< c: what the ramp allows, 0 = open .. 1 = fully coupled
	protected float m_fSinceRestartS = -1;		//!< physics time since the last restart request, -1 = none yet
	protected bool m_bRestartDue;				//!< ConsumeRestart returns it once
	protected int m_iEngineSeen = -1;			//!< engine_on of the last Step: -1 = none yet, 0 off, 1 on
	protected float m_fSlipS;					//!< current span with 0 < c < 1 (see the header)
	protected float m_fMaxSlipS;				//!< longest span since ResetMaxSlip
	protected float m_fMaxSlipTotalS;			//!< longest span since Configure
	protected int m_iStalls;
	protected int m_iRestarts;
	protected int m_iGuardDrops;
	protected bool m_bLocked;					//!< mode slip: coupled (c = 1) until the release
	protected float m_fLastRpm = -1;			//!< mode slip: the last readback, -1 = none yet
	protected float m_fLastMeanRpm = -1;		//!< mode slip: the last mean of two readbacks, -1 = none yet
	protected int m_iHistory;					//!< mode slip: readbacks in the history (at most 3)
	protected float m_fLastSliceS;				//!< mode slip: the last call's duration [s]
	protected float m_fPrevSliceS;				//!< mode slip: the duration of the call before it [s]
	protected int m_iAgreeCalls;				//!< mode slip: calls in a row with the engine at the driveline rpm
	protected float m_fSyncLeftS;				//!< mode slip: time left of an upshift's synchronisation [s]
	protected int m_iLastGear = -1;				//!< mode slip: the last call's gear index, -1 = none
	protected float m_fCallSteps = 1;			//!< mode slip: the held call length in GAIN_STEP_S
	protected int m_iLockups;					//!< mode slip: lock-ups since Configure

	//------------------------------------------------------------------------------------------------
	void DC_LaunchClutch()
	{
		m_Config = new DC_LaunchClutchConfig();
	}

	//------------------------------------------------------------------------------------------------
	//! Takes a checked config (DC_LaunchClutchConfig.Check is the caller's) and clears the state and every counter
	void Configure(notnull DC_LaunchClutchConfig config)
	{
		m_Config = config;
		Reset();
		m_iEngineSeen = -1;
		m_fMaxSlipS = 0;
		m_fMaxSlipTotalS = 0;
		m_iStalls = 0;
		m_iRestarts = 0;
		m_iGuardDrops = 0;
		m_iLockups = 0;
	}

	//------------------------------------------------------------------------------------------------
	//! Open clutch, no slip span, no restart pending or cooling down (an episode start); the counters stay
	void Reset()
	{
		m_fCoupling = 0;
		m_fSinceRestartS = -1;
		m_bRestartDue = false;
		m_fSlipS = 0;
		m_bLocked = false;
		m_iAgreeCalls = 0;
		m_fSyncLeftS = 0;
		m_iLastGear = -1;
		m_fCallSteps = 1;
		ForgetRpm();
	}

	//------------------------------------------------------------------------------------------------
	//! Mode slip: no readback history (no change of the mean, no rate)
	protected void ForgetRpm()
	{
		m_fLastRpm = -1;
		m_fLastMeanRpm = -1;
		m_iHistory = 0;
		m_fLastSliceS = 0;
		m_fPrevSliceS = 0;
	}

	//------------------------------------------------------------------------------------------------
	//! Mode slip: c = 0, not locked, no agreement
	protected void Open()
	{
		m_fCoupling = 0;
		m_bLocked = false;
		m_iAgreeCalls = 0;
	}

	//------------------------------------------------------------------------------------------------
	DC_LaunchClutchConfig GetConfig()
	{
		return m_Config;
	}

	//------------------------------------------------------------------------------------------------
	bool IsRamp()
	{
		return m_Config.IsRamp();
	}

	//------------------------------------------------------------------------------------------------
	//! Mode ramp or slip (the clutch is the helper's); mode off passes the commanded clutch
	bool IsActive()
	{
		return m_Config.IsActive();
	}

	//------------------------------------------------------------------------------------------------
	//! The applied clutch of this step (see the header). dt: the physics time the step covers [s]; inGear: a forward or
	//! reverse gear (not neutral); handbrake: commanded; rpm, engineOn: the last readback (EngineGetRPM, EngineIsOn);
	//! commandedClutch: the caller's clutch, an upper bound (its dip included); dip: the caller's dip is active. Mode
	//! slip also takes drivelineRpm, the engine rpm the driveline turns at in this gear (speed / its m/s per 1000 rpm,
	//! from the same readback; <= 0 = unknown: lock-up then only when c reaches 1), throttle and brake, the values the
	//! caller commands this call, and gear, the applied gear index (-1 = none). Modes off and ramp ignore them
	float Step(float dt, bool inGear, bool handbrake, float rpm, bool engineOn, float commandedClutch, bool dip, float drivelineRpm = -1, float throttle = 1, float brake = 0, int gear = -1)
	{
		float commanded = Math.Clamp(commandedClutch, 0, 1);
		float slice = Math.Max(dt, 0);
		if (m_iEngineSeen == 1 && !engineOn)
			m_iStalls++;

		m_iEngineSeen = 0;
		if (engineOn)
			m_iEngineSeen = 1;

		if (!m_Config.IsActive())
			return commanded;

		// the cooldown is the physics time elapsed at this restart point: the steps the previous calls covered, never
		// this call's (a call covering 3 steps after one covering 1 must not shorten it)
		bool cooled = m_fSinceRestartS < 0 || m_fSinceRestartS >= m_Config.m_fRestartCooldownS;
		if (!engineOn && !handbrake && m_Config.m_bRestartOnStall && cooled)
		{
			m_bRestartDue = true;
			m_iRestarts++;
			m_fSinceRestartS = 0;
			m_fCoupling = 0;
		}

		bool driving = inGear && !handbrake && !dip && engineOn;
		if (m_Config.IsSlip())
		{
			StepSlip(slice, driving, rpm, drivelineRpm, throttle > 0 && brake <= 0, gear);
		}
		else if (!driving)
		{
			m_fCoupling = 0;
		}
		else if (rpm < m_Config.m_fStallGuardRpm)
		{
			if (m_fCoupling > 0)
				m_iGuardDrops++;

			m_fCoupling = 0;
		}
		else if (rpm <= m_Config.m_fEngageRpm)
		{
			// below the band the clutch opens at once: opening never stalls an engine, while a rate-limited release
			// kept loading it under hard braking (the toy model: 606 -> 504 rpm at c 0.333 under a full brake)
			m_fCoupling = 0;
		}
		else
		{
			float target = Math.Clamp((rpm - m_Config.m_fEngageRpm) / (m_Config.m_fFullRpm - m_Config.m_fEngageRpm), 0, 1);
			if (target > m_fCoupling)
				m_fCoupling = Math.Min(target, m_fCoupling + slice / m_Config.m_fCoupleTimeS);
			else
				m_fCoupling = Math.Max(target, m_fCoupling - slice / m_Config.m_fUncoupleTimeS);
		}

		if (driving && m_fCoupling > 0 && m_fCoupling < 1)
			m_fSlipS += slice;
		else
			m_fSlipS = 0;

		if (m_fSinceRestartS >= 0)
			m_fSinceRestartS += slice;

		m_fMaxSlipS = Math.Max(m_fMaxSlipS, m_fSlipS);
		m_fMaxSlipTotalS = Math.Max(m_fMaxSlipTotalS, m_fSlipS);
		return Math.Min(commanded, m_fCoupling);
	}

	//------------------------------------------------------------------------------------------------
	//! Mode slip (DD-14d, MODE SLIP in the header): c for this call. slice: the physics time it covers; driving: in gear,
	//! no handbrake, no dip, engine on; rpm: the last readback; drivelineRpm: <= 0 = unknown; drive: throttle > 0 and no
	//! brake commanded this call; gear: the applied gear index (-1 = none)
	protected void StepSlip(float slice, bool driving, float rpm, float drivelineRpm, bool drive, int gear)
	{
		float engage = m_Config.m_fEngageRpm;
		// an upshift: its synchronisation drop is no fall to predict (a downshift raises the rpm and keeps the history)
		if (gear >= 0 && m_iLastGear >= 0 && gear > m_iLastGear)
		{
			ForgetRpm();
			m_fSyncLeftS = DC_LaunchClutchConfig.SYNC_S;
			m_iAgreeCalls = 0;
		}

		m_iLastGear = gear;

		// the mean of this readback and the last one (the readback alternates between half a step and no step after the
		// setters, DD-14d), its rate over the two calls before [rpm/s], and the rpm predicted for the end of this call
		float last = m_fLastRpm;
		if (last < 0)
			last = rpm;

		float mean = 0.5 * (rpm + last);
		float meanLast = m_fLastMeanRpm;
		if (meanLast < 0)
			meanLast = mean;

		float spanS = m_fLastSliceS + m_fPrevSliceS;
		float rate = 0;
		if (m_iHistory >= 2 && spanS > 0)
			rate = 2 * (mean - meanLast) / spanS;

		m_fLastRpm = rpm;
		m_fLastMeanRpm = mean;
		if (m_iHistory < 3)
			m_iHistory++;

		m_fPrevSliceS = m_fLastSliceS;
		m_fLastSliceS = slice;
		m_fCallSteps = Math.Max(Math.Max(slice / DC_LaunchClutchConfig.GAIN_STEP_S, 1), m_fCallSteps - DC_LaunchClutchConfig.CALL_STEPS_DECAY);
		bool inSync = m_fSyncLeftS > 0;
		m_fSyncLeftS = Math.Max(m_fSyncLeftS - slice, 0);
		float predicted = rpm + DC_LaunchClutchConfig.RELEASE_LOOKAHEAD_CALLS * slice * Math.Min(rate, 0);
		if (!driving)
		{
			Open();
			return;
		}

		if (rpm < m_Config.m_fStallGuardRpm)
		{
			if (m_fCoupling > 0)
				m_iGuardDrops++;

			Open();
			return;
		}

		if (m_bLocked)
		{
			// coupled: open at the floor, at the predicted floor below the lookahead ceiling, and without drive at a fall
			// faster than the plant explains
			float openRpm = engage;
			if (drive)
				openRpm = m_Config.GetLugRpm();

			if (rpm <= openRpm)
			{
				Open();
				return;
			}

			if (!inSync && rpm < m_Config.GetLookaheadCeilingRpm() && predicted <= openRpm)
			{
				Open();
				return;
			}

			if (!drive && !inSync && rate < -DC_LaunchClutchConfig.MAX_DROP_RPM_PER_S)
			{
				Open();
				return;
			}

			m_fCoupling = 1;
			return;
		}

		// slipping (a drive-off): open at engage_rpm, without drive (a stop before the lock-up) and at a predicted lug
		if (rpm <= engage || !drive || (!inSync && predicted <= m_Config.GetLugRpm()))
		{
			Open();
			return;
		}

		// the PI that holds the mean at full_rpm, per GAIN_STEP_S, rate-limited; lock-up once engine and driveline agree
		float kp = m_Config.m_fSlipKp / m_fCallSteps;
		float ki = m_Config.m_fSlipKi / (m_fCallSteps * m_fCallSteps);
		float want = m_fCoupling + slice * (kp * rate + ki * (mean - m_Config.m_fFullRpm));
		float lowest = Math.Max(m_fCoupling - slice / m_Config.m_fUncoupleTimeS, 0);
		float highest = Math.Min(m_fCoupling + slice / m_Config.m_fCoupleTimeS, 1);
		m_fCoupling = Math.Clamp(want, lowest, highest);
		if (m_fCoupling > 0 && drivelineRpm > 0 && Math.AbsFloat(rpm - drivelineRpm) <= m_Config.m_fLockToleranceRpm)
			m_iAgreeCalls++;
		else
			m_iAgreeCalls = 0;

		if (m_iAgreeCalls >= DC_LaunchClutchConfig.LOCK_AGREE_CALLS || m_fCoupling >= 1)
		{
			m_bLocked = true;
			m_fCoupling = 1;
			m_iLockups++;
		}
	}

	//------------------------------------------------------------------------------------------------
	//! true once after a Step that requested a restart: the caller then runs its Mode A start
	bool ConsumeRestart()
	{
		if (!m_bRestartDue)
			return false;

		m_bRestartDue = false;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! c after the last Step (0 in mode off, where nothing is ramped)
	float GetCoupling()
	{
		return m_fCoupling;
	}

	//------------------------------------------------------------------------------------------------
	int GetStalls()
	{
		return m_iStalls;
	}

	//------------------------------------------------------------------------------------------------
	int GetRestarts()
	{
		return m_iRestarts;
	}

	//------------------------------------------------------------------------------------------------
	int GetGuardDrops()
	{
		return m_iGuardDrops;
	}

	//------------------------------------------------------------------------------------------------
	//! Mode slip: lock-ups since Configure (driveline agreement or c reaching 1)
	int GetLockups()
	{
		return m_iLockups;
	}

	//------------------------------------------------------------------------------------------------
	//! Mode slip: coupled after a lock-up, until the release
	bool IsLocked()
	{
		return m_bLocked;
	}

	//------------------------------------------------------------------------------------------------
	//! Longest slip span since the last ResetMaxSlip (a probe phase) [s]
	float GetMaxSlipS()
	{
		return m_fMaxSlipS;
	}

	//------------------------------------------------------------------------------------------------
	//! Longest slip span since Configure [s]
	float GetMaxSlipTotalS()
	{
		return m_fMaxSlipTotalS;
	}

	//------------------------------------------------------------------------------------------------
	//! Starts a new max_slip_s window (a probe phase); a span in progress counts on
	void ResetMaxSlip()
	{
		m_fMaxSlipS = m_fSlipS;
	}

	//------------------------------------------------------------------------------------------------
	//! Run header lines at the probe stop: what the launch clutch did since Configure
	void AppendSummaryLines(notnull array<string> lines, string prefix)
	{
		lines.Insert(prefix + ".stalls=" + m_iStalls.ToString());
		lines.Insert(prefix + ".restarts=" + m_iRestarts.ToString());
		lines.Insert(prefix + ".guard_drops=" + m_iGuardDrops.ToString());
		lines.Insert(prefix + ".max_slip_s=" + m_fMaxSlipTotalS.ToString(-1, 3));
		lines.Insert(prefix + ".lockups=" + m_iLockups.ToString());
	}

	//------------------------------------------------------------------------------------------------
	//! One log fragment: the counters since Configure
	string DescribeCounters()
	{
		return string.Format("launch clutch %1: %2 stalls, %3 restarts, %4 guard drops, max slip %5 s, %6 lock-ups", m_Config.m_sMode, m_iStalls, m_iRestarts, m_iGuardDrops, m_fMaxSlipTotalS.ToString(-1, 2), m_iLockups);
	}
}
