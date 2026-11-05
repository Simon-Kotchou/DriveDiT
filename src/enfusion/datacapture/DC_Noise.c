// ============================================================================
// DC_Noise - N-mode perturbations of the expert's HL command (DD-7-ident component 1: the action-diverse pilot)
// ============================================================================
//
// DC_ExpertDriver (DC_ExpertHL.c) owns one DC_Noise when the trial conf names a perturbation profile ("perturb":
// "pilot_v1"). Without one (T0L, every conf before DD-7-ident: "perturb" absent or "none") nothing here runs and
// every episode is mode E, as before.
//
// CONTRACT (dc-2.0, frozen; nothing is added to it)
//   - The perturbation acts on the HL targets, action_cmd = (kappa_target, a_target, dir_cmd), never on llc_v1's
//     setter inputs. ctl.hl_* is the command applied (what the world model conditions on), ctl.exp_*_now stays the
//     clean expert re-evaluated on the true state (action_label, the policy target: "differs in N/D").
//   - An episode with a profile is mode N (episodes.mode and ctl.mode; r2 3.1 "N noise injection"). Inside it the
//     expert drives clean between disturbances; ctl.noise_active is 1 and ctl.noise_id the disturbance's id on every
//     row whose held command carries a disturbance (the rows after the decision that applies it, through the
//     decision that returns to the clean command).
//   - events perturb_start / perturb_end per disturbance, in the decision callbacks where it is first applied and
//     where the clean command returns (or at the episode end), payload noise_id, kind, amplitude, duration_s (9.5):
//     duration_s is the planned duration at the start and the realised one at the end; amplitude is signed, in the
//     kind's unit (KINDS). The validator's C17 judges mode-E episodes only, so a disturbance is never an expert
//     failure; an episode's terminations stay the expert's (r2 1.3).
//   - Forward only: dir_cmd is +1 or 0 (TBD-7: no session commands -1 before a probe drives reverse), so X mode's
//     reverse is not here; no teleport, no crash episode.
//
// KINDS (one disturbance = one segment; DC_Noise.s_aKindNames; amplitude logged as given here)
//   k_pulse  kappa + half-sine pulse, peak U(k_pulse_amp) x kappa_max(v), U(k_pulse_s) s, sign balanced; amplitude
//            the signed fraction of kappa_max(v) (r2 3.1 dense pulses, U(0.1, 0.5) of kappa_max(v), 0.5-2.5 s)
//   k_step   kappa + a step held U(k_step_s) s, U(k_step_amp) x kappa_max(v), sign balanced, then released: a
//            step-steer (signed fraction)
//   k_ou     kappa + OU noise (stationary sigma k_ou_sigma x kappa_max(v), time constant k_ou_tau_s), U(ou_s) s
//            (amplitude: the sigma fraction)
//   a_pulse  accel + half-sine pulse U(a_pulse_amp) m/s2, U(a_pulse_s) s, sign balanced (accelerate or brake; signed
//            m/s2)
//   a_ou     accel + OU noise (sigma a_ou_sigma m/s2, time constant a_ou_tau_s), U(ou_s) s (amplitude: sigma)
//   speed    speed-target schedule: for U(speed_s) s the accel target tracks f x the expert's own speed profile
//            (route curvature limit and route-end stop) with the expert's gain law, clamped to the expert's accel
//            limits; f = U(speed_f) (amplitude: f)
//   stop     scripted stop and start: dir 0 with a_target -U(stop_decel) m/s2 until the ego stands (below the
//            expert's stop speed), held U(stop_dwell_s) s, at most stop_max_s; then the expert drives off (the
//            start, clean) (amplitude: the a_target, negative)
//   brake    full brake: dir +1 with a_target -brake_mps2 (the S105 card brakes fully from -6.7 m/s2) until the speed
//            falls below brake_end_mps or brake_max_s pass; then the expert accelerates again (amplitude: a_target)
//   The applied kappa is clamped to the expert's +-kappa_max, the applied accel of k_*/a_* kinds to [-decel_cmd_max,
//   accel_cmd_max]. kappa_max(v) = min(expert kappa_max, kappa_alat_mps2 / max(v, min_speed_mps)^2), a stand-in for
//   the identified kappa_max(v) table of E-02Q (DD-7-ident component 2, not measured yet).
//
// SCHEDULE (per decision step of an N episode, decision_hz on the world-time grid; the segment state advances once
// per decision, the applied command is held over the decision interval)
//   - Segments run one at a time. After each one (and at the episode start, start_delay_s) a gap of gap_min_s plus
//     an exponential gap_mean_s; then the next segment starts at the first decision with the lane errors recovered
//     (|lane_lat_err| < recover_lat_m and |lane_head_err| < recover_head_deg held recover_hold_s of world time from
//     the first decision inside, reset by any decision outside: the r2 T3 recovery rule), the clean command forward,
//     the ego at min_speed_mps or faster and end_margin_m or more before the route end. So every disturbance starts
//     near the expert's state distribution and ends in clean recovery.
//   - Kinds are drawn by the weights w_*; the targets of the pilot (research delta 2026-11 section 3.3): at least
//     30% of decision steps off-expert, at least 40% of segments longitudinal, at least 10 stops per hour, both turn
//     directions per lateral kind balanced; tests/test_noise_pilot.py runs this file (DC_NoiseRng, DC_Noise),
//     translated statement for statement, over a simulated hour and checks them.
//   - Recovery is counted per segment: recovered within recover_window_s of its end (reached, then held
//     recover_hold_s), late, or not by the episode end; logged per episode and per session.
//
// DETERMINISM: the schedule (episode mode, kinds, signs, amplitudes, durations, gaps) comes from DC_NoiseRng seeded
// by (trial seed, episode id, stream 0), the OU innovations from stream 1, a fixed number of draws per segment: one
// seed gives one schedule (only when a segment can start depends on the state). Vary the trial seed per session.
//
// TRIAL CONF ($profile:dc_trials/<id>.json, read by DC_Session.ResolveTrial through DC_NoiseConfig.Read; dc-2.0 5.2:
// "the conf holds at least ..."): the top-level keys "perturb": "none" | "pilot_v1" (absent = none) and optionally
// "perturb_params": a flat object of JSON numbers overriding the profile's defaults (DC_NoiseConfig's table, each key
// with a range; an unknown key, a value out of range, a top-level perturb_params twice or without a profile, or a
// misspelt top-level perturb key next to them refuses the session). Only the root object's own keys count: a string
// value, an array item or a nested object's key is never one. session.json trial.conf records both as given (each
// number as written); the console log the resolved parameters and the sha256 of this file.
// ============================================================================

//------------------------------------------------------------------------------------------------
//! xorshift32 (Marsaglia 2003) on Enforce ints: 32-bit wrapping shifts, a logical right shift by masking. SelfTest
//! checks the known first outputs before a schedule is drawn
class DC_NoiseRng
{
	static const int SEED_MIX = 0x5DEECE66;
	static const int SEED_ZERO = 0x6D2B79F5;
	static const int WARMUP_ROUNDS = 8;
	static const float UNIT = 16777216;			//!< 2^24: a draw keeps the top 24 bits (exact in a float)

	protected int m_iState = SEED_ZERO;

	//------------------------------------------------------------------------------------------------
	//! The state of (seed, stream): (seed << 8) ^ stream ^ SEED_MIX (SEED_ZERO instead of 0), then WARMUP_ROUNDS
	//! draws. Distinct for seeds below 2^23 and streams below 256
	void Seed(int seed, int stream)
	{
		int x = (seed << 8) ^ stream ^ SEED_MIX;
		if (x == 0)
			x = SEED_ZERO;

		m_iState = x;
		for (int i = 0; i < WARMUP_ROUNDS; i++)
		{
			NextInt();
		}
	}

	//------------------------------------------------------------------------------------------------
	//! The next state: x ^= x << 13; x ^= x >>> 17; x ^= x << 5
	int NextInt()
	{
		int x = m_iState;
		x = x ^ (x << 13);
		x = x ^ Shr(x, 17);
		x = x ^ (x << 5);
		m_iState = x;
		return x;
	}

	//------------------------------------------------------------------------------------------------
	//! Uniform in [0, 1): the top 24 bits / 2^24
	float Uniform01()
	{
		float bits = Shr(NextInt(), 8);
		return bits / UNIT;
	}

	//------------------------------------------------------------------------------------------------
	//! Uniform in [lo, hi)
	float Uniform(float lo, float hi)
	{
		return lo + (hi - lo) * Uniform01();
	}

	//------------------------------------------------------------------------------------------------
	//! Uniform in (0, 1]: for a logarithm
	float UniformOpen01()
	{
		float bits = Shr(NextInt(), 8) + 1;
		return bits / UNIT;
	}

	//------------------------------------------------------------------------------------------------
	//! Standard normal (Box-Muller, two draws, the cosine branch)
	float Gauss()
	{
		float u1 = UniformOpen01();
		float u2 = Uniform01();
		return Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(Math.PI2 * u2);
	}

	//------------------------------------------------------------------------------------------------
	//! Logical right shift by 1..31 (the >> of an int is arithmetic)
	static int Shr(int value, int count)
	{
		return (value >> count) & (0x7FFFFFFF >> (count - 1));
	}

	//------------------------------------------------------------------------------------------------
	//! The first three NextInt of Seed(1, 0) (tests/test_noise_pilot.py computes them): false (failure) when this
	//! engine's int shifts differ, so no schedule may be drawn
	static bool SelfTest(out string failure)
	{
		failure = string.Empty;
		array<int> expected = {-688395553, 1580203561, -1665444516};
		DC_NoiseRng rng = new DC_NoiseRng();
		rng.Seed(1, 0);
		for (int i = 0; i < expected.Count(); i++)
		{
			int got = rng.NextInt();
			if (got != expected[i])
			{
				failure = string.Format("noise rng self-test %1 gave %2, expected %3", i, got, expected[i]);
				return false;
			}
		}

		failure = string.Empty;
		return true;
	}
}

//------------------------------------------------------------------------------------------------
//! The perturbation profile of a trial conf ("perturb", "perturb_params") with the parameters it resolves to
class DC_NoiseConfig
{
	static const string KEY_PROFILE = "perturb";
	static const string KEY_PARAMS = "perturb_params";
	static const string PROFILE_NONE = "none";
	static const string PROFILE_PILOT_V1 = "pilot_v1";
	static const int MAX_KEY_CHARS = 64;
	static const int CHAR_OBJECT_OPEN = 123;	//!< '{'
	static const int CHAR_OBJECT_CLOSE = 125;	//!< '}'
	static const int CHAR_BACKSLASH = 92;
	static const int MAX_NUMBER_CHARS = 32;
	static const int CHAR_PLUS = 43;
	static const int CHAR_MINUS = 45;
	static const int CHAR_POINT = 46;
	static const int CHAR_ZERO = 48;
	static const int CHAR_NINE = 57;
	static const int CHAR_E_UPPER = 69;
	static const int CHAR_E_LOWER = 101;

	//! The parameters of pilot_v1: key, default, lowest and highest allowed value (index-aligned)
	protected static ref array<string> s_aKeys = {
		"episode_n_share", "w_k_pulse", "w_k_step", "w_k_ou", "w_a_pulse", "w_a_ou", "w_speed", "w_stop", "w_brake",
		"start_delay_s", "gap_min_s", "gap_mean_s", "min_speed_mps", "end_margin_m",
		"recover_lat_m", "recover_head_deg", "recover_hold_s", "recover_window_s",
		"kappa_alat_mps2", "k_pulse_amp_lo", "k_pulse_amp_hi", "k_pulse_s_lo", "k_pulse_s_hi",
		"k_step_amp_lo", "k_step_amp_hi", "k_step_s_lo", "k_step_s_hi", "k_ou_sigma", "k_ou_tau_s",
		"ou_s_lo", "ou_s_hi", "a_pulse_amp_lo", "a_pulse_amp_hi", "a_pulse_s_lo", "a_pulse_s_hi",
		"a_ou_sigma", "a_ou_tau_s", "accel_cmd_max", "decel_cmd_max",
		"speed_f_lo", "speed_f_hi", "speed_s_lo", "speed_s_hi", "speed_gain",
		"stop_decel_lo", "stop_decel_hi", "stop_dwell_s_lo", "stop_dwell_s_hi", "stop_max_s",
		"brake_mps2", "brake_end_mps", "brake_max_s"
	};
	protected static ref array<float> s_aPilotV1 = {
		1, 0.22, 0.1, 0.18, 0.1, 0.08, 0.14, 0.1, 0.08,
		8, 2, 3, 3, 150,
		0.5, 3, 2, 10,
		4, 0.1, 0.5, 0.5, 2.5,
		0.2, 0.5, 0.6, 1.5, 0.2, 1,
		3, 8, 0.5, 2, 1, 3,
		0.8, 1, 2.5, 7,
		0.3, 1.1, 4, 10, 1,
		1.5, 3, 1, 4, 15,
		7, 1, 6
	};
	protected static ref array<float> s_aMin = {
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0.5, 0,
		0.05, 0.1, 0, 1,
		0.5, 0, 0, 0.2, 0.2,
		0, 0, 0.2, 0.2, 0, 0.1,
		0.2, 0.2, 0, 0, 0.2, 0.2,
		0, 0.1, 0.1, 0.1,
		0.1, 0.1, 0.2, 0.2, 0.1,
		0.1, 0.1, 0, 0, 1,
		0.5, 0, 0.2
	};
	protected static ref array<float> s_aMax = {
		1, 100, 100, 100, 100, 100, 100, 100, 100,
		300, 120, 300, 30, 2000,
		5, 45, 30, 120,
		10, 2, 2, 30, 30,
		2, 2, 30, 30, 2, 30,
		120, 120, 8, 8, 30, 30,
		8, 30, 8, 10,
		2, 2, 120, 120, 10,
		8, 8, 15, 15, 18,
		10, 10, 30
	};
	//! lo / hi pairs: lo <= hi
	protected static ref array<string> s_aPairs = {"k_pulse_amp", "k_pulse_s", "k_step_amp", "k_step_s", "ou_s", "a_pulse_amp", "a_pulse_s", "speed_f", "speed_s", "stop_decel", "stop_dwell_s"};

	protected string m_sProfile = PROFILE_NONE;
	protected bool m_bProfileGiven;
	protected ref array<float> m_aValues = {};
	protected ref array<string> m_aGivenKeys = {};
	protected ref array<float> m_aGivenValues = {};
	protected ref array<string> m_aGivenTexts = {};	//!< each override's JSON number text as the conf wrote it
	protected bool m_bParamsGiven;						//!< the conf has "perturb_params" (also an empty one)

	//------------------------------------------------------------------------------------------------
	//! true with a perturbation profile (not none)
	bool IsActive()
	{
		return m_sProfile != PROFILE_NONE;
	}

	//------------------------------------------------------------------------------------------------
	string GetProfile()
	{
		return m_sProfile;
	}

	//------------------------------------------------------------------------------------------------
	//! The resolved value of a parameter of the table; 0 and an ERROR line for a key that is not one
	float Get(string key)
	{
		int index = s_aKeys.Find(key);
		if (index < 0 || index >= m_aValues.Count())
		{
			Print("[DC_Noise] ERROR: no perturbation parameter " + key, LogLevel.ERROR);
			return 0;
		}

		return m_aValues[index];
	}

	//------------------------------------------------------------------------------------------------
	//! Reads "perturb" (context) and "perturb_params" (rawText, the conf as read whole for the key scan; empty when it
	//! was not, and then a conf that sets either key is refused: its keys cannot be checked). false (failure) on an unknown profile, perturb_params without
	//! a profile, an unknown or repeated key, a value that is not a number or out of its range, a lo above its hi,
	//! no positive weight, or stop_dwell_s_hi above stop_max_s
	bool Read(notnull JsonLoadContext context, string rawText, out string failure)
	{
		failure = string.Empty;
		m_sProfile = PROFILE_NONE;
		m_bProfileGiven = false;
		m_aValues.Clear();
		m_aGivenKeys.Clear();
		m_aGivenValues.Clear();
		m_aGivenTexts.Clear();
		m_bParamsGiven = false;
		// fail closed: without the scanned text the key checks below (escaped, repeated, misspelt keys, the top-level
		// perturb_params) cannot run, so a conf that sets a perturb key is refused; one without stays on the main path
		if (rawText.IsEmpty() && (context.DoesKeyExist(KEY_PROFILE) || context.DoesKeyExist(KEY_PARAMS)))
		{
			int scanMax = DC_Session.TRIAL_CONF_SCAN_MAX_BYTES;
			failure = "the conf sets \"" + KEY_PROFILE + "\" or \"" + KEY_PARAMS + "\" but its text was not scanned (more than " + scanMax.ToString() + " bytes, unreadable or not printable ASCII): its perturb keys cannot be checked";
			return false;
		}

		if (context.DoesKeyExist(KEY_PROFILE))
		{
			string profile;
			if (!context.ReadValue(KEY_PROFILE, profile))
			{
				failure = "\"" + KEY_PROFILE + "\" is not a string";
				return false;
			}

			if (profile != PROFILE_NONE && profile != PROFILE_PILOT_V1)
			{
				failure = "\"" + KEY_PROFILE + "\" is '" + profile + "' (known: " + PROFILE_NONE + ", " + PROFILE_PILOT_V1 + ")";
				return false;
			}

			m_sProfile = profile;
			m_bProfileGiven = true;
		}

		m_aValues.Copy(s_aPilotV1);
		// the root object's own keys only: a string value, an array item or a nested object's key is never one
		int paramsAt;
		string misspelt;
		if (!ScanTopLevelKeys(rawText, paramsAt, misspelt, failure))
			return false;

		bool hasParams = paramsAt >= 0 || (rawText.IsEmpty() && context.DoesKeyExist(KEY_PARAMS));
		// a misspelt perturb key refuses a conf that asks for perturbation; without "perturb" and "perturb_params"
		// the conf is read as before DD-7-ident (a WARNING only)
		if (!misspelt.IsEmpty())
		{
			string keyFailure = "the conf has a top-level key starting with \"" + KEY_PROFILE + "\" that is neither \"" + KEY_PROFILE + "\" nor \"" + KEY_PARAMS + "\": \"" + misspelt + "\"";
			if (m_bProfileGiven || hasParams)
			{
				failure = keyFailure;
				return false;
			}

			Print("[DC_Noise] WARNING: " + keyFailure + " (ignored: no perturbation profile)", LogLevel.WARNING);
		}

		if (!hasParams)
			return true;

		if (!IsActive())
		{
			failure = "\"" + KEY_PARAMS + "\" without a perturbation profile";
			return false;
		}

		if (rawText.IsEmpty())
		{
			failure = "\"" + KEY_PARAMS + "\" needs the conf read whole (observed.trial_conf_raw: printable ASCII, at most 4096 bytes)";
			return false;
		}

		if (!ScanParams(rawText, paramsAt, failure))
			return false;

		m_bParamsGiven = true;
		return CheckValues(failure);
	}

	//------------------------------------------------------------------------------------------------
	//! The keys of the conf's root object: a string at depth 1 (inside the root object, outside every string, nested
	//! object and array; a backslash escapes the next character of a string) followed by ':'. A string value, an array
	//! item or a key of a nested object is never one. paramsAt = the index of the top-level "perturb_params" value
	//! (its first character after the ':' and spaces), -1 without one; misspelt = the first top-level key that starts
	//! with "perturb" and is neither "perturb" nor "perturb_params" (JsonLoadContext, which cannot list keys on 1.8,
	//! would ignore it), empty when none. false (failure) when "perturb" or "perturb_params" is a top-level key twice
	//! (which one a JSON loader keeps is not defined), or when a top-level key holds a backslash escape: keys are
	//! compared as written, and "\u0070erturb_params" is the same JSON key as "perturb_params" (RFC 8259 8.3), so an
	//! escaped key is refused rather than misread. Without the text: true, -1 and empty
	static bool ScanTopLevelKeys(string rawText, out int paramsAt, out string misspelt, out string failure)
	{
		paramsAt = -1;
		misspelt = string.Empty;
		failure = string.Empty;
		int length = rawText.Length();
		int depth = 0;
		int index = 0;
		bool profileSeen = false;
		while (index < length)
		{
			int code = rawText.ToAscii(index);
			if (code == CHAR_OBJECT_OPEN || code == DC_JsonScan.CHAR_OPEN)
				depth++;
			else if (code == CHAR_OBJECT_CLOSE || code == DC_JsonScan.CHAR_CLOSE)
				depth--;

			if (code != DC_JsonScan.CHAR_QUOTE)
			{
				index++;
				continue;
			}

			// a string from index: its closing quote (and whether it holds an escape)
			int close = index + 1;
			bool escaped = false;
			while (close < length && rawText.ToAscii(close) != DC_JsonScan.CHAR_QUOTE)
			{
				if (rawText.ToAscii(close) == CHAR_BACKSLASH)
				{
					escaped = true;
					close++;
				}

				close++;
			}

			if (close >= length)
				return true;

			int after = DC_JsonScan.SkipSpace(rawText, close + 1);
			int keyLength = close - index - 1;
			bool isKey = depth == 1 && after < length && rawText.ToAscii(after) == DC_JsonScan.CHAR_COLON;
			if (isKey && escaped)
			{
				failure = "the conf has a top-level key with a backslash escape (\"" + rawText.Substring(index + 1, Math.MinInt(keyLength, MAX_KEY_CHARS)) + "\"): write every top-level key without escapes";
				return false;
			}

			if (isKey && keyLength >= KEY_PROFILE.Length())
			{
				string head = rawText.Substring(index + 1, Math.MinInt(keyLength, MAX_KEY_CHARS));
				bool isProfile = keyLength == KEY_PROFILE.Length() && head == KEY_PROFILE;
				bool isParams = keyLength == KEY_PARAMS.Length() && head == KEY_PARAMS;
				if ((isParams && paramsAt >= 0) || (isProfile && profileSeen))
				{
					failure = "\"" + head + "\" is a top-level key twice";
					return false;
				}

				if (isProfile)
					profileSeen = true;

				if (isParams)
					paramsAt = DC_JsonScan.SkipSpace(rawText, after + 1);
				else if (!isProfile && head.StartsWith(KEY_PROFILE) && misspelt.IsEmpty())
					misspelt = head;
			}

			index = close + 1;
		}

		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! The top-level "perturb_params" value at valueAt (ScanTopLevelKeys): {"key": number, ...}, scanned with
	//! DC_JsonScan (JsonLoadContext cannot list an object's keys on 1.8, and an unknown key must be refused, not
	//! ignored)
	protected bool ScanParams(string text, int valueAt, out string failure)
	{
		failure = string.Empty;
		int index = valueAt;
		if (index < 0 || !DC_JsonScan.Expect(text, index, CHAR_OBJECT_OPEN, index))
		{
			failure = "\"" + KEY_PARAMS + "\" is not an object";
			return false;
		}

		int after;
		if (DC_JsonScan.Expect(text, index, CHAR_OBJECT_CLOSE, after))
		{
			failure = string.Empty;
			return true;
		}

		while (true)
		{
			int close = -1;
			if (DC_JsonScan.Expect(text, index, DC_JsonScan.CHAR_QUOTE, index))
				close = text.IndexOfFrom(index, "\"");

			if (close < index || close - index > MAX_KEY_CHARS)
			{
				failure = "\"" + KEY_PARAMS + "\" has a key that is not a plain string of at most 64 characters";
				return false;
			}

			string key = text.Substring(index, close - index);
			int keyIndex = s_aKeys.Find(key);
			if (keyIndex < 0 || m_aGivenKeys.Find(key) >= 0)
			{
				failure = "\"" + KEY_PARAMS + "\" key '" + key + "' is unknown or repeated";
				return false;
			}

			float value;
			index = close + 1;
			int numberStart = -1;
			if (DC_JsonScan.Expect(text, index, DC_JsonScan.CHAR_COLON, index))
				numberStart = DC_JsonScan.SkipSpace(text, index);

			if (numberStart < 0 || !DC_JsonScan.ReadNumber(text, numberStart, value, index) || !IsNumberText(text, numberStart, index))
			{
				failure = "\"" + KEY_PARAMS + "\" key '" + key + "' has no number of at most 32 characters";
				return false;
			}

			m_aGivenKeys.Insert(key);
			m_aGivenValues.Insert(value);
			m_aGivenTexts.Insert(text.Substring(numberStart, index - numberStart));
			m_aValues[keyIndex] = value;
			if (DC_JsonScan.Expect(text, index, CHAR_OBJECT_CLOSE, index))
				break;

			if (!DC_JsonScan.Expect(text, index, DC_JsonScan.CHAR_COMMA, index))
			{
				failure = "\"" + KEY_PARAMS + "\" key '" + key + "' is not followed by a comma or the closing brace";
				return false;
			}
		}

		failure = string.Empty;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! text[start, end) is one JSON number (RFC 8259: -?(0|[1-9][0-9]*)(.[0-9]+)?([eE][+-]?[0-9]+)?) of at most
	//! MAX_NUMBER_CHARS characters, so session.json can carry it as written (DC_JsonWriter.AddNumberText copies
	//! whatever it is given)
	protected static bool IsNumberText(string text, int start, int end)
	{
		if (end <= start || end - start > MAX_NUMBER_CHARS)
			return false;

		int i = start;
		if (text.ToAscii(i) == CHAR_MINUS)
			i++;

		if (i >= end || !IsDigit(text.ToAscii(i)))
			return false;

		if (text.ToAscii(i) == CHAR_ZERO)
			i++;
		else
			i = SkipDigits(text, i, end);

		if (i < end && text.ToAscii(i) == CHAR_POINT)
		{
			int fraction = i + 1;
			i = SkipDigits(text, fraction, end);
			if (i == fraction)
				return false;
		}

		if (i < end && (text.ToAscii(i) == CHAR_E_LOWER || text.ToAscii(i) == CHAR_E_UPPER))
		{
			i++;
			if (i < end && (text.ToAscii(i) == CHAR_PLUS || text.ToAscii(i) == CHAR_MINUS))
				i++;

			int exponent = i;
			i = SkipDigits(text, exponent, end);
			if (i == exponent)
				return false;
		}

		return i == end;
	}

	//------------------------------------------------------------------------------------------------
	protected static bool IsDigit(int code)
	{
		return code >= CHAR_ZERO && code <= CHAR_NINE;
	}

	//------------------------------------------------------------------------------------------------
	//! The first index at or after i (before end) that is not a digit
	protected static int SkipDigits(string text, int i, int end)
	{
		int at = i;
		while (at < end && IsDigit(text.ToAscii(at)))
		{
			at++;
		}

		return at;
	}

	//------------------------------------------------------------------------------------------------
	//! Every value in its range, every lo <= hi, a positive weight, the stop dwell inside the stop cap
	protected bool CheckValues(out string failure)
	{
		failure = string.Empty;
		for (int i = 0; i < s_aKeys.Count(); i++)
		{
			float value = m_aValues[i];
			if (!(value >= s_aMin[i] && value <= s_aMax[i]))
			{
				failure = string.Format("%1 %2 is outside [%3, %4]", s_aKeys[i], value, s_aMin[i], s_aMax[i]);
				return false;
			}
		}

		foreach (string pair : s_aPairs)
		{
			if (Get(pair + "_lo") > Get(pair + "_hi"))
			{
				failure = string.Format("%1_lo %2 is above %1_hi %3", pair, Get(pair + "_lo"), Get(pair + "_hi"));
				return false;
			}
		}

		float weights = 0;
		foreach (string weightKey : DC_Noise.GetWeightKeys())
		{
			weights += Get(weightKey);
		}

		if (weights <= 0)
		{
			failure = "every kind weight w_* is 0";
			return false;
		}

		if (Get("stop_dwell_s_hi") > Get("stop_max_s"))
		{
			failure = string.Format("stop_dwell_s_hi %1 is above stop_max_s %2", Get("stop_dwell_s_hi"), Get("stop_max_s"));
			return false;
		}

		failure = string.Empty;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! session.json trial.conf: "perturb" and "perturb_params" as the conf gave them: each override's number text
	//! as written (so every accepted value round-trips exactly), an empty object as an empty object
	void WriteConf(notnull DC_JsonWriter json)
	{
		if (m_bProfileGiven)
			json.AddString(KEY_PROFILE, m_sProfile);

		if (!m_bParamsGiven)
			return;

		json.BeginObject(KEY_PARAMS);
		for (int i = 0; i < m_aGivenKeys.Count(); i++)
		{
			json.AddNumberText(m_aGivenKeys[i], m_aGivenTexts[i]);
		}

		json.EndObject();
	}

	//------------------------------------------------------------------------------------------------
	//! One log line: the profile and every resolved parameter (key=value; the overridden ones marked *)
	string Describe()
	{
		if (!IsActive())
			return "perturbation profile " + m_sProfile + ": every episode is mode E";

		string text = "perturbation profile " + m_sProfile + ":";
		for (int i = 0; i < s_aKeys.Count(); i++)
		{
			text += " " + s_aKeys[i] + "=" + m_aValues[i].ToString(-1, 3);
			if (m_aGivenKeys.Find(s_aKeys[i]) >= 0)
				text += "*";
		}

		return text;
	}
}

//------------------------------------------------------------------------------------------------
//! What DC_Noise needs of the state after a decision callback (filled by DC_ExpertDriver)
class DC_NoiseInput
{
	float m_fSpeedMps;
	float m_fRemainingM;		//!< route length minus ordered progress
	float m_fLateralM;			//!< lane_lat_err_m
	float m_fHeadingErrRad;		//!< lane_head_err_rad
	float m_fProfileAheadMps;	//!< the expert's speed profile at its preview point (DC_ExpertHL.ProfileAt)
	int m_iWorldMs;				//!< world_time_ms of the decision callback (the recovery hold is world time)
}

//------------------------------------------------------------------------------------------------
//! The perturbation schedule of N episodes and the command it applies (see the header)
class DC_Noise
{
	static const string VERSION = "noise_pilot_v1";
	static const string SOURCE_FILE = "DC_Noise.c";

	static const int KIND_K_PULSE = 0;
	static const int KIND_K_STEP = 1;
	static const int KIND_K_OU = 2;
	static const int KIND_A_PULSE = 3;
	static const int KIND_A_OU = 4;
	static const int KIND_SPEED = 5;
	static const int KIND_STOP = 6;
	static const int KIND_BRAKE = 7;
	static const int FIRST_LONGITUDINAL = KIND_A_PULSE;	//!< kinds from here on act on a_target or dir_cmd
	static const int SIGN_BALANCE_MAX = 2;				//!< per kind and episode, |#right - #left| never exceeds this

	protected static ref array<string> s_aKindNames = {"k_pulse", "k_step", "k_ou", "a_pulse", "a_ou", "speed", "stop", "brake"};
	protected static ref array<string> s_aWeightKeys = {"w_k_pulse", "w_k_step", "w_k_ou", "w_a_pulse", "w_a_ou", "w_speed", "w_stop", "w_brake"};

	// parameters (Init)
	protected ref DC_NoiseConfig m_Config;
	protected ref array<float> m_aWeights = {};
	protected float m_fDecisionDtS = 0.2;
	protected float m_fKappaCmdMax = 0.2;
	protected float m_fExpertAccelMax = 1.5;
	protected float m_fExpertBrakeMax = 4;
	protected float m_fStopSpeedMps = 0.3;

	// random streams
	protected ref DC_NoiseRng m_Schedule = new DC_NoiseRng();	//!< stream 0: mode, kinds, signs, amplitudes, durations, gaps
	protected ref DC_NoiseRng m_Innovation = new DC_NoiseRng();	//!< stream 1: the OU innovations

	// episode
	protected bool m_bEpisodeN;
	protected ref array<int> m_aBalance = {};		//!< per kind: + right / accelerate, - left / brake
	protected int m_iGapSteps;			//!< decisions still to wait before the next segment may start
	protected int m_iInsideSinceMs = -1;	//!< world time of the first decision of the current run inside the
											//!< recovery bounds; -1 while outside
	// the drawn next (or running) segment
	protected int m_iKind = -1;
	protected float m_fSign = 1;
	protected float m_fAmplitude;		//!< as logged (see the header)
	protected int m_iPlannedSteps;		//!< pulse, step, OU, speed: decisions; stop: the dwell; brake: the cap
	protected int m_iCapSteps;
	// the running segment
	protected bool m_bActive;
	protected int m_iNoiseId = -1;
	protected int m_iStep;
	protected bool m_bStopped;			//!< stop: the ego stands, the dwell runs
	protected int m_iDwellDone;
	protected float m_fOu;
	// recovery after a segment
	protected bool m_bAwaitRecovery;
	protected int m_iEndMs;				//!< world time of the decision that ended the last segment

	// counters (session-wide; the noise id is unique in the session)
	protected int m_iNextNoiseId;
	protected int m_iDecisions;
	protected int m_iPerturbedDecisions;
	protected ref array<int> m_aKindCount = {};
	protected int m_iRecoveredInWindow;
	protected int m_iRecoveredLate;
	protected int m_iNotRecovered;
	protected int m_iEpisodeDecisions;
	protected int m_iEpisodePerturbed;
	protected int m_iEpisodeSegments;
	protected int m_iStops;				//!< stop segments in which the ego stood (a stop, then a start)
	protected int m_iEpisodeStops;

	//------------------------------------------------------------------------------------------------
	static array<string> GetWeightKeys()
	{
		return s_aWeightKeys;
	}

	//------------------------------------------------------------------------------------------------
	static string KindName(int kind)
	{
		if (kind < 0 || kind >= s_aKindNames.Count())
			return "none";

		return s_aKindNames[kind];
	}

	//------------------------------------------------------------------------------------------------
	//! Binds the resolved profile, the decision period [s] and the expert's limits (kappa_max [1/m], accel and brake
	//! command limits [m/s2], stop speed [m/s])
	void Init(notnull DC_NoiseConfig config, float decisionDtS, float kappaCmdMax, float accelMax, float brakeMax, float stopSpeedMps)
	{
		m_Config = config;
		m_fDecisionDtS = decisionDtS;
		m_fKappaCmdMax = kappaCmdMax;
		m_fExpertAccelMax = accelMax;
		m_fExpertBrakeMax = brakeMax;
		m_fStopSpeedMps = stopSpeedMps;
		m_aWeights.Clear();
		m_aKindCount.Clear();
		foreach (string weightKey : s_aWeightKeys)
		{
			m_aWeights.Insert(config.Get(weightKey));
			m_aKindCount.Insert(0);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! Episode start: seeds both streams from (trialSeed, episodeId), draws the episode's mode (N with
	//! probability episode_n_share) and, for N, the first segment after start_delay_s. true for an N episode
	bool BeginEpisode(int trialSeed, int episodeId)
	{
		m_Schedule.Seed(trialSeed, episodeId * 2);
		m_Innovation.Seed(trialSeed, episodeId * 2 + 1);
		m_bEpisodeN = m_Schedule.Uniform01() < m_Config.Get("episode_n_share");
		m_aBalance.Clear();
		for (int k = 0; k < s_aKindNames.Count(); k++)
		{
			m_aBalance.Insert(0);
		}

		m_bActive = false;
		m_iNoiseId = -1;
		m_bAwaitRecovery = false;
		m_iInsideSinceMs = -1;
		m_iEpisodeDecisions = 0;
		m_iEpisodePerturbed = 0;
		m_iEpisodeSegments = 0;
		m_iEpisodeStops = 0;
		DrawSegment();
		m_iGapSteps = Steps(m_Config.Get("start_delay_s"));
		return m_bEpisodeN;
	}

	//------------------------------------------------------------------------------------------------
	//! One decision step of an N episode: applied = clean, or clean with the running segment's disturbance and its
	//! noise flags (DC_HLCommand.m_bNoise, m_iNoiseId). Starts the next segment when it is due and allowed, ends the
	//! running one when it is over (that decision is clean)
	void Decide(notnull DC_HLCommand clean, notnull DC_NoiseInput input, notnull DC_HLCommand applied)
	{
		applied.CopyFrom(clean);
		applied.m_bNoise = false;
		applied.m_iNoiseId = -1;
		if (!m_bEpisodeN)
			return;

		m_iDecisions++;
		m_iEpisodeDecisions++;
		UpdateRecovery(input);
		if (!m_bActive)
		{
			if (m_iGapSteps > 0)
			{
				m_iGapSteps--;
				return;
			}

			if (!CanStart(clean, input))
				return;

			StartSegment();
		}

		if (clean.m_iDir != 1 || !ApplySegment(clean, input, applied))
		{
			EndSegment(input.m_iWorldMs);
			applied.CopyFrom(clean);
			applied.m_bNoise = false;
			applied.m_iNoiseId = -1;
			return;
		}

		applied.m_bNoise = true;
		applied.m_iNoiseId = m_iNoiseId;
		m_iStep++;
		m_iPerturbedDecisions++;
		m_iEpisodePerturbed++;
	}

	//------------------------------------------------------------------------------------------------
	//! The disturbance of the running segment on the clean command; false when the segment is over
	protected bool ApplySegment(notnull DC_HLCommand clean, notnull DC_NoiseInput input, notnull DC_HLCommand applied)
	{
		float speed = input.m_fSpeedMps;
		float decelMax = m_Config.Get("decel_cmd_max");
		float accelMax = m_Config.Get("accel_cmd_max");
		if (m_iKind == KIND_K_PULSE || m_iKind == KIND_K_STEP || m_iKind == KIND_K_OU)
		{
			if (m_iStep >= m_iPlannedSteps)
				return false;

			float kappaDelta = m_fAmplitude * KappaScale(speed);
			if (m_iKind == KIND_K_PULSE)
				kappaDelta = kappaDelta * PulseShape(m_iStep, m_iPlannedSteps);
			else if (m_iKind == KIND_K_OU)
				kappaDelta = NextOu(m_Config.Get("k_ou_tau_s")) * KappaScale(speed);

			applied.m_fKappa = Math.Clamp(clean.m_fKappa + kappaDelta, -m_fKappaCmdMax, m_fKappaCmdMax);
			return true;
		}

		if (m_iKind == KIND_A_PULSE || m_iKind == KIND_A_OU)
		{
			if (m_iStep >= m_iPlannedSteps)
				return false;

			float accelDelta = m_fAmplitude * PulseShape(m_iStep, m_iPlannedSteps);
			if (m_iKind == KIND_A_OU)
				accelDelta = NextOu(m_Config.Get("a_ou_tau_s"));

			applied.m_fAccel = Math.Clamp(clean.m_fAccel + accelDelta, -decelMax, accelMax);
			return true;
		}

		if (m_iKind == KIND_SPEED)
		{
			if (m_iStep >= m_iPlannedSteps)
				return false;

			float target = m_fAmplitude * input.m_fProfileAheadMps;
			applied.m_fAccel = Math.Clamp(m_Config.Get("speed_gain") * (target - speed), -m_fExpertBrakeMax, m_fExpertAccelMax);
			return true;
		}

		if (m_iKind == KIND_STOP)
		{
			if (m_iStep >= m_iCapSteps)
				return false;

			if (!m_bStopped && speed < m_fStopSpeedMps)
			{
				m_bStopped = true;
				m_iStops++;
				m_iEpisodeStops++;
			}

			if (m_bStopped)
			{
				if (m_iDwellDone >= m_iPlannedSteps)
					return false;

				m_iDwellDone++;
			}

			applied.m_iDir = 0;
			applied.m_fAccel = m_fAmplitude;
			return true;
		}

		// KIND_BRAKE
		if (m_iStep >= m_iCapSteps || speed < m_Config.Get("brake_end_mps"))
			return false;

		applied.m_fAccel = m_fAmplitude;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! A segment may start: the clean command forward, the ego fast enough and far enough from the route end, the lane
	//! errors recovered for recover_hold_s
	protected bool CanStart(notnull DC_HLCommand clean, notnull DC_NoiseInput input)
	{
		if (clean.m_iDir != 1 || input.m_fSpeedMps < m_Config.Get("min_speed_mps"))
			return false;

		if (input.m_fRemainingM < m_Config.Get("end_margin_m"))
			return false;

		return IsRecovered(input);
	}

	//------------------------------------------------------------------------------------------------
	protected void StartSegment()
	{
		m_bActive = true;
		m_iNoiseId = m_iNextNoiseId;
		m_iNextNoiseId++;
		m_iStep = 0;
		m_bStopped = false;
		m_iDwellDone = 0;
		m_fOu = 0;
		m_bAwaitRecovery = false;
		m_aKindCount[m_iKind] = m_aKindCount[m_iKind] + 1;
		m_iEpisodeSegments++;
	}

	//------------------------------------------------------------------------------------------------
	//! The running segment ends: recovery is awaited, the next segment and its gap are drawn
	protected void EndSegment(int worldMs)
	{
		m_bActive = false;
		m_iNoiseId = -1;
		m_bAwaitRecovery = true;
		m_iEndMs = worldMs;
		DrawSegment();
	}

	//------------------------------------------------------------------------------------------------
	//! The next segment: five draws of the schedule stream, always (kind, sign, amplitude, duration, gap)
	protected void DrawSegment()
	{
		float uKind = m_Schedule.Uniform01();
		float uSign = m_Schedule.Uniform01();
		float uAmplitude = m_Schedule.Uniform01();
		float uDuration = m_Schedule.Uniform01();
		float uGap = m_Schedule.UniformOpen01();

		// kind: the first whose cumulative weight passes uKind x the total (weights of 0 never chosen)
		float total = 0;
		foreach (float weight : m_aWeights)
		{
			total += weight;
		}

		m_iKind = -1;
		float cumulative = 0;
		for (int k = 0; k < m_aWeights.Count(); k++)
		{
			cumulative += m_aWeights[k];
			if (m_iKind < 0 && m_aWeights[k] > 0 && uKind * total < cumulative)
				m_iKind = k;
		}

		for (int last = m_aWeights.Count() - 1; m_iKind < 0 && last >= 0; last--)
		{
			if (m_aWeights[last] > 0)
				m_iKind = last;
		}

		// sign: uSign, kept within SIGN_BALANCE_MAX of balance per kind and episode (k_pulse, k_step, a_pulse)
		int sign = 1;
		if (uSign >= 0.5)
			sign = -1;

		int balance = m_aBalance[m_iKind];
		if (balance >= SIGN_BALANCE_MAX)
			sign = -1;
		else if (balance <= -SIGN_BALANCE_MAX)
			sign = 1;

		m_fSign = sign;
		m_iCapSteps = 0;
		float durationS = 0;
		if (m_iKind == KIND_K_PULSE)
		{
			m_fAmplitude = sign * Lerp("k_pulse_amp", uAmplitude);
			durationS = Lerp("k_pulse_s", uDuration);
		}
		else if (m_iKind == KIND_K_STEP)
		{
			m_fAmplitude = sign * Lerp("k_step_amp", uAmplitude);
			durationS = Lerp("k_step_s", uDuration);
		}
		else if (m_iKind == KIND_K_OU)
		{
			m_fAmplitude = m_Config.Get("k_ou_sigma");
			durationS = Lerp("ou_s", uDuration);
		}
		else if (m_iKind == KIND_A_PULSE)
		{
			m_fAmplitude = sign * Lerp("a_pulse_amp", uAmplitude);
			durationS = Lerp("a_pulse_s", uDuration);
		}
		else if (m_iKind == KIND_A_OU)
		{
			m_fAmplitude = m_Config.Get("a_ou_sigma");
			durationS = Lerp("ou_s", uDuration);
		}
		else if (m_iKind == KIND_SPEED)
		{
			m_fAmplitude = Lerp("speed_f", uAmplitude);
			durationS = Lerp("speed_s", uDuration);
		}
		else if (m_iKind == KIND_STOP)
		{
			m_fAmplitude = -Lerp("stop_decel", uAmplitude);
			durationS = Lerp("stop_dwell_s", uDuration);
			m_iCapSteps = Steps(m_Config.Get("stop_max_s"));
		}
		else
		{
			m_fAmplitude = -m_Config.Get("brake_mps2");
			durationS = m_Config.Get("brake_max_s");
			m_iCapSteps = Steps(durationS);
		}

		if (m_iKind == KIND_K_PULSE || m_iKind == KIND_K_STEP || m_iKind == KIND_A_PULSE)
			m_aBalance[m_iKind] = balance + sign;

		m_iPlannedSteps = Steps(durationS);
		if (m_iPlannedSteps < 1)
			m_iPlannedSteps = 1;

		float gapS = m_Config.Get("gap_min_s") - m_Config.Get("gap_mean_s") * Math.Log(uGap);
		m_iGapSteps = Steps(gapS);
	}

	//------------------------------------------------------------------------------------------------
	//! <key>_lo + u (<key>_hi - <key>_lo)
	protected float Lerp(string key, float u)
	{
		float lo = m_Config.Get(key + "_lo");
		float hi = m_Config.Get(key + "_hi");
		return lo + (hi - lo) * u;
	}

	//------------------------------------------------------------------------------------------------
	//! Decision steps of a duration: the quotient seconds / dt rounded to the nearest integer, an exact half up, at least
	//! 0. The fractional part (exact in binary32 for a positive quotient) is compared with 0.5, so a quotient just below
	//! a half never rounds up (adding 0.5 first would: 0.4999999702 + 0.5 is 1.0 in binary32), and Math.Round, whose
	//! tie rule the 1.8 API does not document, is not used
	protected int Steps(float seconds)
	{
		float quotient = seconds / m_fDecisionDtS;
		if (!(quotient > 0))
			return 0;

		float whole = Math.Floor(quotient);
		int steps = whole;
		if (quotient - whole >= 0.5)
			steps++;

		return steps;
	}

	//------------------------------------------------------------------------------------------------
	//! Half-sine over n decisions, sampled at the middle of decision i
	protected static float PulseShape(int i, int n)
	{
		return Math.Sin(Math.PI * (i + 0.5) / n);
	}

	//------------------------------------------------------------------------------------------------
	//! kappa_max(v) = min(the expert's kappa_max, kappa_alat_mps2 / max(v, min_speed_mps)^2)
	protected float KappaScale(float speed)
	{
		float v = Math.Max(speed, m_Config.Get("min_speed_mps"));
		return Math.Min(m_fKappaCmdMax, m_Config.Get("kappa_alat_mps2") / (v * v));
	}

	//------------------------------------------------------------------------------------------------
	//! The OU state one decision on: x = rho x + sigma sqrt(1 - rho^2) g, rho = exp(-dt / tau), sigma = the
	//! segment's amplitude (stationary standard deviation)
	protected float NextOu(float tauS)
	{
		float rho = Math.Pow(Math.E, -m_fDecisionDtS / tauS);
		m_fOu = rho * m_fOu + m_fAmplitude * Math.Sqrt(1 - rho * rho) * m_Innovation.Gauss();
		return m_fOu;
	}

	//------------------------------------------------------------------------------------------------
	//! Lane-error recovery (the r2 T3 rule: |lane_lat_err| < recover_lat_m and |lane_head_err| < recover_head_deg
	//! held recover_hold_s) on the state of this decision: the run inside the bounds starts at its first decision
	//! (world time) and ends at the first decision outside; the last segment counts as recovered within the window
	//! when that run began at most recover_window_s after the segment's end (or before it), else late
	protected void UpdateRecovery(notnull DC_NoiseInput input)
	{
		bool inside = Math.AbsFloat(input.m_fLateralM) < m_Config.Get("recover_lat_m") && Math.AbsFloat(input.m_fHeadingErrRad) < m_Config.Get("recover_head_deg") * Math.DEG2RAD;
		if (!inside)
			m_iInsideSinceMs = -1;
		else if (m_iInsideSinceMs < 0)
			m_iInsideSinceMs = input.m_iWorldMs;

		if (!m_bAwaitRecovery || !IsRecovered(input))
			return;

		m_bAwaitRecovery = false;
		if (m_iInsideSinceMs - m_iEndMs <= m_Config.Get("recover_window_s") * 1000)
			m_iRecoveredInWindow++;
		else
			m_iRecoveredLate++;
	}

	//------------------------------------------------------------------------------------------------
	//! The lane errors have been inside the recovery bounds for recover_hold_s of world time, counted from the first
	//! decision of the run (a single decision inside is no time held)
	protected bool IsRecovered(notnull DC_NoiseInput input)
	{
		return m_iInsideSinceMs >= 0 && input.m_iWorldMs - m_iInsideSinceMs >= m_Config.Get("recover_hold_s") * 1000;
	}

	//------------------------------------------------------------------------------------------------
	//! Episode end: a segment awaiting recovery counts as not recovered; a running one ends (the driver logs its
	//! perturb_end). The per-episode log line
	string EndEpisode()
	{
		if (m_bActive)
		{
			m_bActive = false;
			m_iNoiseId = -1;
			m_bAwaitRecovery = true;
		}

		if (m_bAwaitRecovery)
			m_iNotRecovered++;

		m_bAwaitRecovery = false;
		if (!m_bEpisodeN)
			return string.Empty;

		m_bEpisodeN = false;
		float share = 0;
		if (m_iEpisodeDecisions > 0)
			share = m_iEpisodePerturbed / (m_iEpisodeDecisions * 1.0);

		return string.Format("noise: %1 segment(s), %2 of %3 decisions perturbed (share %4), %5 stop(s)", m_iEpisodeSegments, m_iEpisodePerturbed, m_iEpisodeDecisions, share.ToString(-1, 3), m_iEpisodeStops);
	}

	//------------------------------------------------------------------------------------------------
	//! The session's counters: segments per kind, perturbed decisions, recovery
	string Summary()
	{
		string kinds;
		for (int k = 0; k < s_aKindNames.Count(); k++)
		{
			if (k > 0)
				kinds += " ";

			kinds += s_aKindNames[k] + "=" + m_aKindCount[k].ToString();
		}

		float share = 0;
		if (m_iDecisions > 0)
			share = m_iPerturbedDecisions / (m_iDecisions * 1.0);

		return string.Format("noise %1: segments %2; %3 of %4 N-episode decisions perturbed (share %5); %6 stop(s); recovered within the window %7, late %8, not by the episode end %9", VERSION, kinds, m_iPerturbedDecisions, m_iDecisions, share.ToString(-1, 3), m_iStops, m_iRecoveredInWindow, m_iRecoveredLate, m_iNotRecovered);
	}

	//------------------------------------------------------------------------------------------------
	//! "noise_id=..;kind=..;amplitude=.." of the running segment (perturb_start / perturb_end add duration_s)
	string PayloadBase()
	{
		return "noise_id=" + m_iNoiseId.ToString() + ";kind=" + KindName(m_iKind) + ";amplitude=" + m_fAmplitude.ToString(-1, 6);
	}

	//------------------------------------------------------------------------------------------------
	//! The running segment's planned duration [s] (stop: the dwell cap stop_max_s; brake: brake_max_s)
	float GetPlannedDurationS()
	{
		if (m_iKind == KIND_STOP || m_iKind == KIND_BRAKE)
			return m_iCapSteps * m_fDecisionDtS;

		return m_iPlannedSteps * m_fDecisionDtS;
	}

	//------------------------------------------------------------------------------------------------
	int GetActiveNoiseId()
	{
		return m_iNoiseId;
	}
}
