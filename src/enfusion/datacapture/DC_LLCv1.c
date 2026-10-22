// ============================================================================
// DC_LLCv1 - llc_v1, the one in-engine low-level controller of every driver (DD-7-min, r2 3.2)
// ============================================================================
//
// Turns the held HL command (contract action_cmd: kappa [1/m] right-positive, accel [m/s2], dir) into the setter
// inputs of one physics step. The driver interface (DC_StreamRecorder.c) calls it from ApplyStep, i.e. in EOnSimulate
// of every physics step with a held command (the step source DD-4-min uses; no CallLater clock), and records exactly
// what it passed in the ctl llc_* fields. It never calls a setter itself: everything goes through DC_Actuator.Apply
// (Mode A now, Mode B later, r3 P1).
//
// PER PHYSICS STEP (inputs: speed, yaw rate and gear only, read back from the simulation; r2 3.2)
//   steering   contract steer = inverse identified map of the vehicle card: (kappa - kappa_offset(v)) /
//              kappa_per_steer(v), interpolated linearly in speed and clamped at the end bins; plus an optional
//              curvature trim (integral of kappa - yaw_rate / v, card gain, 0 = off); clamped to +-steer_max;
//              llc_steer_raw = steer_sign x that (engine-native sign, TBD-5); then the explicit rate limiter
//              (steer_rate_per_s, 0 = identity) and the first-order lag (steer_lag_tau_s, 0 = identity) give
//              llc_steer_lim, the value passed to SetSteering.
//   longitude  dir +1: a speed reference integrates accel (kept within speed_window_mps of the speed) and a PI on
//              its error plus the accel feedforward gives the effort u [m/s2]; u above the coast band -> throttle
//              (u x throttle_per_mps2, at most throttle_max), below -> brake (-u x brake_per_mps2), never both; with
//              no positive accel left and the reference at 0 the car is held with brake_hold. dir 0: throttle 0,
//              brake from -accel, brake_hold below stop_speed_mps. dir -1 is never commanded while T0-MIN is
//              forward-only (the core fails the session first); llc_v1 brakes.
//   gearbox    the card's gearbox mode (E-04 / E-01): "automatic" = no SetGear and clutch 1 (llc_gear and llc_clutch
//              empty), as the probe kit's speed hold; "scripted" = SetGear every step to the card's forward gears
//              by speed with hysteresis, clutch 1; every scripted gear must be one of the card's measured
//              gear_indices_forward. Forward gears only: no neutral or reverse logic until their indices are
//              measured (TBD-7).
//
// VEHICLE CARD (tools/ident/cards/<pin_id>/<vehicle>.json in the repo, deployed to Configs/ident/cards/; r3 scope note)
//   - loaded once per session (DC_VehicleCard.Load); its sha256 (over the file bytes, the value the DD-5-min reader
//     compares with session.json control.vehicle_card_sha) is logged and stamped;
//   - refused when its build_key differs from the session's; on a dd-launch/1 session (no build_key) its pin_id must
//     be the session's and the build_key is not compared (WARNING);
//   - status "placeholder" (no identification session ids): accepted only for qualification (use_class dev or
//     probe), refused at any other use_class, production included; status "identified" needs the identification
//     session ids and a build_key;
//   - limiter rule (CC-08, M3): llc's own rate limiter is the identity or strictly slower than the engine's
//     identified steering rate limit, so the two never stack; an identified engine path that shapes the input
//     (input_echo false) without an identified limit allows only the identity. The card stores the identified
//     actuator structure (raw engine path and full llc path, per speed and amplitude bin); llc_v1 reads only what it
//     needs. The first-pass fit writes the identity limiter.
//
// PROVENANCE: llc_sha = sha256 of this source file as deployed (DC_Sha256 over its bytes; DC_ExpertDriver reads it),
// so session.json control.llc_sha names the exact controller; VERSION is "llc_v1".
//
// HELPERS: DC_Sha256 (FIPS 180-4 over ASCII text; Enforce has no hash API with a published algorithm, string.Hash()
// is an engine hash), DC_TextFile (whole-file reads, 7-bit ASCII only), DC_JsonScan (flat number and plain-string
// arrays and null values of a JSON text: JsonLoadContext reads the scalar keys, but its handling of JSON arrays and
// of null is not verified on 1.8, so those are scanned with string APIs and checked by the caller).
// ============================================================================

//------------------------------------------------------------------------------------------------
//! SHA-256 of a 7-bit ASCII text (FIPS 180-4), lower-case hex. Assumes 32-bit int addition that wraps and an
//! arithmetic right shift; SelfTest checks that assumption on this engine before any digest is trusted
class DC_Sha256
{
	static const string HEX_DIGITS = "0123456789abcdef";

	//! Round constants (FIPS 180-4 4.2.2)
	protected static ref array<int> s_aK = {
		0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
		0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
		0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
		0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
		0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
		0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
		0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
		0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
	};

	//! Initial hash value (FIPS 180-4 5.3.3)
	protected static ref array<int> s_aH0 = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

	//------------------------------------------------------------------------------------------------
	//! Hex digest of the text's bytes; empty when a character is not 7-bit ASCII (1..127)
	static string HashText(string text)
	{
		int length = text.Length();
		array<int> bytes = {};
		bytes.Reserve(length + 72);
		for (int i = 0; i < length; i++)
		{
			int code = text.ToAscii(i);
			if (code < 1 || code > 127)
				return string.Empty;

			bytes.Insert(code);
		}

		// Padding: 0x80, zeros to 56 mod 64, the bit length as a 64-bit big-endian number
		bytes.Insert(0x80);
		while (bytes.Count() % 64 != 56)
		{
			bytes.Insert(0);
		}

		int bitsHigh = Shr(length, 29);
		int bitsLow = length << 3;
		AppendWord(bytes, bitsHigh);
		AppendWord(bytes, bitsLow);

		array<int> state = {};
		state.Copy(s_aH0);
		array<int> schedule = {};
		schedule.Resize(64);
		for (int block = 0; block < bytes.Count(); block += 64)
		{
			for (int t = 0; t < 16; t++)
			{
				int at = block + 4 * t;
				schedule[t] = (bytes[at] << 24) | (bytes[at + 1] << 16) | (bytes[at + 2] << 8) | bytes[at + 3];
			}

			Compress(state, schedule);
		}

		string digest;
		foreach (int word : state)
		{
			digest += WordHex(word);
		}

		return digest;
	}

	//------------------------------------------------------------------------------------------------
	//! Known answers of FIPS 180-4 / NIST CAVS (empty, one block, two blocks). false (failure) when this engine's
	//! integer arithmetic differs from the assumption, so no digest of DC_Sha256 may be stamped
	static bool SelfTest(out string failure)
	{
		array<string> messages = {"", "abc", "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"};
		array<string> digests = {
			"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
			"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
			"248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
		};
		for (int i = 0; i < messages.Count(); i++)
		{
			string got = HashText(messages[i]);
			if (got != digests[i])
			{
				failure = string.Format("sha256 self-test %1 gave %2, expected %3", i, got, digests[i]);
				return false;
			}
		}

		failure = string.Empty;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! One 512-bit block: schedule[0..15] holds the block's words, [16..63] are filled here
	protected static void Compress(notnull array<int> state, notnull array<int> schedule)
	{
		for (int t = 16; t < 64; t++)
		{
			int w15 = schedule[t - 15];
			int w2 = schedule[t - 2];
			int sigma0 = Rotr(w15, 7) ^ Rotr(w15, 18) ^ Shr(w15, 3);
			int sigma1 = Rotr(w2, 17) ^ Rotr(w2, 19) ^ Shr(w2, 10);
			schedule[t] = schedule[t - 16] + sigma0 + schedule[t - 7] + sigma1;
		}

		int a = state[0];
		int b = state[1];
		int c = state[2];
		int d = state[3];
		int e = state[4];
		int f = state[5];
		int g = state[6];
		int h = state[7];
		for (int r = 0; r < 64; r++)
		{
			int bigSigma1 = Rotr(e, 6) ^ Rotr(e, 11) ^ Rotr(e, 25);
			int choose = (e & f) ^ (~e & g);
			int temp1 = h + bigSigma1 + choose + s_aK[r] + schedule[r];
			int bigSigma0 = Rotr(a, 2) ^ Rotr(a, 13) ^ Rotr(a, 22);
			int majority = (a & b) ^ (a & c) ^ (b & c);
			int temp2 = bigSigma0 + majority;
			h = g;
			g = f;
			f = e;
			e = d + temp1;
			d = c;
			c = b;
			b = a;
			a = temp1 + temp2;
		}

		state[0] = state[0] + a;
		state[1] = state[1] + b;
		state[2] = state[2] + c;
		state[3] = state[3] + d;
		state[4] = state[4] + e;
		state[5] = state[5] + f;
		state[6] = state[6] + g;
		state[7] = state[7] + h;
	}

	//------------------------------------------------------------------------------------------------
	//! Logical right shift by 1..31 (the >> of an int is arithmetic: the sign bits are masked off)
	protected static int Shr(int value, int count)
	{
		return (value >> count) & (0x7FFFFFFF >> (count - 1));
	}

	//------------------------------------------------------------------------------------------------
	//! Rotate right by 1..31
	protected static int Rotr(int value, int count)
	{
		return Shr(value, count) | (value << (32 - count));
	}

	//------------------------------------------------------------------------------------------------
	//! Four bytes, big-endian
	protected static void AppendWord(notnull array<int> bytes, int word)
	{
		bytes.Insert((word >> 24) & 0xFF);
		bytes.Insert((word >> 16) & 0xFF);
		bytes.Insert((word >> 8) & 0xFF);
		bytes.Insert(word & 0xFF);
	}

	//------------------------------------------------------------------------------------------------
	//! Eight lower-case hex digits, most significant first
	protected static string WordHex(int word)
	{
		string text;
		for (int shift = 28; shift >= 0; shift -= 4)
		{
			text += HEX_DIGITS.Substring((word >> shift) & 15, 1);
		}

		return text;
	}
}

//------------------------------------------------------------------------------------------------
//! Whole-file reads of the text inputs of llc_v1 and the expert (vehicle card, route, their own sources)
class DC_TextFile
{
	//------------------------------------------------------------------------------------------------
	//! The first of the comma-separated roots (each ending in '/' or ':') under which fileName exists, joined; empty
	//! when there is none
	static string Resolve(string roots, string fileName)
	{
		array<string> parts = {};
		roots.Split(",", parts, true);
		foreach (string root : parts)
		{
			string candidate = root.Trim() + fileName;
			if (FileIO.FileExists(candidate))
				return candidate;
		}

		return string.Empty;
	}

	//------------------------------------------------------------------------------------------------
	//! The file's bytes as one string: FileHandle.Read of GetLength() bytes, else (a short read) line by line with
	//! ReadLine joined by LF. false (failure) when the file is missing or cannot be opened, when neither read gives
	//! exactly GetLength() bytes, or when a character is not 7-bit ASCII (the inputs are ASCII with LF line ends)
	static bool Read(string path, out string text, out string failure)
	{
		text = string.Empty;
		if (path.IsEmpty() || !FileIO.FileExists(path))
		{
			failure = "missing file " + path;
			return false;
		}

		FileHandle file = FileIO.OpenFile(path, FileMode.READ);
		if (!file)
		{
			failure = "FileIO.OpenFile(" + path + ", READ) returned null";
			return false;
		}

		int length = file.GetLength();
		file.Read(text, length);
		file.Close();
		if (text.Length() != length)
			ReadLines(path, length, text);

		if (text.Length() != length)
		{
			failure = string.Format("%1: read %2 of %3 bytes", path, text.Length(), length);
			return false;
		}

		for (int i = 0; i < length; i++)
		{
			int code = text.ToAscii(i);
			if (code < 1 || code > 127)
			{
				failure = string.Format("%1: byte %2 is not 7-bit ASCII", path, i);
				return false;
			}
		}

		failure = string.Empty;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Fallback: the lines of the file joined by LF, with a final LF when the joined text is one byte short
	protected static void ReadLines(string path, int length, out string text)
	{
		text = string.Empty;
		FileHandle file = FileIO.OpenFile(path, FileMode.READ);
		if (!file)
			return;

		string line;
		bool first = true;
		while (file.ReadLine(line) >= 0)
		{
			if (!first)
				text += "\n";

			text += line;
			first = false;
		}

		file.Close();
		if (text.Length() == length - 1)
			text += "\n";
	}
}

//------------------------------------------------------------------------------------------------
//! Number arrays of a JSON text, found by key and read with string.ToFloat (see the header)
class DC_JsonScan
{
	static const int CHAR_TAB = 9;
	static const int CHAR_LF = 10;
	static const int CHAR_CR = 13;
	static const int CHAR_SPACE = 32;
	static const int CHAR_QUOTE = 34;
	static const int CHAR_COMMA = 44;
	static const int CHAR_COLON = 58;
	static const int CHAR_OPEN = 91;		//!< '['
	static const int CHAR_CLOSE = 93;		//!< ']'

	//------------------------------------------------------------------------------------------------
	//! Index of the first character after `"key":` (and any spaces); -1 when the key is missing or occurs twice
	static int FindValue(string text, string key)
	{
		string quoted = "\"" + key + "\"";
		int at = text.IndexOf(quoted);
		if (at < 0 || text.IndexOfFrom(at + 1, quoted) >= 0)
			return -1;

		int index = SkipSpace(text, at + quoted.Length());
		if (index >= text.Length() || text.ToAscii(index) != CHAR_COLON)
			return -1;

		return SkipSpace(text, index + 1);
	}

	//------------------------------------------------------------------------------------------------
	//! The first index at or after index that is not a space, tab, CR or LF
	static int SkipSpace(string text, int index)
	{
		int length = text.Length();
		while (index < length)
		{
			int code = text.ToAscii(index);
			if (code != CHAR_SPACE && code != CHAR_TAB && code != CHAR_LF && code != CHAR_CR)
				return index;

			index++;
		}

		return length;
	}

	//------------------------------------------------------------------------------------------------
	//! true when the character at index (after spaces) is code; next = the index after it
	static bool Expect(string text, int index, int code, out int next)
	{
		next = SkipSpace(text, index);
		if (next >= text.Length() || text.ToAscii(next) != code)
			return false;

		next++;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! The number starting at index (after spaces): string.ToFloat with its parsed count; false when there is none
	static bool ReadNumber(string text, int index, out float value, out int next)
	{
		int parsed;
		value = text.ToFloat(0, index, parsed);
		next = index + parsed;
		return parsed > 0;
	}

	//------------------------------------------------------------------------------------------------
	//! "key": [n, n, ...] as floats (an empty array gives none). false (failure) when the key is missing or repeated,
	//! or the value is not a flat array of numbers
	static bool ReadNumbers(string text, string key, notnull array<float> values, out string failure)
	{
		values.Clear();
		int index = FindValue(text, key);
		if (index < 0)
		{
			failure = "key \"" + key + "\" is missing or repeated";
			return false;
		}

		if (!Expect(text, index, CHAR_OPEN, index))
		{
			failure = "\"" + key + "\" is not an array";
			return false;
		}

		int after;
		if (Expect(text, index, CHAR_CLOSE, after))
		{
			failure = string.Empty;
			return true;
		}

		while (true)
		{
			float value;
			if (!ReadNumber(text, index, value, index))
			{
				failure = string.Format("\"%1\" item %2 is not a number", key, values.Count());
				return false;
			}

			values.Insert(value);
			if (Expect(text, index, CHAR_CLOSE, index))
				break;

			if (!Expect(text, index, CHAR_COMMA, index))
			{
				failure = string.Format("\"%1\" item %2 is not followed by ',' or ']'", key, values.Count() - 1);
				return false;
			}
		}

		failure = string.Empty;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! "key": ["text", ...] of plain strings (no escape sequences; an empty array gives none). false (failure) when
	//! the key is missing or repeated, or the value is not such an array
	static bool ReadStrings(string text, string key, notnull array<string> values, out string failure)
	{
		values.Clear();
		int index = FindValue(text, key);
		if (index < 0 || !Expect(text, index, CHAR_OPEN, index))
		{
			failure = "\"" + key + "\" is missing, repeated or not an array";
			return false;
		}

		int after;
		if (Expect(text, index, CHAR_CLOSE, after))
		{
			failure = string.Empty;
			return true;
		}

		while (true)
		{
			int close = -1;
			if (Expect(text, index, CHAR_QUOTE, index))
				close = text.IndexOfFrom(index, "\"");

			string item;
			if (close >= 0)
				item = text.Substring(index, close - index);

			if (close < 0 || item.Contains("\\"))
			{
				failure = string.Format("\"%1\" item %2 is not a plain string", key, values.Count());
				return false;
			}

			values.Insert(item);
			index = close + 1;
			if (Expect(text, index, CHAR_CLOSE, index))
				break;

			if (!Expect(text, index, CHAR_COMMA, index))
			{
				failure = string.Format("\"%1\" item %2 is not followed by ',' or ']'", key, values.Count() - 1);
				return false;
			}
		}

		failure = string.Empty;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! true when "key" occurs once and its value is the literal null
	static bool IsNull(string text, string key)
	{
		int index = FindValue(text, key);
		return index >= 0 && index + 4 <= text.Length() && text.Substring(index, 4) == "null";
	}

	//------------------------------------------------------------------------------------------------
	//! A JSON number of an open JsonLoadContext object as float: written as a decimal or as an integer (the
	//! DC_Session.ReadNumber rule)
	static bool ReadContextNumber(notnull JsonLoadContext context, string key, out float value)
	{
		if (!context.DoesKeyExist(key))
			return false;

		if (context.ReadValue(key, value))
			return true;

		int integer;
		if (!context.ReadValue(key, integer))
			return false;

		value = integer;
		return true;
	}
}

//------------------------------------------------------------------------------------------------
//! A vehicle card (drivedit.vehicle_card/1, written by tools/ident/fit_steer_kappa.py). Filled by Load, read-only
//! afterwards. Only the values llc_v1 uses are read; the identified actuator structure stays in the file
class DC_VehicleCard
{
	static const string SCHEMA = "drivedit.vehicle_card/1";
	static const string STATUS_PLACEHOLDER = "placeholder";
	static const string STATUS_IDENTIFIED = "identified";
	static const string GEARBOX_AUTOMATIC = "automatic";
	static const string GEARBOX_SCRIPTED = "scripted";

	string m_sPath;
	string m_sSha256;					//!< sha256 of the file bytes (control.vehicle_card_sha)
	string m_sCardId;
	string m_sVehicle;
	string m_sStatus;
	string m_sPinId;
	string m_sBuildKey;					//!< empty = null (placeholder)
	ref array<string> m_aSessions = {};	//!< identification session ids (empty on a placeholder)
	int m_iSteerSign;					//!< contract steer = sign x engine steer (TBD-5)
	int m_iYawAxisSign;					//!< contract yaw rate = sign x angvel_local_y (TBD-5)
	bool m_bSignsMeasured;
	string m_sGearbox;					//!< automatic or scripted (E-04 / E-01)
	float m_fEngineSteerRateLimit;		//!< identified engine steering rate limit [1/s]; <= 0 = none (or unknown on a placeholder)
	bool m_bInputEchoKnown;				//!< actuator.raw_engine_path.input_echo is true or false (null on a placeholder)
	bool m_bInputEcho;					//!< the engine passes the steering input through unchanged (E-01)
	ref array<float> m_aForwardGears = {};	//!< gear_indices_forward (E-01; empty on a placeholder)

	// kappa_ss: steady-state curvature per contract steer by speed (E-02a)
	ref array<float> m_aSpeedsMps = {};
	ref array<float> m_aKappaPerSteer = {};
	ref array<float> m_aKappaOffset = {};

	// scripted gearbox (forward gears only, TBD-7)
	ref array<float> m_aScriptedGears = {};
	ref array<float> m_aUpshiftMps = {};
	ref array<float> m_aDownshiftMps = {};

	// llc_v1 parameters
	float m_fSteerMax;					//!< |contract steer| limit
	float m_fSteerRatePerS;				//!< llc rate limiter [1/s], 0 = identity
	float m_fSteerLagTauS;				//!< llc first-order lag [s], 0 = identity
	float m_fKappaTrimGain;				//!< curvature trim [steer / (1/m * s)], 0 = off
	float m_fKappaTrimMax;
	float m_fTrimMinSpeedMps;
	float m_fSpeedKp;					//!< [1/s]
	float m_fSpeedKi;					//!< [1/s2]
	float m_fSpeedWindowMps;
	float m_fIntegralMaxMps2;
	float m_fThrottlePerMps2;
	float m_fBrakePerMps2;
	float m_fThrottleMax;
	float m_fCoastBandMps2;
	float m_fBrakeHold;
	float m_fStopSpeedMps;

	//------------------------------------------------------------------------------------------------
	//! Reads and checks the card. false (failure) on a missing file, a parse problem or a value outside its range
	bool Load(string path, out string failure)
	{
		string text;
		if (!DC_TextFile.Read(path, text, failure))
			return false;

		m_sPath = path;
		m_sSha256 = DC_Sha256.HashText(text);
		JsonLoadContext context = new JsonLoadContext(false);
		if (!context.LoadFromString(text))
		{
			failure = path + " does not parse as JSON";
			return false;
		}

		// top-level keys first, then the objects one after another (the DC_LaunchGuard reading order)
		string schema;
		if (!context.ReadValue("schema", schema) || schema != SCHEMA)
		{
			failure = path + ": schema is '" + schema + "', expected " + SCHEMA;
			return false;
		}

		if (!context.ReadValue("card_id", m_sCardId) || !context.ReadValue("vehicle", m_sVehicle) || !context.ReadValue("status", m_sStatus) || !context.ReadValue("pin_id", m_sPinId))
		{
			failure = path + ": card_id, vehicle, status and pin_id are required strings";
			return false;
		}

		// build_key null (a placeholder) and the session ids by the scanner: how JsonLoadContext reads a JSON null
		// into a string, or an array of strings, is not verified on 1.8
		string buildKey;
		if (!DC_JsonScan.IsNull(text, "build_key") && context.DoesKeyExist("build_key") && context.ReadValue("build_key", buildKey))
			m_sBuildKey = buildKey.Trim();

		if (!DC_JsonScan.ReadStrings(text, "identification_sessions", m_aSessions, failure))
		{
			failure = path + ": " + failure;
			return false;
		}

		if (!ReadArrays(text, failure) || !ReadObjects(context, failure))
		{
			failure = path + ": " + failure;
			return false;
		}

		if (!Check(failure))
		{
			failure = path + ": " + failure;
			return false;
		}

		failure = string.Empty;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	protected bool ReadArrays(string text, out string failure)
	{
		if (!DC_JsonScan.ReadNumbers(text, "speeds_mps", m_aSpeedsMps, failure))
			return false;

		if (!DC_JsonScan.ReadNumbers(text, "kappa_per_steer", m_aKappaPerSteer, failure))
			return false;

		if (!DC_JsonScan.ReadNumbers(text, "kappa_offset", m_aKappaOffset, failure))
			return false;

		if (!DC_JsonScan.ReadNumbers(text, "scripted_gears", m_aScriptedGears, failure))
			return false;

		if (!DC_JsonScan.ReadNumbers(text, "upshift_mps", m_aUpshiftMps, failure))
			return false;

		if (!DC_JsonScan.ReadNumbers(text, "downshift_mps", m_aDownshiftMps, failure))
			return false;

		m_bInputEchoKnown = !DC_JsonScan.IsNull(text, "input_echo");
		return DC_JsonScan.ReadNumbers(text, "gear_indices_forward", m_aForwardGears, failure);
	}

	//------------------------------------------------------------------------------------------------
	protected bool ReadObjects(notnull JsonLoadContext context, out string failure)
	{
		if (!context.StartObject("signs"))
		{
			failure = "no signs object";
			return false;
		}

		bool signsOk = context.ReadValue("steer_sign", m_iSteerSign) && context.ReadValue("yaw_axis_sign", m_iYawAxisSign) && context.ReadValue("measured", m_bSignsMeasured);
		context.EndObject();
		if (!signsOk)
		{
			failure = "signs needs steer_sign, yaw_axis_sign (int) and measured (bool)";
			return false;
		}

		if (!context.StartObject("gearbox"))
		{
			failure = "no gearbox object";
			return false;
		}

		bool gearboxOk = context.ReadValue("mode", m_sGearbox);
		context.EndObject();
		if (!gearboxOk)
		{
			failure = "gearbox.mode is required";
			return false;
		}

		// actuator.raw_engine_path: steer_rate_limit_per_s a number, or null when none was found (or unknown);
		// input_echo true, false, or null when unknown
		m_fEngineSteerRateLimit = 0;
		if (context.StartObject("actuator"))
		{
			if (context.StartObject("raw_engine_path"))
			{
				float limit;
				if (DC_JsonScan.ReadContextNumber(context, "steer_rate_limit_per_s", limit))
					m_fEngineSteerRateLimit = limit;

				if (m_bInputEchoKnown && !context.ReadValue("input_echo", m_bInputEcho))
					m_bInputEchoKnown = false;

				context.EndObject();
			}

			context.EndObject();
		}

		if (!context.StartObject("llc_v1"))
		{
			failure = "no llc_v1 object";
			return false;
		}

		bool llcOk = ReadLlcNumbers(context);
		context.EndObject();
		if (!llcOk)
		{
			failure = "llc_v1 misses a parameter (steer_max .. stop_speed_mps)";
			return false;
		}

		failure = string.Empty;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	protected bool ReadLlcNumbers(notnull JsonLoadContext context)
	{
		if (!DC_JsonScan.ReadContextNumber(context, "steer_max", m_fSteerMax) || !DC_JsonScan.ReadContextNumber(context, "steer_rate_per_s", m_fSteerRatePerS) || !DC_JsonScan.ReadContextNumber(context, "steer_lag_tau_s", m_fSteerLagTauS))
			return false;

		if (!DC_JsonScan.ReadContextNumber(context, "kappa_trim_gain", m_fKappaTrimGain) || !DC_JsonScan.ReadContextNumber(context, "kappa_trim_max", m_fKappaTrimMax) || !DC_JsonScan.ReadContextNumber(context, "trim_min_speed_mps", m_fTrimMinSpeedMps))
			return false;

		if (!DC_JsonScan.ReadContextNumber(context, "speed_kp", m_fSpeedKp) || !DC_JsonScan.ReadContextNumber(context, "speed_ki", m_fSpeedKi) || !DC_JsonScan.ReadContextNumber(context, "speed_window_mps", m_fSpeedWindowMps) || !DC_JsonScan.ReadContextNumber(context, "integral_max_mps2", m_fIntegralMaxMps2))
			return false;

		if (!DC_JsonScan.ReadContextNumber(context, "throttle_per_mps2", m_fThrottlePerMps2) || !DC_JsonScan.ReadContextNumber(context, "brake_per_mps2", m_fBrakePerMps2) || !DC_JsonScan.ReadContextNumber(context, "throttle_max", m_fThrottleMax))
			return false;

		return DC_JsonScan.ReadContextNumber(context, "coast_band_mps2", m_fCoastBandMps2) && DC_JsonScan.ReadContextNumber(context, "brake_hold", m_fBrakeHold) && DC_JsonScan.ReadContextNumber(context, "stop_speed_mps", m_fStopSpeedMps);
	}

	//------------------------------------------------------------------------------------------------
	//! Value ranges and the consistency of status, sessions and build key
	protected bool Check(out string failure)
	{
		if (m_sStatus != STATUS_PLACEHOLDER && m_sStatus != STATUS_IDENTIFIED)
		{
			failure = "status '" + m_sStatus + "' is neither placeholder nor identified";
			return false;
		}

		if (m_sStatus == STATUS_PLACEHOLDER && !m_aSessions.IsEmpty())
		{
			failure = "a placeholder card names no identification session";
			return false;
		}

		if (m_sStatus == STATUS_IDENTIFIED && (m_aSessions.IsEmpty() || m_sBuildKey.IsEmpty()))
		{
			failure = "an identified card needs its identification session ids and a build_key";
			return false;
		}

		if (Math.AbsInt(m_iSteerSign) != 1 || Math.AbsInt(m_iYawAxisSign) != 1)
		{
			failure = string.Format("signs must be -1 or +1 (steer_sign %1, yaw_axis_sign %2)", m_iSteerSign, m_iYawAxisSign);
			return false;
		}

		if (!CheckMap(failure) || !CheckGearbox(failure))
			return false;

		bool steerOk = m_fSteerMax > 0 && m_fSteerMax <= 1 && m_fSteerRatePerS >= 0 && m_fSteerLagTauS >= 0 && m_fKappaTrimGain >= 0 && m_fKappaTrimMax >= 0 && m_fTrimMinSpeedMps >= 0;
		bool speedOk = m_fSpeedKp >= 0 && m_fSpeedKi >= 0 && m_fSpeedWindowMps > 0 && m_fIntegralMaxMps2 >= 0 && m_fThrottlePerMps2 > 0 && m_fBrakePerMps2 > 0;
		bool rangeOk = m_fThrottleMax > 0 && m_fThrottleMax <= 1 && m_fCoastBandMps2 >= 0 && m_fBrakeHold >= 0 && m_fBrakeHold <= 1 && m_fStopSpeedMps >= 0;
		if (!steerOk || !speedOk || !rangeOk)
		{
			failure = "an llc_v1 parameter is outside its range (steer_max in (0, 1], throttle_max in (0, 1], brake_hold in [0, 1], the rest >= 0, throttle and brake gains > 0)";
			return false;
		}

		failure = string.Empty;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	protected bool CheckMap(out string failure)
	{
		int count = m_aSpeedsMps.Count();
		if (count < 1 || m_aKappaPerSteer.Count() != count || m_aKappaOffset.Count() != count)
		{
			failure = "kappa_ss needs speeds_mps, kappa_per_steer and kappa_offset of one length >= 1";
			return false;
		}

		for (int i = 0; i < count; i++)
		{
			if (m_aSpeedsMps[i] < 0 || (i > 0 && m_aSpeedsMps[i] <= m_aSpeedsMps[i - 1]))
			{
				failure = "kappa_ss.speeds_mps must be >= 0 and strictly increasing";
				return false;
			}

			if (m_aKappaPerSteer[i] <= 0)
			{
				failure = "kappa_ss.kappa_per_steer must be > 0 (contract sign; the engine sign is signs.steer_sign)";
				return false;
			}
		}

		failure = string.Empty;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	protected bool CheckGearbox(out string failure)
	{
		if (m_sGearbox == GEARBOX_AUTOMATIC)
		{
			failure = string.Empty;
			return true;
		}

		if (m_sGearbox != GEARBOX_SCRIPTED)
		{
			failure = "gearbox.mode '" + m_sGearbox + "' is neither automatic nor scripted";
			return false;
		}

		int count = m_aScriptedGears.Count();
		if (count < 1 || m_aUpshiftMps.Count() != count - 1 || m_aDownshiftMps.Count() != count - 1)
		{
			failure = "a scripted gearbox needs scripted_gears (forward gears, n >= 1) and n - 1 upshift_mps and downshift_mps";
			return false;
		}

		for (int i = 0; i < count - 1; i++)
		{
			if (m_aDownshiftMps[i] >= m_aUpshiftMps[i] || (i > 0 && m_aUpshiftMps[i] <= m_aUpshiftMps[i - 1]))
			{
				failure = "scripted gearbox: downshift_mps[i] < upshift_mps[i], upshift_mps increasing";
				return false;
			}
		}

		// forward-only (TBD-7): every scripted gear is a measured forward gear index, never neutral or reverse
		foreach (float gear : m_aScriptedGears)
		{
			if (gear < 0 || gear != Math.Round(gear) || m_aForwardGears.Find(gear) < 0)
			{
				failure = string.Format("scripted gear %1 is not one of gear_indices_forward (the forward gears E-01 measured)", gear);
				return false;
			}
		}

		failure = string.Empty;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	bool IsPlaceholder()
	{
		return m_sStatus == STATUS_PLACEHOLDER;
	}

	//------------------------------------------------------------------------------------------------
	//! "id1,id2" of the identification sessions, "none" on a placeholder
	string GetSessionsText()
	{
		if (m_aSessions.IsEmpty())
			return "none";

		string text;
		foreach (int i, string session : m_aSessions)
		{
			if (i > 0)
				text += ",";

			text += session;
		}

		return text;
	}
}

//------------------------------------------------------------------------------------------------
//! llc_v1 of one ego (see the header). Init once per session, Reset at every episode start, Step every physics step
//! with a held command
class DC_LLCv1
{
	static const string VERSION = "llc_v1";
	static const string SOURCE_FILE = "DC_LLCv1.c";

	//! use_class values at which a placeholder card is accepted (qualification); every other class refuses it
	protected static ref array<string> s_aQualificationUseClasses = {"dev", "probe"};

	protected DC_VehicleCard m_Card;
	protected DC_Actuator m_Actuator;
	protected VehicleWheeledSimulation m_Simulation;
	protected float m_fTimeSlice;
	protected float m_fLagAlpha;			//!< 1 - exp(-dt / tau); 1 = identity

	// steering state (engine-native sign)
	protected float m_fSteerLimited;
	protected float m_fSteerOut;
	protected float m_fKappaTrim;			//!< contract steer

	// longitudinal state
	protected float m_fSpeedRef;
	protected float m_fIntegral;			//!< [m/s] (integral of the speed error)
	protected int m_iGearSlot;

	//------------------------------------------------------------------------------------------------
	//! The r3 card rules (see the header) for this session. false (failure) refuses the card; warning is a line to
	//! log when the card is accepted with a reservation
	static bool AcceptCard(notnull DC_VehicleCard card, string sessionPinId, string sessionBuildKey, string useClass, out string failure, out string warning)
	{
		warning = string.Empty;
		if (card.IsPlaceholder() && s_aQualificationUseClasses.Find(useClass) < 0)
		{
			failure = "card " + card.m_sCardId + " is a PLACEHOLDER (no identification session): accepted only for qualification (use_class dev or probe), this session is use_class '" + useClass + "'";
			return false;
		}

		if (!sessionBuildKey.IsEmpty())
		{
			if (card.m_sBuildKey.IsEmpty() && !card.IsPlaceholder())
			{
				failure = "card " + card.m_sCardId + " has no build_key";
				return false;
			}

			if (!card.m_sBuildKey.IsEmpty() && card.m_sBuildKey != sessionBuildKey)
			{
				failure = "card " + card.m_sCardId + " has build_key " + card.m_sBuildKey + ", the session " + sessionBuildKey;
				return false;
			}
		}
		else
		{
			if (card.m_sPinId != sessionPinId)
			{
				failure = "card " + card.m_sCardId + " is for pin " + card.m_sPinId + ", the session runs pin " + sessionPinId + " (dd-launch/1: no build_key to compare)";
				return false;
			}

			warning = "the session has no build_key (dd-launch/1 record): card matched by pin_id " + sessionPinId + " only";
		}

		if (card.m_sBuildKey.IsEmpty() && warning.IsEmpty())
			warning = "the placeholder card is not bound to a build (build_key null)";

		// limiter rule: llc's own limiter is the identity or strictly slower than the engine's (CC-08, M3)
		if (card.m_fEngineSteerRateLimit > 0 && card.m_fSteerRatePerS > 0 && card.m_fSteerRatePerS >= card.m_fEngineSteerRateLimit)
		{
			failure = string.Format("llc steer_rate_per_s %1 is not slower than the identified engine limit %2 /s: the two limiters would stack", card.m_fSteerRatePerS, card.m_fEngineSteerRateLimit);
			return false;
		}

		// an engine path that shapes the input without an identified limit leaves only the identity
		if (!card.IsPlaceholder() && card.m_bInputEchoKnown && !card.m_bInputEcho && card.m_fEngineSteerRateLimit <= 0 && card.m_fSteerRatePerS > 0)
		{
			failure = "the engine shapes the steering input (input_echo false) but no limit was identified: llc steer_rate_per_s must be 0 (the identity)";
			return false;
		}

		failure = string.Empty;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! use_class of this session's launch record, read as DC_LaunchGuard reads it (the driver context does not carry
	//! it): the -dcLaunchId record, else (Workbench escape) the newest record of this exe, which must lie in this
	//! Workbench start's window (the guard's G15: a record of another start could carry another class than the one
	//! session.json is stamped with). Empty (failure) when no record can be read, it names another pin or build_key
	//! than the session, or the escape record is outside the window. In the escape this is the guard's own rule, not
	//! the identity of its record: that needs DC_DriverContext.GetUseClass() (an interface addition for the T0-MIN
	//! integration)
	static string ReadUseClass(string sessionPinId, string sessionBuildKey, out string failure)
	{
		string launchId;
		string path;
		bool cli = System.GetCLIParam(DC_LaunchGuard.CLI_LAUNCH_ID, launchId) && DC_LaunchGuard.IsValidLaunchId(launchId.Trim());
		if (cli)
			path = DC_LaunchGuard.LAUNCH_DIR + "/launch_" + launchId.Trim() + ".json";
		else
			path = DC_LaunchGuard.FindNewestLaunchRecord(DC_LaunchGuard.GetExpectedExe());

		DC_LaunchInfo info = new DC_LaunchInfo();
		if (path.IsEmpty() || !FileIO.FileExists(path) || !DC_LaunchGuard.ReadLaunchRecord(path, info))
		{
			failure = "launch record '" + path + "' cannot be read";
			return string.Empty;
		}

		if (info.m_sPinId != sessionPinId || info.m_sBuildKey != sessionBuildKey)
		{
			failure = "launch record " + path + " names pin " + info.m_sPinId + " build_key '" + info.m_sBuildKey + "', the session " + sessionPinId + " '" + sessionBuildKey + "'";
			return string.Empty;
		}

		// the escape record must lie in the guard's G15 window of this Workbench start (at most LAUNCH_LEAD_MAX_S before
		// it, not after it), as DC_LaunchGuard.EvaluateEscape requires of the record it accepted
		int recordUnix = DC_LaunchGuard.UtcTextToUnix(info.m_sRecordUtc);
		int processStartUnix = System.GetUnixTime() - System.GetTickCount() / 1000;
		int leadS = processStartUnix - recordUnix;
		if (!cli && (recordUnix < 0 || leadS > DC_LaunchGuard.LAUNCH_LEAD_MAX_S || leadS < -DC_LaunchGuard.LAUNCH_CLOCK_SLACK_S))
		{
			failure = "the newest launch record " + path + " (utc '" + info.m_sRecordUtc + "') is not in this Workbench start's window: it is not this session's";
			return string.Empty;
		}

		failure = string.Empty;
		return info.m_sUseClass;
	}

	//------------------------------------------------------------------------------------------------
	//! Binds llc_v1 to the ego's actuator with an accepted card. false (failure) without a simulation or a fixed time
	//! slice
	bool Init(notnull DC_VehicleCard card, notnull DC_Actuator actuator, float timeSlice, out string failure)
	{
		m_Card = card;
		m_Actuator = actuator;
		m_Simulation = actuator.GetSimulation();
		m_fTimeSlice = timeSlice;
		if (!m_Simulation)
		{
			failure = "the actuator has no VehicleWheeledSimulation";
			return false;
		}

		if (m_fTimeSlice <= 0)
		{
			failure = "no fixed time slice";
			return false;
		}

		m_fLagAlpha = 1;
		if (card.m_fSteerLagTauS > 0)
			m_fLagAlpha = 1 - Math.Pow(Math.E, -m_fTimeSlice / card.m_fSteerLagTauS);

		Reset();
		failure = string.Empty;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Episode start: steering from centre (the core parked the car with steering 0), speed reference at the speed,
	//! no integral or trim, the gear slot from the speed
	void Reset()
	{
		m_fSteerLimited = 0;
		m_fSteerOut = 0;
		m_fKappaTrim = 0;
		m_fIntegral = 0;
		m_fSpeedRef = 0;
		m_iGearSlot = 0;
		if (!m_Simulation)
			return;

		m_fSpeedRef = GetSpeed();
		while (m_Card && m_iGearSlot < m_Card.m_aUpshiftMps.Count() && m_fSpeedRef >= m_Card.m_aUpshiftMps[m_iGearSlot])
		{
			m_iGearSlot++;
		}
	}

	//------------------------------------------------------------------------------------------------
	//! One physics step: the inputs for the held command, passed through the actuator and recorded in llc
	void Step(notnull DC_HLCommand held, notnull DC_LLCOutput llc)
	{
		if (!m_Card || !m_Actuator || !m_Simulation)
			return;

		float speed = GetSpeed();
		vector angularVelocity = m_Simulation.GetAngularVelocityLocal();
		float yawRate = m_Card.m_iYawAxisSign * angularVelocity[1];

		// steering: inverse map, trim, sign, rate limiter, lag
		float steerContract = SteerOfKappa(held.m_fKappa, speed);
		if (m_Card.m_fKappaTrimGain > 0 && speed >= m_Card.m_fTrimMinSpeedMps && speed > 0)
		{
			float kappaError = held.m_fKappa - yawRate / speed;
			m_fKappaTrim = Math.Clamp(m_fKappaTrim + m_Card.m_fKappaTrimGain * kappaError * m_fTimeSlice, -m_Card.m_fKappaTrimMax, m_Card.m_fKappaTrimMax);
		}

		steerContract = Math.Clamp(steerContract + m_fKappaTrim, -m_Card.m_fSteerMax, m_Card.m_fSteerMax);
		float steerRaw = m_Card.m_iSteerSign * steerContract;
		if (m_Card.m_fSteerRatePerS > 0)
		{
			float maxDelta = m_Card.m_fSteerRatePerS * m_fTimeSlice;
			m_fSteerLimited += Math.Clamp(steerRaw - m_fSteerLimited, -maxDelta, maxDelta);
		}
		else
		{
			m_fSteerLimited = steerRaw;
		}

		m_fSteerOut = Math.Clamp(m_fSteerOut + (m_fSteerLimited - m_fSteerOut) * m_fLagAlpha, -1, 1);

		// longitudinal: throttle or brake, never both
		float throttle;
		float brake;
		if (held.m_iDir > 0)
			Longitudinal(held.m_fAccel, speed, throttle, brake);
		else
			Stop(held.m_fAccel, speed, throttle, brake);

		// gearbox: automatic = no SetGear; scripted = forward gears by speed
		int gear = -1;
		float clutchRecorded = -1;
		if (m_Card.m_sGearbox == DC_VehicleCard.GEARBOX_SCRIPTED)
		{
			gear = ScriptedGear(speed);
			clutchRecorded = 1;
		}

		m_Actuator.Apply(m_fSteerOut, throttle, brake, false, 1, gear);

		llc.m_bActive = true;
		llc.m_fSteerRaw = steerRaw;
		llc.m_fSteerLim = m_fSteerOut;
		llc.m_fThrottle = throttle;
		llc.m_fBrake = brake;
		llc.m_bHandbrake = false;
		llc.m_iGear = gear;
		llc.m_fClutch = clutchRecorded;
	}

	//------------------------------------------------------------------------------------------------
	//! dir +1: speed reference from the accel target, PI on its error plus the accel feedforward
	protected void Longitudinal(float accel, float speed, out float throttle, out float brake)
	{
		throttle = 0;
		brake = 0;
		m_fSpeedRef = Math.Clamp(m_fSpeedRef + accel * m_fTimeSlice, Math.Max(0, speed - m_Card.m_fSpeedWindowMps), speed + m_Card.m_fSpeedWindowMps);
		float error = m_fSpeedRef - speed;
		float effort = accel + m_Card.m_fSpeedKp * error + m_Card.m_fSpeedKi * m_fIntegral;
		bool saturatedUp = effort * m_Card.m_fThrottlePerMps2 >= m_Card.m_fThrottleMax && error > 0;
		bool saturatedDown = -effort * m_Card.m_fBrakePerMps2 >= 1 && error < 0;
		if (!saturatedUp && !saturatedDown)
			m_fIntegral += error * m_fTimeSlice;

		if (m_Card.m_fSpeedKi > 0)
			m_fIntegral = Math.Clamp(m_fIntegral, -m_Card.m_fIntegralMaxMps2 / m_Card.m_fSpeedKi, m_Card.m_fIntegralMaxMps2 / m_Card.m_fSpeedKi);

		effort = accel + m_Card.m_fSpeedKp * error + m_Card.m_fSpeedKi * m_fIntegral;
		if (accel <= 0 && m_fSpeedRef <= 0 && speed < m_Card.m_fStopSpeedMps)
		{
			brake = m_Card.m_fBrakeHold;
			m_fIntegral = 0;
			return;
		}

		if (effort > m_Card.m_fCoastBandMps2)
			throttle = Math.Clamp(effort * m_Card.m_fThrottlePerMps2, 0, m_Card.m_fThrottleMax);
		else if (effort < -m_Card.m_fCoastBandMps2)
			brake = Math.Clamp(-effort * m_Card.m_fBrakePerMps2, 0, 1);
	}

	//------------------------------------------------------------------------------------------------
	//! dir 0 (and the never-commanded -1): no throttle; brake from the deceleration target, brake_hold when slow
	protected void Stop(float accel, float speed, out float throttle, out float brake)
	{
		throttle = 0;
		brake = Math.Clamp(-accel * m_Card.m_fBrakePerMps2, 0, 1);
		if (speed < m_Card.m_fStopSpeedMps)
			brake = Math.Max(brake, m_Card.m_fBrakeHold);

		m_fSpeedRef = speed;
		m_fIntegral = 0;
	}

	//------------------------------------------------------------------------------------------------
	//! Forward gear of a scripted gearbox by speed, with the card's up / down shift hysteresis
	protected int ScriptedGear(float speed)
	{
		int last = m_Card.m_aScriptedGears.Count() - 1;
		if (m_iGearSlot < last && speed >= m_Card.m_aUpshiftMps[m_iGearSlot])
			m_iGearSlot++;
		else if (m_iGearSlot > 0 && speed < m_Card.m_aDownshiftMps[m_iGearSlot - 1])
			m_iGearSlot--;

		int gear = Math.Round(m_Card.m_aScriptedGears[m_iGearSlot]);
		return gear;
	}

	//------------------------------------------------------------------------------------------------
	//! Contract steer for a curvature at a speed: the inverse of kappa_ss (linear in speed between the card's bins,
	//! clamped at the end bins)
	float SteerOfKappa(float kappa, float speed)
	{
		float gain;
		float offset;
		MapAt(speed, gain, offset);
		return (kappa - offset) / gain;
	}

	//------------------------------------------------------------------------------------------------
	//! kappa_per_steer and kappa_offset at a speed
	protected void MapAt(float speed, out float gain, out float offset)
	{
		array<float> speeds = m_Card.m_aSpeedsMps;
		int count = speeds.Count();
		if (speed <= speeds[0] || count == 1)
		{
			gain = m_Card.m_aKappaPerSteer[0];
			offset = m_Card.m_aKappaOffset[0];
			return;
		}

		if (speed >= speeds[count - 1])
		{
			gain = m_Card.m_aKappaPerSteer[count - 1];
			offset = m_Card.m_aKappaOffset[count - 1];
			return;
		}

		int upper = 1;
		while (upper < count - 1 && speeds[upper] < speed)
		{
			upper++;
		}

		float t = (speed - speeds[upper - 1]) / (speeds[upper] - speeds[upper - 1]);
		gain = m_Card.m_aKappaPerSteer[upper - 1] + t * (m_Card.m_aKappaPerSteer[upper] - m_Card.m_aKappaPerSteer[upper - 1]);
		offset = m_Card.m_aKappaOffset[upper - 1] + t * (m_Card.m_aKappaOffset[upper] - m_Card.m_aKappaOffset[upper - 1]);
	}

	//------------------------------------------------------------------------------------------------
	//! GetSpeedKmh in m/s (forward-only: a magnitude)
	protected float GetSpeed()
	{
		return m_Simulation.GetSpeedKmh() / 3.6;
	}

	//------------------------------------------------------------------------------------------------
	//! One line for the log: card, signs, gearbox, limiter and lag
	string Describe()
	{
		if (!m_Card)
			return VERSION + " (no card)";

		return string.Format("%1: card %2 (%3, pin %4, build_key '%5', sessions %6) steer_sign %7 yaw_axis_sign %8 (measured %9)", VERSION, m_Card.m_sCardId, m_Card.m_sStatus, m_Card.m_sPinId, m_Card.m_sBuildKey, m_Card.GetSessionsText(), m_Card.m_iSteerSign, m_Card.m_iYawAxisSign, m_Card.m_bSignsMeasured) + string.Format(", gearbox %1, limiter %2 /s, lag %3 s, engine limit %4 /s, kappa_ss bins %5", m_Card.m_sGearbox, m_Card.m_fSteerRatePerS, m_Card.m_fSteerLagTauS, m_Card.m_fEngineSteerRateLimit, m_Card.m_aSpeedsMps.Count());
	}
}
