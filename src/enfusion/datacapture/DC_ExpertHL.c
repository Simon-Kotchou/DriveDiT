// ============================================================================
// DC_ExpertHL - a minimal expert HL for route following and the dc-2.0 driver component that runs it with llc_v1
// (DD-7-min, r2 3.2 and 3.6)
// ============================================================================
//
// DC_ExpertDriver is the DC_DriverInterface of DD-4-min (DC_StreamRecorder.c) for T0-MIN: attach it to
// GameMode_DataCapture in the DC_Arland layer. It owns the route (DC_Route), the expert (DC_ExpertHL), the vehicle
// card (DC_VehicleCard) and llc_v1 (DC_LLCv1), and is called by the core on its physics-step counters (no CallLater
// clock):
//   GetSpawnPose   the route start (point 0) facing along the route;
//   BeginDriver    sha256 self-test; route (trial conf "route", else m_sDefaultRouteId) checked against the session's
//                  map and the allowed splits (never Everon); vehicle card for the session's pin, accepted by llc_v1's
//                  card rules (use_class from the launch record); llc_sha and expert_sha = sha256 of DC_LLCv1.c and
//                  DC_ExpertHL.c as deployed; stamps and signs to session.json control.*;
//   BeginEpisode   at a decision step after the warm-up, when the ego stands at the route start (distance, heading,
//                  speed), at most m_iMaxEpisodes, m_fEpisodeGapS after the previous one: mode E, driver expert;
//   ApplyStep      every physics step with a held command: llc_v1 (EOnSimulate);
//   PostStep       every physics step of the episode (EOnPostSimulate): ordered route progress and lane errors on
//                  the state after the step, the expert evaluated on that state (exp_*_now), terminations;
//   Decide         every decision step (P mod N == 0; N = 6 at 60 Hz, 10 Hz): the command evaluated in PostStep of
//                  the same step, so in mode E action_cmd of decision t equals exp_*_now at P_t (dc-2.0 section 6);
//   EndEpisode / EndDriver   log lines (the core parks the ego).
//
// EXPERT (pure pursuit and a speed profile on the route polyline, the driving line of the drivedit_nav route)
//   progress   ordered projection as nav_v1: inside +-PROGRESS_WINDOW_M around the previous progress, never decreasing
//              (a loop's start and end are told apart by it);
//   kappa      pure pursuit on the route point at progress + lookahead (lookahead = m_fLookaheadS x speed, clamped
//              to [m_fLookaheadMinM, m_fLookaheadMaxM]): kappa = 2 y / d^2 in the ego frame (y right), clamped to
//              +-m_fKappaMax;
//   accel      speed profile per route point: min(cruise, sqrt(a_lat,max / |kappa_route|)) with the route curvature
//              over +-CURVATURE_HALF_WINDOW_M (C17's definition), 0 at the route end (its route_end stop), limited
//              backwards by m_fProfileDecelMps2; accel = m_fSpeedGain x (profile at progress + speed x
//              m_fSpeedPreviewS - speed), clamped to [-m_fBrakeMaxMps2, m_fAccelMaxMps2];
//   dir        +1 (forward-only, T0-MIN); 0 once the ego stands within m_fEndToleranceM of the route end;
//   held       the command of decision t is held over (P_t, P_t + N] by the core (zero-order hold, [t, t + 100 ms)).
// Terminations (the first one reported wins, at the next decision step): route_complete (at the end and stopped),
// route_deviation (|lane_lat_err_m| > m_fDeviationM), stuck (slower than m_fStuckKmh for m_fStuckS while dir +1),
// timeout (m_fEpisodeTimeoutS, the r2 3.6 cap of 8 min), rollover (up axis tilted beyond m_fRolloverDeg),
// engine_off (engine off for m_fEngineOffS).
// Every constant is a parameter with a stated basis (r2 3.6, style_v1 nominal a_lat,max 2.75 m/s2, the S105); none
// is ported from the Everon-built navigation code (EL-002).
//
// ROUTE FILE (drivedit.route/1, tools/routes/make_t0_loop.py): scalar keys through JsonLoadContext, the polyline by
// DC_JsonScan (number arrays, see DC_LLCv1.c), checked against source.n_points and length_m. episodes.route_sha256
// = sha256 of the file without its final LF, which is drivedit_nav's Route.sha256 for a file in canonical form (the
// generator writes it so; tests/test_t0_loop_route.py checks it).
// ============================================================================

//------------------------------------------------------------------------------------------------
//! A drivedit.route/1 route: the driving line (x, y, z), arc lengths (horizontal), curvature per point
class DC_Route
{
	static const string SCHEMA = "drivedit.route/1";
	static const float PROGRESS_WINDOW_M = 30;		//!< drivedit_nav PROGRESS_WINDOW_M (nav_v1 ordered progress)
	static const float CURVATURE_HALF_WINDOW_M = 5;	//!< C17 route_curvature: heading change over +-5 m
	static const float LENGTH_TOL_M = 0.5;			//!< length_m against the float sum of the segments
	static const float CLOSED_TOL_M = 1;			//!< drivedit_nav CLOSED_TOL_M

	string m_sPath;
	string m_sSha256;			//!< sha256 of the file without its final LF (drivedit_nav Route.sha256 of a canonical file)
	string m_sRouteId;
	string m_sMapId;
	string m_sSplit;
	string m_sLaneSide;
	bool m_bClosed;
	float m_fLengthM;
	ref array<vector> m_aPoints = {};
	ref array<float> m_aS = {};
	ref array<float> m_aKappa = {};

	//------------------------------------------------------------------------------------------------
	//! Reads and checks a route file. false (failure) on a missing file, a parse problem or an inconsistent polyline
	bool Load(string path, out string failure)
	{
		string text;
		if (!DC_TextFile.Read(path, text, failure))
			return false;

		m_sPath = path;
		string hashed = text;
		if (text.EndsWith("\n"))
			hashed = text.Substring(0, text.Length() - 1);

		m_sSha256 = DC_Sha256.HashText(hashed);
		JsonLoadContext context = new JsonLoadContext(false);
		if (!context.LoadFromString(text))
		{
			failure = path + " does not parse as JSON";
			return false;
		}

		string schema;
		if (!context.ReadValue("schema", schema) || schema != SCHEMA)
		{
			failure = path + ": schema is '" + schema + "', expected " + SCHEMA;
			return false;
		}

		bool keysOk = context.ReadValue("route_id", m_sRouteId) && context.ReadValue("map_id", m_sMapId) && context.ReadValue("split", m_sSplit);
		keysOk = keysOk && context.ReadValue("lane_side", m_sLaneSide) && context.ReadValue("closed", m_bClosed);
		if (!keysOk || !DC_JsonScan.ReadContextNumber(context, "length_m", m_fLengthM))
		{
			failure = path + ": route_id, map_id, split, lane_side, closed and length_m are required";
			return false;
		}

		int declaredPoints = -1;
		if (context.StartObject("source"))
		{
			if (context.DoesKeyExist("n_points"))
				context.ReadValue("n_points", declaredPoints);

			context.EndObject();
		}

		if (!ReadPolyline(text, failure))
		{
			failure = path + ": " + failure;
			return false;
		}

		if (declaredPoints >= 0 && declaredPoints != m_aPoints.Count())
		{
			failure = string.Format("%1: polyline has %2 points, source.n_points says %3", path, m_aPoints.Count(), declaredPoints);
			return false;
		}

		if (!Measure(failure))
		{
			failure = path + ": " + failure;
			return false;
		}

		failure = string.Empty;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! "polyline": [[x, y, z], ...] by DC_JsonScan
	protected bool ReadPolyline(string text, out string failure)
	{
		m_aPoints.Clear();
		int index = DC_JsonScan.FindValue(text, "polyline");
		if (index < 0 || !DC_JsonScan.Expect(text, index, DC_JsonScan.CHAR_OPEN, index))
		{
			failure = "no polyline array";
			return false;
		}

		while (true)
		{
			float x;
			float y;
			float z;
			bool pointOk = DC_JsonScan.Expect(text, index, DC_JsonScan.CHAR_OPEN, index) && DC_JsonScan.ReadNumber(text, index, x, index);
			pointOk = pointOk && DC_JsonScan.Expect(text, index, DC_JsonScan.CHAR_COMMA, index) && DC_JsonScan.ReadNumber(text, index, y, index);
			pointOk = pointOk && DC_JsonScan.Expect(text, index, DC_JsonScan.CHAR_COMMA, index) && DC_JsonScan.ReadNumber(text, index, z, index);
			if (!pointOk || !DC_JsonScan.Expect(text, index, DC_JsonScan.CHAR_CLOSE, index))
			{
				failure = string.Format("polyline point %1 is not [x, y, z]", m_aPoints.Count());
				return false;
			}

			m_aPoints.Insert(Vector(x, y, z));
			if (DC_JsonScan.Expect(text, index, DC_JsonScan.CHAR_CLOSE, index))
				break;

			if (!DC_JsonScan.Expect(text, index, DC_JsonScan.CHAR_COMMA, index))
			{
				failure = string.Format("polyline point %1 is not followed by ',' or ']'", m_aPoints.Count() - 1);
				return false;
			}
		}

		failure = string.Empty;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Arc lengths, the length check, the closed check and the curvature per point
	protected bool Measure(out string failure)
	{
		int count = m_aPoints.Count();
		if (count < 2)
		{
			failure = "the polyline needs at least 2 points";
			return false;
		}

		m_aS.Clear();
		m_aS.Insert(0);
		for (int i = 1; i < count; i++)
		{
			vector a = m_aPoints[i - 1];
			vector b = m_aPoints[i];
			float segment = Math.Sqrt((b[0] - a[0]) * (b[0] - a[0]) + (b[2] - a[2]) * (b[2] - a[2]));
			if (segment < 0.01)
			{
				failure = string.Format("polyline points %1 and %2 coincide", i - 1, i);
				return false;
			}

			m_aS.Insert(m_aS[i - 1] + segment);
		}

		float measured = m_aS[count - 1];
		if (Math.AbsFloat(measured - m_fLengthM) > LENGTH_TOL_M)
		{
			failure = string.Format("length_m %1 differs from the polyline length %2", m_fLengthM, measured);
			return false;
		}

		m_fLengthM = measured;
		vector first = m_aPoints[0];
		vector last = m_aPoints[count - 1];
		float gap = Math.Sqrt((last[0] - first[0]) * (last[0] - first[0]) + (last[2] - first[2]) * (last[2] - first[2]));
		if (m_bClosed && gap > CLOSED_TOL_M)
		{
			failure = string.Format("closed route: first and last point are %1 m apart", gap);
			return false;
		}

		m_aKappa.Clear();
		for (int j = 0; j < count; j++)
		{
			float lo = Math.Max(m_aS[j] - CURVATURE_HALF_WINDOW_M, 0);
			float hi = Math.Min(m_aS[j] + CURVATURE_HALF_WINDOW_M, m_fLengthM);
			float kappa = 0;
			if (hi - lo > 0.001)
				kappa = WrapAngle(HeadingAt(hi) - HeadingAt(lo)) / (hi - lo);

			m_aKappa.Insert(kappa);
		}

		failure = string.Empty;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	int GetPointCount()
	{
		return m_aPoints.Count();
	}

	//------------------------------------------------------------------------------------------------
	//! Index i of the segment [s_i, s_i+1] holding arc length s (clamped to the first and last segment)
	int FindSegment(float s)
	{
		int lo = 0;
		int hi = m_aS.Count() - 1;
		if (s <= m_aS[lo])
			return 0;

		if (s >= m_aS[hi])
			return hi - 1;

		while (hi - lo > 1)
		{
			int mid = (lo + hi) / 2;
			if (m_aS[mid] <= s)
				lo = mid;
			else
				hi = mid;
		}

		return lo;
	}

	//------------------------------------------------------------------------------------------------
	//! Point at arc length s, clamped to [0, length]
	vector PointAt(float s)
	{
		int i = FindSegment(s);
		float segment = m_aS[i + 1] - m_aS[i];
		float t = Math.Clamp((s - m_aS[i]) / segment, 0, 1);
		vector a = m_aPoints[i];
		vector b = m_aPoints[i + 1];
		return a + (b - a) * t;
	}

	//------------------------------------------------------------------------------------------------
	//! Heading of the segment at arc length s [rad], 0 = +Z, clockwise seen from above
	float HeadingAt(float s)
	{
		int i = FindSegment(s);
		vector a = m_aPoints[i];
		vector b = m_aPoints[i + 1];
		return Math.Atan2(b[0] - a[0], b[2] - a[2]);
	}

	//------------------------------------------------------------------------------------------------
	//! Route curvature of point i [1/m], right positive
	float GetKappa(int i)
	{
		return m_aKappa[i];
	}

	//------------------------------------------------------------------------------------------------
	float GetS(int i)
	{
		return m_aS[i];
	}

	//------------------------------------------------------------------------------------------------
	//! Closest point to (x, z) among arc lengths [sLo, sHi] (drivedit_nav Route.project; a later segment wins only when
	//! strictly closer): its arc length, distance, signed lateral offset (right of the route positive) and the
	//! segment heading. false when the interval holds no segment
	bool Project(float x, float z, float sLo, float sHi, out float s, out float distance, out float lateral, out float tangentHeading)
	{
		float from = Math.Max(0, sLo);
		float to = Math.Min(m_fLengthM, sHi);
		bool found = false;
		int count = m_aPoints.Count();
		int i = FindSegment(from);
		while (i < count - 1 && m_aS[i] <= to)
		{
			vector a = m_aPoints[i];
			vector b = m_aPoints[i + 1];
			float segment = m_aS[i + 1] - m_aS[i];
			float tLo = Math.Max(0, (from - m_aS[i]) / segment);
			float tHi = Math.Min(1, (to - m_aS[i]) / segment);
			if (tLo <= tHi)
			{
				float dx = b[0] - a[0];
				float dz = b[2] - a[2];
				float t = Math.Clamp(((x - a[0]) * dx + (z - a[2]) * dz) / (segment * segment), tLo, tHi);
				float px = a[0] + t * dx;
				float pz = a[2] + t * dz;
				float d = Math.Sqrt((x - px) * (x - px) + (z - pz) * (z - pz));
				if (!found || d < distance)
				{
					found = true;
					distance = d;
					s = m_aS[i] + t * segment;
					lateral = ((x - px) * dz - (z - pz) * dx) / segment;
					tangentHeading = Math.Atan2(dx, dz);
				}
			}

			i++;
		}

		return found;
	}

	//------------------------------------------------------------------------------------------------
	//! An angle wrapped to (-pi, pi]
	static float WrapAngle(float angle)
	{
		while (angle > Math.PI)
		{
			angle -= Math.PI2;
		}

		while (angle <= -Math.PI)
		{
			angle += Math.PI2;
		}

		return angle;
	}

	//------------------------------------------------------------------------------------------------
	//! A route id of drivedit_nav ([a-z0-9][a-z0-9_.-]{0,63}); it becomes part of a file path
	static bool IsValidRouteId(string routeId)
	{
		int length = routeId.Length();
		if (length < 1 || length > 64)
			return false;

		for (int i = 0; i < length; i++)
		{
			int code = routeId.ToAscii(i);
			bool lower = code >= 97 && code <= 122;
			bool digit = code >= 48 && code <= 57;
			bool mark = code == 95 || code == 46 || code == 45;
			if (!lower && !digit && !(mark && i > 0))
				return false;
		}

		return true;
	}
}

//------------------------------------------------------------------------------------------------
//! The minimal expert (see the header): progress and lane errors on the route, pure pursuit and the speed profile.
//! The driver sets the public parameters, then Init; Reset at every episode start
class DC_ExpertHL
{
	static const string VERSION = "expert_min_v1";
	static const string SOURCE_FILE = "DC_ExpertHL.c";
	static const float MIN_TARGET_DISTANCE_M = 1;	//!< closer pursuit targets (the route end) give kappa 0

	// parameters (DC_ExpertDriver attributes)
	float m_fCruiseMps = 11.1;
	float m_fAlatMaxMps2 = 2.75;
	float m_fAccelMaxMps2 = 1.5;
	float m_fBrakeMaxMps2 = 4;
	float m_fProfileDecelMps2 = 2;
	float m_fSpeedGain = 1;
	float m_fSpeedPreviewS = 1;
	float m_fLookaheadS = 0.9;
	float m_fLookaheadMinM = 6;
	float m_fLookaheadMaxM = 15;
	float m_fKappaMax = 0.2;
	float m_fEndToleranceM = 5;
	float m_fStopSpeedMps = 0.3;

	protected DC_Route m_Route;
	protected ref array<float> m_aProfile = {};

	// state of the episode
	protected float m_fProgress;
	protected float m_fDistance;
	protected float m_fLateral;
	protected float m_fHeadingError;
	protected int m_iDir = 1;

	//------------------------------------------------------------------------------------------------
	//! Binds the route and builds the speed profile from the parameters
	void Init(notnull DC_Route route)
	{
		m_Route = route;
		m_aProfile.Clear();
		int count = route.GetPointCount();
		for (int i = 0; i < count; i++)
		{
			float speed = m_fCruiseMps;
			float kappa = Math.AbsFloat(route.GetKappa(i));
			if (kappa > 0.0001)
				speed = Math.Min(speed, Math.Sqrt(m_fAlatMaxMps2 / kappa));

			m_aProfile.Insert(speed);
		}

		// the route_end stop, then the deceleration limit backwards
		m_aProfile[count - 1] = 0;
		for (int j = count - 2; j >= 0; j--)
		{
			float ds = route.GetS(j + 1) - route.GetS(j);
			float reachable = Math.Sqrt(m_aProfile[j + 1] * m_aProfile[j + 1] + 2 * m_fProfileDecelMps2 * ds);
			m_aProfile[j] = Math.Min(m_aProfile[j], reachable);
		}

		Reset();
	}

	//------------------------------------------------------------------------------------------------
	//! Episode start: progress 0 (nav_v1's first call), forward
	void Reset()
	{
		m_fProgress = 0;
		m_fDistance = 0;
		m_fLateral = 0;
		m_fHeadingError = 0;
		m_iDir = 1;
	}

	//------------------------------------------------------------------------------------------------
	//! Ordered progress (nav_v1 rule) and the lane errors of the state after a physics step
	void Update(vector position, float headingRad)
	{
		float s;
		float distance;
		float lateral;
		float tangent;
		if (!m_Route.Project(position[0], position[2], m_fProgress - DC_Route.PROGRESS_WINDOW_M, m_fProgress + DC_Route.PROGRESS_WINDOW_M, s, distance, lateral, tangent))
			return;

		m_fProgress = Math.Max(m_fProgress, s);
		m_fDistance = distance;
		m_fLateral = lateral;
		m_fHeadingError = DC_Route.WrapAngle(headingRad - tangent);
	}

	//------------------------------------------------------------------------------------------------
	//! The expert command on the state after a physics step (call Update first)
	void Evaluate(vector position, float headingRad, float speed, notnull DC_HLCommand command)
	{
		float remaining = m_Route.m_fLengthM - m_fProgress;
		if (remaining <= m_fEndToleranceM && speed < m_fStopSpeedMps)
			m_iDir = 0;

		// speed profile, looked up ahead by the preview time
		float target = ProfileAt(m_fProgress + speed * m_fSpeedPreviewS);
		float accel = Math.Clamp(m_fSpeedGain * (target - speed), -m_fBrakeMaxMps2, m_fAccelMaxMps2);

		// pure pursuit on the route point one lookahead ahead
		float lookahead = Math.Clamp(m_fLookaheadS * speed, m_fLookaheadMinM, m_fLookaheadMaxM);
		vector point = m_Route.PointAt(m_fProgress + lookahead);
		float dx = point[0] - position[0];
		float dz = point[2] - position[2];
		float sinH = Math.Sin(headingRad);
		float cosH = Math.Cos(headingRad);
		float forward = dx * sinH + dz * cosH;
		float right = dx * cosH - dz * sinH;
		float distance2 = forward * forward + right * right;
		float kappa = 0;
		if (distance2 > MIN_TARGET_DISTANCE_M * MIN_TARGET_DISTANCE_M)
			kappa = Math.Clamp(2 * right / distance2, -m_fKappaMax, m_fKappaMax);

		if (m_iDir == 0)
		{
			kappa = 0;
			accel = -m_fProfileDecelMps2;
		}

		command.m_fKappa = kappa;
		command.m_fAccel = accel;
		command.m_iDir = m_iDir;
	}

	//------------------------------------------------------------------------------------------------
	//! The speed profile at arc length s (linear between the route points)
	float ProfileAt(float s)
	{
		int i = m_Route.FindSegment(s);
		float s0 = m_Route.GetS(i);
		float s1 = m_Route.GetS(i + 1);
		float t = Math.Clamp((s - s0) / (s1 - s0), 0, 1);
		return m_aProfile[i] + t * (m_aProfile[i + 1] - m_aProfile[i]);
	}

	//------------------------------------------------------------------------------------------------
	float GetProgress()
	{
		return m_fProgress;
	}

	//------------------------------------------------------------------------------------------------
	//! Distance to the route at the projection [m]
	float GetDistance()
	{
		return m_fDistance;
	}

	//------------------------------------------------------------------------------------------------
	//! lane_lat_err_m: right of the driving line positive
	float GetLateral()
	{
		return m_fLateral;
	}

	//------------------------------------------------------------------------------------------------
	//! lane_head_err_rad: heading minus route tangent, wrapped, right positive
	float GetHeadingError()
	{
		return m_fHeadingError;
	}

	//------------------------------------------------------------------------------------------------
	//! At the route end and stopped (dir 0)
	bool IsFinished()
	{
		return m_iDir == 0;
	}

	//------------------------------------------------------------------------------------------------
	//! Lowest and highest value of the speed profile [m/s] (for the log)
	void GetProfileRange(out float lowest, out float highest)
	{
		lowest = m_fCruiseMps;
		highest = 0;
		for (int i = 0; i < m_aProfile.Count() - 1; i++)
		{
			lowest = Math.Min(lowest, m_aProfile[i]);
			highest = Math.Max(highest, m_aProfile[i]);
		}
	}
}

//------------------------------------------------------------------------------------------------
[ComponentEditorProps(category: "GameScripted/DataCapture", description: "dc-2.0 driver: the minimal route-following expert HL with llc_v1 (DD-7-min); attach to the capture game mode")]
class DC_ExpertDriverClass : DC_DriverInterfaceClass
{
}

//------------------------------------------------------------------------------------------------
class DC_ExpertDriver : DC_DriverInterface
{
	// === VEHICLE CARD (llc_v1) ===
	[Attribute("S105", UIWidgets.EditBox, "Vehicle card id: tools/ident/cards/<pin_id>/<id>.json, deployed to Configs/ident/cards/<pin_id>/<id>.json (the session's pin_id)", category: "Card")]
	protected string m_sCardId;

	[Attribute("$DataCapture:Configs/ident/cards/,Configs/ident/cards/", UIWidgets.EditBox, "Folders searched for <pin_id>/<card id>.json, in this order (comma-separated, each ending in / or :)", category: "Card")]
	protected string m_sCardRoots;

	[Attribute("$DataCapture:Scripts/Game/DataCapture/,Scripts/Game/DataCapture/", UIWidgets.EditBox, "Folders searched for DC_LLCv1.c and DC_ExpertHL.c as deployed: llc_sha and expert_sha are their sha256", category: "Card")]
	protected string m_sSourceRoots;

	// === ROUTE ===
	[Attribute("arland_t0_loop", UIWidgets.EditBox, "Route id when the trial conf names no route (file <id>.json)", category: "Route")]
	protected string m_sDefaultRouteId;

	[Attribute("$DataCapture:Configs/routes/,Configs/routes/", UIWidgets.EditBox, "Folders searched for <route id>.json, in this order (comma-separated, each ending in / or :)", category: "Route")]
	protected string m_sRouteRoots;

	[Attribute("train", UIWidgets.EditBox, "Route splits this driver may drive (comma-separated); Everon splits are always refused", category: "Route")]
	protected string m_sAllowedSplits;

	// === EXPERT ===
	[Attribute("40", UIWidgets.Slider, "Cruise speed on straights [km/h] (style_v1 speed factor 1)", "5 90 1", category: "Expert")]
	protected float m_fCruiseKmh;

	[Attribute("2.75", UIWidgets.Slider, "Lateral acceleration limit of the speed profile [m/s2] (style_v1 a_lat,max nominal 2.75)", "0.5 6 0.05", category: "Expert")]
	protected float m_fAlatMaxMps2;

	[Attribute("1.5", UIWidgets.Slider, "Acceleration command limit [m/s2]", "0.1 5 0.1", category: "Expert")]
	protected float m_fAccelMaxMps2;

	[Attribute("4", UIWidgets.Slider, "Deceleration command limit [m/s2]", "0.5 8 0.1", category: "Expert")]
	protected float m_fBrakeMaxMps2;

	[Attribute("2", UIWidgets.Slider, "Deceleration the speed profile plans with (curves, the route end) [m/s2]", "0.5 6 0.1", category: "Expert")]
	protected float m_fProfileDecelMps2;

	[Attribute("1", UIWidgets.Slider, "Speed error gain of the acceleration command [1/s]", "0.1 5 0.05", category: "Expert")]
	protected float m_fSpeedGain;

	[Attribute("1", UIWidgets.Slider, "Speed profile preview [s of travel at the current speed]", "0 5 0.1", category: "Expert")]
	protected float m_fSpeedPreviewS;

	[Attribute("0.9", UIWidgets.Slider, "Pure pursuit lookahead [s of travel]", "0.1 3 0.05", category: "Expert")]
	protected float m_fLookaheadS;

	[Attribute("6", UIWidgets.Slider, "Shortest pure pursuit lookahead [m] (shorter ones oscillate in slow S-bends with the actuator delay)", "1 20 0.5", category: "Expert")]
	protected float m_fLookaheadMinM;

	[Attribute("15", UIWidgets.Slider, "Longest pure pursuit lookahead [m]", "2 50 0.5", category: "Expert")]
	protected float m_fLookaheadMaxM;

	[Attribute("0.2", UIWidgets.Slider, "Curvature command limit [1/m] (about the S105's tightest turn, radius 5 m)", "0.01 0.5 0.01", category: "Expert")]
	protected float m_fKappaMax;

	[Attribute("5", UIWidgets.Slider, "The route counts as complete within this distance of its end, stopped [m] (the preview of the stop profile stops the car up to a few metres early)", "0.5 20 0.5", category: "Expert")]
	protected float m_fEndToleranceM;

	[Attribute("1", UIWidgets.Slider, "Stopped below this speed [km/h]", "0.1 5 0.1", category: "Expert")]
	protected float m_fStopKmh;

	// === TERMINATIONS ===
	[Attribute("6", UIWidgets.Slider, "route_deviation beyond this lateral distance from the driving line [m] (the far edge of an 8 m road from the right lane centre)", "1 30 0.5", category: "Terminations")]
	protected float m_fDeviationM;

	[Attribute("10", UIWidgets.Slider, "stuck after this long below m_fStuckKmh while the expert drives forward [s]", "1 60 1", category: "Terminations")]
	protected float m_fStuckS;

	[Attribute("2", UIWidgets.Slider, "stuck speed [km/h]", "0.1 10 0.1", category: "Terminations")]
	protected float m_fStuckKmh;

	[Attribute("480", UIWidgets.Slider, "timeout of an episode [s] (r2 3.6: hard cap 8 min)", "10 1800 10", category: "Terminations")]
	protected float m_fEpisodeTimeoutS;

	[Attribute("60", UIWidgets.Slider, "rollover when the up axis tilts beyond this [deg]", "10 90 1", category: "Terminations")]
	protected float m_fRolloverDeg;

	[Attribute("2", UIWidgets.Slider, "engine_off after the engine is off this long [s]", "0.1 30 0.1", category: "Terminations")]
	protected float m_fEngineOffS;

	// === EPISODES ===
	[Attribute("0", UIWidgets.Slider, "Episodes per session, 0 = as many as the trial allows (one lap each)", "0 100 1", category: "Episodes")]
	protected int m_iMaxEpisodes;

	[Attribute("3", UIWidgets.Slider, "Pause after an episode before the next one may begin [s of world time]", "0 60 0.5", category: "Episodes")]
	protected float m_fEpisodeGapS;

	[Attribute("8", UIWidgets.Slider, "An episode begins only within this distance of the route start [m] (a closed route: the previous lap stops within m_fEndToleranceM of it)", "0.5 50 0.5", category: "Episodes")]
	protected float m_fStartMaxDistM;

	[Attribute("30", UIWidgets.Slider, "... and within this heading of the route start [deg]", "1 180 1", category: "Episodes")]
	protected float m_fStartMaxHeadingDeg;

	[Attribute("1", UIWidgets.Slider, "... and below this speed [km/h]", "0.1 10 0.1", category: "Episodes")]
	protected float m_fStartMaxKmh;

	// === CONSTANTS ===
	protected static ref array<string> s_aSealedSplits = {"everon_harness", "everon_test"};

	// === STATE ===
	protected DC_DriverContext m_Context;
	protected ref DC_Route m_Route;
	protected ref DC_VehicleCard m_Card;
	protected ref DC_LLCv1 m_LLC;
	protected ref DC_ExpertHL m_Expert;
	protected ref DC_HLCommand m_Decision = new DC_HLCommand();
	protected string m_sRouteFailure;
	protected bool m_bRouteTried;
	protected bool m_bReady;
	protected string m_sLlcSha;
	protected string m_sExpertSha;
	protected string m_sUseClass;

	// episode
	protected bool m_bInEpisode;
	protected int m_iEpisodes;
	protected int m_iEpisodeStartMs;
	protected int m_iLastEndMs = -1;
	protected int m_iDecisionStep = -1;
	protected int m_iSlowSinceMs = -1;
	protected int m_iEngineOffSinceMs = -1;
	protected string m_sWaiting;
	protected int m_iSamples;
	protected float m_fSumAbsLateral;
	protected float m_fMaxAbsLateral;

	//------------------------------------------------------------------------------------------------
	//! The route start facing along the route (y from the surface, by DC_EgoSession)
	override bool GetSpawnPose(out vector position, out float headingDeg)
	{
		if (SCR_Global.IsEditMode() || !LoadRoute())
			return false;

		position = m_Route.PointAt(0);
		headingDeg = m_Route.HeadingAt(0) * Math.RAD2DEG;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Readiness: self-test, route, card, llc_v1, expert, source hashes, stamps (see the header)
	override bool BeginDriver(notnull DC_DriverContext ctx)
	{
		m_Context = ctx;
		string failure;
		if (!DC_Sha256.SelfTest(failure))
		{
			ctx.Error(failure + ": no digest of this engine can be stamped");
			return false;
		}

		if (!LoadRoute() || !CheckRoute(ctx, failure))
		{
			if (failure.IsEmpty())
				failure = m_sRouteFailure;

			ctx.Error("route: " + failure);
			return false;
		}

		if (!LoadCard(ctx, failure))
		{
			ctx.Error("vehicle card: " + failure);
			return false;
		}

		m_LLC = new DC_LLCv1();
		if (!m_LLC.Init(m_Card, ctx.GetActuator(), ctx.GetFixedTimeSliceS(), failure))
		{
			ctx.Error("llc_v1: " + failure);
			return false;
		}

		m_Expert = new DC_ExpertHL();
		m_Expert.m_fCruiseMps = m_fCruiseKmh / 3.6;
		m_Expert.m_fAlatMaxMps2 = m_fAlatMaxMps2;
		m_Expert.m_fAccelMaxMps2 = m_fAccelMaxMps2;
		m_Expert.m_fBrakeMaxMps2 = m_fBrakeMaxMps2;
		m_Expert.m_fProfileDecelMps2 = m_fProfileDecelMps2;
		m_Expert.m_fSpeedGain = m_fSpeedGain;
		m_Expert.m_fSpeedPreviewS = m_fSpeedPreviewS;
		m_Expert.m_fLookaheadS = m_fLookaheadS;
		m_Expert.m_fLookaheadMinM = m_fLookaheadMinM;
		m_Expert.m_fLookaheadMaxM = m_fLookaheadMaxM;
		m_Expert.m_fKappaMax = m_fKappaMax;
		m_Expert.m_fEndToleranceM = m_fEndToleranceM;
		m_Expert.m_fStopSpeedMps = m_fStopKmh / 3.6;
		m_Expert.Init(m_Route);

		m_sLlcSha = HashSource(DC_LLCv1.SOURCE_FILE, failure);
		if (m_sLlcSha.IsEmpty())
		{
			ctx.Error("llc_sha: " + failure);
			return false;
		}

		m_sExpertSha = HashSource(DC_ExpertHL.SOURCE_FILE, failure);
		if (m_sExpertSha.IsEmpty())
		{
			ctx.Error("expert_sha: " + failure);
			return false;
		}

		ctx.SetStamps(m_sLlcSha, m_sExpertSha, m_Card.m_sCardId, m_Card.m_sSha256);
		ctx.SetSigns(m_Card.m_iSteerSign, m_Card.m_iYawAxisSign);
		float lowest;
		float highest;
		m_Expert.GetProfileRange(lowest, highest);
		float lowestKmh = lowest * 3.6;
		float highestKmh = highest * 3.6;
		float lengthM = m_Route.m_fLengthM;
		ctx.Log(m_LLC.Describe());
		ctx.Log(string.Format("vehicle card %1 sha256 %2 (use_class %3)", m_Card.m_sPath, m_Card.m_sSha256, m_sUseClass));
		ctx.Log(string.Format("route %1 (%2, split %3, %4 m, %5 points) sha256 %6; speed profile %7..%8 km/h", m_Route.m_sRouteId, m_Route.m_sPath, m_Route.m_sSplit, lengthM.ToString(-1, 1), m_Route.GetPointCount(), m_Route.m_sSha256, lowestKmh.ToString(-1, 1), highestKmh.ToString(-1, 1)));
		ctx.Log(string.Format("%1 llc_sha %2, %3 expert_sha %4", DC_LLCv1.VERSION, m_sLlcSha, DC_ExpertHL.VERSION, m_sExpertSha));
		m_bReady = true;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! A new episode when the ego stands at the route start (see the header)
	override bool BeginEpisode(int episodeId, int firstPhysicsStep, notnull DC_EpisodeInfo info)
	{
		if (!m_bReady || m_bInEpisode)
			return false;

		if (m_iMaxEpisodes > 0 && m_iEpisodes >= m_iMaxEpisodes)
		{
			NoteWaiting(string.Format("%1 episode(s) done (m_iMaxEpisodes)", m_iEpisodes));
			return false;
		}

		int now = m_Context.GetWorldMs();
		if (m_iLastEndMs >= 0 && now - m_iLastEndMs < m_fEpisodeGapS * 1000)
			return false;

		string notAtStart = CheckStart();
		if (!notAtStart.IsEmpty())
		{
			NoteWaiting(notAtStart);
			return false;
		}

		m_LLC.Reset();
		m_Expert.Reset();
		info.m_sMode = "E";
		info.m_sDriver = "expert";
		info.m_sVehicleCardId = m_Card.m_sCardId;
		info.m_sRouteId = m_Route.m_sRouteId;
		info.m_sRouteSha256 = m_Route.m_sSha256;
		info.m_sExpertSha = m_sExpertSha;
		info.m_sLlcSha = m_sLlcSha;
		info.m_fStyleSpeedFactor = 1;
		info.m_fStyleLaneOffsetM = 0;
		info.m_fStyleAlatMaxMps2 = m_fAlatMaxMps2;

		m_bInEpisode = true;
		m_iEpisodeStartMs = now;
		m_iDecisionStep = -1;
		m_iSlowSinceMs = -1;
		m_iEngineOffSinceMs = -1;
		m_sWaiting = string.Empty;
		m_iSamples = 0;
		m_fSumAbsLateral = 0;
		m_fMaxAbsLateral = 0;
		m_Context.Log(string.Format("episode %1 begins at physics step %2 on route %3 (seed %4)", episodeId, firstPhysicsStep, m_Route.m_sRouteId, info.m_iSeed));
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! llc_v1 for the held command (EOnSimulate)
	override void ApplyStep(int physicsStep, float timeSlice, notnull DC_HLCommand held, notnull DC_LLCOutput llc)
	{
		if (!m_bInEpisode || !m_LLC)
			return;

		m_LLC.Step(held, llc);
	}

	//------------------------------------------------------------------------------------------------
	//! Progress, lane errors, the expert on the state after the step, terminations (EOnPostSimulate)
	override void PostStep(int physicsStep, float timeSlice, notnull DC_DriverPost post)
	{
		if (!m_bInEpisode)
			return;

		IEntity ego = m_Context.GetEgo();
		DC_Actuator actuator = m_Context.GetActuator();
		if (!ego || !actuator || !actuator.GetSimulation())
			return;

		VehicleWheeledSimulation simulation = actuator.GetSimulation();
		vector transform[4];
		ego.GetWorldTransform(transform);
		vector position = transform[3];
		vector forward = transform[2];
		float headingRad = Math.Atan2(forward[0], forward[2]);
		float speed = simulation.GetSpeedKmh() / 3.6;

		m_Expert.Update(position, headingRad);
		m_Expert.Evaluate(position, headingRad, speed, m_Decision);
		m_iDecisionStep = physicsStep;

		post.m_bHasExpert = true;
		post.m_fExpKappa = m_Decision.m_fKappa;
		post.m_fExpAccel = m_Decision.m_fAccel;
		post.m_iExpDir = m_Decision.m_iDir;
		post.m_bHasRoute = true;
		post.m_fLaneLatErrM = m_Expert.GetLateral();
		post.m_fLaneHeadErrRad = m_Expert.GetHeadingError();
		post.m_fRouteProgressM = m_Expert.GetProgress();
		post.m_sTermination = CheckTermination(transform[1], speed, simulation);

		float absLateral = Math.AbsFloat(m_Expert.GetLateral());
		m_iSamples++;
		m_fSumAbsLateral += absLateral;
		m_fMaxAbsLateral = Math.Max(m_fMaxAbsLateral, absLateral);
	}

	//------------------------------------------------------------------------------------------------
	//! The command evaluated in PostStep of this decision step (held over (P, P + N])
	override bool Decide(int physicsStep, notnull DC_HLCommand command)
	{
		if (!m_bInEpisode || m_iDecisionStep != physicsStep)
			return false;

		command.CopyFrom(m_Decision);
		return true;
	}

	//------------------------------------------------------------------------------------------------
	override void EndEpisode(int episodeId, int lastPhysicsStep, string terminationReason)
	{
		if (!m_bInEpisode)
			return;

		m_bInEpisode = false;
		m_iEpisodes++;
		m_iLastEndMs = m_Context.GetWorldMs();
		float meanAbs = 0;
		if (m_iSamples > 0)
			meanAbs = m_fSumAbsLateral / m_iSamples;

		float progress = m_Expert.GetProgress();
		float lengthM = m_Route.m_fLengthM;
		m_Context.Log(string.Format("episode %1 ended at physics step %2: %3, progress %4 of %5 m, |lane_lat_err_m| mean %6 max %7", episodeId, lastPhysicsStep, terminationReason, progress.ToString(-1, 1), lengthM.ToString(-1, 1), meanAbs.ToString(-1, 3), m_fMaxAbsLateral.ToString(-1, 3)));
	}

	//------------------------------------------------------------------------------------------------
	override void EndDriver(string endReason)
	{
		if (m_Context)
			m_Context.Log(string.Format("driver ends (%1) after %2 episode(s)", endReason, m_iEpisodes));

		m_bReady = false;
		m_bInEpisode = false;
		m_Context = null;
	}

	//------------------------------------------------------------------------------------------------
	//! The route of the trial conf ("route": an id or <id>.json), else m_sDefaultRouteId; once per session
	protected bool LoadRoute()
	{
		if (m_bRouteTried)
			return m_Route != null;

		m_bRouteTried = true;
		string routeId = m_sDefaultRouteId;
		string trialRoute;
		DC_Session session = DC_Session.Cast(GetOwner().FindComponent(DC_Session));
		if (session)
			trialRoute = session.GetTrialRoute();

		trialRoute = trialRoute.Trim();
		if (!trialRoute.IsEmpty())
			routeId = trialRoute;

		if (routeId.EndsWith(".json"))
			routeId = routeId.Substring(0, routeId.Length() - 5);

		if (!DC_Route.IsValidRouteId(routeId))
		{
			m_sRouteFailure = "route id '" + routeId + "' is not a drivedit_nav route id";
			return false;
		}

		string path = DC_TextFile.Resolve(m_sRouteRoots, routeId + ".json");
		if (path.IsEmpty())
		{
			m_sRouteFailure = "no " + routeId + ".json under " + m_sRouteRoots;
			return false;
		}

		DC_Route route = new DC_Route();
		string failure;
		if (!route.Load(path, failure))
		{
			m_sRouteFailure = failure;
			return false;
		}

		if (route.m_sRouteId != routeId)
		{
			m_sRouteFailure = path + " holds route " + route.m_sRouteId;
			return false;
		}

		m_Route = route;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! The route's map is the session's, its split is allowed, and it is never on the sealed map
	protected bool CheckRoute(notnull DC_DriverContext ctx, out string failure)
	{
		if (m_Route.m_sMapId == "everon" || s_aSealedSplits.Find(m_Route.m_sSplit) >= 0 || m_Route.m_sSplit.StartsWith("everon"))
		{
			failure = "route " + m_Route.m_sRouteId + " is on the sealed test map (Everon): never driven here";
			return false;
		}

		if (m_Route.m_sMapId != ctx.GetMapId())
		{
			failure = "route " + m_Route.m_sRouteId + " is on map " + m_Route.m_sMapId + ", the session on " + ctx.GetMapId();
			return false;
		}

		array<string> splits = {};
		m_sAllowedSplits.Split(",", splits, true);
		bool allowed = false;
		foreach (string split : splits)
		{
			if (split.Trim() == m_Route.m_sSplit)
				allowed = true;
		}

		if (!allowed)
		{
			failure = "route " + m_Route.m_sRouteId + " has split " + m_Route.m_sSplit + ", allowed: " + m_sAllowedSplits;
			return false;
		}

		failure = string.Empty;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! The card of the session's pin, accepted by llc_v1's rules for the session's build_key and use_class
	protected bool LoadCard(notnull DC_DriverContext ctx, out string failure)
	{
		string pinId = ctx.GetPinId();
		string fileName = pinId + "/" + m_sCardId + ".json";
		string path = DC_TextFile.Resolve(m_sCardRoots, fileName);
		if (pinId.IsEmpty() || path.IsEmpty())
		{
			failure = "no " + fileName + " under " + m_sCardRoots + " (the card is deployed by its manifest config line)";
			return false;
		}

		m_Card = new DC_VehicleCard();
		if (!m_Card.Load(path, failure))
			return false;

		if (m_Card.m_sCardId != m_sCardId)
		{
			failure = path + " holds card " + m_Card.m_sCardId;
			return false;
		}

		m_sUseClass = DC_LLCv1.ReadUseClass(pinId, ctx.GetBuildKey(), failure);
		if (m_sUseClass.IsEmpty())
		{
			failure = "use_class unknown: " + failure + " (a placeholder card needs a qualification use_class)";
			return false;
		}

		string warning;
		if (!DC_LLCv1.AcceptCard(m_Card, pinId, ctx.GetBuildKey(), m_sUseClass, failure, warning))
			return false;

		if (!warning.IsEmpty())
			ctx.Warn("vehicle card " + m_Card.m_sCardId + ": " + warning);

		if (m_Card.IsPlaceholder())
			ctx.Warn("vehicle card " + m_Card.m_sCardId + " is a PLACEHOLDER (no identification session): qualification data only");

		failure = string.Empty;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! sha256 of a deployed source file; empty (failure) when it cannot be read
	protected string HashSource(string fileName, out string failure)
	{
		string path = DC_TextFile.Resolve(m_sSourceRoots, fileName);
		string text;
		if (path.IsEmpty())
		{
			failure = "no " + fileName + " under " + m_sSourceRoots;
			return string.Empty;
		}

		if (!DC_TextFile.Read(path, text, failure))
			return string.Empty;

		return DC_Sha256.HashText(text);
	}

	//------------------------------------------------------------------------------------------------
	//! Empty when the ego stands at the route start, else why not
	protected string CheckStart()
	{
		IEntity ego = m_Context.GetEgo();
		DC_Actuator actuator = m_Context.GetActuator();
		VehicleWheeledSimulation simulation;
		if (actuator)
			simulation = actuator.GetSimulation();

		if (!ego || !simulation)
			return "no ego";

		vector transform[4];
		ego.GetWorldTransform(transform);
		vector position = transform[3];
		vector forward = transform[2];
		vector start = m_Route.PointAt(0);
		float distance = Math.Sqrt((position[0] - start[0]) * (position[0] - start[0]) + (position[2] - start[2]) * (position[2] - start[2]));
		float headingError = DC_Route.WrapAngle(Math.Atan2(forward[0], forward[2]) - m_Route.HeadingAt(0)) * Math.RAD2DEG;
		float speedKmh = simulation.GetSpeedKmh();
		if (distance > m_fStartMaxDistM)
			return string.Format("the ego is %1 m from the route start (at most %2 m)", distance.ToString(-1, 1), m_fStartMaxDistM);

		if (Math.AbsFloat(headingError) > m_fStartMaxHeadingDeg)
			return string.Format("the ego heading is %1 deg off the route start (at most %2 deg)", headingError.ToString(-1, 1), m_fStartMaxHeadingDeg);

		if (speedKmh > m_fStartMaxKmh)
			return string.Format("the ego moves at %1 km/h", speedKmh.ToString(-1, 1));

		return string.Empty;
	}

	//------------------------------------------------------------------------------------------------
	//! The first terminal event of this step, empty when none (the core keeps the first one it gets)
	protected string CheckTermination(vector up, float speed, notnull VehicleWheeledSimulation simulation)
	{
		int now = m_Context.GetWorldMs();
		if (m_Expert.IsFinished())
			return "route_complete";

		if (Math.AbsFloat(m_Expert.GetLateral()) > m_fDeviationM)
			return "route_deviation";

		if (up[1] < Math.Cos(m_fRolloverDeg * Math.DEG2RAD))
			return "rollover";

		if (!simulation.EngineIsOn())
		{
			if (m_iEngineOffSinceMs < 0)
				m_iEngineOffSinceMs = now;
			else if (now - m_iEngineOffSinceMs >= m_fEngineOffS * 1000)
				return "engine_off";
		}
		else
		{
			m_iEngineOffSinceMs = -1;
		}

		if (speed < m_fStuckKmh / 3.6)
		{
			if (m_iSlowSinceMs < 0)
				m_iSlowSinceMs = now;
			else if (now - m_iSlowSinceMs >= m_fStuckS * 1000)
				return "stuck";
		}
		else
		{
			m_iSlowSinceMs = -1;
		}

		if (now - m_iEpisodeStartMs >= m_fEpisodeTimeoutS * 1000)
			return "timeout";

		return string.Empty;
	}

	//------------------------------------------------------------------------------------------------
	//! One log line per new reason why no episode begins
	protected void NoteWaiting(string reason)
	{
		if (reason == m_sWaiting)
			return;

		m_sWaiting = reason;
		m_Context.Log("no episode yet: " + reason);
	}
}
