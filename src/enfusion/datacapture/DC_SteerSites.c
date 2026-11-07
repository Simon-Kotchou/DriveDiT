// ============================================================================
// DC_SteerSites - runtime test-site search of the steering probes (DD-14b, CC-07 b/c, CC-08)
// ============================================================================
//
// CC-07 (b) needs >= 200 m of flat straight ground per zero-command segment, CC-07 (c) a cross-slope of >= 5% at 20
// and 40 km/h, CC-07 (d) and CC-08 room for circles and steering steps. The e01 test area (the Arland main runway) is
// flat but has no cross-slope, so the probes look for their sites at run time instead of hard-coding them:
//
//   1. Explicit candidates (start x z, heading) are evaluated first (the runway for the flat site);
//   2. if none passes, a grid over a configurable region: at each grid point the terrain gradient from four
//      BaseWorld.GetSurfaceY samples (central differences); a cross-slope strip is laid along the contour through
//      the point with the downhill on the configured side (and +-a heading spread), a flat strip in four headings;
//   3. TERRAIN. Every strip is sampled every step_m at lateral offsets: the two TYRE TRACKS (+-track_half_m) around
//      each of corridor_points positions spread over the driving corridor (+-corridor_half_m, the part of the area the
//      car may drive in; an odd count, so the centre tyre pair is always sampled, at most 2 x track_half_m apart, so
//      no tyre position of the corridor is farther than track_half_m from a sample), the centre line, and the area
//      edges (+-half_width_m, surrounding-terrain safety only). The
//      cross-slope that counts is the one between the two tyre tracks, at every corridor position and every station:
//      track cross = (y(p - h) - y(p + h)) / 2h, positive = the right side lower. Criteria (DC_SteerTerrainMath): mean
//      and windowed grade of the centre line; side x track cross within cross_min..cross_max at every corridor
//      position of every measure-zone station (flat: |track cross| <= cross_max); |track cross| and the edge-to-edge
//      cross <= cross_abs_max_all over the whole strip; the rms about a plane fit; every sample above min_surface_y.
//      A flat road between sloping shoulders therefore has a track cross of 0 and is refused as a cross-slope site;
//   4. OBSTACLES. The strips that pass are ranked by score; the best ones are swept with an oriented box of the car's
//      size (TraceOBB: half width box_half_width_m, from box_bottom_m to box_top_m above the surface; BaseWorld.
//      TraceMove, TraceFlags.ENTS, LayerMask EPhysicsLayerPresets.Vehicle, the ego and its children excluded) along
//      lines every sweep_line_step_m (< 2 x the half width, so neighbouring sweeps overlap) across the area widened
//      by the lateral stopping envelope, from the strip start to its end plus the longitudinal stopping envelope
//      (the car still brakes after it leaves the area; the probe sets both from its top speed). The first box of
//      every line is also checked for an initial overlap (TracePosition). A hit rejects the candidate, unless the hit
//      entity's class or prefab is listed as benign (the probe's m_sBenignColliders, empty by default): then that one
//      entity is excluded and the same sweep is traced again, so every remaining check still runs. The first clean
//      strip is the site.
// The work is spread over render frames with one budget per frame shared by every search of the probe
// (DC_SteerBudget): each GetSurfaceY takes one sample and each TracePosition / TraceMove, benign retries included,
// one trace, taken before the call; a strip's terrain and a candidate's sweeps continue in the next frame where they
// stopped (DC_SteerSweepPlan keeps the pending call).
// The chosen site, its measured slopes, its centre tyre-pair height profile (track_profile, for the same-ground check
// of a baseline), its definition as an explicit candidate (explicit = x z heading) and the search counts go into the
// run header (site.<kind>.*).
//
// BUILD-AGNOSTIC: GetSurfaceY, TraceMove, TracePosition, TraceOBB (Mins, Maxs, Mat), TraceParam (Start, End, Flags,
// LayerMask, ExcludeArray, TraceEnt), TraceFlags.ENTS, EPhysicsLayerPresets.Vehicle, IEntity.GetPrefabData and
// EntityPrefabData.GetPrefabName have the same declarations in 1.8.0.13 and 1.9.0.37 (Core/generated/World/BaseWorld.c
// :24, :229, :237; TraceOBB.c; TraceBox.c; TraceParam.c; TraceFlags.c; GameLib/gameLib.c:24; IEntity.c:181;
// EntityPrefabData.c:20).
// ============================================================================

//------------------------------------------------------------------------------------------------
//! What a site of one kind must meet (flat or cross); filled by the probe from its attributes
class DC_SteerSiteCriteria
{
	string m_sKind = "flat";			//!< flat | cross
	float m_fLengthM = 1000;
	float m_fHalfWidthM = 20;			//!< the area: edge samples at +-this, the run aborts outside it (minus the margin)
	float m_fCorridorHalfM = 17;		//!< the tyre-track check covers positions within +-this
	int m_iCorridorPoints = 9;			//!< corridor positions (odd, >= 1; spread evenly, the ends at +-m_fCorridorHalfM)
	float m_fTrackHalfM = 0.65;			//!< half the track width (S105 front track about 1.3 m)
	float m_fZoneFromM = 0;				//!< measure zone along the strip (cross criteria; flat: the whole strip)
	float m_fZoneToM = 1000;
	float m_fStepM = 5;
	float m_fGradeWindowM = 20;
	float m_fGradeMax = 0.005;			//!< |mean grade| (regression of the centre line)
	float m_fGradeWindowMax = 0.01;		//!< |grade| over any window inside the zone
	float m_fGradeWindowMaxAll = 0.02;	//!< |grade| over any window of the whole strip
	float m_fCrossMin = 0;				//!< cross: side x track cross at every zone station and corridor position >= this
	float m_fCrossMax = 0.01;			//!< cross: side x track cross <= this; flat: |track cross| <= this
	float m_fCrossAbsMaxAll = 0.02;		//!< |track cross| and |edge cross| at every station of the whole strip
	float m_fCrossTarget = 0;			//!< cross: score target of the zone mean
	float m_fRmsMax = 0.1;				//!< rms of all samples about the plane fit [m]
	float m_fMinSurfaceY = 3;			//!< every sample above this [m] (sea level 0)
	int m_iSide = 1;					//!< cross: +1 downhill to the right of the heading, -1 to the left
	// obstacle sweeps
	float m_fEnvelopeLatM = 0;			//!< lateral stopping envelope beyond the area [m]
	float m_fEnvelopeLongM = 0;			//!< longitudinal stopping envelope beyond the strip end [m]
	float m_fEnvelopeBackM = 0;			//!< stopping envelope before the strip start [m] (probes that may exit backwards)
	float m_fBoxHalfWidthM = 1.0;		//!< S105 about 1.6 m wide, plus a margin
	float m_fBoxHalfLengthM = 0.5;		//!< the box is swept along the line, its own length only overlaps the segments
	float m_fBoxBottomM = 0.25;			//!< above the surface: lower obstructions (kerbs) are not obstacles
	float m_fBoxTopM = 1.6;				//!< S105 about 1.4 m tall, plus a margin
	float m_fSweepLineStepM = 1.8;		//!< < 2 x m_fBoxHalfWidthM: neighbouring sweeps overlap
	float m_fSweepSegmentM = 20;
	ref array<string> m_aBenign = {};	//!< class or prefab names that are never obstacles (empty by default)

	//------------------------------------------------------------------------------------------------
	bool IsCross()
	{
		return m_sKind == "cross";
	}

	//------------------------------------------------------------------------------------------------
	//! Lateral offsets sampled per station: 0 = -half width, 1 = +half width, 2 = centre, then per corridor position k
	//! the left (3 + 2k) and right (4 + 2k) tyre track
	int OffsetCount()
	{
		return 3 + 2 * Math.MaxInt(1, m_iCorridorPoints);
	}

	//------------------------------------------------------------------------------------------------
	//! The lateral offset of sample index j (right positive)
	float OffsetAt(int j)
	{
		if (j == 0)
			return -m_fHalfWidthM;

		if (j == 1)
			return m_fHalfWidthM;

		if (j == 2)
			return 0;

		int points = Math.MaxInt(1, m_iCorridorPoints);
		int k = (j - 3) / 2;
		float position = 0;
		if (points > 1)
			position = -m_fCorridorHalfM + k * (2 * m_fCorridorHalfM / (points - 1));

		if ((j - 3) % 2 == 0)
			return position - m_fTrackHalfM;

		return position + m_fTrackHalfM;
	}

	//------------------------------------------------------------------------------------------------
	int StationCount()
	{
		if (m_fStepM <= 0)
			return 0;

		int intervals = Math.Floor(m_fLengthM / m_fStepM);
		return intervals + 1;
	}

	//------------------------------------------------------------------------------------------------
	//! GetSurfaceY calls of one strip's terrain evaluation
	int StripCost()
	{
		return StationCount() * OffsetCount();
	}

	//------------------------------------------------------------------------------------------------
	//! true for a hit entity whose class or prefab name is listed as benign
	bool IsBenign(IEntity entity)
	{
		if (!entity || m_aBenign.IsEmpty())
			return false;

		string className = entity.ClassName();
		if (m_aBenign.Find(className) >= 0)
			return true;

		EntityPrefabData prefab = entity.GetPrefabData();
		if (!prefab)
			return false;

		string prefabName = prefab.GetPrefabName();
		return m_aBenign.Find(prefabName) >= 0;
	}

	//------------------------------------------------------------------------------------------------
	//! <prefix>.crit.* lines
	void AppendHeaderLines(notnull array<string> lines, string prefix)
	{
		string p = prefix + ".crit.";
		lines.Insert(p + "length_m=" + m_fLengthM.ToString(-1, 1));
		lines.Insert(p + "half_width_m=" + m_fHalfWidthM.ToString(-1, 2));
		lines.Insert(p + "corridor_half_m=" + m_fCorridorHalfM.ToString(-1, 2));
		lines.Insert(p + "corridor_points=" + m_iCorridorPoints.ToString());
		lines.Insert(p + "track_half_m=" + m_fTrackHalfM.ToString(-1, 3));
		lines.Insert(p + "zone_from_m=" + m_fZoneFromM.ToString(-1, 1));
		lines.Insert(p + "zone_to_m=" + m_fZoneToM.ToString(-1, 1));
		lines.Insert(p + "step_m=" + m_fStepM.ToString(-1, 2));
		lines.Insert(p + "grade_window_m=" + m_fGradeWindowM.ToString(-1, 1));
		lines.Insert(p + "grade_max=" + m_fGradeMax.ToString(-1, 4));
		lines.Insert(p + "grade_window_max=" + m_fGradeWindowMax.ToString(-1, 4));
		lines.Insert(p + "grade_window_max_all=" + m_fGradeWindowMaxAll.ToString(-1, 4));
		lines.Insert(p + "cross_min=" + m_fCrossMin.ToString(-1, 4));
		lines.Insert(p + "cross_max=" + m_fCrossMax.ToString(-1, 4));
		lines.Insert(p + "cross_abs_max_all=" + m_fCrossAbsMaxAll.ToString(-1, 4));
		lines.Insert(p + "cross_target=" + m_fCrossTarget.ToString(-1, 4));
		lines.Insert(p + "rms_max_m=" + m_fRmsMax.ToString(-1, 3));
		lines.Insert(p + "min_surface_y=" + m_fMinSurfaceY.ToString(-1, 2));
		lines.Insert(p + "side=" + m_iSide.ToString());
		lines.Insert(p + "envelope_lat_m=" + m_fEnvelopeLatM.ToString(-1, 2));
		lines.Insert(p + "envelope_long_m=" + m_fEnvelopeLongM.ToString(-1, 2));
		lines.Insert(p + "envelope_back_m=" + m_fEnvelopeBackM.ToString(-1, 2));
		lines.Insert(p + "box_half_width_m=" + m_fBoxHalfWidthM.ToString(-1, 2));
		lines.Insert(p + "box_heights_m=" + m_fBoxBottomM.ToString(-1, 2) + "," + m_fBoxTopM.ToString(-1, 2));
		lines.Insert(p + "sweep_line_step_m=" + m_fSweepLineStepM.ToString(-1, 2));
		lines.Insert(p + "sweep_segment_m=" + m_fSweepSegmentM.ToString(-1, 1));
		string benign;
		foreach (int i, string name : m_aBenign)
		{
			if (i > 0)
				benign += ";";

			benign += name;
		}

		lines.Insert(p + "benign=" + DC_ProbeCsv.CleanField(benign));
		lines.Insert(p + "rule=track cross = (y(p - track_half) - y(p + track_half)) / (2 track_half) at corridor_points positions p over +-corridor_half (positive = right side lower); the edge samples at +-half_width only bound the surrounding terrain");
	}
}

//------------------------------------------------------------------------------------------------
//! One candidate strip: start, heading, length, half width and what its terrain and obstacle checks measured
class DC_SteerSite
{
	string m_sSource;					//!< explicit | grid
	vector m_vStart;					//!< x, z used
	float m_fHeadingDeg;
	float m_fLengthM;
	float m_fHalfWidthM;
	float m_fZoneFromM;
	float m_fZoneToM;
	int m_iSide = 1;

	float m_fGrade;						//!< centre-line regression slope, uphill along the heading positive
	float m_fGradeWindowMax;			//!< zone
	float m_fGradeWindowMaxAll;
	float m_fCrossMean;					//!< zone mean of the track cross (positive = the right side lower), unsigned
	float m_fCrossMin;					//!< zone min / max of side x track cross
	float m_fCrossMax;
	float m_fCrossAbsMaxAll;			//!< |track cross| over the whole strip
	float m_fWideCrossAbsMax;			//!< |edge-to-edge cross| over the whole strip
	float m_fRmsM;
	float m_fMinY;
	float m_fMaxY;
	int m_iStations;
	bool m_bTerrainOk;
	string m_sTerrainReason;
	float m_fScore;
	int m_iObstacleTraces;
	int m_iObstacleHits = -1;			//!< -1 = not checked
	int m_iBenignHits;
	string m_sObstacle;
	ref array<float> m_aTrackProfile = {};	//!< centre tyre pair heights per station: left, right, left, ...

	//------------------------------------------------------------------------------------------------
	void DC_SteerSite(string source, vector start, float headingDeg, notnull DC_SteerSiteCriteria criteria)
	{
		m_sSource = source;
		m_vStart = Vector(start[0], 0, start[2]);
		m_fHeadingDeg = NormalizeDeg(headingDeg);
		m_fLengthM = criteria.m_fLengthM;
		m_fHalfWidthM = criteria.m_fHalfWidthM;
		m_fZoneFromM = criteria.m_fZoneFromM;
		m_fZoneToM = criteria.m_fZoneToM;
		m_iSide = criteria.m_iSide;
	}

	//------------------------------------------------------------------------------------------------
	//! 0 <= deg < 360
	static float NormalizeDeg(float deg)
	{
		float value = Math.Mod(deg, 360);
		if (value < 0)
			value += 360;

		return value;
	}

	//------------------------------------------------------------------------------------------------
	//! The probe area of this site (DC_ProbeArea, shared with DC_ProbeSequences)
	DC_ProbeArea MakeArea(float halfWidthM)
	{
		DC_ProbeArea area = new DC_ProbeArea();
		area.Set(m_vStart, m_fHeadingDeg, m_fLengthM, halfWidthM);
		return area;
	}

	//------------------------------------------------------------------------------------------------
	//! <prefix>.* lines (the run header's site.<kind>.*)
	void AppendHeaderLines(notnull array<string> lines, string prefix)
	{
		lines.Insert(prefix + ".source=" + m_sSource);
		lines.Insert(prefix + ".start=" + DC_ProbeArea.FormatVector(m_vStart, 3));
		lines.Insert(prefix + ".heading_deg=" + m_fHeadingDeg.ToString(-1, 3));
		lines.Insert(prefix + ".length_m=" + m_fLengthM.ToString(-1, 1));
		lines.Insert(prefix + ".half_width_m=" + m_fHalfWidthM.ToString(-1, 2));
		lines.Insert(prefix + ".zone_from_m=" + m_fZoneFromM.ToString(-1, 1));
		lines.Insert(prefix + ".zone_to_m=" + m_fZoneToM.ToString(-1, 1));
		lines.Insert(prefix + ".side=" + m_iSide.ToString());
		lines.Insert(prefix + ".grade=" + m_fGrade.ToString(-1, 5));
		lines.Insert(prefix + ".grade_window_max=" + m_fGradeWindowMax.ToString(-1, 5));
		lines.Insert(prefix + ".grade_window_max_all=" + m_fGradeWindowMaxAll.ToString(-1, 5));
		lines.Insert(prefix + ".cross_mean=" + m_fCrossMean.ToString(-1, 5));
		lines.Insert(prefix + ".cross_min_signed=" + m_fCrossMin.ToString(-1, 5));
		lines.Insert(prefix + ".cross_max_signed=" + m_fCrossMax.ToString(-1, 5));
		lines.Insert(prefix + ".cross_abs_max_all=" + m_fCrossAbsMaxAll.ToString(-1, 5));
		lines.Insert(prefix + ".wide_cross_abs_max=" + m_fWideCrossAbsMax.ToString(-1, 5));
		lines.Insert(prefix + ".rms_m=" + m_fRmsM.ToString(-1, 4));
		lines.Insert(prefix + ".min_y=" + m_fMinY.ToString(-1, 3));
		lines.Insert(prefix + ".max_y=" + m_fMaxY.ToString(-1, 3));
		lines.Insert(prefix + ".stations=" + m_iStations.ToString());
		lines.Insert(prefix + ".score=" + m_fScore.ToString(-1, 4));
		lines.Insert(prefix + ".obstacle_traces=" + m_iObstacleTraces.ToString());
		lines.Insert(prefix + ".obstacle_hits=" + m_iObstacleHits.ToString());
		lines.Insert(prefix + ".benign_hits=" + m_iBenignHits.ToString());
		float startX = m_vStart[0];
		float startZ = m_vStart[2];
		lines.Insert(prefix + ".explicit=" + startX.ToString(-1, 3) + " " + startZ.ToString(-1, 3) + " " + m_fHeadingDeg.ToString(-1, 4));
		string profile;
		foreach (float height : m_aTrackProfile)
		{
			profile += height.ToString(-1, 3) + " ";
		}

		lines.Insert(prefix + ".track_profile=" + profile.Trim());
		lines.Insert(prefix + ".cross_rule=the tyre-track cross over the driving corridor (criteria line rule); *_signed = side x track cross over the measure zone");
	}

	//------------------------------------------------------------------------------------------------
	string DescribeSite()
	{
		string text = string.Format("%1 site start %2 heading %3 deg, %4 m x +-%5 m: grade %6, track cross mean %7 (zone %8..%9)", m_sSource, DC_ProbeArea.FormatVector(m_vStart, 1), m_fHeadingDeg.ToString(-1, 2), m_fLengthM.ToString(-1, 0), m_fHalfWidthM.ToString(-1, 1), m_fGrade.ToString(-1, 4), m_fCrossMean.ToString(-1, 4), m_fCrossMin.ToString(-1, 4), m_fCrossMax.ToString(-1, 4));
		return text + string.Format(", rms %1 m, y %2..%3, %4 sweeps, %5 benign hits", m_fRmsM.ToString(-1, 3), m_fMinY.ToString(-1, 1), m_fMaxY.ToString(-1, 1), m_iObstacleTraces, m_iBenignHits);
	}
}

//------------------------------------------------------------------------------------------------
//! The native-call budget of one render frame, shared by every search of the probe in that frame (the probe resets it
//! once per frame and hands the same object to each unfinished search in turn). Every GetSurfaceY of the terrain and
//! of a sweep's end points takes one sample, every TracePosition / TraceMove (also each benign retry) one trace,
//! taken before the call. Pure arithmetic: tests/enforce_subset.py runs it
class DC_SteerBudget
{
	int m_iSamples;
	int m_iTraces;
	int m_iSamplesUsed;
	int m_iTracesUsed;

	//------------------------------------------------------------------------------------------------
	void Reset(int samples, int traces)
	{
		m_iSamples = samples;
		m_iTraces = traces;
		m_iSamplesUsed = 0;
		m_iTracesUsed = 0;
	}

	//------------------------------------------------------------------------------------------------
	//! true (and charged) when count samples are left in this frame
	bool TakeSamples(int count)
	{
		if (count > m_iSamples)
			return false;

		m_iSamples -= count;
		m_iSamplesUsed += count;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! true (and charged) when one trace is left in this frame
	bool TakeTrace()
	{
		if (m_iTraces < 1)
			return false;

		m_iTraces -= 1;
		m_iTracesUsed += 1;
		return true;
	}
}

//------------------------------------------------------------------------------------------------
//! The obstacle sweeps of one candidate as a sequence of single native calls (pure arithmetic: tests/enforce_subset.py
//! runs it with a mock tracer). Sweep i covers line i / segments, segment i % segments; segment 0 of every line first
//! checks the start box for an initial overlap (CALL_POSITION), then every sweep traces its move (CALL_MOVE). The
//! caller makes the pending call (charging one trace first) and reports it: a benign hit repeats the same call (the
//! entity is excluded by then), at most MAX_BENIGN_RETRIES times per sweep; a blocking hit ends the plan. Nothing
//! is lost when the frame's traces run out between two calls: the next frame continues with the pending one
class DC_SteerSweepPlan
{
	static const int CALL_POSITION = 1;
	static const int CALL_MOVE = 2;
	static const int RESULT_CLEAR = 0;
	static const int RESULT_BENIGN = 1;
	static const int RESULT_BLOCKED = 2;
	static const int MAX_BENIGN_RETRIES = 8;

	int m_iLines = 1;
	int m_iSegments = 1;
	int m_iSweep;
	int m_iCall = 1;
	int m_iRetries;
	int m_iCalls;
	int m_iBenign;
	bool m_bBlocked;
	bool m_bRetryCap;

	//------------------------------------------------------------------------------------------------
	void Reset(int lines, int segments)
	{
		m_iLines = lines;
		if (m_iLines < 1)
			m_iLines = 1;

		m_iSegments = segments;
		if (m_iSegments < 1)
			m_iSegments = 1;

		m_iSweep = 0;
		m_iCall = CALL_POSITION;
		m_iRetries = 0;
		m_iCalls = 0;
		m_iBenign = 0;
		m_bBlocked = false;
		m_bRetryCap = false;
	}

	//------------------------------------------------------------------------------------------------
	int GetTotal()
	{
		return m_iLines * m_iSegments;
	}

	//------------------------------------------------------------------------------------------------
	bool IsDone()
	{
		return m_bBlocked || m_bRetryCap || m_iSweep >= GetTotal();
	}

	//------------------------------------------------------------------------------------------------
	//! Every sweep traced without a blocking hit
	bool IsClear()
	{
		return !m_bBlocked && !m_bRetryCap && m_iSweep >= GetTotal();
	}

	//------------------------------------------------------------------------------------------------
	int GetLine()
	{
		return m_iSweep / m_iSegments;
	}

	//------------------------------------------------------------------------------------------------
	int GetSegment()
	{
		return m_iSweep % m_iSegments;
	}

	//------------------------------------------------------------------------------------------------
	//! The pending native call (CALL_POSITION or CALL_MOVE)
	int GetCall()
	{
		return m_iCall;
	}

	//------------------------------------------------------------------------------------------------
	//! The pending call was made: RESULT_CLEAR, RESULT_BENIGN or RESULT_BLOCKED
	void Report(int result)
	{
		m_iCalls += 1;
		if (result == RESULT_BLOCKED)
		{
			m_bBlocked = true;
			return;
		}

		if (result == RESULT_BENIGN)
		{
			m_iBenign += 1;
			m_iRetries += 1;
			if (m_iRetries > MAX_BENIGN_RETRIES)
				m_bRetryCap = true;

			return;
		}

		if (m_iCall == CALL_POSITION)
		{
			m_iCall = CALL_MOVE;
			return;
		}

		m_iSweep += 1;
		m_iRetries = 0;
		m_iCall = CALL_MOVE;
		if (m_iSweep < GetTotal() && GetSegment() == 0)
			m_iCall = CALL_POSITION;
	}
}

//------------------------------------------------------------------------------------------------
//! The search for one site kind: explicit candidates, then the grid; the terrain of a strip is sampled across as many
//! frames as it needs, the passing strips are ranked and swept for obstacles one native call at a time
class DC_SteerSiteSearch
{
	static const int STAGE_EXPLICIT = 0;
	static const int STAGE_GRID = 1;
	static const int STAGE_OBSTACLES = 2;
	static const int STAGE_DONE = 3;
	//! Strips kept for the obstacle stage (the best by score)
	static const int MAX_RANKED = 64;

	protected ref DC_SteerSiteCriteria m_Criteria;
	protected ref array<ref DC_SteerSite> m_aExplicit = {};
	protected ref array<ref DC_SteerSite> m_aPending = {};
	protected ref array<ref DC_SteerSite> m_aRanked = {};
	protected ref array<IEntity> m_aExcludes = {};
	protected ref DC_SteerSite m_Selected;
	protected ref TraceOBB m_Box;
	protected ref DC_SteerSweepPlan m_Plan = new DC_SteerSweepPlan();

	protected bool m_bGrid;
	protected bool m_bGridDone;
	protected float m_fGridMinX;
	protected float m_fGridMinZ;
	protected float m_fGridSpacingM;
	protected float m_fGradientHalfM = 15;
	protected float m_fHeadingSpreadDeg = 10;
	protected int m_iGridNx;
	protected int m_iGridNz;
	protected int m_iGridCursor;
	protected int m_iMaxStrips = 3000;
	protected int m_iMaxObstacleChecks = 48;
	protected int m_iStripCost;

	protected int m_iStage = STAGE_EXPLICIT;
	protected int m_iExplicitCursor;

	// the strip whose terrain is being sampled (across frames)
	protected ref DC_SteerSite m_Sampling;
	protected ref DC_ProbeArea m_SamplingStrip;
	protected ref array<float> m_aAlong = {};
	protected ref array<float> m_aLateral = {};
	protected ref array<float> m_aHeights = {};
	protected int m_iSampleIndex;
	protected bool m_bSamplingExplicit;

	// the candidate being swept
	protected bool m_bPlanReady;
	protected int m_iCachedSweep = -1;
	protected float m_fSweepCorridor;
	protected float m_fSweepFrom;
	protected float m_fSweepSegmentLength;
	protected float m_fSweepLateral;
	protected float m_fSweepA0;
	protected float m_fSweepA1;
	protected vector m_vSweepStart;
	protected vector m_vSweepEnd;

	// Counters (site.<kind>.search.*)
	protected int m_iGridPoints;
	protected int m_iGradientOk;
	protected int m_iStrips;
	protected int m_iTerrainOk;
	protected int m_iObstacleChecked;
	protected int m_iObstacleRejected;
	protected int m_iSamples;
	protected int m_iTraces;
	protected int m_iCalls;
	protected int m_iRankedDropped;
	protected int m_iMaxCallSamples;		//!< the most GetSurfaceY calls of one StepSearch call (<= the frame budget)
	protected int m_iMaxCallTraces;			//!< the most trace calls of one StepSearch call (<= the frame budget)
	protected string m_sFirstReject;		//!< first terrain refusal reason of an explicit candidate
	protected string m_sFirstObstacle;		//!< first blocking obstacle

	//------------------------------------------------------------------------------------------------
	//! excludes: the entities the obstacle sweeps ignore (the ego and its children)
	void Configure(notnull DC_SteerSiteCriteria criteria, notnull array<IEntity> excludes, int maxStrips, int maxObstacleChecks)
	{
		m_Criteria = criteria;
		m_aExcludes = {};
		foreach (IEntity entity : excludes)
		{
			m_aExcludes.Insert(entity);
		}

		m_iMaxStrips = maxStrips;
		m_iMaxObstacleChecks = maxObstacleChecks;
		m_iStripCost = criteria.StripCost();
		float halfHeight = (criteria.m_fBoxTopM - criteria.m_fBoxBottomM) * 0.5;
		m_Box = new TraceOBB();
		m_Box.Flags = TraceFlags.ENTS;
		m_Box.LayerMask = EPhysicsLayerPresets.Vehicle;
		m_Box.ExcludeArray = m_aExcludes;
		m_Box.Mins = Vector(-criteria.m_fBoxHalfWidthM, -halfHeight, -criteria.m_fBoxHalfLengthM);
		m_Box.Maxs = Vector(criteria.m_fBoxHalfWidthM, halfHeight, criteria.m_fBoxHalfLengthM);
	}

	//------------------------------------------------------------------------------------------------
	//! GetSurfaceY calls of one strip (sampled across frames when it exceeds a frame's budget)
	int GetStripCost()
	{
		return m_iStripCost;
	}

	//------------------------------------------------------------------------------------------------
	void AddExplicit(vector start, float headingDeg)
	{
		m_aExplicit.Insert(new DC_SteerSite("explicit", start, headingDeg, m_Criteria));
	}

	//------------------------------------------------------------------------------------------------
	//! The grid fallback over [minX, maxX] x [minZ, maxZ] every spacingM; gradientHalfM: half distance of the central
	//! differences; headingSpreadDeg: the cross strips are also tried at +- this from the contour
	void SetGrid(float minX, float minZ, float maxX, float maxZ, float spacingM, float gradientHalfM, float headingSpreadDeg)
	{
		if (spacingM <= 0 || maxX <= minX || maxZ <= minZ)
			return;

		m_bGrid = true;
		m_fGridMinX = minX;
		m_fGridMinZ = minZ;
		m_fGridSpacingM = spacingM;
		m_fGradientHalfM = gradientHalfM;
		m_fHeadingSpreadDeg = headingSpreadDeg;
		int cellsX = Math.Floor((maxX - minX) / spacingM);
		int cellsZ = Math.Floor((maxZ - minZ) / spacingM);
		m_iGridNx = cellsX + 1;
		m_iGridNz = cellsZ + 1;
	}

	//------------------------------------------------------------------------------------------------
	//! One slice of work within the frame's shared budget; true once the search is over (GetSelected)
	bool StepSearch(notnull BaseWorld world, notnull DC_SteerBudget budget)
	{
		if (m_iStage == STAGE_DONE)
			return true;

		m_iCalls++;
		int samplesBefore = budget.m_iSamplesUsed;
		int tracesBefore = budget.m_iTracesUsed;
		StepStages(world, budget);
		m_iMaxCallSamples = Math.MaxInt(m_iMaxCallSamples, budget.m_iSamplesUsed - samplesBefore);
		m_iMaxCallTraces = Math.MaxInt(m_iMaxCallTraces, budget.m_iTracesUsed - tracesBefore);
		return m_iStage == STAGE_DONE;
	}

	//------------------------------------------------------------------------------------------------
	protected void StepStages(notnull BaseWorld world, notnull DC_SteerBudget budget)
	{
		if (m_iStage == STAGE_EXPLICIT)
		{
			while (true)
			{
				if (m_Sampling)
				{
					if (!ContinueSampling(world, budget))
						return;

					continue;
				}

				if (m_iExplicitCursor >= m_aExplicit.Count())
					break;

				StartSampling(m_aExplicit[m_iExplicitCursor], true);
				m_iExplicitCursor++;
			}

			if (!m_aRanked.IsEmpty())
				m_iStage = STAGE_OBSTACLES;
			else if (m_bGrid)
				m_iStage = STAGE_GRID;
			else
				m_iStage = STAGE_DONE;

			return;
		}

		if (m_iStage == STAGE_GRID)
		{
			StepGrid(world, budget);
			if (!m_bGridDone)
				return;

			if (m_aRanked.IsEmpty())
				m_iStage = STAGE_DONE;
			else
				m_iStage = STAGE_OBSTACLES;

			return;
		}

		// STAGE_OBSTACLES: the best remaining candidate, swept one native call at a time
		StepObstacles(world, budget);
	}

	//------------------------------------------------------------------------------------------------
	//! Grid points (4 samples each) and the generated strips' terrain until the frame's samples run out
	protected void StepGrid(notnull BaseWorld world, notnull DC_SteerBudget budget)
	{
		while (true)
		{
			if (m_Sampling)
			{
				if (!ContinueSampling(world, budget))
					return;

				continue;
			}

			if (!m_aPending.IsEmpty())
			{
				DC_SteerSite site = m_aPending[0];
				m_aPending.RemoveOrdered(0);
				StartSampling(site, false);
				continue;
			}

			if (m_iGridCursor >= m_iGridNx * m_iGridNz || m_iStrips >= m_iMaxStrips)
			{
				m_bGridDone = true;
				return;
			}

			if (!budget.TakeSamples(4))
				return;

			int ix = m_iGridCursor % m_iGridNx;
			int iz = m_iGridCursor / m_iGridNx;
			m_iGridCursor++;
			m_iGridPoints++;
			float x = m_fGridMinX + ix * m_fGridSpacingM;
			float z = m_fGridMinZ + iz * m_fGridSpacingM;
			GenerateStrips(world, x, z);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! Central-difference gradient at (x, z) (4 GetSurfaceY calls, charged by the caller) and the strips centred
	//! there that it allows (pending)
	protected void GenerateStrips(notnull BaseWorld world, float x, float z)
	{
		float d = m_fGradientHalfM;
		float gx = (world.GetSurfaceY(x + d, z) - world.GetSurfaceY(x - d, z)) / (2 * d);
		float gz = (world.GetSurfaceY(x, z + d) - world.GetSurfaceY(x, z - d)) / (2 * d);
		m_iSamples += 4;
		float g = Math.Sqrt(gx * gx + gz * gz);
		if (m_Criteria.IsCross())
		{
			if (g < 0.8 * m_Criteria.m_fCrossMin || g > 1.15 * m_Criteria.m_fCrossMax || g <= 0)
				return;

			m_iGradientOk++;
			// downhill (-gradient) on the side of the heading: right(h) = (cos h, -sin h) = -side x gradient / g
			float side = m_Criteria.m_iSide;
			float heading = Math.Atan2(side * gz / g, -side * gx / g) * Math.RAD2DEG;
			AddCentredStrip(x, z, heading);
			if (m_fHeadingSpreadDeg > 0)
			{
				AddCentredStrip(x, z, heading + m_fHeadingSpreadDeg);
				AddCentredStrip(x, z, heading - m_fHeadingSpreadDeg);
			}

			return;
		}

		if (g > m_Criteria.m_fCrossMax)
			return;

		m_iGradientOk++;
		AddCentredStrip(x, z, 0);
		AddCentredStrip(x, z, 45);
		AddCentredStrip(x, z, 90);
		AddCentredStrip(x, z, 135);
	}

	//------------------------------------------------------------------------------------------------
	protected void AddCentredStrip(float x, float z, float headingDeg)
	{
		if (m_iStrips >= m_iMaxStrips)
			return;

		float rad = headingDeg * Math.DEG2RAD;
		float half = m_Criteria.m_fLengthM * 0.5;
		vector start = Vector(x - Math.Sin(rad) * half, 0, z - Math.Cos(rad) * half);
		m_aPending.Insert(new DC_SteerSite("grid", start, headingDeg, m_Criteria));
		m_iStrips++;
	}

	//------------------------------------------------------------------------------------------------
	//! Starts the terrain sampling of a strip at every offset of the criteria (DC_SteerSiteCriteria.OffsetAt)
	protected void StartSampling(notnull DC_SteerSite site, bool isExplicit)
	{
		DC_SteerSiteCriteria c = m_Criteria;
		m_Sampling = site;
		m_bSamplingExplicit = isExplicit;
		m_SamplingStrip = site.MakeArea(site.m_fHalfWidthM);
		m_iSampleIndex = 0;
		m_aAlong = {};
		m_aLateral = {};
		m_aHeights = {};
		int offsets = c.OffsetCount();
		for (int j = 0; j < offsets; j++)
		{
			m_aLateral.Insert(c.OffsetAt(j));
		}

		int stations = c.StationCount();
		for (int i = 0; i < stations; i++)
		{
			m_aAlong.Insert(i * c.m_fStepM);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! Samples the current strip one GetSurfaceY at a time while the frame has samples; when it is complete, applies the
	//! criteria (DC_SteerTerrainMath) and ranks it. true when the strip is finished, false when the samples ran out
	protected bool ContinueSampling(notnull BaseWorld world, notnull DC_SteerBudget budget)
	{
		int offsets = m_aLateral.Count();
		int total = m_aAlong.Count() * offsets;
		while (m_iSampleIndex < total)
		{
			if (!budget.TakeSamples(1))
				return false;

			int station = m_iSampleIndex / offsets;
			int offset = m_iSampleIndex % offsets;
			vector point = m_SamplingStrip.PointAt(m_aAlong[station], m_aLateral[offset]);
			m_aHeights.Insert(world.GetSurfaceY(point[0], point[2]));
			m_iSampleIndex++;
			m_iSamples++;
		}

		DC_SteerSite site = m_Sampling;
		m_Sampling = null;
		site.m_iStations = m_aAlong.Count();
		DC_SteerTerrainMath.Apply(site, m_Criteria, m_aAlong, m_aLateral, m_aHeights);
		if (site.m_bTerrainOk)
		{
			m_iTerrainOk++;
			AddRanked(site);
		}
		else if (m_bSamplingExplicit && m_sFirstReject.IsEmpty())
		{
			m_sFirstReject = site.m_sTerrainReason;
		}

		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Keeps m_aRanked sorted by score (ascending), at most MAX_RANKED
	protected void AddRanked(notnull DC_SteerSite site)
	{
		int index = 0;
		while (index < m_aRanked.Count() && m_aRanked[index].m_fScore <= site.m_fScore)
		{
			index++;
		}

		if (index >= MAX_RANKED)
		{
			m_iRankedDropped++;
			return;
		}

		if (index >= m_aRanked.Count())
			m_aRanked.Insert(site);
		else
			m_aRanked.InsertAt(site, index);

		while (m_aRanked.Count() > MAX_RANKED)
		{
			m_aRanked.Remove(m_aRanked.Count() - 1);
			m_iRankedDropped++;
		}
	}

	//------------------------------------------------------------------------------------------------
	//! The best ranked candidate's obstacle corridor, one native call at a time (DC_SteerSweepPlan): the area widened by
	//! the lateral envelope, from the backward envelope before the start to the longitudinal envelope past the end. A
	//! blocking hit rejects the candidate, a clean one is the site. When the ranked list runs out: the grid next (if
	//! not run yet), else no site
	protected void StepObstacles(notnull BaseWorld world, notnull DC_SteerBudget budget)
	{
		DC_SteerSiteCriteria c = m_Criteria;
		while (true)
		{
			if (!m_bPlanReady)
			{
				bool capped = m_iObstacleChecked >= m_iMaxObstacleChecks;
				if (m_aRanked.IsEmpty() || capped)
				{
					if (m_bGrid && !m_bGridDone && !capped)
					{
						m_iStage = STAGE_GRID;
						return;
					}

					m_iStage = STAGE_DONE;
					return;
				}

				DC_SteerSite next = m_aRanked[0];
				m_fSweepCorridor = next.m_fHalfWidthM + c.m_fEnvelopeLatM;
				m_fSweepFrom = -c.m_fEnvelopeBackM;
				float sweepLength = next.m_fLengthM + c.m_fEnvelopeLongM + c.m_fEnvelopeBackM;
				int lineGaps = Math.Ceil(2 * m_fSweepCorridor / c.m_fSweepLineStepM);
				int segments = Math.Ceil(sweepLength / c.m_fSweepSegmentM);
				if (segments < 1)
					segments = 1;

				m_fSweepSegmentLength = sweepLength / segments;
				m_Plan.Reset(lineGaps + 1, segments);
				m_iCachedSweep = -1;
				m_bPlanReady = true;
				m_iObstacleChecked++;
				next.m_iObstacleHits = 0;
				next.m_iObstacleTraces = 0;
				next.m_iBenignHits = 0;
			}

			DC_SteerSite site = m_aRanked[0];
			if (m_Plan.IsDone())
			{
				m_bPlanReady = false;
				site.m_iObstacleTraces = m_Plan.m_iCalls;
				if (m_Plan.IsClear())
				{
					m_Selected = site;
					m_iStage = STAGE_DONE;
					return;
				}

				if (m_Plan.m_bRetryCap)
				{
					int retries = DC_SteerSweepPlan.MAX_BENIGN_RETRIES;
					site.m_sObstacle = "more than " + retries.ToString() + " benign hits on one sweep";
				}

				site.m_iObstacleHits = 1;
				m_iObstacleRejected++;
				if (m_sFirstObstacle.IsEmpty())
					m_sFirstObstacle = site.m_sObstacle;

				m_aRanked.RemoveOrdered(0);
				continue;
			}

			int sweep = m_Plan.m_iSweep;
			if (sweep != m_iCachedSweep)
			{
				if (!budget.TakeSamples(2))
					return;

				SweepEnds(world, site);
				m_iCachedSweep = sweep;
			}

			if (!budget.TakeTrace())
				return;

			m_Plan.Report(TraceOnce(world, site));
		}
	}

	//------------------------------------------------------------------------------------------------
	//! The end points of the plan's current sweep, the box above the surface (2 GetSurfaceY calls, charged by the
	//! caller), and the box orientation
	protected void SweepEnds(notnull BaseWorld world, notnull DC_SteerSite site)
	{
		DC_SteerSiteCriteria c = m_Criteria;
		DC_ProbeArea strip = site.MakeArea(site.m_fHalfWidthM);
		int line = m_Plan.GetLine();
		int segment = m_Plan.GetSegment();
		m_fSweepLateral = Math.Min(-m_fSweepCorridor + line * c.m_fSweepLineStepM, m_fSweepCorridor);
		m_fSweepA0 = m_fSweepFrom + segment * m_fSweepSegmentLength;
		m_fSweepA1 = m_fSweepA0 + m_fSweepSegmentLength;
		float lift = c.m_fBoxBottomM + (c.m_fBoxTopM - c.m_fBoxBottomM) * 0.5;
		vector q0 = strip.PointAt(m_fSweepA0, m_fSweepLateral);
		vector q1 = strip.PointAt(m_fSweepA1, m_fSweepLateral);
		m_vSweepStart = Vector(q0[0], world.GetSurfaceY(q0[0], q0[2]) + lift, q0[2]);
		m_vSweepEnd = Vector(q1[0], world.GetSurfaceY(q1[0], q1[2]) + lift, q1[2]);
		m_iSamples += 2;
		m_Box.Mat[0] = strip.GetRight();
		m_Box.Mat[1] = Vector(0, 1, 0);
		m_Box.Mat[2] = strip.GetDirection();
	}

	//------------------------------------------------------------------------------------------------
	//! The plan's pending native call (one trace, charged by the caller): DC_SteerSweepPlan.RESULT_*; a benign hit is
	//! excluded for the retry, a blocking one described on the site
	protected int TraceOnce(notnull BaseWorld world, notnull DC_SteerSite site)
	{
		m_iTraces++;
		m_Box.Start = m_vSweepStart;
		m_Box.End = m_vSweepEnd;
		m_Box.TraceEnt = null;
		bool blocked = false;
		float fraction = 0;
		if (m_Plan.GetCall() == DC_SteerSweepPlan.CALL_POSITION)
		{
			blocked = world.TracePosition(m_Box, null) < 0;
		}
		else
		{
			fraction = world.TraceMove(m_Box, null);
			blocked = fraction < 1;
		}

		if (!blocked)
			return DC_SteerSweepPlan.RESULT_CLEAR;

		IEntity hit = m_Box.TraceEnt;
		if (hit && m_Criteria.IsBenign(hit))
		{
			site.m_iBenignHits++;
			m_aExcludes.Insert(hit);
			m_Box.ExcludeArray = m_aExcludes;
			return DC_SteerSweepPlan.RESULT_BENIGN;
		}

		string what = "unknown";
		if (hit)
			what = hit.ClassName() + "@" + DC_ProbeArea.FormatVector(hit.GetOrigin(), 1);

		float hitAlong = m_fSweepA0 + fraction * (m_fSweepA1 - m_fSweepA0);
		site.m_sObstacle = string.Format("%1 at %2 m along, %3 m lateral (site %4)", what, hitAlong.ToString(-1, 1), m_fSweepLateral.ToString(-1, 1), DC_ProbeArea.FormatVector(site.m_vStart, 1));
		return DC_SteerSweepPlan.RESULT_BLOCKED;
	}

	//------------------------------------------------------------------------------------------------
	//! null when no candidate passed
	DC_SteerSite GetSelected()
	{
		return m_Selected;
	}

	//------------------------------------------------------------------------------------------------
	bool IsDone()
	{
		return m_iStage == STAGE_DONE;
	}

	//------------------------------------------------------------------------------------------------
	//! <prefix>.found and the site lines, the criteria and the search counts
	void AppendHeaderLines(notnull array<string> lines, string prefix)
	{
		lines.Insert(prefix + ".kind=" + m_Criteria.m_sKind);
		lines.Insert(prefix + ".found=" + DC_LaunchGuard.BoolText(m_Selected != null));
		if (m_Selected)
			m_Selected.AppendHeaderLines(lines, prefix);

		m_Criteria.AppendHeaderLines(lines, prefix);
		string s = prefix + ".search.";
		int explicitCount = m_aExplicit.Count();
		lines.Insert(s + "explicit=" + explicitCount.ToString());
		lines.Insert(s + "grid=" + DC_LaunchGuard.BoolText(m_bGrid));
		lines.Insert(s + "grid_points=" + m_iGridPoints.ToString());
		lines.Insert(s + "gradient_ok=" + m_iGradientOk.ToString());
		lines.Insert(s + "strips=" + m_iStrips.ToString());
		lines.Insert(s + "strip_cost=" + m_iStripCost.ToString());
		lines.Insert(s + "max_call_samples=" + m_iMaxCallSamples.ToString());
		lines.Insert(s + "max_call_traces=" + m_iMaxCallTraces.ToString());
		lines.Insert(s + "terrain_ok=" + m_iTerrainOk.ToString());
		lines.Insert(s + "ranked_dropped=" + m_iRankedDropped.ToString());
		lines.Insert(s + "obstacle_checked=" + m_iObstacleChecked.ToString());
		lines.Insert(s + "obstacle_rejected=" + m_iObstacleRejected.ToString());
		lines.Insert(s + "samples=" + m_iSamples.ToString());
		lines.Insert(s + "traces=" + m_iTraces.ToString());
		lines.Insert(s + "calls=" + m_iCalls.ToString());
		lines.Insert(s + "first_terrain_reject=" + DC_ProbeCsv.CleanField(m_sFirstReject));
		lines.Insert(s + "first_obstacle=" + DC_ProbeCsv.CleanField(m_sFirstObstacle));
	}

	//------------------------------------------------------------------------------------------------
	string DescribeSearch()
	{
		string found = "no site";
		if (m_Selected)
			found = m_Selected.DescribeSite();

		int explicitCount = m_aExplicit.Count();
		return string.Format("%1 search: %2 explicit, %3 grid points (%4 with a usable gradient), %5 strips, %6 terrain ok, %7 obstacle-checked (%8 rejected): %9", m_Criteria.m_sKind, explicitCount, m_iGridPoints, m_iGradientOk, m_iStrips, m_iTerrainOk, m_iObstacleChecked, m_iObstacleRejected, found);
	}
}

//------------------------------------------------------------------------------------------------
//! The terrain criteria of one strip from its samples (pure arithmetic on the sample arrays, no engine call; tests
//! run it through tests/enforce_subset.py). along[i] = i x step; lateral[j] = the offset of sample j
//! (DC_SteerSiteCriteria.OffsetAt: 0 / 1 the area edges, 2 the centre, 3 + 2k / 4 + 2k the tyre tracks of corridor
//! position k); heights[i x lateral.Count() + j] = the surface y of station i at offset j
class DC_SteerTerrainMath
{
	//------------------------------------------------------------------------------------------------
	//! Fills site.m_fGrade .. m_fRmsM, m_fMinY / m_fMaxY, m_bTerrainOk, m_sTerrainReason and m_fScore
	static void Apply(DC_SteerSite site, DC_SteerSiteCriteria c, array<float> along, array<float> lateral, array<float> heights)
	{
		int n = along.Count();
		int m = lateral.Count();
		int points = (m - 3) / 2;
		site.m_bTerrainOk = false;
		site.m_fScore = 1000000;
		if (n < 3 || m < 5 || heights.Count() != n * m || c.m_fStepM <= 0 || c.m_fTrackHalfM <= 0)
		{
			site.m_sTerrainReason = "too_short";
			return;
		}

		// centre-line regression slope; extremes over every sample
		float sumA = 0;
		float sumY = 0;
		float minY = heights[0];
		float maxY = heights[0];
		for (int i = 0; i < n; i++)
		{
			sumA += along[i];
			sumY += heights[i * m + 2];
		}

		for (int s = 0; s < n * m; s++)
		{
			minY = Math.Min(minY, heights[s]);
			maxY = Math.Max(maxY, heights[s]);
		}

		float meanA = sumA / n;
		float meanY = sumY / n;
		float sxy = 0;
		float sxx = 0;
		for (int j = 0; j < n; j++)
		{
			float da = along[j] - meanA;
			sxy += da * (heights[j * m + 2] - meanY);
			sxx += da * da;
		}

		float grade = 0;
		if (sxx > 0)
			grade = sxy / sxx;

		// tyre-track cross-slope at every corridor position and station; the edge-to-edge cross for safety
		float side = 1;
		if (c.IsCross())
			side = site.m_iSide;

		float halfTrack = c.m_fTrackHalfM;
		float halfWidth = (lateral[1] - lateral[0]) * 0.5;
		float crossSum = 0;
		int crossCount = 0;
		float crossMin = 0;
		float crossMax = 0;
		float crossAbsMaxAll = 0;
		float wideAbsMax = 0;
		for (int k = 0; k < n; k++)
		{
			if (halfWidth > 0)
				wideAbsMax = Math.Max(wideAbsMax, Math.AbsFloat((heights[k * m] - heights[k * m + 1]) / (2 * halfWidth)));

			bool inZone = along[k] >= site.m_fZoneFromM && along[k] <= site.m_fZoneToM;
			for (int p = 0; p < points; p++)
			{
				float cross = (heights[k * m + 3 + 2 * p] - heights[k * m + 4 + 2 * p]) / (2 * halfTrack);
				crossAbsMaxAll = Math.Max(crossAbsMaxAll, Math.AbsFloat(cross));
				if (inZone)
				{
					float signedCross = side * cross;
					if (crossCount == 0)
					{
						crossMin = signedCross;
						crossMax = signedCross;
					}
					else
					{
						crossMin = Math.Min(crossMin, signedCross);
						crossMax = Math.Max(crossMax, signedCross);
					}

					crossSum += cross;
					crossCount++;
				}
			}
		}

		float crossMean = 0;
		if (crossCount > 0)
			crossMean = crossSum / crossCount;

		// windowed grade of the centre line (zone and whole strip)
		int windowSteps = Math.Round(c.m_fGradeWindowM / c.m_fStepM);
		if (windowSteps < 1)
			windowSteps = 1;

		float gradeWindowMax = 0;
		float gradeWindowMaxAll = 0;
		for (int w = 0; w + windowSteps < n; w++)
		{
			float windowGrade = Math.AbsFloat((heights[(w + windowSteps) * m + 2] - heights[w * m + 2]) / (along[w + windowSteps] - along[w]));
			gradeWindowMaxAll = Math.Max(gradeWindowMaxAll, windowGrade);
			if (along[w] >= site.m_fZoneFromM && along[w + windowSteps] <= site.m_fZoneToM)
				gradeWindowMax = Math.Max(gradeWindowMax, windowGrade);
		}

		// rms of every sample about the plane y = meanY + grade (a - meanA) - crossMean x lateral (right positive)
		float sq = 0;
		for (int r = 0; r < n; r++)
		{
			float planeY = meanY + grade * (along[r] - meanA);
			for (int q = 0; q < m; q++)
			{
				float e = heights[r * m + q] - (planeY - crossMean * lateral[q]);
				sq += e * e;
			}
		}

		float rms = Math.Sqrt(sq / (n * m));
		// the centre tyre pair (corridor position (points - 1) / 2) at every station: the same-ground check of a baseline
		int centre = (points - 1) / 2;
		site.m_aTrackProfile.Clear();
		for (int t = 0; t < n; t++)
		{
			site.m_aTrackProfile.Insert(heights[t * m + 3 + 2 * centre]);
			site.m_aTrackProfile.Insert(heights[t * m + 4 + 2 * centre]);
		}

		site.m_fGrade = grade;
		site.m_fGradeWindowMax = gradeWindowMax;
		site.m_fGradeWindowMaxAll = gradeWindowMaxAll;
		site.m_fCrossMean = crossMean;
		site.m_fCrossMin = crossMin;
		site.m_fCrossMax = crossMax;
		site.m_fCrossAbsMaxAll = crossAbsMaxAll;
		site.m_fWideCrossAbsMax = wideAbsMax;
		site.m_fRmsM = rms;
		site.m_fMinY = minY;
		site.m_fMaxY = maxY;

		// criteria, first failure named
		if (crossCount < 2)
		{
			site.m_sTerrainReason = "zone_empty";
			return;
		}

		if (minY < c.m_fMinSurfaceY)
		{
			site.m_sTerrainReason = "below_min_surface_y";
			return;
		}

		if (Math.AbsFloat(grade) > c.m_fGradeMax)
		{
			site.m_sTerrainReason = "grade";
			return;
		}

		if (gradeWindowMax > c.m_fGradeWindowMax || gradeWindowMaxAll > c.m_fGradeWindowMaxAll)
		{
			site.m_sTerrainReason = "grade_window";
			return;
		}

		if (crossAbsMaxAll > c.m_fCrossAbsMaxAll || wideAbsMax > c.m_fCrossAbsMaxAll)
		{
			site.m_sTerrainReason = "cross_all";
			return;
		}

		if (c.IsCross())
		{
			if (crossMin < c.m_fCrossMin || crossMax > c.m_fCrossMax)
			{
				site.m_sTerrainReason = "cross_zone";
				return;
			}
		}
		else if (Math.Max(Math.AbsFloat(crossMin), Math.AbsFloat(crossMax)) > c.m_fCrossMax)
		{
			site.m_sTerrainReason = "cross_zone";
			return;
		}

		if (rms > c.m_fRmsMax)
		{
			site.m_sTerrainReason = "rms";
			return;
		}

		site.m_bTerrainOk = true;
		site.m_sTerrainReason = "ok";
		float score = Math.AbsFloat(grade) / Math.Max(c.m_fGradeMax, 0.0001) + rms / Math.Max(c.m_fRmsMax, 0.0001);
		if (c.IsCross())
		{
			float target = Math.Max(c.m_fCrossTarget, 0.0001);
			score += Math.AbsFloat(side * crossMean - c.m_fCrossTarget) / target + (crossMax - crossMin) / target;
		}
		else
		{
			score += crossAbsMaxAll / Math.Max(c.m_fCrossMax, 0.0001);
		}

		site.m_fScore = score;
	}
}
