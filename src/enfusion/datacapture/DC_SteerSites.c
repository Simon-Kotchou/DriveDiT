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
//      cross <= cross_abs_max_all over the whole strip; the rms of the measure zone's samples about its plane fit
//      (flat: the zone is the whole strip; the run-up and stop parts of a cross strip only meet the lax criteria);
//      every sample above min_surface_y. A flat road between sloping shoulders therefore has a track cross of 0 and
//      is refused as a cross-slope site. A grid strip first gets an exact pre-check: the centre tyre pair at every
//      measure-zone station against the zone's cross window (a subset of the full check, so it refuses nothing the
//      full check would accept; reason cross_zone_pre) and is sampled in full only when it passes. Every refused
//      candidate is counted by its first failed criterion (site.<kind>.search.rejects), the first refusal of any
//      candidate and of an explicit one are kept, and so is the near miss (the fully sampled refused candidate with
//      the largest share of zone tyre-pair samples inside the cross window), so a search without a site says why;
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
// The work is spread over RENDER frames (the probe's OnProbeFrame, EOnFrame) with one budget per frame shared by every
// search of the probe (DC_SteerBudget): each GetSurfaceY takes one sample and each TracePosition / TraceMove, benign
// retries included, one trace, taken before the call; a strip's terrain and a candidate's sweeps continue in the
// next frame where they stopped (DC_SteerSweepPlan keeps the pending call). A search therefore takes frames / fps of
// world time (the first 1.9.0.37 Play: 497 frames in 9.0 s, 55 fps at 6000 samples per frame), and the probe caps the
// world time of all its searches (DC_ProbeSteerBase m_fSiteSearchMaxS): a search it stops ends with status time_cap.
// The header (site.<kind>.search.*) says how a search ended: status done, time_cap or the probe's stop reason when
// the probe stopped while it searched (timeout, play_end, ...: its counts, refusals, near miss and the strip it was
// sampling are written then), the stage it was in, and grid_complete 1 only when every grid point was visited and
// the strip cap suppressed no heading (strips_capped).
// The chosen site, its measured slopes, its centre tyre-pair height profile (track_profile, for the same-ground check
// of a baseline), its definition as an explicit candidate (explicit = x z heading) and the search counts go into the
// run header (site.<kind>.*).
// SURVEY (DD-14f; DC_SteerSurvey, site.<kind>.survey.*, WriteSurvey): so that a search without a site says what a
// changed design would find, without another search. A strip the pre-check refuses goes on sampling its zone's centre
// tyre pairs while its longest run of consecutive in-window zone stations can still grow (the refusal itself is the
// same as before); the runs go into a histogram over every refused strip. Every fully sampled refusal lists all its
// failed criteria (DC_SteerSite.m_sFailed), not only the first. WriteSurvey writes, at most 10 rows each, the best near
// misses with every measured value, the terrain-ok strips an obstacle refused, the longest runs, and the histogram; the
// extra samples are at most the zone's centre pairs per refused strip, fewer once its run cannot grow (on the 1.9.0.37
// cross grids of the order of the search's own samples: seconds at 6000 per frame, far inside m_fSiteSearchMaxS).
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
	float m_fZoneShare;					//!< share of the zone tyre-pair samples inside the cross window (diagnostic)
	bool m_bTerrainOk;
	string m_sTerrainReason;
	string m_sFailed;					//!< every failed terrain criterion, '|'-joined in the order Apply checks them (survey)
	int m_iFailedCount;					//!< how many (the survey ranks near misses of equal zone share by fewer)
	int m_iPreRun = -1;					//!< cross_zone_pre: the longest run of zone stations whose centre tyre pair is in the window
	float m_fPreRunFromM;				//!< cross_zone_pre: where that run starts along the strip [m]
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
		lines.Insert(prefix + ".zone_share=" + m_fZoneShare.ToString(-1, 4));
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
	//! A refused candidate in one line (the search's near miss)
	string DescribeRefusal()
	{
		return string.Format("%1 start %2 heading %3: %4, zone share %5, side x cross %6..%7, grade %8, rms %9 m", m_sSource, DC_ProbeArea.FormatVector(m_vStart, 1), m_fHeadingDeg.ToString(-1, 2), m_sTerrainReason, m_fZoneShare.ToString(-1, 3), m_fCrossMin.ToString(-1, 4), m_fCrossMax.ToString(-1, 4), m_fGrade.ToString(-1, 4), m_fRmsM.ToString(-1, 3)) + string.Format(", grade windows %1 / %2, |cross| all %3, edge cross %4", m_fGradeWindowMax.ToString(-1, 4), m_fGradeWindowMaxAll.ToString(-1, 4), m_fCrossAbsMaxAll.ToString(-1, 4), m_fWideCrossAbsMax.ToString(-1, 4));
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
//! The site survey of one search (DD-14f; pure arithmetic: tests/enforce_subset.py runs it): for the strips the
//! pre-check refuses (cross_zone_pre), the longest run of consecutive zone stations whose centre tyre pair lies inside
//! the zone's cross window and a histogram of those runs over every refused strip, so a search without a site says
//! whether a shorter measure zone would exist (the centre pair only: a necessary condition, the full check samples
//! every corridor position); and where a refused or ranked item goes in a bounded top-TOP list
class DC_SteerSurvey
{
	static const int TOP = 10;				//!< rows per list in the survey CSV

	ref array<int> m_aRunHistogram = {};	//!< index = longest run in stations, value = refused strips
	int m_iPreRejected;
	int m_iLongestRun;						//!< the longest run of any refused strip
	// the strip being pre-checked
	int m_iRun;
	int m_iBestRun;
	int m_iBestRunFrom = -1;				//!< zone station index where the best run starts, -1 = none
	int m_iStation;

	//------------------------------------------------------------------------------------------------
	//! A new strip's pre-check
	void StartStrip()
	{
		m_iRun = 0;
		m_iBestRun = 0;
		m_iBestRunFrom = -1;
		m_iStation = 0;
	}

	//------------------------------------------------------------------------------------------------
	//! The next zone station of the strip: its centre tyre pair inside the window or not
	void AddStation(bool inWindow)
	{
		if (inWindow)
		{
			m_iRun++;
			if (m_iRun > m_iBestRun)
			{
				m_iBestRun = m_iRun;
				m_iBestRunFrom = m_iStation - m_iRun + 1;
			}
		}
		else
		{
			m_iRun = 0;
		}

		m_iStation++;
	}

	//------------------------------------------------------------------------------------------------
	//! true while the stations left could still make a longer run than the best so far (the pre-check of a refused
	//! strip stops sampling once they cannot: its longest run is then final)
	bool CanImprove(int stationsLeft)
	{
		return m_iRun + stationsLeft > m_iBestRun;
	}

	//------------------------------------------------------------------------------------------------
	//! The refused strip's longest run into the histogram
	void CountRefused()
	{
		while (m_aRunHistogram.Count() <= m_iBestRun)
		{
			m_aRunHistogram.Insert(0);
		}

		m_aRunHistogram[m_iBestRun] = m_aRunHistogram[m_iBestRun] + 1;
		m_iPreRejected++;
		m_iLongestRun = Math.MaxInt(m_iLongestRun, m_iBestRun);
	}

	//------------------------------------------------------------------------------------------------
	//! Index of key in keys sorted descending (ties after the earlier ones), -1 when a list of cap entries is full
	//! and key does not beat its last
	static int RankPosition(notnull array<float> keys, float key, int cap)
	{
		int index = 0;
		while (index < keys.Count() && keys[index] >= key)
		{
			index++;
		}

		if (index >= cap)
			return -1;

		return index;
	}

	//------------------------------------------------------------------------------------------------
	//! 'run:strips;...' of the non-empty histogram bins
	string HistogramText()
	{
		string text;
		for (int run = 0; run < m_aRunHistogram.Count(); run++)
		{
			int count = m_aRunHistogram[run];
			if (count > 0)
			{
				if (text != string.Empty)
					text += ";";

				text += run.ToString() + ":" + count.ToString();
			}
		}

		return text;
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

	//! Reason of a grid strip refused by the pre-check (the centre tyre pair of the zone)
	static const string REASON_PRE = "cross_zone_pre";

	// the strip whose terrain is being sampled (across frames)
	protected ref DC_SteerSite m_Sampling;
	protected ref DC_ProbeArea m_SamplingStrip;
	protected ref array<float> m_aAlong = {};
	protected ref array<float> m_aLateral = {};
	protected ref array<float> m_aHeights = {};
	protected int m_iSampleIndex;
	protected bool m_bSamplingExplicit;
	// the pre-check of a grid strip: the centre tyre pair at the zone stations, one sample at a time
	protected bool m_bPreCheck;
	protected bool m_bPreRejected;
	protected ref array<int> m_aPreStations = {};
	protected int m_iPreIndex;
	protected float m_fPreLeftY;

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
	protected string m_sFirstReject;		//!< first terrain refusal reason of any candidate (explicit or grid)
	protected string m_sFirstExplicitReject;	//!< first terrain refusal reason of an explicit candidate
	protected string m_sFirstObstacle;		//!< first blocking obstacle
	protected int m_iStripsCapped;			//!< headings the strip cap suppressed (grid_complete needs none)
	protected string m_sStopReason;			//!< why the search ended early (time_cap, the probe's stop reason); empty: it ran out
	protected ref array<string> m_aRejectReasons = {};	//!< every refusal reason seen, with its count
	protected ref array<int> m_aRejectCounts = {};
	protected ref DC_SteerSite m_NearMiss;	//!< the fully sampled refused candidate with the largest zone share
	// the survey (DD-14f): site.<kind>.survey.* and WriteSurvey; every list bounded to DC_SteerSurvey.TOP
	protected ref DC_SteerSurvey m_Survey = new DC_SteerSurvey();
	protected ref array<ref DC_SteerSite> m_aNearList = {};		//!< fully sampled refusals, the largest zone share first
	protected ref array<float> m_aNearKeys = {};
	protected ref array<ref DC_SteerSite> m_aRunList = {};		//!< cross_zone_pre refusals, the longest run first
	protected ref array<float> m_aRunKeys = {};
	protected ref array<ref DC_SteerSite> m_aObstacleList = {};	//!< terrain-ok strips refused by an obstacle, in sweep order
	protected int m_iZoneStations;				//!< zone stations of a grid strip's pre-check

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
		{
			m_iStripsCapped++;
			return;
		}

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

		// grid strips: the exact pre-check of the zone's centre tyre pair first (explicit candidates are sampled in
		// full at once, so their refusal names the first failed criterion with every metric)
		m_aPreStations = {};
		m_iPreIndex = 0;
		m_bPreRejected = false;
		if (!isExplicit)
		{
			foreach (int station, float along : m_aAlong)
			{
				if (along >= site.m_fZoneFromM && along <= site.m_fZoneToM)
					m_aPreStations.Insert(station);
			}
		}

		m_bPreCheck = !m_aPreStations.IsEmpty();
		m_iZoneStations = Math.MaxInt(m_iZoneStations, m_aPreStations.Count());
		m_Survey.StartStrip();
	}

	//------------------------------------------------------------------------------------------------
	//! The pre-check of the strip being sampled, one GetSurfaceY at a time while the frame has samples: the centre
	//! tyre pair (corridor position (points - 1) / 2, on the centre line) at every zone station against the zone's
	//! cross window (DC_SteerTerrainMath.ZoneCrossOk, the full check's own test). false when the samples ran out;
	//! true when it ended (m_bPreRejected when a pair lay outside the window). A refused strip samples on only while
	//! its longest run of in-window stations can still grow (the survey, DC_SteerSurvey; the refusal itself is decided
	//! at the first pair outside, as before), then its run goes on the site and into the histogram
	protected bool ContinuePreCheck(notnull BaseWorld world, notnull DC_SteerBudget budget)
	{
		DC_SteerSiteCriteria c = m_Criteria;
		int centre = (Math.MaxInt(1, c.m_iCorridorPoints) - 1) / 2;
		float leftLateral = c.OffsetAt(3 + 2 * centre);
		float rightLateral = c.OffsetAt(4 + 2 * centre);
		int total = 2 * m_aPreStations.Count();
		while (m_iPreIndex < total)
		{
			// the stations left, the pair in progress included
			int stationsLeft = (total - m_iPreIndex + 1) / 2;
			if (m_bPreRejected && !m_Survey.CanImprove(stationsLeft))
				break;

			if (!budget.TakeSamples(1))
				return false;

			int station = m_aPreStations[m_iPreIndex / 2];
			bool left = m_iPreIndex % 2 == 0;
			float lateral = rightLateral;
			if (left)
				lateral = leftLateral;

			vector point = m_SamplingStrip.PointAt(m_aAlong[station], lateral);
			float y = world.GetSurfaceY(point[0], point[2]);
			m_iSamples++;
			m_iPreIndex++;
			if (left)
			{
				m_fPreLeftY = y;
				continue;
			}

			float cross = (m_fPreLeftY - y) / (2 * c.m_fTrackHalfM);
			bool inWindow = DC_SteerTerrainMath.ZoneCrossOk(c, m_Sampling.m_iSide, cross);
			m_Survey.AddStation(inWindow);
			if (!inWindow)
				m_bPreRejected = true;
		}

		m_bPreCheck = false;
		if (m_bPreRejected)
		{
			m_Sampling.m_iPreRun = m_Survey.m_iBestRun;
			m_Sampling.m_fPreRunFromM = -1;
			if (m_Survey.m_iBestRunFrom >= 0)
				m_Sampling.m_fPreRunFromM = m_aAlong[m_aPreStations[m_Survey.m_iBestRunFrom]];

			m_Survey.CountRefused();
		}

		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! A refused candidate: its reason counted, the first refusal (of any candidate and of an explicit one) kept, and
	//! the near miss (fully sampled candidates only: the pre-check measures no zone share)
	protected void CountReject(notnull DC_SteerSite site, bool isExplicit)
	{
		string reason = site.m_sTerrainReason;
		int index = m_aRejectReasons.Find(reason);
		if (index < 0)
		{
			m_aRejectReasons.Insert(reason);
			m_aRejectCounts.Insert(1);
		}
		else
		{
			m_aRejectCounts[index] = m_aRejectCounts[index] + 1;
		}

		if (m_sFirstReject.IsEmpty())
			m_sFirstReject = reason;

		if (isExplicit && m_sFirstExplicitReject.IsEmpty())
			m_sFirstExplicitReject = reason;

		if (reason == REASON_PRE)
		{
			AddToList(m_aRunList, m_aRunKeys, site, site.m_iPreRun);
			return;
		}

		if (!m_NearMiss || site.m_fZoneShare > m_NearMiss.m_fZoneShare)
			m_NearMiss = site;

		// equal zone shares: the fewer failed criteria first (the shares step by 1 / zone samples, far above this)
		AddToList(m_aNearList, m_aNearKeys, site, site.m_fZoneShare - 0.0001 * site.m_iFailedCount);
	}

	//------------------------------------------------------------------------------------------------
	//! site into a survey list kept sorted by key (descending), at most DC_SteerSurvey.TOP entries
	protected void AddToList(notnull array<ref DC_SteerSite> list, notnull array<float> keys, notnull DC_SteerSite site, float key)
	{
		int index = DC_SteerSurvey.RankPosition(keys, key, DC_SteerSurvey.TOP);
		if (index < 0)
			return;

		if (index >= list.Count())
		{
			list.Insert(site);
			keys.Insert(key);
		}
		else
		{
			list.InsertAt(site, index);
			keys.InsertAt(key, index);
		}

		while (list.Count() > DC_SteerSurvey.TOP)
		{
			list.Remove(list.Count() - 1);
			keys.Remove(keys.Count() - 1);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! 'reason:count;...' in the order the reasons first appeared
	string DescribeRejects()
	{
		string text;
		foreach (int i, string reason : m_aRejectReasons)
		{
			if (i > 0)
				text += ";";

			int count = m_aRejectCounts[i];
			text += reason + ":" + count.ToString();
		}

		return text;
	}

	//------------------------------------------------------------------------------------------------
	//! Samples the current strip one GetSurfaceY at a time while the frame has samples; when it is complete, applies the
	//! criteria (DC_SteerTerrainMath) and ranks it. true when the strip is finished, false when the samples ran out
	protected bool ContinueSampling(notnull BaseWorld world, notnull DC_SteerBudget budget)
	{
		if (m_bPreCheck)
		{
			if (!ContinuePreCheck(world, budget))
				return false;

			if (m_bPreRejected)
			{
				DC_SteerSite refused = m_Sampling;
				m_Sampling = null;
				refused.m_bTerrainOk = false;
				refused.m_sTerrainReason = REASON_PRE;
				CountReject(refused, m_bSamplingExplicit);
				return true;
			}
		}

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
		else
		{
			CountReject(site, m_bSamplingExplicit);
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

				if (m_aObstacleList.Count() < DC_SteerSurvey.TOP)
					m_aObstacleList.Insert(site);

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
	//! Ends an unfinished search now with reason (time_cap: the probe's search time is used up). No candidate is
	//! selected that has not passed its obstacle sweeps; the counts so far stay for the header
	void StopSearch(string reason)
	{
		if (m_iStage == STAGE_DONE)
			return;

		m_sStopReason = reason;
		m_iStage = STAGE_DONE;
	}

	//------------------------------------------------------------------------------------------------
	//! The probe stopped while this search was unfinished (its stop reason): the header lines then say so
	void NoteProbeStop(string reason)
	{
		if (m_iStage == STAGE_DONE)
			return;

		m_sStopReason = reason;
	}

	//------------------------------------------------------------------------------------------------
	//! done (the search ran out), or why it ended or was interrupted
	string GetStatus()
	{
		if (!m_sStopReason.IsEmpty())
			return m_sStopReason;

		if (m_iStage == STAGE_DONE)
			return "done";

		return "running";
	}

	//------------------------------------------------------------------------------------------------
	static string StageName(int stage)
	{
		if (stage == STAGE_EXPLICIT)
			return "explicit";

		if (stage == STAGE_GRID)
			return "grid";

		if (stage == STAGE_OBSTACLES)
			return "obstacles";

		return "done";
	}

	//------------------------------------------------------------------------------------------------
	//! Every grid point visited, every strip generated (no heading suppressed by the cap), nothing pending, and the
	//! search not stopped early
	bool IsGridComplete()
	{
		int total = m_iGridNx * m_iGridNz;
		return m_bGrid && m_bGridDone && m_iGridCursor >= total && m_iStripsCapped == 0 && m_sStopReason.IsEmpty();
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
		lines.Insert(s + "first_explicit_reject=" + DC_ProbeCsv.CleanField(m_sFirstExplicitReject));
		lines.Insert(s + "rejects=" + DC_ProbeCsv.CleanField(DescribeRejects()));
		int gridTotal = m_iGridNx * m_iGridNz;
		lines.Insert(s + "grid_total=" + gridTotal.ToString());
		lines.Insert(s + "strips_capped=" + m_iStripsCapped.ToString());
		lines.Insert(s + "grid_complete=" + DC_LaunchGuard.BoolText(IsGridComplete()));
		lines.Insert(s + "status=" + DC_ProbeCsv.CleanField(GetStatus()));
		lines.Insert(s + "stage=" + StageName(m_iStage));
		string pending;
		if (m_Sampling)
		{
			int sampled = m_iSampleIndex;
			int total = m_aAlong.Count() * m_aLateral.Count();
			float pendingHeading = m_Sampling.m_fHeadingDeg;
			pending = string.Format("%1 start %2 heading %3, pre-check %4, %5 of %6 samples", m_Sampling.m_sSource, DC_ProbeArea.FormatVector(m_Sampling.m_vStart, 1), pendingHeading.ToString(-1, 2), DC_LaunchGuard.BoolText(m_bPreCheck), sampled, total);
		}

		lines.Insert(s + "pending=" + DC_ProbeCsv.CleanField(pending));
		string nearMiss;
		if (m_NearMiss)
			nearMiss = m_NearMiss.DescribeRefusal();

		lines.Insert(s + "near_miss=" + DC_ProbeCsv.CleanField(nearMiss));
		lines.Insert(s + "first_obstacle=" + DC_ProbeCsv.CleanField(m_sFirstObstacle));
		AppendSurveyLines(lines, prefix + ".survey.");
		lines.Insert(s + "rule=rejects counts every refused candidate by its first failed criterion (cross_zone_pre: a grid strip whose centre tyre pair left the zone's cross window at a zone station, before the full sampling); first_terrain_reject is the first refusal of any candidate, first_explicit_reject of an explicit one; near_miss the fully sampled refused candidate with the largest zone share; grid_complete 1 only when every grid point was visited, the strip cap suppressed no heading (strips_capped 0) and the search was not stopped; status done, time_cap (the probe's search time ran out) or the probe's stop reason (the probe stopped while it searched: stage and pending say where)");
	}

	//------------------------------------------------------------------------------------------------
	//! The survey's summary lines (<prefix>*): the pre-check refusals, the zone stations, the longest run of
	//! in-window zone stations of any refused strip, the run histogram and the list sizes of WriteSurvey
	protected void AppendSurveyLines(notnull array<string> lines, string p)
	{
		int refused = m_Survey.m_iPreRejected;
		int longest = m_Survey.m_iLongestRun;
		float longestM = Math.Max(longest - 1, 0) * m_Criteria.m_fStepM;
		int nearListed = m_aNearList.Count();
		int obstacleListed = m_aObstacleList.Count();
		int runsListed = m_aRunList.Count();
		lines.Insert(p + "pre_rejected=" + refused.ToString());
		lines.Insert(p + "zone_stations=" + m_iZoneStations.ToString());
		lines.Insert(p + "longest_run_stations=" + longest.ToString());
		lines.Insert(p + "longest_run_m=" + longestM.ToString(-1, 1));
		lines.Insert(p + "run_histogram=" + DC_ProbeCsv.CleanField(m_Survey.HistogramText()));
		lines.Insert(p + "near_listed=" + nearListed.ToString());
		lines.Insert(p + "obstacle_listed=" + obstacleListed.ToString());
		lines.Insert(p + "runs_listed=" + runsListed.ToString());
		lines.Insert(p + "rule=run = the longest run of consecutive measure-zone stations whose centre tyre pair is inside the cross window, of a strip the pre-check refused (cross_zone_pre; the centre pair only, so a necessary condition for a shorter zone, never a site); run_m = (run - 1) x step_m; run_histogram = run:strips; the survey CSV lists the best near misses (fully sampled refusals by zone share, every failed criterion), the strips an obstacle refused and the longest runs, at most 10 each");
	}

	//------------------------------------------------------------------------------------------------
	//! The survey CSV's columns (WriteSurvey)
	static void AppendSurveyColumns(notnull array<string> columns)
	{
		array<string> names = {"search", "list", "rank", "source", "start_x", "start_z", "heading_deg", "failed", "zone_share", "cross_min_signed", "cross_max_signed", "cross_mean", "grade", "grade_window_max", "grade_window_max_all", "cross_abs_max_all", "wide_cross_abs_max", "rms_m", "min_y", "max_y", "obstacle", "run_stations", "run_m", "run_from_m", "zone_stations", "strips"};
		foreach (string name : names)
		{
			columns.Insert(name);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! The survey rows of this search (search: its name in the CSV): near_miss (every failed criterion and the measured
	//! values), obstacle (the obstacle that refused a terrain-ok strip), pre_run (a pre-check refusal's longest run)
	//! and run_histogram (one row per non-empty run length); at most DC_SteerSurvey.TOP rows per list. false when a
	//! row could not be written
	bool WriteSurvey(notnull DC_ProbeCsv csv, string search)
	{
		bool ok = true;
		foreach (int i, DC_SteerSite near : m_aNearList)
		{
			if (!WriteSurveySite(csv, search, "near_miss", i + 1, near, near.m_sFailed))
				ok = false;
		}

		foreach (int j, DC_SteerSite blocked : m_aObstacleList)
		{
			if (!WriteSurveySite(csv, search, "obstacle", j + 1, blocked, "obstacle"))
				ok = false;
		}

		foreach (int k, DC_SteerSite run : m_aRunList)
		{
			if (!WriteSurveySite(csv, search, "pre_run", k + 1, run, REASON_PRE))
				ok = false;
		}

		for (int r = 0; r < m_Survey.m_aRunHistogram.Count(); r++)
		{
			int strips = m_Survey.m_aRunHistogram[r];
			if (strips == 0)
				continue;

			DC_ProbeRow row = new DC_ProbeRow();
			row.AddString(DC_ProbeCsv.CleanField(search));
			row.AddString("run_histogram");
			row.AddInt(r);
			for (int e = 0; e < 18; e++)
			{
				row.AddEmpty();
			}

			float runM = Math.Max(r - 1, 0) * m_Criteria.m_fStepM;
			row.AddInt(r);
			row.AddFloat(runM, 1);
			row.AddEmpty();
			row.AddInt(m_iZoneStations);
			row.AddInt(strips);
			if (!csv.WriteRow(row))
				ok = false;
		}

		return ok;
	}

	//------------------------------------------------------------------------------------------------
	//! One survey row of a candidate; a pre_run row has no full-strip values (only its pre-check was sampled)
	protected bool WriteSurveySite(notnull DC_ProbeCsv csv, string search, string list, int rank, notnull DC_SteerSite site, string failed)
	{
		DC_ProbeRow row = new DC_ProbeRow();
		float startX = site.m_vStart[0];
		float startZ = site.m_vStart[2];
		row.AddString(DC_ProbeCsv.CleanField(search));
		row.AddString(list);
		row.AddInt(rank);
		row.AddString(DC_ProbeCsv.CleanField(site.m_sSource));
		row.AddFloat(startX, 1);
		row.AddFloat(startZ, 1);
		row.AddFloat(site.m_fHeadingDeg, 2);
		row.AddString(DC_ProbeCsv.CleanField(failed));
		if (list == "pre_run")
		{
			for (int e = 0; e < 12; e++)
			{
				row.AddEmpty();
			}

			row.AddEmpty();
			float runM = Math.Max(site.m_iPreRun - 1, 0) * m_Criteria.m_fStepM;
			row.AddInt(site.m_iPreRun);
			row.AddFloat(runM, 1);
			row.AddFloat(site.m_fPreRunFromM, 1);
			row.AddInt(m_iZoneStations);
			row.AddEmpty();
			return csv.WriteRow(row);
		}

		row.AddFloat(site.m_fZoneShare, 4);
		row.AddFloat(site.m_fCrossMin, 4);
		row.AddFloat(site.m_fCrossMax, 4);
		row.AddFloat(site.m_fCrossMean, 4);
		row.AddFloat(site.m_fGrade, 4);
		row.AddFloat(site.m_fGradeWindowMax, 4);
		row.AddFloat(site.m_fGradeWindowMaxAll, 4);
		row.AddFloat(site.m_fCrossAbsMaxAll, 4);
		row.AddFloat(site.m_fWideCrossAbsMax, 4);
		row.AddFloat(site.m_fRmsM, 3);
		row.AddFloat(site.m_fMinY, 2);
		row.AddFloat(site.m_fMaxY, 2);
		row.AddString(DC_ProbeCsv.CleanField(site.m_sObstacle));
		row.AddEmpty();
		row.AddEmpty();
		row.AddEmpty();
		row.AddInt(m_iZoneStations);
		row.AddEmpty();
		return csv.WriteRow(row);
	}

	//------------------------------------------------------------------------------------------------
	string DescribeSearch()
	{
		string found = "no site";
		if (m_Selected)
			found = m_Selected.DescribeSite();

		int explicitCount = m_aExplicit.Count();
		string text = string.Format("%1 search: %2 explicit, %3 grid points (%4 with a usable gradient), %5 strips, %6 terrain ok, %7 obstacle-checked (%8 rejected): %9", m_Criteria.m_sKind, explicitCount, m_iGridPoints, m_iGradientOk, m_iStrips, m_iTerrainOk, m_iObstacleChecked, m_iObstacleRejected, found);
		if (!m_sStopReason.IsEmpty())
			text += string.Format("; stopped (%1) in stage %2", m_sStopReason, StageName(m_iStage));

		if (m_iStripsCapped > 0)
			text += string.Format("; %1 headings over the strip cap", m_iStripsCapped);

		if (!m_aRejectReasons.IsEmpty())
			text += "; terrain refusals " + DescribeRejects();

		if (!m_Selected && m_NearMiss)
			text += "; near miss " + m_NearMiss.DescribeRefusal();

		if (!m_Selected && !m_sFirstObstacle.IsEmpty())
			text += "; first obstacle " + m_sFirstObstacle;

		return text;
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
	//! One zone tyre-pair cross against the zone's window: cross criteria side x cross within cross_min..cross_max,
	//! flat |cross| <= cross_max (the test of Apply's cross_zone criterion and of the search's pre-check)
	static bool ZoneCrossOk(DC_SteerSiteCriteria c, int side, float cross)
	{
		if (c.IsCross())
		{
			float signedCross = side * cross;
			return signedCross >= c.m_fCrossMin && signedCross <= c.m_fCrossMax;
		}

		return Math.AbsFloat(cross) <= c.m_fCrossMax;
	}

	//------------------------------------------------------------------------------------------------
	//! list + '|' + reason (reason alone for an empty list)
	static string AddFailure(string list, string reason)
	{
		if (list == string.Empty)
			return reason;

		return list + "|" + reason;
	}

	//------------------------------------------------------------------------------------------------
	//! Fills site.m_fGrade .. m_fRmsM, m_fMinY / m_fMaxY, m_bTerrainOk, m_sTerrainReason (the first failed criterion),
	//! m_sFailed (every failed criterion) and m_fScore
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
		int crossInWindow = 0;
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
					if (ZoneCrossOk(c, site.m_iSide, cross))
						crossInWindow++;
				}
			}
		}

		float crossMean = 0;
		float zoneShare = 0;
		if (crossCount > 0)
		{
			crossMean = crossSum / crossCount;
			zoneShare = crossInWindow;
			zoneShare = zoneShare / crossCount;
		}

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

		// rms of the measure zone's samples about its plane y = meanY_z + grade_z (a - meanA_z) - crossMean x lateral
		// (right positive; the centre-line regression of the zone stations): the run-up and stop parts of a cross
		// strip only meet the lax criteria, and the flat zone is the whole strip (the same plane as the mean grade's)
		float sumAz = 0;
		float sumYz = 0;
		int nz = 0;
		for (int iz = 0; iz < n; iz++)
		{
			if (along[iz] >= site.m_fZoneFromM && along[iz] <= site.m_fZoneToM)
			{
				sumAz += along[iz];
				sumYz += heights[iz * m + 2];
				nz++;
			}
		}

		float rms = 0;
		if (nz > 0)
		{
			float meanAz = sumAz / nz;
			float meanYz = sumYz / nz;
			float sxyz = 0;
			float sxxz = 0;
			for (int jz = 0; jz < n; jz++)
			{
				if (along[jz] >= site.m_fZoneFromM && along[jz] <= site.m_fZoneToM)
				{
					float daz = along[jz] - meanAz;
					sxyz += daz * (heights[jz * m + 2] - meanYz);
					sxxz += daz * daz;
				}
			}

			float gradeZone = 0;
			if (sxxz > 0)
				gradeZone = sxyz / sxxz;

			float sq = 0;
			for (int r = 0; r < n; r++)
			{
				if (along[r] >= site.m_fZoneFromM && along[r] <= site.m_fZoneToM)
				{
					float planeY = meanYz + gradeZone * (along[r] - meanAz);
					for (int q = 0; q < m; q++)
					{
						float e = heights[r * m + q] - (planeY - crossMean * lateral[q]);
						sq += e * e;
					}
				}
			}

			rms = Math.Sqrt(sq / (nz * m));
		}
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
		site.m_fZoneShare = zoneShare;
		site.m_fMinY = minY;
		site.m_fMaxY = maxY;

		// criteria: every failure is listed (m_sFailed, the survey's near misses), the first one names the refusal
		bool zoneEmpty = crossCount < 2;
		bool lowSurface = minY < c.m_fMinSurfaceY;
		bool gradeFails = Math.AbsFloat(grade) > c.m_fGradeMax;
		bool windowFails = gradeWindowMax > c.m_fGradeWindowMax || gradeWindowMaxAll > c.m_fGradeWindowMaxAll;
		bool crossAllFails = crossAbsMaxAll > c.m_fCrossAbsMaxAll || wideAbsMax > c.m_fCrossAbsMaxAll;
		bool crossZoneFails = false;
		if (c.IsCross())
			crossZoneFails = crossMin < c.m_fCrossMin || crossMax > c.m_fCrossMax;
		else
			crossZoneFails = Math.Max(Math.AbsFloat(crossMin), Math.AbsFloat(crossMax)) > c.m_fCrossMax;

		bool rmsFails = rms > c.m_fRmsMax;
		string failed = string.Empty;
		int failedCount = 0;
		if (zoneEmpty)
		{
			failed = AddFailure(failed, "zone_empty");
			failedCount++;
		}

		if (lowSurface)
		{
			failed = AddFailure(failed, "below_min_surface_y");
			failedCount++;
		}

		if (gradeFails)
		{
			failed = AddFailure(failed, "grade");
			failedCount++;
		}

		if (windowFails)
		{
			failed = AddFailure(failed, "grade_window");
			failedCount++;
		}

		if (crossAllFails)
		{
			failed = AddFailure(failed, "cross_all");
			failedCount++;
		}

		if (crossZoneFails)
		{
			failed = AddFailure(failed, "cross_zone");
			failedCount++;
		}

		if (rmsFails)
		{
			failed = AddFailure(failed, "rms");
			failedCount++;
		}

		site.m_sFailed = failed;
		site.m_iFailedCount = failedCount;
		if (zoneEmpty)
		{
			site.m_sTerrainReason = "zone_empty";
			return;
		}

		if (lowSurface)
		{
			site.m_sTerrainReason = "below_min_surface_y";
			return;
		}

		if (gradeFails)
		{
			site.m_sTerrainReason = "grade";
			return;
		}

		if (windowFails)
		{
			site.m_sTerrainReason = "grade_window";
			return;
		}

		if (crossAllFails)
		{
			site.m_sTerrainReason = "cross_all";
			return;
		}

		if (crossZoneFails)
		{
			site.m_sTerrainReason = "cross_zone";
			return;
		}

		if (rmsFails)
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
