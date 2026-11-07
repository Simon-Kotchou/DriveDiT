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
//   3. every strip is sampled on three lines (lateral -w, 0, +w) every step_m and must meet the kind's criteria
//      (DC_SteerSiteCriteria): mean and windowed grade, the cross-slope at every station of the measure zone
//      (cross = (y(-w) - y(+w)) / 2w, positive = the right side lower, multiplied by the side), the cross-slope and
//      grade of the run-up and stop parts, the rms about a plane fit, a minimum surface height (no shore or water);
//   4. the strips that pass are ranked by score; the best ones are checked for obstacles with BaseWorld.TraceMove
//      (TraceFlags.ENTS, LayerMask EPhysicsLayerPresets.Vehicle, the ego and its children excluded): lines every
//      line_step_m across the strip's width at two heights above the surface, in segments of segment_m. The first
//      strip without a hit is the site. With SetTrustExplicit (the flat site's e01 runway) an explicit candidate that
//      passes the terrain check is used even with a hit, which is recorded (obstacle_hits 1, first_obstacle).
// The work is spread over render frames (StepSearch with a sample and a trace budget per call), so no frame stalls.
// The chosen site and its measured slopes go into the run header (site.<kind>.*), with the search counts.
//
// BUILD-AGNOSTIC: GetSurfaceY, TraceMove, TraceParam (Start, End, Flags, LayerMask, ExcludeArray), TraceFlags.ENTS and
// EPhysicsLayerPresets.Vehicle have the same declarations in 1.8.0.13 and 1.9.0.37 (Core/generated/World/BaseWorld.c
// :24, :237; TraceParam.c; TraceFlags.c; GameLib/gameLib.c:24).
// ============================================================================

//------------------------------------------------------------------------------------------------
//! What a site of one kind must meet (flat or cross); filled by the probe from its attributes
class DC_SteerSiteCriteria
{
	string m_sKind = "flat";			//!< flat | cross
	float m_fLengthM = 1000;
	float m_fHalfWidthM = 20;			//!< terrain lines at +-this, obstacle lines across +-this
	float m_fZoneFromM = 0;				//!< measure zone along the strip (cross criteria; flat: the whole strip)
	float m_fZoneToM = 1000;
	float m_fStepM = 5;
	float m_fGradeWindowM = 20;
	float m_fGradeMax = 0.005;			//!< |mean grade| (regression of the centre line)
	float m_fGradeWindowMax = 0.01;		//!< |grade| over any window inside the zone
	float m_fGradeWindowMaxAll = 0.02;	//!< |grade| over any window of the whole strip
	float m_fCrossMin = 0;				//!< cross: side x cross at every zone station >= this
	float m_fCrossMax = 0.01;			//!< cross: side x cross <= this; flat: |cross| <= this at every station
	float m_fCrossAbsMaxAll = 0.02;		//!< |cross| at every station of the whole strip
	float m_fCrossTarget = 0;			//!< cross: score target of the zone mean
	float m_fRmsMax = 0.1;				//!< rms of all samples about the plane fit [m]
	float m_fMinSurfaceY = 3;			//!< every sample above this [m] (sea level 0)
	int m_iSide = 1;					//!< cross: +1 downhill to the right of the heading, -1 to the left
	float m_fObstacleLineStepM = 2.5;
	float m_fObstacleSegmentM = 20;
	float m_fObstacleLowM = 0.5;
	float m_fObstacleHighM = 1.3;

	//------------------------------------------------------------------------------------------------
	bool IsCross()
	{
		return m_sKind == "cross";
	}

	//------------------------------------------------------------------------------------------------
	//! <prefix>.crit.* lines
	void AppendHeaderLines(notnull array<string> lines, string prefix)
	{
		string p = prefix + ".crit.";
		lines.Insert(p + "length_m=" + m_fLengthM.ToString(-1, 1));
		lines.Insert(p + "half_width_m=" + m_fHalfWidthM.ToString(-1, 2));
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
		lines.Insert(p + "obstacle_line_step_m=" + m_fObstacleLineStepM.ToString(-1, 2));
		lines.Insert(p + "obstacle_segment_m=" + m_fObstacleSegmentM.ToString(-1, 1));
		lines.Insert(p + "obstacle_heights_m=" + m_fObstacleLowM.ToString(-1, 2) + "," + m_fObstacleHighM.ToString(-1, 2));
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
	float m_fCrossMean;					//!< zone mean of cross (positive = the right side lower), unsigned by side
	float m_fCrossMin;					//!< zone min / max of side x cross
	float m_fCrossMax;
	float m_fCrossAbsMaxAll;
	float m_fRmsM;
	float m_fMinY;
	float m_fMaxY;
	int m_iStations;
	bool m_bTerrainOk;
	string m_sTerrainReason;
	float m_fScore;
	int m_iObstacleTraces;
	int m_iObstacleHits = -1;			//!< -1 = not checked
	string m_sObstacle;

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
		lines.Insert(prefix + ".rms_m=" + m_fRmsM.ToString(-1, 4));
		lines.Insert(prefix + ".min_y=" + m_fMinY.ToString(-1, 3));
		lines.Insert(prefix + ".max_y=" + m_fMaxY.ToString(-1, 3));
		lines.Insert(prefix + ".stations=" + m_iStations.ToString());
		lines.Insert(prefix + ".score=" + m_fScore.ToString(-1, 4));
		lines.Insert(prefix + ".obstacle_traces=" + m_iObstacleTraces.ToString());
		lines.Insert(prefix + ".obstacle_hits=" + m_iObstacleHits.ToString());
		lines.Insert(prefix + ".cross_rule=cross = (y(-w) - y(+w)) / 2w per station, positive = the right side of the heading lower; *_signed = side x cross over the measure zone");
	}

	//------------------------------------------------------------------------------------------------
	string DescribeSite()
	{
		string text = string.Format("%1 site start %2 heading %3 deg, %4 m x +-%5 m: grade %6, cross mean %7 (zone %8..%9)", m_sSource, DC_ProbeArea.FormatVector(m_vStart, 1), m_fHeadingDeg.ToString(-1, 2), m_fLengthM.ToString(-1, 0), m_fHalfWidthM.ToString(-1, 1), m_fGrade.ToString(-1, 4), m_fCrossMean.ToString(-1, 4), m_fCrossMin.ToString(-1, 4), m_fCrossMax.ToString(-1, 4));
		return text + string.Format(", rms %1 m, y %2..%3, %4 obstacle traces", m_fRmsM.ToString(-1, 3), m_fMinY.ToString(-1, 1), m_fMaxY.ToString(-1, 1), m_iObstacleTraces);
	}
}

//------------------------------------------------------------------------------------------------
//! Incremental search for one site of one kind (see the file header). Configure, AddExplicit, optionally SetGrid,
//! then StepSearch once per render frame until it returns true; GetSelected is null when nothing passed.
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
	protected ref TraceParam m_Trace;

	protected bool m_bGrid;
	protected bool m_bGridDone;
	protected bool m_bTrustExplicit;		//!< an obstacle hit on an explicit candidate is logged, not a rejection
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

	protected int m_iStage = STAGE_EXPLICIT;
	protected int m_iExplicitCursor;
	protected int m_iObstacleIndex;			//!< trace index inside the current candidate (segment x line x height)

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
	protected string m_sFirstReject;		//!< first terrain refusal reason of an explicit candidate
	protected string m_sFirstObstacle;		//!< first obstacle hit

	//------------------------------------------------------------------------------------------------
	//! excludes: the entities the obstacle traces ignore (the ego and its children)
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
		m_Trace = new TraceParam();
		m_Trace.Flags = TraceFlags.ENTS;
		m_Trace.LayerMask = EPhysicsLayerPresets.Vehicle;
		m_Trace.ExcludeArray = m_aExcludes;
	}

	//------------------------------------------------------------------------------------------------
	//! true: an explicit candidate that passes the terrain check is used even with an obstacle hit (it is recorded:
	//! obstacle_hits 1, the first hit in first_obstacle); for a candidate known to be drivable (the e01 runway)
	void SetTrustExplicit(bool trust)
	{
		m_bTrustExplicit = trust;
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
		m_iGridNx = Math.Floor((maxX - minX) / spacingM) + 1;
		m_iGridNz = Math.Floor((maxZ - minZ) / spacingM) + 1;
	}

	//------------------------------------------------------------------------------------------------
	//! One slice of work; true once the search is over (GetSelected). sampleBudget: GetSurfaceY calls, traceBudget:
	//! TraceMove calls of this call (each at least one unit of work is done)
	bool StepSearch(notnull BaseWorld world, int sampleBudget, int traceBudget)
	{
		if (m_iStage == STAGE_DONE)
			return true;

		m_iCalls++;
		int samples = 0;
		if (m_iStage == STAGE_EXPLICIT)
		{
			while (m_iExplicitCursor < m_aExplicit.Count() && samples < sampleBudget)
			{
				DC_SteerSite site = m_aExplicit[m_iExplicitCursor];
				m_iExplicitCursor++;
				samples += EvaluateTerrain(world, site);
				if (site.m_bTerrainOk)
					AddRanked(site);
				else if (m_sFirstReject.IsEmpty())
					m_sFirstReject = site.m_sTerrainReason;
			}

			if (m_iExplicitCursor < m_aExplicit.Count())
				return false;

			if (!m_aRanked.IsEmpty())
				m_iStage = STAGE_OBSTACLES;
			else if (m_bGrid)
				m_iStage = STAGE_GRID;
			else
				m_iStage = STAGE_DONE;

			return m_iStage == STAGE_DONE;
		}

		if (m_iStage == STAGE_GRID)
		{
			StepGrid(world, sampleBudget);
			if (!m_bGridDone)
				return false;

			if (m_aRanked.IsEmpty())
				m_iStage = STAGE_DONE;
			else
				m_iStage = STAGE_OBSTACLES;

			return m_iStage == STAGE_DONE;
		}

		// STAGE_OBSTACLES: the best remaining candidate, traced in slices
		StepObstacles(world, traceBudget);
		return m_iStage == STAGE_DONE;
	}

	//------------------------------------------------------------------------------------------------
	//! Grid points first (gradient and strip generation), the generated strips' terrain within the budget
	protected void StepGrid(notnull BaseWorld world, int sampleBudget)
	{
		int samples = 0;
		while (samples < sampleBudget)
		{
			if (!m_aPending.IsEmpty())
			{
				DC_SteerSite site = m_aPending[0];
				m_aPending.Remove(0);
				samples += EvaluateTerrain(world, site);
				if (site.m_bTerrainOk)
					AddRanked(site);

				continue;
			}

			if (m_iGridCursor >= m_iGridNx * m_iGridNz || m_iStrips >= m_iMaxStrips)
			{
				m_bGridDone = true;
				return;
			}

			int ix = m_iGridCursor % m_iGridNx;
			int iz = m_iGridCursor / m_iGridNx;
			m_iGridCursor++;
			m_iGridPoints++;
			float x = m_fGridMinX + ix * m_fGridSpacingM;
			float z = m_fGridMinZ + iz * m_fGridSpacingM;
			samples += GenerateStrips(world, x, z);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! Central-difference gradient at (x, z) and the strips centred there that it allows (pending); samples used
	protected int GenerateStrips(notnull BaseWorld world, float x, float z)
	{
		float d = m_fGradientHalfM;
		float gx = (world.GetSurfaceY(x + d, z) - world.GetSurfaceY(x - d, z)) / (2 * d);
		float gz = (world.GetSurfaceY(x, z + d) - world.GetSurfaceY(x, z - d)) / (2 * d);
		m_iSamples += 4;
		float g = Math.Sqrt(gx * gx + gz * gz);
		if (m_Criteria.IsCross())
		{
			if (g < 0.8 * m_Criteria.m_fCrossMin || g > 1.15 * m_Criteria.m_fCrossMax || g <= 0)
				return 4;

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

			return 4;
		}

		if (g > m_Criteria.m_fCrossMax)
			return 4;

		m_iGradientOk++;
		AddCentredStrip(x, z, 0);
		AddCentredStrip(x, z, 45);
		AddCentredStrip(x, z, 90);
		AddCentredStrip(x, z, 135);
		return 4;
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
	//! Samples the strip on three lines and applies the criteria (site.m_bTerrainOk, m_sTerrainReason, m_fScore);
	//! returns the GetSurfaceY calls used
	protected int EvaluateTerrain(notnull BaseWorld world, notnull DC_SteerSite site)
	{
		DC_SteerSiteCriteria c = m_Criteria;
		DC_ProbeArea strip = site.MakeArea(site.m_fHalfWidthM);
		float w = site.m_fHalfWidthM;
		int stations = Math.Floor(site.m_fLengthM / c.m_fStepM) + 1;
		array<float> along = {};
		array<float> left = {};
		array<float> centre = {};
		array<float> right = {};
		float minY = 0;
		float maxY = 0;
		for (int i = 0; i < stations; i++)
		{
			float a = i * c.m_fStepM;
			vector pl = strip.PointAt(a, -w);
			vector pc = strip.PointAt(a, 0);
			vector pr = strip.PointAt(a, w);
			float yl = world.GetSurfaceY(pl[0], pl[2]);
			float yc = world.GetSurfaceY(pc[0], pc[2]);
			float yr = world.GetSurfaceY(pr[0], pr[2]);
			along.Insert(a);
			left.Insert(yl);
			centre.Insert(yc);
			right.Insert(yr);
			if (i == 0)
			{
				minY = Math.Min(yl, Math.Min(yc, yr));
				maxY = Math.Max(yl, Math.Max(yc, yr));
			}
			else
			{
				minY = Math.Min(minY, Math.Min(yl, Math.Min(yc, yr)));
				maxY = Math.Max(maxY, Math.Max(yl, Math.Max(yc, yr)));
			}
		}

		int used = 3 * stations;
		m_iSamples += used;
		site.m_iStations = stations;
		site.m_fMinY = minY;
		site.m_fMaxY = maxY;
		DC_SteerTerrainMath.Apply(site, c, along, left, centre, right);
		if (site.m_bTerrainOk)
			m_iTerrainOk++;

		return used;
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
	//! Traces the best ranked candidate in slices; a hit rejects it, a clean candidate is the site. When the ranked
	//! list runs out: the grid next (if not run yet), else the search ends without a site
	protected void StepObstacles(notnull BaseWorld world, int traceBudget)
	{
		int traces = 0;
		while (traces < traceBudget)
		{
			bool capped = m_iObstacleIndex == 0 && m_iObstacleChecked >= m_iMaxObstacleChecks;
			if (m_aRanked.IsEmpty() || capped)
			{
				if (m_bGrid && !m_bGridDone && m_iObstacleChecked < m_iMaxObstacleChecks)
				{
					m_iStage = STAGE_GRID;
					return;
				}

				m_iStage = STAGE_DONE;
				return;
			}

			DC_SteerSite site = m_aRanked[0];
			DC_ProbeArea strip = site.MakeArea(site.m_fHalfWidthM);
			int lines = Math.Floor(2 * site.m_fHalfWidthM / m_Criteria.m_fObstacleLineStepM) + 1;
			int segments = Math.Floor(site.m_fLengthM / m_Criteria.m_fObstacleSegmentM);
			if (segments < 1)
				segments = 1;

			int total = segments * lines * 2;
			if (m_iObstacleIndex == 0)
			{
				m_iObstacleChecked++;
				site.m_iObstacleHits = 0;
				site.m_iObstacleTraces = 0;
			}

			while (m_iObstacleIndex < total && traces < traceBudget)
			{
				int height = m_iObstacleIndex % 2;
				int line = (m_iObstacleIndex / 2) % lines;
				int segment = m_iObstacleIndex / (2 * lines);
				m_iObstacleIndex++;
				traces++;
				if (TraceSegment(world, site, strip, segment, line, height))
					continue;

				// hit: reject this candidate (a trusted explicit one is used anyway, with the hit recorded)
				site.m_iObstacleHits = 1;
				if (m_sFirstObstacle.IsEmpty())
					m_sFirstObstacle = site.m_sObstacle;

				if (m_bTrustExplicit && site.m_sSource == "explicit")
				{
					m_Selected = site;
					m_iStage = STAGE_DONE;
					return;
				}

				m_iObstacleRejected++;

				m_aRanked.RemoveOrdered(0);
				m_iObstacleIndex = 0;
				break;
			}

			if (m_iObstacleIndex >= total)
			{
				m_Selected = site;
				m_iStage = STAGE_DONE;
				return;
			}
		}
	}

	//------------------------------------------------------------------------------------------------
	//! One obstacle trace of segment x line x height of the site; true when nothing is hit
	protected bool TraceSegment(notnull BaseWorld world, notnull DC_SteerSite site, notnull DC_ProbeArea strip, int segment, int line, int height)
	{
		DC_SteerSiteCriteria c = m_Criteria;
		float lateral = -site.m_fHalfWidthM + line * c.m_fObstacleLineStepM;
		if (lateral > site.m_fHalfWidthM)
			lateral = site.m_fHalfWidthM;

		float segmentLength = site.m_fLengthM / Math.Max(1, Math.Floor(site.m_fLengthM / c.m_fObstacleSegmentM));
		float a0 = segment * segmentLength;
		float a1 = a0 + segmentLength;
		float lift = c.m_fObstacleLowM;
		if (height == 1)
			lift = c.m_fObstacleHighM;

		vector q0 = strip.PointAt(a0, lateral);
		vector q1 = strip.PointAt(a1, lateral);
		vector p0 = Vector(q0[0], world.GetSurfaceY(q0[0], q0[2]) + lift, q0[2]);
		vector p1 = Vector(q1[0], world.GetSurfaceY(q1[0], q1[2]) + lift, q1[2]);
		m_iSamples += 2;
		m_iTraces++;
		site.m_iObstacleTraces++;
		m_Trace.Start = p0;
		m_Trace.End = p1;
		m_Trace.TraceEnt = null;
		float fraction = world.TraceMove(m_Trace, null);
		if (fraction >= 1)
			return true;

		string what = "unknown";
		IEntity hit = m_Trace.TraceEnt;
		if (hit)
			what = hit.ClassName() + "@" + DC_ProbeArea.FormatVector(hit.GetOrigin(), 1);

		float hitAlong = a0 + fraction * (a1 - a0);
		site.m_sObstacle = string.Format("%1 at %2 m along, %3 m lateral, %4 m up (%5)", what, hitAlong.ToString(-1, 1), lateral.ToString(-1, 1), lift.ToString(-1, 1), DC_ProbeArea.FormatVector(site.m_vStart, 1));
		return false;
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
		lines.Insert(s + "trust_explicit=" + DC_LaunchGuard.BoolText(m_bTrustExplicit));
		lines.Insert(s + "grid_points=" + m_iGridPoints.ToString());
		lines.Insert(s + "gradient_ok=" + m_iGradientOk.ToString());
		lines.Insert(s + "strips=" + m_iStrips.ToString());
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
//! run it through tests/enforce_subset.py). along[i] = i x step, left / centre / right = the surface y at lateral
//! -w, 0, +w of station i
class DC_SteerTerrainMath
{
	//------------------------------------------------------------------------------------------------
	//! Fills site.m_fGrade .. m_fRmsM, m_bTerrainOk, m_sTerrainReason and m_fScore (m_fMinY / m_fMaxY are set by the
	//! caller from the same samples)
	static void Apply(DC_SteerSite site, DC_SteerSiteCriteria c, array<float> along, array<float> left, array<float> centre, array<float> right)
	{
		int n = along.Count();
		float w = site.m_fHalfWidthM;
		site.m_bTerrainOk = false;
		site.m_fScore = 1000000;
		if (n < 3 || w <= 0 || c.m_fStepM <= 0)
		{
			site.m_sTerrainReason = "too_short";
			return;
		}

		// centre-line regression slope
		float sumA = 0;
		float sumY = 0;
		for (int i = 0; i < n; i++)
		{
			sumA += along[i];
			sumY += centre[i];
		}

		float meanA = sumA / n;
		float meanY = sumY / n;
		float sxy = 0;
		float sxx = 0;
		for (int j = 0; j < n; j++)
		{
			float da = along[j] - meanA;
			sxy += da * (centre[j] - meanY);
			sxx += da * da;
		}

		float grade = 0;
		if (sxx > 0)
			grade = sxy / sxx;

		// cross-slope per station, zone statistics
		float side = 1;
		if (c.IsCross())
			side = site.m_iSide;

		float crossSum = 0;
		int crossCount = 0;
		float crossMin = 0;
		float crossMax = 0;
		float crossAbsMaxAll = 0;
		for (int k = 0; k < n; k++)
		{
			float cross = (left[k] - right[k]) / (2 * w);
			crossAbsMaxAll = Math.Max(crossAbsMaxAll, Math.AbsFloat(cross));
			bool inZone = along[k] >= site.m_fZoneFromM && along[k] <= site.m_fZoneToM;
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

		float crossMean = 0;
		if (crossCount > 0)
			crossMean = crossSum / crossCount;

		// windowed grade (zone and whole strip)
		int windowSteps = Math.Round(c.m_fGradeWindowM / c.m_fStepM);
		if (windowSteps < 1)
			windowSteps = 1;

		float gradeWindowMax = 0;
		float gradeWindowMaxAll = 0;
		for (int m = 0; m + windowSteps < n; m++)
		{
			float windowGrade = Math.AbsFloat((centre[m + windowSteps] - centre[m]) / (along[m + windowSteps] - along[m]));
			gradeWindowMaxAll = Math.Max(gradeWindowMaxAll, windowGrade);
			if (along[m] >= site.m_fZoneFromM && along[m + windowSteps] <= site.m_fZoneToM)
				gradeWindowMax = Math.Max(gradeWindowMax, windowGrade);
		}

		// rms about the plane y = meanY + grade (a - meanA) + slope_lat x lateral, slope_lat = -crossMean (right +)
		float slopeLat = -crossMean;
		float sq = 0;
		for (int r = 0; r < n; r++)
		{
			float planeY = meanY + grade * (along[r] - meanA);
			float el = left[r] - (planeY - slopeLat * w);
			float ec = centre[r] - planeY;
			float er = right[r] - (planeY + slopeLat * w);
			sq += el * el + ec * ec + er * er;
		}

		float rms = Math.Sqrt(sq / (3 * n));
		site.m_fGrade = grade;
		site.m_fGradeWindowMax = gradeWindowMax;
		site.m_fGradeWindowMaxAll = gradeWindowMaxAll;
		site.m_fCrossMean = crossMean;
		site.m_fCrossMin = crossMin;
		site.m_fCrossMax = crossMax;
		site.m_fCrossAbsMaxAll = crossAbsMaxAll;
		site.m_fRmsM = rms;

		// criteria, first failure named
		if (crossCount < 2)
		{
			site.m_sTerrainReason = "zone_empty";
			return;
		}

		if (site.m_fMinY < c.m_fMinSurfaceY)
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

		if (crossAbsMaxAll > c.m_fCrossAbsMaxAll)
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
