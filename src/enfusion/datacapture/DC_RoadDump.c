// ============================================================================
// DC_RoadDump - T0-ENG probe E-08a: in-engine road dump of the current map
// ============================================================================
//
// A DC_ProbeComponent on the probe game mode (GameMode_DCProbes in DC_Probes_Arland), started and stopped by
// DC_ProbeRunner (PROBE_DESIGN.md 4.4, 5.6). It reads the AI road network of the world the way vanilla
// SCR_ResupplyTaskSolver.c:411-418 reaches it (SCR_AIWorld.Cast(GetGame().GetAIWorld()).GetRoadNetworkManager()),
// so the probe layer must hold the SCR_AIWorld copied from DC_Arland. tools/roads/compare_dump_decoder.py compares
// the dump with the decoder's Arland_road_polylines.tsv.
//
// WHAT IT DOES (authority, play mode, only while the runner has started the probe)
//   - covers BaseWorld.GetBoundBox in x and z with tiles of m_fTileM (y from m_fAabbMinY to m_fAabbMaxY) and queries
//     m_iTilesPerFrame tiles per render frame with RoadNetworkManager.GetRoadsInAABB;
//   - BaseRoad has no id (BaseRoad.c:18-20: GetWidth and GetPoints only), so a road is recognised twice: a BaseRoad
//     object returned again (array.Find, identity) is a repeat across tiles (n_dup_identity, counted in its
//     n_tiles_seen and written once); a new object whose content key equals an earlier road's key is a content
//     duplicate (n_dup_content, written with dup_content_of = the earlier road_idx). Content key (TBD-9):
//     <n_pts>/<x0>:<z0>/<x1>:<z1> with 2 decimals, first and last point (the separators of the design's
//     "<n_pts>|<x0>,<z0>|<x1>,<z1>" are replaced because CSV fields carry no ',' or '|', PROBE_DESIGN.md 4.1);
//   - cap check: a tile that returns m_iCapSuspect or more roads is queried again as 4 sub-tiles; when the union of
//     the sub-tile results holds a key the tile result lacks, the tile was capped (cap_confirmed) and the missing
//     roads are added to the dump (footer roads_subtile_added);
//   - writes <map_id>_road_points.csv while it discovers roads, <map_id>_tiles.csv per tile, and at the end
//     <map_id>_roads.csv (n_tiles_seen is final only then) and <map_id>_roads_footer.txt (complete=1).
//
// FAILURE HANDLING: OnProbeStart refuses (false, ERROR) without the AI world or its road network manager, with an
// empty world bound box, a non-positive tile size or more than MAX_TILES tiles. A stop before the last tile (timeout,
// play_end, write_fail) still writes the roads found so far and the footer with complete=0; a stop on world teardown
// (reason delete) only closes the files (the base class's OnDelete reaches OnProbeStop). Both builds: every call has
// the same signature in 1.8.0.13 and 1.9.0.10 (PROBE_APIS.md 4.6); E-08a@1.9 diffs this dump against the 1.8 one.
// ============================================================================

//------------------------------------------------------------------------------------------------
//! One road written by DC_RoadDump (one BaseRoad object)
class DC_RoadDumpRoad
{
	int m_iIdx;						//!< road_idx: discovery order
	string m_sKey;					//!< content key <n_pts>/<x0>:<z0>/<x1>:<z1>
	float m_fWidthM;
	int m_iPoints;
	float m_fLengthM;				//!< sum of 3D point distances
	vector m_vFirst;
	vector m_vLast;
	int m_iFirstTileX;
	int m_iFirstTileZ;
	int m_iTilesSeen = 1;			//!< tile queries that returned this object (sub-tile queries of the cap check not counted)
	int m_iDupContentOf = -1;		//!< road_idx of the earlier road with the same key, -1 = none
}

//------------------------------------------------------------------------------------------------
[ComponentEditorProps(category: "GameScripted/DataCapture", description: "T0-ENG probe E-08a: in-engine road dump (GetRoadsInAABB per tile over the world bounds) for the comparison with the decoder")]
class DC_RoadDumpClass : DC_ProbeComponentClass
{
}

//------------------------------------------------------------------------------------------------
class DC_RoadDump : DC_ProbeComponent
{
	static const string PROBE_E08A = "e08a";
	static const float TIMEOUT_S = 600;
	//! Refuse a bound box that needs more tiles than this (a wrong bound box would take hours)
	static const int MAX_TILES = 16384;

	// === E-08a ===
	[Attribute("256", UIWidgets.Slider, "Tile size of the GetRoadsInAABB queries [m]", "16 4096 1", category: "E-08a")]
	protected float m_fTileM;

	[Attribute("4", UIWidgets.Slider, "Tiles queried per render frame", "1 64 1", category: "E-08a")]
	protected int m_iTilesPerFrame;

	[Attribute("-500", UIWidgets.Slider, "Lower y of every query box [m]", "-10000 10000 1", category: "E-08a")]
	protected float m_fAabbMinY;

	[Attribute("2000", UIWidgets.Slider, "Upper y of every query box [m]", "-10000 10000 1", category: "E-08a")]
	protected float m_fAabbMaxY;

	[Attribute("64", UIWidgets.Slider, "A tile returning this many roads or more is queried again as 4 sub-tiles (cap check)", "1 10000 1", category: "E-08a")]
	protected int m_iCapSuspect;

	// === PROBE STATE ===
	protected string m_sActiveProbe;
	protected ref DC_ProbeContext m_Ctx;
	protected string m_sMapId;
	protected ref DC_ProbeCsv m_RoadsCsv;
	protected ref DC_ProbeCsv m_PointsCsv;
	protected ref DC_ProbeCsv m_TilesCsv;
	protected ref DC_ProbeRow m_Row;
	protected ref array<BaseRoad> m_aSeenRoads;				//!< every BaseRoad object written, index = road_idx
	protected ref array<ref DC_RoadDumpRoad> m_aRoads;		//!< same index
	protected ref array<string> m_aKeys;					//!< same index
	protected vector m_vBoundMin;
	protected vector m_vBoundMax;
	protected int m_iTilesX;
	protected int m_iTilesZ;
	protected int m_iTileCount;
	protected int m_iNextTile;
	protected bool m_bDone;
	protected bool m_bFinished;
	protected int m_iDupIdentity;
	protected int m_iDupContent;
	protected int m_iSubtileAdded;
	protected int m_iPointsTotal;
	protected int m_iMaxReturned;
	protected int m_iCapSuspectTiles;
	protected int m_iCapConfirmed;

	//------------------------------------------------------------------------------------------------
	override void GetProbeIds(notnull array<string> outIds)
	{
		outIds.Insert(PROBE_E08A);
	}

	//------------------------------------------------------------------------------------------------
	override float GetProbeTimeoutS(string probeId)
	{
		return TIMEOUT_S;
	}

	//------------------------------------------------------------------------------------------------
	override void GetProbeParams(string probeId, notnull array<string> outKeyValues)
	{
		outKeyValues.Insert("m_fTileM=" + m_fTileM.ToString(-1, 3));
		outKeyValues.Insert("m_iTilesPerFrame=" + m_iTilesPerFrame.ToString());
		outKeyValues.Insert("m_fAabbMinY=" + m_fAabbMinY.ToString(-1, 3));
		outKeyValues.Insert("m_fAabbMaxY=" + m_fAabbMaxY.ToString(-1, 3));
		outKeyValues.Insert("m_iCapSuspect=" + m_iCapSuspect.ToString());
		outKeyValues.Insert("road_key=<n_pts>/<x0>:<z0>/<x1>:<z1>, 2 decimals, first and last point");
	}

	//------------------------------------------------------------------------------------------------
	override bool OnProbeStart(string probeId, notnull DC_ProbeContext ctx)
	{
		if (probeId != PROBE_E08A)
			return false;

		ResetProbeState();
		m_Ctx = ctx;

		if (m_fTileM <= 0 || m_iTilesPerFrame < 1)
			return RefuseStart(string.Format("m_fTileM %1 / m_iTilesPerFrame %2 must be positive - probe not started", m_fTileM, m_iTilesPerFrame));

		if (!GetRoadManager())
			return RefuseStart("no SCR_AIWorld or no road network manager in this world (the probe layer must hold the SCR_AIWorld of DC_Arland) - probe not started");

		World world = GetGame().GetWorld();
		if (!world)
			return RefuseStart("no game world - probe not started");

		world.GetBoundBox(m_vBoundMin, m_vBoundMax);
		float sizeX = m_vBoundMax[0] - m_vBoundMin[0];
		float sizeZ = m_vBoundMax[2] - m_vBoundMin[2];
		if (sizeX <= 0 || sizeZ <= 0)
			return RefuseStart("empty world bound box " + FormatVector(m_vBoundMin, 1) + " .. " + FormatVector(m_vBoundMax, 1) + " - probe not started");

		m_iTilesX = Math.Ceil(sizeX / m_fTileM);
		m_iTilesZ = Math.Ceil(sizeZ / m_fTileM);
		m_iTilesX = Math.MaxInt(m_iTilesX, 1);
		m_iTilesZ = Math.MaxInt(m_iTilesZ, 1);
		m_iTileCount = m_iTilesX * m_iTilesZ;
		if (m_iTileCount > MAX_TILES)
			return RefuseStart(string.Format("%1 x %2 tiles of %3 m exceed %4 (bound box %5 .. %6) - probe not started", m_iTilesX, m_iTilesZ, m_fTileM, MAX_TILES, FormatVector(m_vBoundMin, 1), FormatVector(m_vBoundMax, 1)));

		m_sMapId = ctx.GetMapId();
		if (m_sMapId.IsEmpty())
			m_sMapId = "map";

		m_Row = new DC_ProbeRow();
		array<string> pointColumns = {"road_idx", "pt_idx", "x", "y", "z"};
		m_PointsCsv = ctx.OpenCsv(m_sMapId + "_road_points", pointColumns);
		array<string> tileColumns = {"tile_ix", "tile_iz", "min_x", "min_z", "max_x", "max_z", "ret_value", "n_returned", "n_new", "n_dup_identity", "n_dup_content", "subdivided", "n_sub_union"};
		m_TilesCsv = ctx.OpenCsv(m_sMapId + "_tiles", tileColumns);
		array<string> roadColumns = {"road_idx", "road_key", "width_m", "n_pts", "length_m", "x0", "y0", "z0", "x1", "y1", "z1", "first_tile_ix", "first_tile_iz", "n_tiles_seen", "dup_content_of"};
		m_RoadsCsv = ctx.OpenCsv(m_sMapId + "_roads", roadColumns);
		if (!m_PointsCsv || !m_TilesCsv || !m_RoadsCsv)
			return RefuseStart("the road dump CSVs could not be created - probe not started");

		m_sActiveProbe = probeId;
		ctx.Log(string.Format("bound box %1 .. %2, %3 x %4 tiles of %5 m, %6 per frame, cap check at %7 roads", FormatVector(m_vBoundMin, 1), FormatVector(m_vBoundMax, 1), m_iTilesX, m_iTilesZ, m_fTileM, m_iTilesPerFrame, m_iCapSuspect));
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! m_iTilesPerFrame tiles per render frame; after the last one the roads CSV and the footer are written
	override void OnProbeFrame(string probeId, float timeSlice, int renderFrame)
	{
		if (probeId != m_sActiveProbe || m_bDone)
			return;

		RoadNetworkManager manager = GetRoadManager();
		if (!manager)
		{
			LogError("the road network manager disappeared - dump incomplete");
			Finish(false);
			m_bDone = true;
			return;
		}

		for (int i = 0; i < m_iTilesPerFrame && m_iNextTile < m_iTileCount; i++)
		{
			ProcessTile(manager, m_iNextTile);
			m_iNextTile++;
		}

		if (m_iNextTile >= m_iTileCount)
		{
			Finish(true);
			m_bDone = true;
		}
	}

	//------------------------------------------------------------------------------------------------
	override bool IsProbeDone(string probeId)
	{
		if (probeId != m_sActiveProbe)
			return true;

		return m_bDone;
	}

	//------------------------------------------------------------------------------------------------
	//! Idempotent. Writes what was found so far (footer complete=0) unless the world is being deleted
	override void OnProbeStop(string probeId, string reason)
	{
		if (probeId != m_sActiveProbe || m_sActiveProbe.IsEmpty())
			return;

		if (reason != DC_ProbeRunner.REASON_DELETE && !m_bFinished)
		{
			LogWarning(string.Format("stopped (%1) after %2 of %3 tiles - writing a partial dump with complete=0", reason, m_iNextTile, m_iTileCount));
			Finish(false);
		}

		if (m_PointsCsv)
			m_PointsCsv.Close();

		if (m_TilesCsv)
			m_TilesCsv.Close();

		if (m_RoadsCsv)
			m_RoadsCsv.Close();

		m_sActiveProbe = string.Empty;
		m_bDone = true;
		m_Ctx = null;
		m_PointsCsv = null;
		m_TilesCsv = null;
		m_RoadsCsv = null;
		m_aSeenRoads = null;
	}

	//------------------------------------------------------------------------------------------------
	protected void ResetProbeState()
	{
		m_sActiveProbe = string.Empty;
		m_Ctx = null;
		m_sMapId = string.Empty;
		m_RoadsCsv = null;
		m_PointsCsv = null;
		m_TilesCsv = null;
		m_Row = null;
		m_aSeenRoads = {};
		m_aRoads = {};
		m_aKeys = {};
		m_iTilesX = 0;
		m_iTilesZ = 0;
		m_iTileCount = 0;
		m_iNextTile = 0;
		m_bDone = false;
		m_bFinished = false;
		m_iDupIdentity = 0;
		m_iDupContent = 0;
		m_iSubtileAdded = 0;
		m_iPointsTotal = 0;
		m_iMaxReturned = 0;
		m_iCapSuspectTiles = 0;
		m_iCapConfirmed = 0;
	}

	//------------------------------------------------------------------------------------------------
	//! Logs the refusal and returns false (OnProbeStart result)
	protected bool RefuseStart(string reason)
	{
		LogError(reason);
		m_sActiveProbe = string.Empty;
		m_bDone = true;
		m_Ctx = null;
		return false;
	}

	//------------------------------------------------------------------------------------------------
	//! As vanilla SCR_ResupplyTaskSolver.c:411-418; null without an SCR_AIWorld or its manager
	protected RoadNetworkManager GetRoadManager()
	{
		SCR_AIWorld aiWorld = SCR_AIWorld.Cast(GetGame().GetAIWorld());
		if (!aiWorld)
			return null;

		return aiWorld.GetRoadNetworkManager();
	}

	//------------------------------------------------------------------------------------------------
	//! Queries one tile (and its 4 sub-tiles when it returned m_iCapSuspect or more roads) and writes its tile row
	protected void ProcessTile(RoadNetworkManager manager, int tile)
	{
		int ix = tile % m_iTilesX;
		int iz = tile / m_iTilesX;
		float minX = m_vBoundMin[0] + ix * m_fTileM;
		float minZ = m_vBoundMin[2] + iz * m_fTileM;
		float maxX = Math.Min(minX + m_fTileM, m_vBoundMax[0]);
		float maxZ = Math.Min(minZ + m_fTileM, m_vBoundMax[2]);

		array<BaseRoad> roads = {};
		int returnValue = manager.GetRoadsInAABB(Vector(minX, m_fAabbMinY, minZ), Vector(maxX, m_fAabbMaxY, maxZ), roads);
		int returned = roads.Count();
		if (returned > m_iMaxReturned)
			m_iMaxReturned = returned;

		int added = 0;
		int dupIdentity = 0;
		int dupContent = 0;
		array<string> tileKeys = {};
		foreach (BaseRoad road : roads)
		{
			if (!road)
				continue;

			int seen = m_aSeenRoads.Find(road);
			if (seen >= 0)
			{
				dupIdentity++;
				DC_RoadDumpRoad seenRecord = m_aRoads[seen];
				seenRecord.m_iTilesSeen++;
				tileKeys.Insert(m_aKeys[seen]);
				continue;
			}

			DC_RoadDumpRoad record = AddRoad(road, ix, iz);
			tileKeys.Insert(record.m_sKey);
			if (record.m_iDupContentOf >= 0)
				dupContent++;
			else
				added++;
		}

		m_iDupIdentity += dupIdentity;

		bool subdivided = returned >= m_iCapSuspect;
		int subUnion = -1;
		if (subdivided)
		{
			m_iCapSuspectTiles++;
			subUnion = CheckCap(manager, ix, iz, minX, minZ, maxX, maxZ, tileKeys);
		}

		m_Row.Clear();
		m_Row.AddInt(ix);
		m_Row.AddInt(iz);
		m_Row.AddFloat(minX, 3);
		m_Row.AddFloat(minZ, 3);
		m_Row.AddFloat(maxX, 3);
		m_Row.AddFloat(maxZ, 3);
		m_Row.AddInt(returnValue);
		m_Row.AddInt(returned);
		m_Row.AddInt(added);
		m_Row.AddInt(dupIdentity);
		m_Row.AddInt(dupContent);
		m_Row.AddBool(subdivided);
		if (subdivided)
			m_Row.AddInt(subUnion);
		else
			m_Row.AddEmpty();

		m_TilesCsv.WriteRow(m_Row);
	}

	//------------------------------------------------------------------------------------------------
	//! Queries the 4 sub-tiles of a suspect tile; adds roads the tile result lacked (capped) and returns the number
	//! of distinct keys in the union of the sub-tile results
	protected int CheckCap(RoadNetworkManager manager, int ix, int iz, float minX, float minZ, float maxX, float maxZ, notnull array<string> tileKeys)
	{
		float midX = (minX + maxX) * 0.5;
		float midZ = (minZ + maxZ) * 0.5;
		array<string> unionKeys = {};
		bool capped = false;
		for (int quarter = 0; quarter < 4; quarter++)
		{
			float subMinX = minX;
			float subMaxX = midX;
			if (quarter % 2 == 1)
			{
				subMinX = midX;
				subMaxX = maxX;
			}

			float subMinZ = minZ;
			float subMaxZ = midZ;
			if (quarter >= 2)
			{
				subMinZ = midZ;
				subMaxZ = maxZ;
			}

			array<BaseRoad> subRoads = {};
			manager.GetRoadsInAABB(Vector(subMinX, m_fAabbMinY, subMinZ), Vector(subMaxX, m_fAabbMaxY, subMaxZ), subRoads);
			foreach (BaseRoad subRoad : subRoads)
			{
				if (!subRoad)
					continue;

				string key;
				int known = m_aSeenRoads.Find(subRoad);
				if (known >= 0)
				{
					key = m_aKeys[known];
				}
				else
				{
					DC_RoadDumpRoad extra = AddRoad(subRoad, ix, iz);
					key = extra.m_sKey;
					m_iSubtileAdded++;
				}

				if (unionKeys.Find(key) < 0)
					unionKeys.Insert(key);

				if (tileKeys.Find(key) < 0)
					capped = true;
			}
		}

		if (capped)
		{
			m_iCapConfirmed++;
			LogWarning(string.Format("tile %1,%2 returned %3 roads and its sub-tiles found more: the query result is capped", ix, iz, tileKeys.Count()));
		}

		return unionKeys.Count();
	}

	//------------------------------------------------------------------------------------------------
	//! Records a new BaseRoad object: points to the points CSV, key, length, content-duplicate check
	protected DC_RoadDumpRoad AddRoad(BaseRoad road, int ix, int iz)
	{
		array<vector> points = {};
		road.GetPoints(points);
		int pointCount = points.Count();

		DC_RoadDumpRoad record = new DC_RoadDumpRoad();
		record.m_iIdx = m_aRoads.Count();
		record.m_fWidthM = road.GetWidth();
		record.m_iPoints = pointCount;
		record.m_iFirstTileX = ix;
		record.m_iFirstTileZ = iz;
		record.m_sKey = MakeKey(points);
		record.m_iDupContentOf = m_aKeys.Find(record.m_sKey);
		if (record.m_iDupContentOf >= 0)
			m_iDupContent++;

		float length = 0;
		for (int i = 0; i < pointCount; i++)
		{
			if (i > 0)
				length += vector.Distance(points[i - 1], points[i]);

			m_Row.Clear();
			m_Row.AddInt(record.m_iIdx);
			m_Row.AddInt(i);
			m_Row.AddVector(points[i], 3);
			m_PointsCsv.WriteRow(m_Row);
		}

		record.m_fLengthM = length;
		if (pointCount > 0)
		{
			record.m_vFirst = points[0];
			record.m_vLast = points[pointCount - 1];
		}

		m_iPointsTotal += pointCount;
		m_aSeenRoads.Insert(road);
		m_aKeys.Insert(record.m_sKey);
		m_aRoads.Insert(record);
		return record;
	}

	//------------------------------------------------------------------------------------------------
	//! Content key (TBD-9) <n_pts>/<x0>:<z0>/<x1>:<z1>, 2 decimals; "0" for a road without points
	protected static string MakeKey(notnull array<vector> points)
	{
		int count = points.Count();
		if (count == 0)
			return "0";

		vector first = points[0];
		vector last = points[count - 1];
		float x0 = first[0];
		float z0 = first[2];
		float x1 = last[0];
		float z1 = last[2];
		return count.ToString() + "/" + x0.ToString(-1, 2) + ":" + z0.ToString(-1, 2) + "/" + x1.ToString(-1, 2) + ":" + z1.ToString(-1, 2);
	}

	//------------------------------------------------------------------------------------------------
	//! Writes <map>_roads.csv and <map>_roads_footer.txt once; complete = every tile was queried
	protected void Finish(bool complete)
	{
		if (m_bFinished)
			return;

		m_bFinished = true;
		int unique = 0;
		float metres = 0;
		if (m_RoadsCsv && m_aRoads)
		{
			foreach (DC_RoadDumpRoad record : m_aRoads)
			{
				if (record.m_iDupContentOf < 0)
				{
					unique++;
					metres += record.m_fLengthM;
				}

				WriteRoadRow(record);
			}
		}

		float km = metres / 1000;
		float tileM = m_fTileM;
		float minY = m_fAabbMinY;
		float maxY = m_fAabbMaxY;
		int written = 0;
		if (m_aRoads)
			written = m_aRoads.Count();

		array<string> lines = {};
		lines.Insert("map_id=" + m_sMapId);
		lines.Insert("bound_min=" + FormatVector(m_vBoundMin, 3));
		lines.Insert("bound_max=" + FormatVector(m_vBoundMax, 3));
		lines.Insert("tile_m=" + tileM.ToString(-1, 3));
		lines.Insert("aabb_min_y=" + minY.ToString(-1, 3));
		lines.Insert("aabb_max_y=" + maxY.ToString(-1, 3));
		lines.Insert("tiles=" + m_iTileCount.ToString());
		lines.Insert("tiles_done=" + m_iNextTile.ToString());
		lines.Insert("roads_unique=" + unique.ToString());
		lines.Insert("roads_written=" + written.ToString());
		lines.Insert("roads_dup_identity=" + m_iDupIdentity.ToString());
		lines.Insert("roads_dup_content=" + m_iDupContent.ToString());
		lines.Insert("roads_subtile_added=" + m_iSubtileAdded.ToString());
		lines.Insert("points_total=" + m_iPointsTotal.ToString());
		lines.Insert("km_total=" + km.ToString(-1, 3));
		lines.Insert("max_returned_per_tile=" + m_iMaxReturned.ToString());
		lines.Insert("cap_suspect=" + m_iCapSuspect.ToString());
		lines.Insert("cap_suspect_tiles=" + m_iCapSuspectTiles.ToString());
		lines.Insert("cap_confirmed=" + m_iCapConfirmed.ToString());
		if (complete)
			lines.Insert("complete=1");
		else
			lines.Insert("complete=0");

		bool footerOk = false;
		if (m_Ctx)
			footerOk = m_Ctx.WriteTextFile(m_sMapId + "_roads_footer.txt", lines);

		if (!footerOk)
			LogError(m_sMapId + "_roads_footer.txt could not be written");

		LogInfo(string.Format("%1 roads (%2 unique, %3 km), %4 repeats across tiles, %5 content duplicates, %6 points, max %7 per tile, cap suspects %8, capped %9", written, unique, km.ToString(-1, 2), m_iDupIdentity, m_iDupContent, m_iPointsTotal, m_iMaxReturned, m_iCapSuspectTiles, m_iCapConfirmed));
		if (complete && written == 0)
			LogWarning("the road network returned no road at all (not built yet, or the layer has no SCR_AIWorld navmesh setup)");
	}

	//------------------------------------------------------------------------------------------------
	protected void WriteRoadRow(notnull DC_RoadDumpRoad record)
	{
		m_Row.Clear();
		m_Row.AddInt(record.m_iIdx);
		m_Row.AddString(record.m_sKey);
		m_Row.AddFloat(record.m_fWidthM, 3);
		m_Row.AddInt(record.m_iPoints);
		m_Row.AddFloat(record.m_fLengthM, 3);
		if (record.m_iPoints > 0)
		{
			m_Row.AddVector(record.m_vFirst, 3);
			m_Row.AddVector(record.m_vLast, 3);
		}
		else
		{
			for (int i = 0; i < 6; i++)
			{
				m_Row.AddEmpty();
			}
		}

		m_Row.AddInt(record.m_iFirstTileX);
		m_Row.AddInt(record.m_iFirstTileZ);
		m_Row.AddInt(record.m_iTilesSeen);
		m_Row.AddInt(record.m_iDupContentOf);
		m_RoadsCsv.WriteRow(m_Row);
	}

	//------------------------------------------------------------------------------------------------
	//! "x y z" with the given decimals (footer and log values)
	protected static string FormatVector(vector value, int decimals)
	{
		float x = value[0];
		float y = value[1];
		float z = value[2];
		return x.ToString(-1, decimals) + " " + y.ToString(-1, decimals) + " " + z.ToString(-1, decimals);
	}

	//------------------------------------------------------------------------------------------------
	protected void LogInfo(string message)
	{
		if (m_Ctx)
			m_Ctx.Log(message);
	}

	//------------------------------------------------------------------------------------------------
	protected void LogWarning(string message)
	{
		if (m_Ctx)
			m_Ctx.Warn(message);
	}

	//------------------------------------------------------------------------------------------------
	protected void LogError(string message)
	{
		if (m_Ctx)
			m_Ctx.Error(message);
	}
}
