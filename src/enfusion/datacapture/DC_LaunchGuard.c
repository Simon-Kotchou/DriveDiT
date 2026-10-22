// ============================================================================
// DC_LaunchGuard - fail-closed launch guard of the T0-ENG probes (DD-4-probe)
// ============================================================================
//
// Decides whether this Play may write anything at all. DC_ProbeRunner calls DC_LaunchGuard.Evaluate() once per Play,
// logs the result and, when it is refused, writes nothing (not even a directory).
//
// THE RULE (PROBE_DESIGN.md 4.6, checks G1-G14 in this order; the first failure refuses with its reason)
//   CLI mode: -dcLaunchId <id> is on the command line (System.GetCLIParam). The id must be a launch id, -profile must
//     be visible and lie inside the required profile root (E:/DriveDiT), in Workbench the resolved $profile too, and
//     $profile:dc_launch/launch_<id>.json must exist and be a valid launch record (status ok, same id, exe wb in
//     Workbench and client outside). tools/launch/wb_R1.bat copies the preflight record there and passes the id
//     (DD-0 follow-up).
//   Workbench escape: only when the command line is not visible at all (E-11: whether Workbench Play passes its own
//     command line to GetCLIParam is unknown until the first Play), only with m_bAllowWorkbenchWithoutLaunchId, only
//     in Workbench, only with a $profile inside the root, and only with the newest -wb- launch record in
//     $profile:dc_launch, which must be valid and younger than the deploy (DC_BuildInfo.DEPLOY_UTC). Escape runs are
//     marked launch.mode=wb_escape in every run header.
//   A visible command line without -dcLaunchId always refuses, escape or not: Workbench was then started by hand.
//
// WHAT IT READS from a launch record (dd-launch/1, tools/pin/preflight.py make_record; dd-launch/2 adds the optional
//   top-level build_key, channel and use_class): schema, launch_id, status, utc, exe, pin.pin_id, pin.sha256,
//   observed.game.buildid, observed.tools.buildid. host, argv and checks are never read and never copied (the record
//   holds the host name).
//
// Evaluate() never writes and never prints; the caller logs. The E-11 line the runner prints shows $profile, which
// is a C: path with the Windows user name when Workbench was started from Steam: that line is never pasted into a
// PR, a commit or a report.
// ============================================================================

//------------------------------------------------------------------------------------------------
//! Result of one launch-guard evaluation. Filled by DC_LaunchGuard.Evaluate, read-only afterwards.
class DC_LaunchInfo
{
	bool m_bAllowed;
	string m_sMode;				//!< "cli", "wb_escape" or "refused"
	string m_sReason;			//!< one-line refusal reason, empty when allowed
	string m_sLaunchId;
	string m_sRecordPath;		//!< $profile:dc_launch/launch_<id>.json
	bool m_bCliVisible;			//!< GetCLIParam saw -profile or -dcLaunchId (E-11 fact)
	bool m_bCliLaunchId;		//!< -dcLaunchId seen
	bool m_bCliProfile;			//!< -profile seen
	string m_sProfileCli;		//!< -profile as seen by GetCLIParam, empty when not visible
	string m_sProfileAbs;		//!< $profile resolved by Workbench.GetAbsolutePath (Workbench only), else empty
	string m_sRecordSchema;		//!< "dd-launch/1"
	string m_sRecordStatus;		//!< must be "ok"
	string m_sRecordExe;		//!< "wb" or "client"
	string m_sRecordUtc;		//!< "YYYY-MM-DDTHH:MM:SSZ"
	string m_sRecordLaunchId;	//!< launch_id inside the record (must equal m_sLaunchId)
	string m_sPinId;			//!< pin.pin_id
	string m_sPinSha256;		//!< pin.sha256
	string m_sBuildKey;			//!< top-level build_key (dd-launch/2), empty on dd-launch/1
	string m_sChannel = "stable";	//!< top-level channel, "stable" when absent (dd-launch/1, dc-2.0 draft 5.1)
	string m_sUseClass = "dev";	//!< top-level use_class, "dev" when absent
	int m_iGameBuildId = -1;	//!< observed.game.buildid
	int m_iToolsBuildId = -1;	//!< observed.tools.buildid

	//------------------------------------------------------------------------------------------------
	//! Appends the launch.* key=value lines of PROBE_DESIGN.md 4.5 to lines
	void AppendHeaderLines(notnull array<string> lines)
	{
		lines.Insert("launch.mode=" + m_sMode);
		lines.Insert("launch.id=" + m_sLaunchId);
		lines.Insert("launch.record=" + m_sRecordPath);
		lines.Insert("launch.cli_visible=" + DC_LaunchGuard.BoolText(m_bCliVisible));
		lines.Insert("launch.cli_launch_id=" + DC_LaunchGuard.BoolText(m_bCliLaunchId));
		lines.Insert("launch.profile_cli=" + m_sProfileCli);
		lines.Insert("launch.profile_abs=" + m_sProfileAbs);
		lines.Insert("launch.record_schema=" + m_sRecordSchema);
		lines.Insert("launch.record_exe=" + m_sRecordExe);
		lines.Insert("launch.record_utc=" + m_sRecordUtc);
		lines.Insert("launch.pin_id=" + m_sPinId);
		lines.Insert("launch.pin_sha256=" + m_sPinSha256);
		lines.Insert("launch.build_key=" + m_sBuildKey);
		lines.Insert("launch.channel=" + m_sChannel);
		lines.Insert("launch.use_class=" + m_sUseClass);
		lines.Insert("launch.game_buildid=" + m_iGameBuildId.ToString());
		lines.Insert("launch.tools_buildid=" + m_iToolsBuildId.ToString());
	}
}

//------------------------------------------------------------------------------------------------
class DC_LaunchGuard
{
	static const string CLI_LAUNCH_ID = "dcLaunchId";
	static const string CLI_PROFILE = "profile";
	static const string LAUNCH_DIR = "$profile:dc_launch";

	static const string MODE_CLI = "cli";
	static const string MODE_ESCAPE = "wb_escape";
	static const string MODE_REFUSED = "refused";

	//! Launch ids are YYYYMMDDTHHMMSSZ-<exe>-<hex>: 24 characters today, 20..64 accepted
	static const int LAUNCH_ID_MIN_LENGTH = 20;
	static const int LAUNCH_ID_MAX_LENGTH = 64;
	//! "YYYY-MM-DDTHH:MM:SSZ"
	static const int UTC_LENGTH = 20;

	//------------------------------------------------------------------------------------------------
	//! Evaluates the guard (PROBE_DESIGN.md 4.6, G1-G14). Never writes and never prints; the caller logs the result.
	//! \param[in] requiredProfileRoot folder -profile and $profile must lie in (DC_ProbeRunner.m_sRequiredProfileRoot)
	//! \param[in] allowWorkbenchWithoutLaunchId enables the Workbench escape for an invisible command line
	static DC_LaunchInfo Evaluate(string requiredProfileRoot, bool allowWorkbenchWithoutLaunchId)
	{
		DC_LaunchInfo info = new DC_LaunchInfo();
		info.m_sMode = MODE_REFUSED;

		// G1: the E-11 facts. GetCLIParam: Core/generated/System/System.c:33, IsCLIParam :45 (both unchanged in 1.9)
		string launchId;
		info.m_bCliLaunchId = System.GetCLIParam(CLI_LAUNCH_ID, launchId);
		launchId = launchId.Trim();

		string profile;
		info.m_bCliProfile = System.GetCLIParam(CLI_PROFILE, profile);
		if (info.m_bCliProfile)
			info.m_sProfileCli = profile.Trim();

		info.m_bCliVisible = info.m_bCliLaunchId || info.m_bCliProfile || System.IsCLIParam(CLI_LAUNCH_ID);
		info.m_sProfileAbs = ResolveProfileAbsolute();

		// G2: the probe guard has no off switch
		if (NormalizePathForCompare(requiredProfileRoot).IsEmpty())
			return Refuse(info, "required profile root is empty (the probe guard has no off switch)");

		if (info.m_bCliLaunchId)
			return EvaluateCli(info, launchId, requiredProfileRoot);

		// G8: a visible command line without -dcLaunchId means Workbench was not started by the patched launcher
		if (info.m_bCliProfile)
			return Refuse(info, "started without -dcLaunchId: start Workbench with E:\\DriveDiT\\tools\\launch\\wb_R1.bat");

		// G9
		if (!allowWorkbenchWithoutLaunchId)
			return Refuse(info, "no -dcLaunchId and the Workbench escape is off");

		return EvaluateEscape(info, requiredProfileRoot);
	}

	//------------------------------------------------------------------------------------------------
	//! G3-G7: -dcLaunchId is on the command line
	protected static DC_LaunchInfo EvaluateCli(notnull DC_LaunchInfo info, string launchId, string requiredProfileRoot)
	{
		info.m_sLaunchId = launchId;

		// G3: the id becomes part of a file path, so only id characters pass
		if (!IsValidLaunchId(launchId))
			return Refuse(info, "-dcLaunchId '" + launchId + "' is not a launch id");

		// G4
		if (!info.m_bCliProfile)
			return Refuse(info, "-dcLaunchId visible but -profile is not");

		if (!IsPathInside(info.m_sProfileCli, requiredProfileRoot))
			return Refuse(info, "-profile '" + info.m_sProfileCli + "' is not inside " + requiredProfileRoot);

		// G5: in Workbench the resolved $profile as well, when it could be resolved
		if (!info.m_sProfileAbs.IsEmpty() && !IsPathInside(info.m_sProfileAbs, requiredProfileRoot))
			return Refuse(info, "$profile '" + info.m_sProfileAbs + "' is not inside " + requiredProfileRoot);

		// G6
		info.m_sRecordPath = LAUNCH_DIR + "/launch_" + launchId + ".json";
		if (!FileIO.FileExists(info.m_sRecordPath))
			return Refuse(info, "no launch record " + info.m_sRecordPath + " (Workbench not started by the patched wb_R1.bat, or the record was removed)");

		// G7
		string problem = CheckLaunchRecord(info, launchId);
		if (!problem.IsEmpty())
			return Refuse(info, "launch record " + info.m_sRecordPath + " " + problem);

		info.m_bAllowed = true;
		info.m_sMode = MODE_CLI;
		return info;
	}

	//------------------------------------------------------------------------------------------------
	//! G10-G14: no command line visible, escape enabled
	protected static DC_LaunchInfo EvaluateEscape(notnull DC_LaunchInfo info, string requiredProfileRoot)
	{
#ifndef WORKBENCH
		// G10
		return Refuse(info, "no -dcLaunchId (the escape exists only in Workbench)");
#else
		// G11
		if (info.m_sProfileAbs.IsEmpty())
			return Refuse(info, "$profile could not be resolved");

		if (!IsPathInside(info.m_sProfileAbs, requiredProfileRoot))
			return Refuse(info, "$profile '" + info.m_sProfileAbs + "' is not inside " + requiredProfileRoot);

		// G12
		string recordPath = FindNewestLaunchRecord(GetExpectedExe());
		if (recordPath.IsEmpty())
			return Refuse(info, "no launch record under " + LAUNCH_DIR);

		// G13: the id comes from the file name and must match the record's own launch_id
		string fileName = FilePath.StripPath(recordPath);
		string launchId = fileName.Substring(7, fileName.Length() - 12);	// "launch_" + id + ".json"
		info.m_sLaunchId = launchId;
		info.m_sRecordPath = recordPath;
		string problem = CheckLaunchRecord(info, launchId);
		if (!problem.IsEmpty())
			return Refuse(info, "launch record " + recordPath + " " + problem);

		// G14: a record copied before the deploy belongs to an earlier Workbench start (both YYYY-MM-DDTHH:MM:SSZ)
		string deployUtc = DC_BuildInfo.DEPLOY_UTC;
		if (deployUtc.IsEmpty() || info.m_sRecordUtc.Compare(deployUtc) < 0)
			return Refuse(info, "newest launch record " + launchId + " predates the deploy " + deployUtc + ": restart Workbench with wb_R1.bat");

		info.m_bAllowed = true;
		info.m_sMode = MODE_ESCAPE;
		return info;
#endif
	}

	//------------------------------------------------------------------------------------------------
	//! Reads the record at info.m_sRecordPath into info and checks it against launchId and this exe.
	//! \return empty string when the record is acceptable, otherwise what is wrong (one line)
	protected static string CheckLaunchRecord(notnull DC_LaunchInfo info, string launchId)
	{
		if (!ReadLaunchRecord(info.m_sRecordPath, info))
			return "is unreadable or misses schema, launch_id, status, exe, utc, pin.pin_id or pin.sha256";

		if (!info.m_sRecordSchema.StartsWith("dd-launch/"))
			return "has schema '" + info.m_sRecordSchema + "', not dd-launch/<n>";

		if (info.m_sRecordStatus != "ok")
			return "has status '" + info.m_sRecordStatus + "', not ok";

		if (!IsValidLaunchId(info.m_sRecordLaunchId) || info.m_sRecordLaunchId != launchId)
			return "is for launch '" + info.m_sRecordLaunchId + "', not '" + launchId + "'";

		string exe = GetExpectedExe();
		if (info.m_sRecordExe != exe)
			return "is for exe '" + info.m_sRecordExe + "', this is '" + exe + "'";

		// G14 compares it as text, so the form must be exact
		if (info.m_sRecordUtc.Length() != UTC_LENGTH || !info.m_sRecordUtc.EndsWith("Z"))
			return "has utc '" + info.m_sRecordUtc + "', not YYYY-MM-DDTHH:MM:SSZ";

		return string.Empty;
	}

	//------------------------------------------------------------------------------------------------
	//! Marks info refused with reason (mode "refused")
	//! \return info
	protected static DC_LaunchInfo Refuse(notnull DC_LaunchInfo info, string reason)
	{
		info.m_bAllowed = false;
		info.m_sMode = MODE_REFUSED;
		info.m_sReason = reason;
		return info;
	}

	//------------------------------------------------------------------------------------------------
	//! \return "wb" in Workbench, "client" otherwise (the exe a launch record must name)
	static string GetExpectedExe()
	{
#ifdef WORKBENCH
		return "wb";
#else
		return "client";
#endif
	}

	//------------------------------------------------------------------------------------------------
	//! Same rule as SCR_MLDataCollector.NormalizePathForCompare (DD-1, SCR_MLDataCollector.c:201): trim, drop '"',
	//! '\' -> '/', lower case, no trailing '/'. "E:\DriveDiT\" -> "e:/drivedit"
	static string NormalizePathForCompare(string path)
	{
		string normalized = path.Trim();
		normalized.Replace("\"", "");
		normalized.Replace("\\", "/");
		normalized.ToLower();
		while (normalized.EndsWith("/"))
		{
			normalized = normalized.Substring(0, normalized.Length() - 1);
		}

		return normalized;
	}

	//------------------------------------------------------------------------------------------------
	//! \return true when path equals root or lies below it (both normalised); false when either is empty. Unlike DD-1's
	//! IsProfileAllowed an empty root refuses: the probe guard has no off switch.
	static bool IsPathInside(string path, string root)
	{
		string normalizedPath = NormalizePathForCompare(path);
		string normalizedRoot = NormalizePathForCompare(root);
		if (normalizedPath.IsEmpty() || normalizedRoot.IsEmpty())
			return false;

		return normalizedPath == normalizedRoot || normalizedPath.StartsWith(normalizedRoot + "/");
	}

	//------------------------------------------------------------------------------------------------
	//! Launch ids are YYYYMMDDTHHMMSSZ-<exe>-<hex> (tools/pin/preflight.py new_launch_id): 20..64 characters, only
	//! [0-9A-Za-z-]. Rejects anything that could change the record path ('/', '\', '.', ':', spaces)
	static bool IsValidLaunchId(string launchId)
	{
		int length = launchId.Length();
		if (length < LAUNCH_ID_MIN_LENGTH || length > LAUNCH_ID_MAX_LENGTH)
			return false;

		for (int i = 0; i < length; i++)
		{
			int c = launchId.ToAscii(i);
			bool digit = c >= 48 && c <= 57;		// 0-9
			bool upper = c >= 65 && c <= 90;		// A-Z
			bool lower = c >= 97 && c <= 122;		// a-z
			if (!digit && !upper && !lower && c != 45)	// 45 = '-'
				return false;
		}

		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Parses a launch record (JsonLoadContext.LoadFromFile) into info.
	//! \return false when unreadable or when schema, launch_id, status, exe, utc, pin.pin_id or pin.sha256 is missing
	static bool ReadLaunchRecord(string path, notnull DC_LaunchInfo info)
	{
		// skipEmptyObjects = false: StartObject fails on a missing object instead of reading defaults
		JsonLoadContext context = new JsonLoadContext(false);
		if (!context.LoadFromFile(path))
			return false;

		string schema;
		string launchId;
		string status;
		string utc;
		string exe;
		if (!context.ReadValue("schema", schema) || !context.ReadValue("launch_id", launchId) || !context.ReadValue("status", status))
			return false;

		if (!context.ReadValue("utc", utc) || !context.ReadValue("exe", exe))
			return false;

		info.m_sRecordSchema = schema;
		info.m_sRecordLaunchId = launchId;
		info.m_sRecordStatus = status;
		info.m_sRecordUtc = utc;
		info.m_sRecordExe = exe;

		// Optional dd-launch/2 keys, read before entering any child object
		string text;
		if (context.DoesKeyExist("build_key") && context.ReadValue("build_key", text))
			info.m_sBuildKey = text;

		if (context.DoesKeyExist("channel") && context.ReadValue("channel", text))
			info.m_sChannel = text;

		if (context.DoesKeyExist("use_class") && context.ReadValue("use_class", text))
			info.m_sUseClass = text;

		if (!context.StartObject("pin"))
			return false;

		string pinId;
		string pinSha256;
		bool pinOk = context.ReadValue("pin_id", pinId) && context.ReadValue("sha256", pinSha256);
		context.EndObject();
		if (!pinOk || pinId.IsEmpty() || pinSha256.IsEmpty())
			return false;

		info.m_sPinId = pinId;
		info.m_sPinSha256 = pinSha256;

		// observed.<app>.buildid: logged only, never required
		if (context.DoesKeyExist("observed") && context.StartObject("observed"))
		{
			info.m_iGameBuildId = ReadBuildId(context, "game");
			info.m_iToolsBuildId = ReadBuildId(context, "tools");
			context.EndObject();
		}

		return !schema.IsEmpty() && !launchId.IsEmpty() && !status.IsEmpty() && !utc.IsEmpty() && !exe.IsEmpty();
	}

	//------------------------------------------------------------------------------------------------
	//! \return observed.<app>.buildid of the open "observed" object, -1 when absent
	protected static int ReadBuildId(notnull JsonLoadContext context, string app)
	{
		if (!context.DoesKeyExist(app) || !context.StartObject(app))
			return -1;

		int buildId = -1;
		if (!context.ReadValue("buildid", buildId))
			buildId = -1;

		context.EndObject();
		return buildId;
	}

	//------------------------------------------------------------------------------------------------
	//! Escape mode: the record $profile:dc_launch/launch_*-<exe>-*.json with the greatest name (ids sort by UTC);
	//! empty when there is none (FileIO.FindFiles + FilePath.StripPath + string.Compare)
	static string FindNewestLaunchRecord(string exe)
	{
		array<string> files = {};
		FileIO.FindFiles(files.Insert, LAUNCH_DIR, ".json");	// vanilla pattern: SCR_ControllerPresetsSettingsSubMenu.c:240

		string marker = "-" + exe + "-";
		string newest;
		foreach (string file : files)
		{
			string name = FilePath.StripPath(file);
			if (!name.StartsWith("launch_") || !name.EndsWith(".json") || !name.Contains(marker))
				continue;

			string launchId = name.Substring(7, name.Length() - 12);
			if (!IsValidLaunchId(launchId))
				continue;

			if (newest.IsEmpty() || name.Compare(newest) > 0)
				newest = name;
		}

		if (newest.IsEmpty())
			return string.Empty;

		return LAUNCH_DIR + "/" + newest;
	}

	//------------------------------------------------------------------------------------------------
	//! Workbench only (#ifdef WORKBENCH): Workbench.GetAbsolutePath("$profile:", abs, false); empty otherwise.
	//! Vanilla calls GetAbsolutePath under the same define (Game/Helpers/SCR_FileIOHelper.c:84-91).
	static string ResolveProfileAbsolute()
	{
#ifdef WORKBENCH
		string absolutePath;
		if (Workbench.GetAbsolutePath("$profile:", absolutePath, false) && !absolutePath.IsEmpty())
			return absolutePath;

		// Fallback: a folder inside the profile still answers "is $profile inside the root"
		if (Workbench.GetAbsolutePath(LAUNCH_DIR, absolutePath, false) && !absolutePath.IsEmpty())
			return absolutePath;
#endif
		return string.Empty;
	}

	//------------------------------------------------------------------------------------------------
	//! "1" / "0" for key=value files and CSVs
	static string BoolText(bool value)
	{
		if (value)
			return "1";

		return "0";
	}
}
