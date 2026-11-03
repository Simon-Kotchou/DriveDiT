// ============================================================================
// DC_FiducialShape - fiducial v2 drawn as debug Shapes in front of the rig camera (DD-4-probe)
// ============================================================================
//
// Second carrier of fiducial v2; DC_Fiducial.c is the widget carrier of r2 3.4. Probe run 1 (2026-10-22, Workbench
// 1.8.0.13) found no widget patch in any System.MakeScreenshot file, while the E-06A debug spheres
// (Shape.CreateSphere) kept their exact colour in the same files (t0/run1/FIDUCIAL.md 1.3). This class draws the same
// code with the API that demonstrably reaches the files. Any change re-runs E-05, E-06 and E-06A.
//
// WHAT IT DRAWS
//   - the v2 code of DC_Fiducial: code = render_frame mod 2^18 (DC_Fiducial.CodeFor), 6 patches x 3 bits, patch 0 on
//     top with the 3 least significant bits (DC_Fiducial.PatchValue), a value is an RGB-cube corner: bit 0 red, bit 1
//     green, bit 2 blue, ARGB(255, 0 or 255, 0 or 255, 0 or 255) (Core/proto/proto.c:67); the right band mirrors the
//     left one;
//   - the r2 3.4 layout rule on the canvas W x H = WorkspaceWidget.GetWidth() x GetHeight() (physical px, the
//     screenshot size in Workbench, WorkspaceWidget.c:37-40): left x0 = M, right x0 = W - M - P,
//     y_i = (H - (6 P + 5 G)) / 2 + i (P + G) (DC_Fiducial.LayoutTopPx), with its own margin M (the probes default to
//     64 = beside the widget bands at 16, so one image can carry both carriers);
//   - each patch is one flat box (Shape.Create(ShapeType.BBOX, color, flags, mins, maxs), Shape.c:19, ShapeType.c:17;
//     a flat box with min z = max z in a camera-relative frame, then SetMatrix, as vanilla
//     SCR_UnitDisplayComponent does, UnitDisplayComponent.c:149-158) at depth D in the camera frame (x right, y up,
//     z forward). With f = (H / 2) / tan(vFOV / 2) the pinhole model that E-06A verified against
//     ProjWorldToScreenNative (run 1: p95 0.046 px) maps pixel column u to X = (u - W / 2) D / f and row v to
//     Y = (H / 2 - v) D / f, so a patch covers exactly its P x P pixels; D must exceed the near plane;
//   - flags NOZBUFFER | NOZWRITE | NOOUTLINE | DOUBLESIDE | NOCULL (ShapeFlags.c:16, 18, 24, 28, 32): no depth test
//     (drawn over the bonnet and anything closer than D), solid faces only, never culled.
//
// HOW
//   - Show(renderFrame) sets the 12 colours (Shape.SetColor, Shape.c:112) from 8 cached ARGB values. It re-creates the
//     shapes on the new layout when the workspace size changed (window resize);
//   - Place(mat) moves all 12 with one camera world transform (Shape.SetMatrix, Shape.c:110). When to place is the
//     caller's choice: the camera applies its transform in EOnPostFrame (CameraBase.c:18), so the probes place after
//     the physics update and measure the result (DC_FiducialCarriers, DC_ProbeFiducial.c);
//   - ResidualPx(cameraMat) is the largest image distance, over the 12 patch centres, between where the shapes of the
//     last Place appear to a camera with transform cameraMat and where the layout puts them (0 = exact);
//   - Destroy() drops the shapes (a Shape is drawn while a reference to it lives, Shape.c:15-17). Idempotent.
//
// Failure handling: Create returns false (GetFailure) for a bad parameter, a depth not beyond the near plane, a canvas
// too small for the bands or a Shape.Create that returned null; nothing is left behind then.
// ============================================================================

//------------------------------------------------------------------------------------------------
//! Which fiducial v2 carrier a probe draws (DC_ProbeImages, DC_ProbeMarkers: m_eFiducialCarrier)
enum DC_EFiducialCarrier
{
	WIDGET,		//!< DC_Fiducial: 12 ImageWidgets in the game workspace (the r2 3.4 design)
	SHAPE,		//!< DC_FiducialShape: 12 debug Shapes in front of the rig camera
	BOTH		//!< both, at different margins, so one Play compares them
}

//------------------------------------------------------------------------------------------------
class DC_FiducialShape
{
	static const int PATCH_COUNT = 6;
	//! Smallest depth along the reference camera's forward axis that ResidualPx projects [m]
	static const float MIN_PROJECT_DEPTH_M = 0.001;
	//! ResidualPx when a patch centre lies behind the reference camera [px]
	static const float BEHIND_PX = 100000;
	static const string FLAGS_TEXT = "NOZBUFFER|NOZWRITE|NOOUTLINE|DOUBLESIDE|NOCULL";

	protected ref array<ref Shape> m_aShapes = {};		//!< left band top to bottom, then the right band
	protected ref array<int> m_aColors = {};			//!< ARGB, index = patch value 0..7
	protected ref array<int> m_aPatchX = {};			//!< per shape: physical x of the top-left pixel
	protected ref array<int> m_aPatchY = {};			//!< per shape: physical y of the top-left pixel
	protected ref array<int> m_aPatchIndex = {};		//!< per shape: patch index 0..5 (0 = top)
	protected ref array<vector> m_aLocalCentre = {};	//!< per shape: patch centre in the camera frame

	protected bool m_bCreated;
	protected bool m_bPlaced;
	protected vector m_vPlaced[4];						//!< camera world transform of the last Place
	protected int m_iPatchPx;
	protected int m_iMarginPx;
	protected int m_iGapPx;
	protected float m_fDepthM;
	protected float m_fVFovDeg;
	protected float m_fNearM;
	protected int m_iCanvasW;
	protected int m_iCanvasH;
	protected float m_fFocalPx;
	protected int m_iLeftX;
	protected int m_iRightX;
	protected int m_iTopY;
	protected int m_iLastCode = -1;
	protected int m_iRelayouts;
	protected int m_iPlaceCount;
	protected string m_sFailure;

	//------------------------------------------------------------------------------------------------
	//! Creates the 12 patch shapes on the canvasW x canvasH layout for a camera with vertical FOV vFovDeg.
	//! \return false (GetFailure) on any failure; nothing is left behind then
	bool Create(int canvasW, int canvasH, float vFovDeg, float nearM, int patchPx, int marginPx, int gapPx, float depthM)
	{
		if (m_bCreated)
		{
			m_sFailure = "shape fiducial already created";
			return false;
		}

		if (patchPx < 8 || marginPx < 0 || gapPx < 0)
		{
			m_sFailure = "shape fiducial patch " + patchPx.ToString() + " px, margin " + marginPx.ToString() + " px, gap " + gapPx.ToString() + " px: patch must be >= 8 px, margin and gap >= 0";
			return false;
		}

		if (vFovDeg <= 0 || vFovDeg >= 180)
		{
			m_sFailure = "shape fiducial: camera vertical FOV " + vFovDeg.ToString(-1, 3) + " deg is not in (0, 180)";
			return false;
		}

		if (depthM <= nearM)
		{
			m_sFailure = "shape fiducial depth " + depthM.ToString(-1, 3) + " m is not beyond the near plane " + nearM.ToString(-1, 3) + " m";
			return false;
		}

		m_iPatchPx = patchPx;
		m_iMarginPx = marginPx;
		m_iGapPx = gapPx;
		m_fDepthM = depthM;
		m_fVFovDeg = vFovDeg;
		m_fNearM = nearM;

		m_aColors.Clear();
		for (int value = 0; value < 8; value++)
		{
			m_aColors.Insert(PatchArgb(value));
		}

		if (!BuildShapes(canvasW, canvasH))
		{
			Destroy();
			return false;
		}

		m_bCreated = true;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! \return ARGB of a patch value: bit 0 red, bit 1 green, bit 2 blue; each channel 0 or 255; alpha 255
	static int PatchArgb(int value)
	{
		int red = 0;
		int green = 0;
		int blue = 0;
		if ((value & 1) != 0)
			red = 255;

		if ((value & 2) != 0)
			green = 255;

		if ((value & 4) != 0)
			blue = 255;

		return ARGB(255, red, green, blue);
	}

	//------------------------------------------------------------------------------------------------
	//! Layout of the canvas and one shape per patch (colours of the last code, transform of the last Place)
	//! \return false (GetFailure) when the canvas is too small or a shape cannot be created
	protected bool BuildShapes(int canvasW, int canvasH)
	{
		m_aShapes.Clear();
		m_aPatchX.Clear();
		m_aPatchY.Clear();
		m_aPatchIndex.Clear();
		m_aLocalCentre.Clear();

		m_iCanvasW = canvasW;
		m_iCanvasH = canvasH;
		m_fFocalPx = 0.5 * canvasH / Math.Tan(0.5 * m_fVFovDeg * Math.DEG2RAD);
		m_iLeftX = m_iMarginPx;
		m_iRightX = canvasW - m_iMarginPx - m_iPatchPx;
		m_iTopY = DC_Fiducial.LayoutTopPx(canvasH, m_iPatchPx, m_iGapPx);
		if (canvasW <= 0 || canvasH <= 0 || m_iTopY < 0 || m_iRightX < m_iLeftX + m_iPatchPx)
		{
			m_sFailure = "shape fiducial canvas " + canvasW.ToString() + "x" + canvasH.ToString() + " px is too small for the bands at margin " + m_iMarginPx.ToString() + " px";
			return false;
		}

		ShapeFlags flags = ShapeFlags.NOZBUFFER | ShapeFlags.NOZWRITE | ShapeFlags.NOOUTLINE | ShapeFlags.DOUBLESIDE | ShapeFlags.NOCULL;
		float metresPerPx = m_fDepthM / m_fFocalPx;
		float halfW = 0.5 * canvasW;
		float halfH = 0.5 * canvasH;
		int startColor = m_aColors[0];
		for (int band = 0; band < 2; band++)
		{
			int x0 = m_iLeftX;
			if (band == 1)
				x0 = m_iRightX;

			for (int i = 0; i < PATCH_COUNT; i++)
			{
				int y0 = m_iTopY + i * (m_iPatchPx + m_iGapPx);
				if (m_iLastCode >= 0)
					startColor = m_aColors[DC_Fiducial.PatchValue(m_iLastCode, i)];

				vector mins = Vector((x0 - halfW) * metresPerPx, (halfH - y0 - m_iPatchPx) * metresPerPx, m_fDepthM);
				vector maxs = Vector((x0 + m_iPatchPx - halfW) * metresPerPx, (halfH - y0) * metresPerPx, m_fDepthM);
				Shape patch = Shape.Create(ShapeType.BBOX, startColor, flags, mins, maxs);
				if (!patch)
				{
					m_sFailure = "Shape.Create(ShapeType.BBOX) returned null for patch " + i.ToString() + " of band " + band.ToString();
					return false;
				}

				if (m_bPlaced)
					patch.SetMatrix(m_vPlaced);

				m_aShapes.Insert(patch);
				m_aPatchX.Insert(x0);
				m_aPatchY.Insert(y0);
				m_aPatchIndex.Insert(i);
				float centreX = x0 + 0.5 * m_iPatchPx;
				float centreY = y0 + 0.5 * m_iPatchPx;
				m_aLocalCentre.Insert(Vector((centreX - halfW) * metresPerPx, (halfH - centreY) * metresPerPx, m_fDepthM));
			}
		}

		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Sets the 12 colours to the code of renderFrame; re-creates the shapes first when the workspace size changed
	void Show(int renderFrame)
	{
		if (!m_bCreated)
			return;

		WorkspaceWidget workspace = GetWorkspace();
		if (workspace && (workspace.GetWidth() != m_iCanvasW || workspace.GetHeight() != m_iCanvasH))
		{
			int canvasW = workspace.GetWidth();
			int canvasH = workspace.GetHeight();
			m_iRelayouts++;
			if (!BuildShapes(canvasW, canvasH))
			{
				// The new size cannot hold the bands: the carrier stops drawing, GetFailure says why
				m_aShapes.Clear();
				m_bCreated = false;
				return;
			}
		}

		int code = DC_Fiducial.CodeFor(renderFrame);
		for (int i = 0, count = m_aShapes.Count(); i < count; i++)
		{
			m_aShapes[i].SetColor(m_aColors[DC_Fiducial.PatchValue(code, m_aPatchIndex[i])]);
		}

		m_iLastCode = code;
	}

	//------------------------------------------------------------------------------------------------
	//! Moves the 12 shapes with the camera world transform mat (rows right, up, forward, position)
	void Place(vector mat[4])
	{
		if (!m_bCreated)
			return;

		m_vPlaced[0] = mat[0];
		m_vPlaced[1] = mat[1];
		m_vPlaced[2] = mat[2];
		m_vPlaced[3] = mat[3];
		m_bPlaced = true;
		m_iPlaceCount++;
		foreach (Shape patch : m_aShapes)
		{
			patch.SetMatrix(m_vPlaced);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! Largest image distance [px] over the 12 patch centres between their layout position and where the shapes of
	//! the last Place appear to a pinhole camera (this canvas and focal length) with world transform cameraMat.
	//! \return -1 before the first Place, BEHIND_PX when a centre lies behind that camera
	float ResidualPx(vector cameraMat[4])
	{
		if (!m_bPlaced || m_aLocalCentre.IsEmpty())
			return -1;

		// Relative to the reference camera first: world coordinates are ~1e3 m, the patches ~1 m from the camera
		vector offset = m_vPlaced[3] - cameraMat[3];
		float halfW = 0.5 * m_iCanvasW;
		float halfH = 0.5 * m_iCanvasH;
		float halfPatch = 0.5 * m_iPatchPx;
		float worst = 0;
		for (int i = 0, count = m_aLocalCentre.Count(); i < count; i++)
		{
			vector centre = m_aLocalCentre[i];
			vector rel = m_vPlaced[0] * centre[0] + m_vPlaced[1] * centre[1] + m_vPlaced[2] * centre[2] + offset;
			float along = vector.Dot(rel, cameraMat[2]);
			if (along <= MIN_PROJECT_DEPTH_M)
				return BEHIND_PX;

			float u = halfW + m_fFocalPx * vector.Dot(rel, cameraMat[0]) / along;
			float v = halfH - m_fFocalPx * vector.Dot(rel, cameraMat[1]) / along;
			float du = u - (m_aPatchX[i] + halfPatch);
			float dv = v - (m_aPatchY[i] + halfPatch);
			float dist = Math.Sqrt(du * du + dv * dv);
			if (dist > worst)
				worst = dist;
		}

		return worst;
	}

	//------------------------------------------------------------------------------------------------
	bool IsCreated()
	{
		return m_bCreated;
	}

	//------------------------------------------------------------------------------------------------
	//! \return code shown by the last Show, -1 before the first
	int GetLastCode()
	{
		return m_iLastCode;
	}

	//------------------------------------------------------------------------------------------------
	//! \return number of times Show re-created the shapes after a workspace size change
	int GetRelayouts()
	{
		return m_iRelayouts;
	}

	//------------------------------------------------------------------------------------------------
	int GetPlaceCount()
	{
		return m_iPlaceCount;
	}

	//------------------------------------------------------------------------------------------------
	string GetFailure()
	{
		return m_sFailure;
	}

	//------------------------------------------------------------------------------------------------
	//! Drops the 12 shapes. Idempotent.
	void Destroy()
	{
		m_aShapes.Clear();
		m_bCreated = false;
	}

	//------------------------------------------------------------------------------------------------
	//! Appends the fiducial.shape.* key=value lines (geometry in physical px, top-left corners) to lines
	void AppendHeaderLines(notnull array<string> lines)
	{
		lines.Insert("fiducial.shape.version=v2");
		lines.Insert("fiducial.shape.created=" + BoolText(m_bCreated));
		lines.Insert("fiducial.shape.type=BBOX");
		lines.Insert("fiducial.shape.flags=" + FLAGS_TEXT);
		lines.Insert("fiducial.shape.patch_px=" + m_iPatchPx.ToString());
		lines.Insert("fiducial.shape.margin_px=" + m_iMarginPx.ToString());
		lines.Insert("fiducial.shape.gap_px=" + m_iGapPx.ToString());
		lines.Insert("fiducial.shape.canvas_w=" + m_iCanvasW.ToString());
		lines.Insert("fiducial.shape.canvas_h=" + m_iCanvasH.ToString());
		lines.Insert("fiducial.shape.canvas_source=GetWidth_GetHeight_physical");
		lines.Insert("fiducial.shape.vfov_deg=" + m_fVFovDeg.ToString(-1, 6));
		lines.Insert("fiducial.shape.focal_px=" + m_fFocalPx.ToString(-1, 4));
		lines.Insert("fiducial.shape.depth_m=" + m_fDepthM.ToString(-1, 4));
		lines.Insert("fiducial.shape.near_m=" + m_fNearM.ToString(-1, 4));
		lines.Insert("fiducial.shape.left_x=" + m_iLeftX.ToString());
		lines.Insert("fiducial.shape.right_x=" + m_iRightX.ToString());
		for (int i = 0; i < PATCH_COUNT; i++)
		{
			int y = m_iTopY + i * (m_iPatchPx + m_iGapPx);
			lines.Insert("fiducial.shape.y" + i.ToString() + "=" + y.ToString());
		}
	}

	//------------------------------------------------------------------------------------------------
	protected WorkspaceWidget GetWorkspace()
	{
		ArmaReforgerScripted game = GetGame();
		if (!game)
			return null;

		return game.GetWorkspace();
	}

	//------------------------------------------------------------------------------------------------
	protected static string BoolText(bool value)
	{
		if (value)
			return "1";

		return "0";
	}
}
