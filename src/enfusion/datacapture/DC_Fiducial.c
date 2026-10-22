// ============================================================================
// DC_Fiducial - fiducial v2: the render-frame code drawn into every capture image (DD-4-probe, r2 3.4)
// ============================================================================
//
// First version of a production class: DD-4-min reuses it, and any change re-runs E-05, E-06 and E-06A.
//
// WHAT IT DRAWS
//   - code = render_frame mod 2^18 (unique for 73 min at 60 FPS; sessions are at most 30 min);
//   - 6 patches x 3 bits in a left band, mirrored (same values, same order) in a right band; patch 0 is the top
//     patch and carries the 3 least significant bits: value_i = (code >> (3 * i)) & 7;
//   - a patch value is a corner of the RGB cube: bit 0 red, bit 1 green, bit 2 blue, each channel 0 or 255;
//   - each patch is P x P physical pixels. Layout on the physical canvas W x H (the workspace in physical pixels),
//     integer arithmetic: left x0 = M, right x0 = W - M - P, y_i = (H - (6 P + 5 G)) / 2 + i (P + G), i = 0..5 from
//     the top. P = 32, M = 16, G = 8 by default; for the 1408x720 client class the patches sit inside the 64 px
//     bands (left x0 16, right x0 1360, y0 244). The Python decoder (data/dc2/fiducial.py) uses the same rule.
//
// HOW (vanilla 1.8.0.13 APIs, PROBE_APIS.md 4.4)
//   - the user DPI scale is saved and forced to 1.0 (WorkspaceWidget.SetUserDPIScale) so one reference unit maps to
//     a fixed number of physical pixels; the physical canvas is DPIScale(GetWidth()) x DPIScale(GetHeight());
//   - 12 textureless ImageWidgets (WorkspaceWidget.CreateWidgetInWorkspace, ImageWidgetTypeID, as vanilla
//     SCR_SlotLabelsComponent2.c:22 uses a coloured ImageWidget) are placed with FrameSlot.SetPos/SetSize in
//     reference units = DPIUnscale(physical px). Widgets are pixel-perfect by default, so they snap to whole pixels;
//   - Show(renderFrame) sets the 12 colours from 8 cached Colors. Call it first in every EOnFrame. It re-runs the
//     layout when the workspace size changes (window resize, or a DPI change that only applied a frame later);
//   - Destroy() removes the widgets and restores the saved user DPI scale. Idempotent.
//
// Failure handling: Create returns false (GetFailure) when there is no workspace, a widget cannot be created or the
// canvas is too small for the bands; it then removes what it created and restores the DPI scale.
// ============================================================================

//------------------------------------------------------------------------------------------------
class DC_Fiducial
{
	static const int CODE_BITS = 18;
	static const int CODE_MODULUS = 262144;
	static const int PATCH_COUNT = 6;
	static const int BITS_PER_PATCH = 3;

	//! Widget sort order of the patches: in front of the game UI (vanilla autotest overlays use 1024)
	static const int WIDGET_SORT = 1024;

	protected ref array<Widget> m_aLeft = {};		//!< top to bottom
	protected ref array<Widget> m_aRight = {};		//!< top to bottom
	protected ref array<ref Color> m_aColors = {};	//!< index = patch value 0..7
	protected ref array<int> m_aY = {};				//!< physical y of each patch row, top to bottom

	protected bool m_bCreated;
	protected bool m_bDpiSaved;
	protected float m_fDpiUserBefore = 1;
	protected float m_fDpiUserNow = 1;
	protected int m_iPatchPx;
	protected int m_iMarginPx;
	protected int m_iGapPx;
	protected int m_iWorkspaceW;					//!< WorkspaceWidget.GetWidth() at the last layout (reference units)
	protected int m_iWorkspaceH;
	protected int m_iPhysicalW;						//!< DPIScale(GetWidth()) rounded, physical px
	protected int m_iPhysicalH;
	protected float m_fPatchRefUnits;
	protected int m_iLeftX;
	protected int m_iRightX;
	protected int m_iLastCode = -1;
	protected int m_iRelayouts;
	protected string m_sFailure;

	//------------------------------------------------------------------------------------------------
	//! \return renderFrame mod 2^18, never negative
	static int CodeFor(int renderFrame)
	{
		int code = renderFrame % CODE_MODULUS;
		if (code < 0)
			code += CODE_MODULUS;

		return code;
	}

	//------------------------------------------------------------------------------------------------
	//! \return value 0..7 of patch index (0 = top patch) of a code
	static int PatchValue(int code, int index)
	{
		return (code >> (BITS_PER_PATCH * index)) & 7;
	}

	//------------------------------------------------------------------------------------------------
	//! \return colour of a patch value: bit 0 red, bit 1 green, bit 2 blue; each channel 0 or 255; alpha 255
	static Color PatchColor(int value)
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

		return Color.FromRGBA(red, green, blue, 255);
	}

	//------------------------------------------------------------------------------------------------
	//! Saves the user DPI scale, forces it to 1.0 and creates the 12 patch widgets.
	//! \return false (GetFailure) on any failure; nothing is left behind then
	bool Create(int patchPx, int marginPx, int gapPx)
	{
		if (m_bCreated)
		{
			m_sFailure = "fiducial already created";
			return false;
		}

		if (patchPx < 8 || marginPx < 0 || gapPx < 0)
		{
			m_sFailure = "fiducial patch " + patchPx.ToString() + " px, margin " + marginPx.ToString() + " px, gap " + gapPx.ToString() + " px: patch must be >= 8 px, margin and gap >= 0";
			return false;
		}

		WorkspaceWidget workspace = GetWorkspace();
		if (!workspace)
		{
			m_sFailure = "no workspace widget (GetGame().GetWorkspace() is null)";
			return false;
		}

		m_iPatchPx = patchPx;
		m_iMarginPx = marginPx;
		m_iGapPx = gapPx;

		m_fDpiUserBefore = workspace.GetUserDPIScale();
		m_bDpiSaved = true;
		workspace.SetUserDPIScale(1.0);
		m_fDpiUserNow = workspace.GetUserDPIScale();

		m_aColors.Clear();
		for (int value = 0; value < 8; value++)
		{
			m_aColors.Insert(PatchColor(value));
		}

		if (!ComputeLayout(workspace))
		{
			Destroy();
			return false;
		}

		// Integer reference units here; PlacePatches then sets the exact float positions and sizes.
		int sizeRef = Math.Round(m_fPatchRefUnits);
		int leftRef = Math.Round(workspace.DPIUnscale(m_iLeftX));
		int rightRef = Math.Round(workspace.DPIUnscale(m_iRightX));
		for (int i = 0; i < PATCH_COUNT; i++)
		{
			int topRef = Math.Round(workspace.DPIUnscale(m_aY[i]));
			Widget left = workspace.CreateWidgetInWorkspace(WidgetType.ImageWidgetTypeID, leftRef, topRef, sizeRef, sizeRef, WidgetFlags.VISIBLE | WidgetFlags.IGNORE_CURSOR | WidgetFlags.NOFOCUS, m_aColors[0], WIDGET_SORT);
			Widget right = workspace.CreateWidgetInWorkspace(WidgetType.ImageWidgetTypeID, rightRef, topRef, sizeRef, sizeRef, WidgetFlags.VISIBLE | WidgetFlags.IGNORE_CURSOR | WidgetFlags.NOFOCUS, m_aColors[0], WIDGET_SORT);
			if (left)
				m_aLeft.Insert(left);

			if (right)
				m_aRight.Insert(right);

			if (!left || !right)
			{
				m_sFailure = "CreateWidgetInWorkspace(ImageWidgetTypeID) returned null for patch " + i.ToString();
				Destroy();
				return false;
			}
		}

		m_bCreated = true;
		PlacePatches(workspace);
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Physical canvas and patch positions from the current workspace size (r2 3.4 layout, integer arithmetic).
	//! \return false (GetFailure) when the canvas is too small for both bands
	protected bool ComputeLayout(notnull WorkspaceWidget workspace)
	{
		m_iWorkspaceW = workspace.GetWidth();
		m_iWorkspaceH = workspace.GetHeight();
		m_iPhysicalW = Math.Round(workspace.DPIScale(m_iWorkspaceW));
		m_iPhysicalH = Math.Round(workspace.DPIScale(m_iWorkspaceH));
		m_fPatchRefUnits = workspace.DPIUnscale(m_iPatchPx);

		int columnPx = PATCH_COUNT * m_iPatchPx + (PATCH_COUNT - 1) * m_iGapPx;
		int topPx = (m_iPhysicalH - columnPx) / 2;
		m_iLeftX = m_iMarginPx;
		m_iRightX = m_iPhysicalW - m_iMarginPx - m_iPatchPx;

		m_aY.Clear();
		for (int i = 0; i < PATCH_COUNT; i++)
		{
			m_aY.Insert(topPx + i * (m_iPatchPx + m_iGapPx));
		}

		if (topPx < 0 || m_iRightX < m_iLeftX + m_iPatchPx)
		{
			m_sFailure = "physical canvas " + m_iPhysicalW.ToString() + "x" + m_iPhysicalH.ToString() + " px is too small for the fiducial bands";
			return false;
		}

		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Exact float positions and sizes in reference units (the integer arguments of CreateWidgetInWorkspace round)
	protected void PlacePatches(notnull WorkspaceWidget workspace)
	{
		for (int i = 0, count = m_aLeft.Count(); i < count; i++)
		{
			float top = workspace.DPIUnscale(m_aY[i]);
			FrameSlot.SetPos(m_aLeft[i], workspace.DPIUnscale(m_iLeftX), top);
			FrameSlot.SetSize(m_aLeft[i], m_fPatchRefUnits, m_fPatchRefUnits);
			FrameSlot.SetPos(m_aRight[i], workspace.DPIUnscale(m_iRightX), top);
			FrameSlot.SetSize(m_aRight[i], m_fPatchRefUnits, m_fPatchRefUnits);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! Sets the 12 patch colours to the code of renderFrame. Call it first in every EOnFrame.
	void Show(int renderFrame)
	{
		if (!m_bCreated)
			return;

		WorkspaceWidget workspace = GetWorkspace();
		if (workspace && (workspace.GetWidth() != m_iWorkspaceW || workspace.GetHeight() != m_iWorkspaceH))
		{
			// Window resize, or the forced DPI scale only applied after Create: lay the bands out again.
			if (ComputeLayout(workspace))
				PlacePatches(workspace);

			m_iRelayouts++;
		}

		int code = CodeFor(renderFrame);
		for (int i = 0, count = m_aLeft.Count(); i < count; i++)
		{
			Color color = m_aColors[PatchValue(code, i)];
			m_aLeft[i].SetColor(color);
			m_aRight[i].SetColor(color);
		}

		m_iLastCode = code;
	}

	//------------------------------------------------------------------------------------------------
	//! \return code shown by the last Show, -1 before the first
	int GetLastCode()
	{
		return m_iLastCode;
	}

	//------------------------------------------------------------------------------------------------
	//! \return number of times Show re-ran the layout after a workspace size change
	int GetRelayouts()
	{
		return m_iRelayouts;
	}

	//------------------------------------------------------------------------------------------------
	bool IsCreated()
	{
		return m_bCreated;
	}

	//------------------------------------------------------------------------------------------------
	string GetFailure()
	{
		return m_sFailure;
	}

	//------------------------------------------------------------------------------------------------
	//! Removes the patch widgets and restores the user DPI scale saved by Create. Idempotent.
	void Destroy()
	{
		foreach (Widget left : m_aLeft)
		{
			if (left)
				left.RemoveFromHierarchy();
		}

		foreach (Widget right : m_aRight)
		{
			if (right)
				right.RemoveFromHierarchy();
		}

		m_aLeft.Clear();
		m_aRight.Clear();
		m_bCreated = false;

		if (!m_bDpiSaved)
			return;

		m_bDpiSaved = false;
		WorkspaceWidget workspace = GetWorkspace();
		if (workspace)
			workspace.SetUserDPIScale(m_fDpiUserBefore);
	}

	//------------------------------------------------------------------------------------------------
	//! Appends the fiducial.* key=value lines (geometry in physical px, top-left corners) to lines
	void AppendHeaderLines(notnull array<string> lines)
	{
		lines.Insert("fiducial.version=v2");
		lines.Insert("fiducial.code_bits=" + CODE_BITS.ToString());
		lines.Insert("fiducial.patch_count=" + PATCH_COUNT.ToString());
		lines.Insert("fiducial.bits_per_patch=" + BITS_PER_PATCH.ToString());
		lines.Insert("fiducial.created=" + BoolText(m_bCreated));
		lines.Insert("fiducial.patch_px=" + m_iPatchPx.ToString());
		lines.Insert("fiducial.margin_px=" + m_iMarginPx.ToString());
		lines.Insert("fiducial.gap_px=" + m_iGapPx.ToString());
		lines.Insert("fiducial.workspace_w=" + m_iWorkspaceW.ToString());
		lines.Insert("fiducial.workspace_h=" + m_iWorkspaceH.ToString());
		lines.Insert("fiducial.physical_w=" + m_iPhysicalW.ToString());
		lines.Insert("fiducial.physical_h=" + m_iPhysicalH.ToString());
		lines.Insert("fiducial.dpi_user_before=" + m_fDpiUserBefore.ToString(-1, 4));
		lines.Insert("fiducial.dpi_user_now=" + m_fDpiUserNow.ToString(-1, 4));
		lines.Insert("fiducial.patch_ref_units=" + m_fPatchRefUnits.ToString(-1, 6));
		lines.Insert("fiducial.left_x=" + m_iLeftX.ToString());
		lines.Insert("fiducial.right_x=" + m_iRightX.ToString());
		for (int i = 0, count = m_aY.Count(); i < count; i++)
		{
			lines.Insert("fiducial.y" + i.ToString() + "=" + m_aY[i].ToString());
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
