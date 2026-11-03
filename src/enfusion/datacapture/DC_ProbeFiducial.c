// ============================================================================
// DC_ProbeFiducial - fiducial v2 carriers and their diagnostics in the image probes (DD-4-probe)
// ============================================================================
//
// Used by e05 (DC_ProbeImages) and e06a (DC_ProbeMarkers) through m_eFiducialCarrier: WIDGET (DC_Fiducial, the r2 3.4
// ImageWidgets), SHAPE (DC_FiducialShape, debug Shapes in front of the rig camera) or BOTH (default for the Play after
// probe run 3: the shape bands sit at their own margin, beside the widget bands, so one Play compares the carriers).
// Both carriers show the same code of the same render frame.
//
// WHEN THE SHAPES MOVE
//   The camera applies its transform in EOnPostFrame (CameraBase.c:18; ScriptCamera.c:60-77 copies GetWorldTransform
//   into BaseWorld.SetCameraEx there), after the physics update of the frame, so shapes placed in EOnFrame trail the
//   rendered camera by one physics update. The shapes are therefore placed three times per frame, the last placement
//   wins: in EOnFrame (OnFrame; also the only one when nothing else arrives), after the frame's physics step
//   (OnPostStep, from OnProbePostSimulate: EOnPostFixedFrame in the 1.8 fixedframe fallback) and in EOnPostFrame
//   (OnPostFrame, from OnProbePostFrame, if the engine delivers POSTFRAME to the game mode entity). Each placement uses
//   the rig camera's entity transform (IEntity.GetWorldTransform, IEntity.c:203), the transform the camera applies.
//   RESIDUAL: at the next EOnFrame, before anything moves, the last placement is projected with the camera the frame
//   was rendered with (BaseWorld.GetCurrentCamera, BaseWorld.c:170) and with the rig camera transform that hf logs
//   (CameraBase.GetWorldCameraTransform, CameraBase.c:21): the largest patch-centre distance in pixels
//   (DC_FiducialShape.ResidualPx), split by the event of the last placement.
//
// WIDGET DIAGNOSTICS (m_bWidgetDiagnostics, with the widget carrier)
//   Probe runs 1 and 2 could not tell "never drawn" from "drawn but not captured" (t0/run1/RUN1_SUMMARY.md 2, H1 vs
//   H2b). Three control widgets in the same workspace and sort order, colour (255, 128, 0), in a row at the top
//   centre: a textureless ImageWidget (as the patches), a textured ImageWidget (LoadImageTexture, ImageWidget.c:106)
//   and a TextWidget (SetExactFontSize, SetText; vanilla Screenshot_Autotest.c:132-133). Once per second of world time
//   one line reads back the workspace size and DPI, Widget.IsVisibleInHierarchy (Widget.c:56) and
//   GetScreenPos / GetScreenSize (physical px, Widget.c:161-164) of the 12 patches and the 3 controls.
//
// LOG LINES (once per second of world time, through the probe context)
//   "[DC_Probe <id>] fiducial widgets: ..."  readback (widget carrier with diagnostics)
//   "[DC_Probe <id>] fiducial shapes: ..."   placements and residuals of the last second (shape carrier)
//   "[DC_Probe <id>] fiducial summary: ..."  once at the stop; the same totals go to run_header.txt
//   (fiducial.diag.*, fiducial.shape.place_* / last_* / residual_*)
//
// Failure handling: Create returns false (GetFailure) when a requested carrier cannot be created or the two carriers
// would overlap; it removes what it created. A control widget that cannot be created only logs a warning.
// ============================================================================

//------------------------------------------------------------------------------------------------
//! The three control widgets of the widget diagnostics: same workspace, sort order and colour rule as the patches
class DC_FiducialWidgetControls
{
	static const string TEXT = "DC FID CONTROL";
	//! Vanilla placeholder texture (SCR_BaseTextureResource.c:10)
	static const string TEXTURE = "{AC7E384FF9D8016A}Common/Textures/placeholder_BCR.edds";
	static const int BLOCK_PX = 64;
	static const int TEXT_W_PX = 320;
	static const int FONT_PX = 32;
	static const int COLOR_R = 255;
	static const int COLOR_G = 128;
	static const int COLOR_B = 0;

	protected Widget m_wFlat;
	protected ImageWidget m_wTextured;
	protected TextWidget m_wText;
	protected bool m_bTextureLoaded;
	protected int m_iTextureW;
	protected int m_iTextureH;
	protected int m_iTopY;
	protected int m_iFlatX;
	protected int m_iTexturedX;
	protected int m_iTextX;
	protected int m_iPlacedW;
	protected int m_iPlacedH;
	protected float m_fPlacedDpi;
	protected int m_iReplacements;
	protected int m_iFlatVisible;
	protected int m_iTexturedVisible;
	protected int m_iTextVisible;
	protected string m_sFailure;

	//------------------------------------------------------------------------------------------------
	//! Creates the controls in a row at the top centre of the canvas, topPx below its top edge.
	//! \return false (GetFailure) when the canvas is too narrow or a widget cannot be created; nothing is left then
	bool Create(notnull WorkspaceWidget workspace, int topPx)
	{
		int canvasW = workspace.GetWidth();
		int centreX = canvasW / 2;
		m_iTopY = topPx;
		m_iFlatX = centreX - 200;
		m_iTexturedX = centreX - 120;
		m_iTextX = centreX - 40;
		if (m_iFlatX < 0 || m_iTextX + TEXT_W_PX > canvasW)
		{
			m_sFailure = "canvas " + canvasW.ToString() + " px is too narrow for the control widgets";
			return false;
		}

		Color color = Color.FromRGBA(COLOR_R, COLOR_G, COLOR_B, 255);
		int blockRef = Math.Round(workspace.DPIUnscale(BLOCK_PX));
		int textRefW = Math.Round(workspace.DPIUnscale(TEXT_W_PX));
		int topRef = Math.Round(workspace.DPIUnscale(m_iTopY));
		int flatRef = Math.Round(workspace.DPIUnscale(m_iFlatX));
		int texturedRef = Math.Round(workspace.DPIUnscale(m_iTexturedX));
		int textRef = Math.Round(workspace.DPIUnscale(m_iTextX));
		m_wFlat = workspace.CreateWidgetInWorkspace(WidgetType.ImageWidgetTypeID, flatRef, topRef, blockRef, blockRef, WidgetFlags.VISIBLE | WidgetFlags.IGNORE_CURSOR | WidgetFlags.NOFOCUS, color, DC_Fiducial.WIDGET_SORT);
		m_wTextured = ImageWidget.Cast(workspace.CreateWidgetInWorkspace(WidgetType.ImageWidgetTypeID, texturedRef, topRef, blockRef, blockRef, WidgetFlags.VISIBLE | WidgetFlags.IGNORE_CURSOR | WidgetFlags.NOFOCUS, color, DC_Fiducial.WIDGET_SORT));
		m_wText = TextWidget.Cast(workspace.CreateWidgetInWorkspace(WidgetType.TextWidgetTypeID, textRef, topRef, textRefW, blockRef, WidgetFlags.VISIBLE | WidgetFlags.IGNORE_CURSOR | WidgetFlags.NOFOCUS, color, DC_Fiducial.WIDGET_SORT));
		if (!m_wFlat || !m_wTextured || !m_wText)
		{
			m_sFailure = "CreateWidgetInWorkspace returned null (flat " + BoolText(m_wFlat != null) + ", textured " + BoolText(m_wTextured != null) + ", text " + BoolText(m_wText != null) + ")";
			Destroy();
			return false;
		}

		m_bTextureLoaded = m_wTextured.LoadImageTexture(0, TEXTURE);
		m_wTextured.GetImageSize(0, m_iTextureW, m_iTextureH);
		int fontRef = Math.Round(workspace.DPIUnscale(FONT_PX));
		m_wText.SetExactFontSize(fontRef);
		m_wText.SetText(TEXT);
		PlaceControls(workspace);
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Exact float positions and sizes in reference units (DPIUnscale of the physical layout), as DC_Fiducial does
	protected void PlaceControls(notnull WorkspaceWidget workspace)
	{
		float top = workspace.DPIUnscale(m_iTopY);
		float block = workspace.DPIUnscale(BLOCK_PX);
		FrameSlot.SetPos(m_wFlat, workspace.DPIUnscale(m_iFlatX), top);
		FrameSlot.SetSize(m_wFlat, block, block);
		FrameSlot.SetPos(m_wTextured, workspace.DPIUnscale(m_iTexturedX), top);
		FrameSlot.SetSize(m_wTextured, block, block);
		FrameSlot.SetPos(m_wText, workspace.DPIUnscale(m_iTextX), top);
		FrameSlot.SetSize(m_wText, workspace.DPIUnscale(TEXT_W_PX), block);
		m_iPlacedW = workspace.GetWidth();
		m_iPlacedH = workspace.GetHeight();
		m_fPlacedDpi = workspace.DPIScale(1);
	}

	//------------------------------------------------------------------------------------------------
	//! One readback of the three controls (re-placed first when the workspace size or DPI changed since placement)
	string ReadBack(WorkspaceWidget workspace)
	{
		if (!m_wFlat || !m_wTextured || !m_wText)
			return "controls none";

		if (workspace)
		{
			float dpi = workspace.DPIScale(1);
			if (workspace.GetWidth() != m_iPlacedW || workspace.GetHeight() != m_iPlacedH || Math.AbsFloat(dpi - m_fPlacedDpi) > 0.0001)
			{
				PlaceControls(workspace);
				m_iReplacements++;
			}
		}

		if (m_wFlat.IsVisibleInHierarchy())
			m_iFlatVisible++;

		if (m_wTextured.IsVisibleInHierarchy())
			m_iTexturedVisible++;

		if (m_wText.IsVisibleInHierarchy())
			m_iTextVisible++;

		return "controls text " + DC_FiducialCarriers.DescribeRect(m_wText) + ", image_textured " + DC_FiducialCarriers.DescribeRect(m_wTextured) + " (texture loaded " + BoolText(m_bTextureLoaded) + ", " + m_iTextureW.ToString() + "x" + m_iTextureH.ToString() + "), image_flat " + DC_FiducialCarriers.DescribeRect(m_wFlat);
	}

	//------------------------------------------------------------------------------------------------
	string GetFailure()
	{
		return m_sFailure;
	}

	//------------------------------------------------------------------------------------------------
	//! Appends the fiducial.control.* lines: colour, text, texture and the physical rectangles (x y w h)
	void AppendHeaderLines(notnull array<string> lines)
	{
		lines.Insert("fiducial.control.color=" + COLOR_R.ToString() + "," + COLOR_G.ToString() + "," + COLOR_B.ToString());
		lines.Insert("fiducial.control.flat=" + m_iFlatX.ToString() + " " + m_iTopY.ToString() + " " + BLOCK_PX.ToString() + " " + BLOCK_PX.ToString());
		lines.Insert("fiducial.control.textured=" + m_iTexturedX.ToString() + " " + m_iTopY.ToString() + " " + BLOCK_PX.ToString() + " " + BLOCK_PX.ToString());
		lines.Insert("fiducial.control.text=" + m_iTextX.ToString() + " " + m_iTopY.ToString() + " " + TEXT_W_PX.ToString() + " " + BLOCK_PX.ToString());
		lines.Insert("fiducial.control.text_string=" + TEXT);
		lines.Insert("fiducial.control.font_px=" + FONT_PX.ToString());
		lines.Insert("fiducial.control.texture=" + TEXTURE);
		lines.Insert("fiducial.control.texture_loaded=" + BoolText(m_bTextureLoaded));
		lines.Insert("fiducial.control.texture_size=" + m_iTextureW.ToString() + "x" + m_iTextureH.ToString());
	}

	//------------------------------------------------------------------------------------------------
	//! Appends the stop totals: readbacks in which each control was visible in the hierarchy, re-placements
	void AppendSummaryLines(notnull array<string> lines)
	{
		lines.Insert("fiducial.diag.control_flat_visible=" + m_iFlatVisible.ToString());
		lines.Insert("fiducial.diag.control_textured_visible=" + m_iTexturedVisible.ToString());
		lines.Insert("fiducial.diag.control_text_visible=" + m_iTextVisible.ToString());
		lines.Insert("fiducial.diag.control_replacements=" + m_iReplacements.ToString());
	}

	//------------------------------------------------------------------------------------------------
	//! Removes the controls. Idempotent; not for world teardown.
	void Destroy()
	{
		if (m_wFlat)
			m_wFlat.RemoveFromHierarchy();

		if (m_wTextured)
			m_wTextured.RemoveFromHierarchy();

		if (m_wText)
			m_wText.RemoveFromHierarchy();

		m_wFlat = null;
		m_wTextured = null;
		m_wText = null;
	}

	//------------------------------------------------------------------------------------------------
	protected static string BoolText(bool value)
	{
		if (value)
			return "1";

		return "0";
	}
}

//------------------------------------------------------------------------------------------------
//! The probe side of fiducial v2: the carriers of m_eFiducialCarrier, the shape placement and residual, the widget
//! diagnostics and their log lines. One per running image probe.
class DC_FiducialCarriers
{
	static const float DIAG_PERIOD_MS = 1000;
	//! A shape residual above this counts as off [px]
	static const float RESIDUAL_OFF_PX = 1;
	//! A patch widget is at its layout when position and size are within this of the layout [px]
	static const float WIDGET_TOL_PX = 0.5;
	//! Placement events; the index into the per-event counters
	static const int EVENT_NONE = 0;
	static const int EVENT_FRAME = 1;
	static const int EVENT_POSTSTEP = 2;
	static const int EVENT_POSTFRAME = 3;
	static const int EVENT_COUNT = 4;

	protected DC_EFiducialCarrier m_eCarrier;
	protected ref DC_Fiducial m_Widgets;
	protected ref DC_FiducialShape m_Shapes;
	protected ref DC_FiducialWidgetControls m_Controls;
	protected DC_RigR1 m_Rig;					//!< owned by the probe
	protected DC_ProbeContext m_Ctx;			//!< owned by the runner; null = no log lines
	protected bool m_bCreated;
	protected bool m_bDiagnostics;
	protected float m_fDepthM;
	protected int m_iShapeMarginPx;
	protected int m_iLastCode = -1;
	protected string m_sFailure;

	protected int m_iLastEvent;							//!< event of the latest placement since the last residual
	protected ref array<int> m_aWinPlaced = {};			//!< per event: placements in the current window
	protected ref array<int> m_aTotPlaced = {};			//!< per event: placements in the run
	protected ref array<int> m_aWinLast = {};			//!< per event: frames whose last placement it was (window)
	protected ref array<int> m_aTotLast = {};
	protected ref array<float> m_aWinMax = {};			//!< per event: largest residual of those frames (window)
	protected ref array<float> m_aTotMax = {};
	protected int m_iWinFrames;
	protected float m_fWinSum;
	protected int m_iWinOff;
	protected float m_fWinRigMax;
	protected int m_iTotFrames;
	protected float m_fTotSum;
	protected int m_iTotOff;
	protected float m_fTotRigMax;
	protected int m_iNoWorldCamera;

	protected bool m_bDiagStarted;
	protected float m_fNextDiagMs;
	protected int m_iReadbacks;
	protected int m_iVisibleMin = -1;
	protected int m_iVisibleMax = -1;
	protected int m_iAtLayoutMin = -1;
	protected int m_iAtLayoutMax = -1;
	protected float m_fOffsetMaxPx;
	protected string m_sLastWidgetLine;

	//------------------------------------------------------------------------------------------------
	void DC_FiducialCarriers()
	{
		for (int i = 0; i < EVENT_COUNT; i++)
		{
			m_aWinPlaced.Insert(0);
			m_aTotPlaced.Insert(0);
			m_aWinLast.Insert(0);
			m_aTotLast.Insert(0);
			m_aWinMax.Insert(0);
			m_aTotMax.Insert(0);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! "widget", "shape" or "both" (run header rig.fiducial.carrier)
	static string CarrierText(DC_EFiducialCarrier carrier)
	{
		if (carrier == DC_EFiducialCarrier.SHAPE)
			return "shape";

		if (carrier == DC_EFiducialCarrier.BOTH)
			return "both";

		return "widget";
	}

	//------------------------------------------------------------------------------------------------
	//! Creates the carriers of carrier on rig R1 (attached) and, with diagnostics and the widget carrier, the controls.
	//! The widget bands use marginPx, the shape bands shapeMarginPx; with BOTH they must not overlap.
	//! \return false (GetFailure) on any failure; nothing is left behind then
	bool Create(DC_EFiducialCarrier carrier, notnull DC_RigR1 rig, DC_ProbeContext ctx, int patchPx, int marginPx, int gapPx, int shapeMarginPx, float shapeDepthM, bool diagnostics)
	{
		m_eCarrier = carrier;
		m_Rig = rig;
		m_Ctx = ctx;
		m_fDepthM = shapeDepthM;
		m_iShapeMarginPx = shapeMarginPx;
		bool withWidgets = carrier != DC_EFiducialCarrier.SHAPE;
		bool withShapes = carrier != DC_EFiducialCarrier.WIDGET;

		if (withWidgets && withShapes && Math.AbsInt(shapeMarginPx - marginPx) < patchPx)
		{
			m_sFailure = "shape margin " + shapeMarginPx.ToString() + " px and widget margin " + marginPx.ToString() + " px are less than one patch (" + patchPx.ToString() + " px) apart: the bands of carrier BOTH would overlap";
			return false;
		}

		if (withWidgets)
		{
			m_Widgets = new DC_Fiducial();
			if (!m_Widgets.Create(patchPx, marginPx, gapPx))
			{
				m_sFailure = "widget carrier: " + m_Widgets.GetFailure();
				m_Widgets = null;
				Destroy();
				return false;
			}
		}

		WorkspaceWidget workspace = GetWorkspace();
		if (withShapes)
		{
			CameraBase camera = rig.GetCamera();
			if (!camera || !workspace)
			{
				m_sFailure = "shape carrier: no rig camera or no workspace widget";
				Destroy();
				return false;
			}

			m_Shapes = new DC_FiducialShape();
			if (!m_Shapes.Create(workspace.GetWidth(), workspace.GetHeight(), camera.GetVerticalFOV(), camera.GetNearPlane(), patchPx, shapeMarginPx, gapPx, shapeDepthM))
			{
				m_sFailure = "shape carrier: " + m_Shapes.GetFailure();
				m_Shapes = null;
				Destroy();
				return false;
			}
		}

		m_bDiagnostics = diagnostics && m_Widgets != null;
		if (m_bDiagnostics && workspace)
		{
			m_Controls = new DC_FiducialWidgetControls();
			if (!m_Controls.Create(workspace, marginPx))
			{
				if (m_Ctx)
					m_Ctx.Warn("fiducial widget diagnostics: " + m_Controls.GetFailure() + " - no control widgets");

				m_Controls = null;
			}
		}

		m_bCreated = true;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! First call of every EOnFrame (OnProbeFrame): the residual of the last frame's placement, the code of
	//! renderFrame on every carrier, the EOnFrame placement and, once per DIAG_PERIOD_MS of world time, the log lines
	void OnFrame(int renderFrame, float worldMs)
	{
		if (!m_bCreated)
			return;

		if (m_Shapes)
		{
			MeasureResidual();
			m_Shapes.Show(renderFrame);
			PlaceShapes(EVENT_FRAME);
		}

		if (m_Widgets)
			m_Widgets.Show(renderFrame);

		m_iLastCode = DC_Fiducial.CodeFor(renderFrame);

		if (!m_bDiagStarted)
		{
			m_bDiagStarted = true;
			m_fNextDiagMs = worldMs + DIAG_PERIOD_MS;
			return;
		}

		if (worldMs < m_fNextDiagMs)
			return;

		while (m_fNextDiagMs <= worldMs)
		{
			m_fNextDiagMs += DIAG_PERIOD_MS;
		}

		LogDiagnostics();
	}

	//------------------------------------------------------------------------------------------------
	//! After the frame's physics step (OnProbePostSimulate): places the shapes again
	void OnPostStep()
	{
		if (m_bCreated && m_Shapes)
			PlaceShapes(EVENT_POSTSTEP);
	}

	//------------------------------------------------------------------------------------------------
	//! EOnPostFrame (OnProbePostFrame), after the physics update: places the shapes again
	void OnPostFrame()
	{
		if (m_bCreated && m_Shapes)
			PlaceShapes(EVENT_POSTFRAME);
	}

	//------------------------------------------------------------------------------------------------
	protected void PlaceShapes(int eventId)
	{
		if (!m_Rig)
			return;

		CameraBase camera = m_Rig.GetCamera();
		if (!camera)
			return;

		vector mat[4];
		camera.GetWorldTransform(mat);
		m_Shapes.Place(mat);
		m_iLastEvent = eventId;
		m_aWinPlaced[eventId] = m_aWinPlaced[eventId] + 1;
		m_aTotPlaced[eventId] = m_aTotPlaced[eventId] + 1;
	}

	//------------------------------------------------------------------------------------------------
	//! The last placement against the camera the previous frame was rendered with (nothing moved since)
	protected void MeasureResidual()
	{
		int lastEvent = m_iLastEvent;
		if (lastEvent == EVENT_NONE)
			return;

		m_iLastEvent = EVENT_NONE;
		ArmaReforgerScripted game = GetGame();
		if (!game)
			return;

		BaseWorld world = game.GetWorld();
		if (!world)
			return;

		vector applied[4];
		world.GetCurrentCamera(applied);
		vector forward = applied[2];
		if (forward.Length() < 0.5)
		{
			m_iNoWorldCamera++;
			return;
		}

		float residual = m_Shapes.ResidualPx(applied);
		if (residual < 0)
			return;

		vector rigMat[4];
		m_Rig.GetCameraTransform(rigMat);
		float rigResidual = m_Shapes.ResidualPx(rigMat);

		m_iWinFrames++;
		m_iTotFrames++;
		m_fWinSum += residual;
		m_fTotSum += residual;
		if (residual > RESIDUAL_OFF_PX)
		{
			m_iWinOff++;
			m_iTotOff++;
		}

		m_aWinLast[lastEvent] = m_aWinLast[lastEvent] + 1;
		m_aTotLast[lastEvent] = m_aTotLast[lastEvent] + 1;
		if (residual > m_aWinMax[lastEvent])
			m_aWinMax[lastEvent] = residual;

		if (residual > m_aTotMax[lastEvent])
			m_aTotMax[lastEvent] = residual;

		if (rigResidual > m_fWinRigMax)
			m_fWinRigMax = rigResidual;

		if (rigResidual > m_fTotRigMax)
			m_fTotRigMax = rigResidual;
	}

	//------------------------------------------------------------------------------------------------
	protected void LogDiagnostics()
	{
		if (m_bDiagnostics)
		{
			m_sLastWidgetLine = ReadBackWidgets();
			if (m_Ctx)
				m_Ctx.Log("fiducial widgets: " + m_sLastWidgetLine);
		}

		if (m_Shapes)
		{
			if (m_Ctx)
				m_Ctx.Log("fiducial shapes: " + DescribeShapes(m_iWinFrames, m_fWinSum, m_iWinOff, m_fWinRigMax, m_aWinLast, m_aWinMax, m_aWinPlaced));

			m_iWinFrames = 0;
			m_fWinSum = 0;
			m_iWinOff = 0;
			m_fWinRigMax = 0;
			for (int i = 0; i < EVENT_COUNT; i++)
			{
				m_aWinLast[i] = 0;
				m_aWinMax[i] = 0;
				m_aWinPlaced[i] = 0;
			}
		}
	}

	//------------------------------------------------------------------------------------------------
	//! One readback of the workspace, the 12 patch widgets and the controls; updates the run totals
	protected string ReadBackWidgets()
	{
		WorkspaceWidget workspace = GetWorkspace();
		string text = "workspace none";
		if (workspace)
		{
			int canvasW = workspace.GetWidth();
			int canvasH = workspace.GetHeight();
			float dpiScale = workspace.DPIScale(1);
			float dpiUser = workspace.GetUserDPIScale();
			text = "workspace " + canvasW.ToString() + "x" + canvasH.ToString() + " visible " + BoolText(workspace.IsVisible()) + " (DPIScale(1) " + dpiScale.ToString(-1, 4) + ", user " + dpiUser.ToString(-1, 4) + ")";
		}

		int visible = 0;
		int atLayout = 0;
		float worstOffset = 0;
		int patchPx = m_Widgets.GetPatchPx();
		for (int band = 0; band < 2; band++)
		{
			bool rightBand = band == 1;
			for (int i = 0; i < DC_Fiducial.PATCH_COUNT; i++)
			{
				Widget patch = m_Widgets.GetPatchWidget(rightBand, i);
				if (!patch)
					continue;

				if (patch.IsVisibleInHierarchy())
					visible++;

				float posX;
				float posY;
				float sizeW;
				float sizeH;
				patch.GetScreenPos(posX, posY);
				patch.GetScreenSize(sizeW, sizeH);
				float offset = Math.AbsFloat(posX - m_Widgets.GetPatchX(rightBand));
				offset = Math.Max(offset, Math.AbsFloat(posY - m_Widgets.GetPatchY(i)));
				offset = Math.Max(offset, Math.AbsFloat(sizeW - patchPx));
				offset = Math.Max(offset, Math.AbsFloat(sizeH - patchPx));
				if (offset <= WIDGET_TOL_PX)
					atLayout++;

				if (offset > worstOffset)
					worstOffset = offset;
			}
		}

		m_iReadbacks++;
		if (m_iVisibleMin < 0 || visible < m_iVisibleMin)
			m_iVisibleMin = visible;

		if (visible > m_iVisibleMax)
			m_iVisibleMax = visible;

		if (m_iAtLayoutMin < 0 || atLayout < m_iAtLayoutMin)
			m_iAtLayoutMin = atLayout;

		if (atLayout > m_iAtLayoutMax)
			m_iAtLayoutMax = atLayout;

		if (worstOffset > m_fOffsetMaxPx)
			m_fOffsetMaxPx = worstOffset;

		int total = 2 * DC_Fiducial.PATCH_COUNT;
		int lastIndex = DC_Fiducial.PATCH_COUNT - 1;
		text += "; patches visible_in_hierarchy " + visible.ToString() + "/" + total.ToString() + ", at layout " + atLayout.ToString() + "/" + total.ToString() + " (max offset " + worstOffset.ToString(-1, 2) + " px)";
		text += "; L0 " + DescribeRect(m_Widgets.GetPatchWidget(false, 0)) + "; R" + lastIndex.ToString() + " " + DescribeRect(m_Widgets.GetPatchWidget(true, lastIndex));
		if (m_Controls)
			text += "; " + m_Controls.ReadBack(workspace);
		else
			text += "; controls none";

		return text;
	}

	//------------------------------------------------------------------------------------------------
	//! "vis <IsVisibleInHierarchy> at <x>,<y> <w>x<h>" in physical px (GetScreenPos / GetScreenSize), "none" for null
	static string DescribeRect(Widget target)
	{
		if (!target)
			return "none";

		float posX;
		float posY;
		float sizeW;
		float sizeH;
		target.GetScreenPos(posX, posY);
		target.GetScreenSize(sizeW, sizeH);
		return "vis " + BoolText(target.IsVisibleInHierarchy()) + " at " + posX.ToString(-1, 2) + "," + posY.ToString(-1, 2) + " " + sizeW.ToString(-1, 2) + "x" + sizeH.ToString(-1, 2);
	}

	//------------------------------------------------------------------------------------------------
	//! Placement and residual text of a window or of the run
	protected string DescribeShapes(int frames, float sum, int off, float rigMax, notnull array<int> last, notnull array<float> maxima, notnull array<int> placed)
	{
		float worst = 0;
		for (int i = 1; i < EVENT_COUNT; i++)
		{
			if (last[i] > 0 && maxima[i] > worst)
				worst = maxima[i];
		}

		float mean = 0;
		if (frames > 0)
			mean = sum / frames;

		string text = frames.ToString() + " frames, last placed by postframe " + last[EVENT_POSTFRAME].ToString() + ", poststep " + last[EVENT_POSTSTEP].ToString() + ", frame " + last[EVENT_FRAME].ToString();
		text += "; residual vs the world camera max " + worst.ToString(-1, 3) + " px (postframe " + MaxText(last, maxima, EVENT_POSTFRAME) + ", poststep " + MaxText(last, maxima, EVENT_POSTSTEP) + ", frame " + MaxText(last, maxima, EVENT_FRAME) + "), mean " + mean.ToString(-1, 3) + " px, over " + RESIDUAL_OFF_PX.ToString(-1, 1) + " px " + off.ToString();
		text += "; vs the rig camera max " + rigMax.ToString(-1, 3) + " px";
		text += "; placements frame " + placed[EVENT_FRAME].ToString() + ", poststep " + placed[EVENT_POSTSTEP].ToString() + ", postframe " + placed[EVENT_POSTFRAME].ToString();
		if (m_iNoWorldCamera > 0)
			text += "; no world camera " + m_iNoWorldCamera.ToString();

		return text;
	}

	//------------------------------------------------------------------------------------------------
	protected static string MaxText(notnull array<int> last, notnull array<float> maxima, int eventId)
	{
		if (last[eventId] <= 0)
			return "-";

		float value = maxima[eventId];
		return value.ToString(-1, 3);
	}

	//------------------------------------------------------------------------------------------------
	//! Log line and run header lines of the run totals; call once in OnProbeStop before Destroy (not on teardown)
	void ReportSummary()
	{
		if (!m_Ctx)
			return;

		string text = "carrier " + CarrierText(m_eCarrier);
		if (m_Shapes)
			text += "; shapes " + DescribeShapes(m_iTotFrames, m_fTotSum, m_iTotOff, m_fTotRigMax, m_aTotLast, m_aTotMax, m_aTotPlaced);

		if (m_bDiagnostics)
			text += "; widgets " + m_iReadbacks.ToString() + " readbacks, visible " + m_iVisibleMin.ToString() + ".." + m_iVisibleMax.ToString() + ", at layout " + m_iAtLayoutMin.ToString() + ".." + m_iAtLayoutMax.ToString() + ", max offset " + m_fOffsetMaxPx.ToString(-1, 2) + " px";

		m_Ctx.Log("fiducial summary: " + text);

		array<string> lines = {};
		AppendSummaryLines(lines);
		if (!m_Ctx.AppendHeaderLines(lines))
			m_Ctx.Warn("could not append the fiducial summary to run_header.txt");
	}

	//------------------------------------------------------------------------------------------------
	//! The run totals as fiducial.shape.* and fiducial.diag.* lines
	void AppendSummaryLines(notnull array<string> lines)
	{
		if (m_Shapes)
		{
			float mean = 0;
			if (m_iTotFrames > 0)
				mean = m_fTotSum / m_iTotFrames;

			float worst = 0;
			for (int i = 1; i < EVENT_COUNT; i++)
			{
				if (m_aTotMax[i] > worst)
					worst = m_aTotMax[i];
			}

			lines.Insert("fiducial.shape.place_frame=" + m_aTotPlaced[EVENT_FRAME].ToString());
			lines.Insert("fiducial.shape.place_poststep=" + m_aTotPlaced[EVENT_POSTSTEP].ToString());
			lines.Insert("fiducial.shape.place_postframe=" + m_aTotPlaced[EVENT_POSTFRAME].ToString());
			lines.Insert("fiducial.shape.last_frame=" + m_aTotLast[EVENT_FRAME].ToString());
			lines.Insert("fiducial.shape.last_poststep=" + m_aTotLast[EVENT_POSTSTEP].ToString());
			lines.Insert("fiducial.shape.last_postframe=" + m_aTotLast[EVENT_POSTFRAME].ToString());
			lines.Insert("fiducial.shape.residual_frames=" + m_iTotFrames.ToString());
			lines.Insert("fiducial.shape.residual_max_px=" + worst.ToString(-1, 3));
			lines.Insert("fiducial.shape.residual_mean_px=" + mean.ToString(-1, 3));
			lines.Insert("fiducial.shape.residual_over_1px=" + m_iTotOff.ToString());
			lines.Insert("fiducial.shape.residual_max_px_postframe=" + MaxText(m_aTotLast, m_aTotMax, EVENT_POSTFRAME));
			lines.Insert("fiducial.shape.residual_max_px_poststep=" + MaxText(m_aTotLast, m_aTotMax, EVENT_POSTSTEP));
			lines.Insert("fiducial.shape.residual_max_px_frame=" + MaxText(m_aTotLast, m_aTotMax, EVENT_FRAME));
			lines.Insert("fiducial.shape.residual_rig_max_px=" + m_fTotRigMax.ToString(-1, 3));
			int relayouts = m_Shapes.GetRelayouts();
			lines.Insert("fiducial.shape.no_world_camera=" + m_iNoWorldCamera.ToString());
			lines.Insert("fiducial.shape.relayouts=" + relayouts.ToString());
			lines.Insert("fiducial.shape.created_at_stop=" + BoolText(m_Shapes.IsCreated()));
			if (!m_Shapes.IsCreated())
				lines.Insert("fiducial.shape.failure=" + m_Shapes.GetFailure());
		}

		if (m_bDiagnostics)
		{
			lines.Insert("fiducial.diag.readbacks=" + m_iReadbacks.ToString());
			lines.Insert("fiducial.diag.patches_visible_min=" + m_iVisibleMin.ToString());
			lines.Insert("fiducial.diag.patches_visible_max=" + m_iVisibleMax.ToString());
			lines.Insert("fiducial.diag.patches_at_layout_min=" + m_iAtLayoutMin.ToString());
			lines.Insert("fiducial.diag.patches_at_layout_max=" + m_iAtLayoutMax.ToString());
			lines.Insert("fiducial.diag.patch_offset_max_px=" + m_fOffsetMaxPx.ToString(-1, 3));
			if (m_Controls)
				m_Controls.AppendSummaryLines(lines);

			lines.Insert("fiducial.diag.last=" + m_sLastWidgetLine);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! \return code shown by the last OnFrame (the same on every carrier), -1 before the first
	int GetLastCode()
	{
		return m_iLastCode;
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
	//! Appends rig.fiducial.carrier / depth_m and the fiducial.* (widget), fiducial.shape.* and fiducial.control.*
	//! lines; a carrier that is not drawn writes its created=0 key only
	void AppendHeaderLines(notnull array<string> lines)
	{
		lines.Insert("rig.fiducial.carrier=" + CarrierText(m_eCarrier));
		lines.Insert("rig.fiducial.depth_m=" + m_fDepthM.ToString(-1, 4));
		if (m_Widgets)
			m_Widgets.AppendHeaderLines(lines);
		else
			lines.Insert("fiducial.created=0");

		if (m_Shapes)
		{
			m_Shapes.AppendHeaderLines(lines);
			lines.Insert("fiducial.shape.update=frame,poststep,postframe (the last placement of a frame wins)");
			lines.Insert("fiducial.shape.residual_reference=BaseWorld.GetCurrentCamera at the next EOnFrame");
		}
		else
		{
			lines.Insert("fiducial.shape.created=0");
		}

		lines.Insert("fiducial.diag.enabled=" + BoolText(m_bDiagnostics));
		if (m_Controls)
			m_Controls.AppendHeaderLines(lines);
	}

	//------------------------------------------------------------------------------------------------
	//! Removes the controls, the patch widgets (restoring the user DPI scale) and the shapes. Idempotent; not for
	//! world teardown (the probe then only drops its reference to this object).
	void Destroy()
	{
		if (m_Controls)
			m_Controls.Destroy();

		if (m_Widgets)
			m_Widgets.Destroy();

		if (m_Shapes)
			m_Shapes.Destroy();

		m_Controls = null;
		m_Widgets = null;
		m_Shapes = null;
		m_bCreated = false;
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
