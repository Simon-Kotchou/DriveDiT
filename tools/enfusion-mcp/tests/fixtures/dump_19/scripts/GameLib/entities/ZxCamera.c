// Synthetic fixture (DD-14a).
class ZxCameraClass: ZxCameraEntityClass
{
}

class ZxCamera: ZxCameraEntity
{
	[Attribute("45", UIWidgets.Slider, "Synthetic view angle", "0 90 1")]
	float Fov;
	protected float m_fRate = 1; // "{not a brace}"
	void ZxCamera(ZxSource src, ZxEntity parent)
	{
		if (src) { Tick(1); }
	}

	override void OnTick(ZxEntity owner, float dt)
	{
		string s = "}";
	}
}
