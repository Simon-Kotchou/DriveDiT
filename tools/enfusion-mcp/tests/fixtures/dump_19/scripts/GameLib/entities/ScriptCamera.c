class ScriptCameraClass: GenericCameraEntityClass
{
}

class ScriptCamera: GenericCameraEntity
{
	[Attribute("60", UIWidgets.Slider, "Field of view", "0 180 1")]
	float FOV;
	protected float m_fSpeed = 1; // "{not a brace}"
	void ScriptCamera(IEntitySource src, IEntity parent)
	{
		if (src) { SetEventMask(EntityEvent.FRAME); }
	}

	override void EOnFrame(IEntity owner, float timeSlice)
	{
		string s = "}";
	}
}
