class VehicleWheelEffectComponent: GameComponent
{
	proto external IEntity GetOwner();
	proto external float GetWheelDust(int wheelIdx);
	proto external void EnableSlidingTrack(bool enabled);
}
