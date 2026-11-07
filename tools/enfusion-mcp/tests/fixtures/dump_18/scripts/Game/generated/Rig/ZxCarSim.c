// Synthetic fixture for tools/enfusion-mcp/tests (DD-14a): invented classes in the shape of an Enforce
// script dump. No line is taken from any game's scripts.

class ZxCarSim: ZxSimBase
{
	//! Reads the steer input of the synthetic car
	proto external float GetSteer();
	proto external void SetSteer(float s);
	//! Sets the synthetic brakes
	proto external void SetBrakes(float level, bool park);
	proto external int GetGearIndex();
}
