/*
===========================================
Do not modify, this script is generated
===========================================
*/

/*!
\addtogroup Vehicle
\{
*/

class VehicleWheeledSimulation: VehicleBaseSimulation
{
	//! Returns current steering input in range <-1,1>
	proto external float GetSteering();
	proto external void SetSteering(float steering);
	//! Sets brake inputs
	proto external void SetBreak(float in, bool hb);
	proto external int GetGear();
}

/*!
\}
*/
