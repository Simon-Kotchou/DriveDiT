// Synthetic fixture (DD-14a).
class ZxTint
{
	static const ref ZxTint Dark = new ZxTint(0.1, 0.1, 0.1, 1.0); // dark
	proto external ZxEntity Spawn(string what, vector at, vector turn = "0 90 0");
}
