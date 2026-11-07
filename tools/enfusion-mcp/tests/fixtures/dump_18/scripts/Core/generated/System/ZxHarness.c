// Synthetic fixture (DD-14a).
class ZxHarness
{
	static proto int CountSuites();
	static proto void Start(array<ref ZxSuite> suites = null);
	static proto ZxSuite CurrentSuite();
}
