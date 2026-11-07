class TestHarness
{
	static proto int GetNSuites();
	static proto void Begin(array<ref TestSuite> suites = null);
	static proto TestSuite ActiveSuite();
}
