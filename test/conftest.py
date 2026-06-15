def pytest_runtest_logreport(report):
    if report.when != "call":
        return
    cls    = report.nodeid.split("::")[-2] if "::" in report.nodeid else "Global"
    name   = report.nodeid.split("::")[-1]
    status = "PASS ✓" if report.passed else "FAIL ✗" if report.failed else "SKIP"
    # humanise test name as fallback description
    desc   = name.replace("test_", "").replace("_", " ").capitalize()
    print(f"  {status}  [{cls}]  {name}  —  {desc}")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    passed = len(terminalreporter.stats.get("passed", []))
    failed = len(terminalreporter.stats.get("failed", []))
    error  = len(terminalreporter.stats.get("error",  []))
    total  = passed + failed + error
    pct    = (passed / total * 100) if total else 0.0
    terminalreporter.write_sep("=", f"RESULT: {passed}/{total} passed  ({pct:.0f}%)", bold=True)
