# DriveDiT Data Capture - Release Checklist

Use this checklist before publishing a new version of the mod.

## Pre-Release Validation

### Code Quality

- [ ] All scripts compile without errors
- [ ] No compiler warnings in scripts
- [ ] All TODO/FIXME items addressed or documented
- [ ] Code review completed
- [ ] Naming conventions followed (SCR_ prefix)

### Validation Checks

```bash
# Run validation script
python tools/validate.py --strict --report pre_release_validation.json
```

- [ ] Validation passes with 0 errors
- [ ] All warnings reviewed and addressed
- [ ] Directory structure verified
- [ ] Required files present

### Testing

- [ ] Minimal profile tested
- [ ] Research profile tested
- [ ] Production profile tested
- [ ] AI driving simulation tested
- [ ] Depth capture verified (if enabled)
- [ ] Scene enumeration verified (if enabled)
- [ ] Data output verified and readable
- [ ] Memory usage acceptable
- [ ] No memory leaks detected
- [ ] Performance within acceptable limits

### Documentation

- [ ] README.md updated
- [ ] Version number updated in .gproj
- [ ] Changelog updated
- [ ] Configuration options documented
- [ ] Known issues documented

## Build Process

### Local Build

```bash
# Clean and build
python tools/build.py --clean --validate --platform PC --config release
```

- [ ] Build completes successfully
- [ ] Output directory contains all required files
- [ ] Version file generated correctly
- [ ] No build warnings

### Test Build

- [ ] Load mod in Arma Reforger
- [ ] Verify all components initialize
- [ ] Test basic capture functionality
- [ ] Verify data output

## Git Workflow

### Version Control

- [ ] All changes committed
- [ ] Commit messages follow convention
- [ ] Feature branch merged to main
- [ ] Version tag created

```bash
# Create version tag
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

### Branch Status

- [ ] Main branch is clean
- [ ] No uncommitted changes
- [ ] All tests pass on main branch

## Package Creation

### Create Release Package

```bash
# Build and package
python tools/build.py --clean --validate --package
```

- [ ] Package created successfully
- [ ] Package contains all required files
- [ ] Package size reasonable (< 10MB typically)
- [ ] Package extracts correctly

### Package Contents Verification

```
DriveDiT_DataCapture_v1.0.0_pc_YYYYMMDD.zip
├── DriveDiT_DataCapture.gproj
├── Scripts/
│   └── Game/
│       └── DataCapture/
│           ├── SCR_CaptureOrchestrator.c
│           ├── SCR_MLDataCollector.c
│           ├── SCR_AIDrivingSimulator.c
│           ├── SCR_DepthRaycaster.c
│           ├── SCR_SceneEnumerator.c
│           ├── SCR_BinarySerializer.c
│           └── SCR_DrivingSimDebugUI.c
├── Prefabs/
│   └── DataCapture/
│       └── DataCaptureGameMode.et
├── Configs/
│   └── CaptureProfiles/
│       ├── minimal.conf
│       ├── research.conf
│       └── production.conf
├── version.json
└── README.md (optional in package)
```

## Workshop Publishing

### Pre-Publishing

- [ ] Mod title finalized: "DriveDiT Data Capture"
- [ ] Description written (max 2000 chars)
- [ ] Thumbnail image prepared (512x512 recommended)
- [ ] Category selected
- [ ] Tags added

### Workshop Fields

| Field | Value |
|-------|-------|
| **Title** | DriveDiT Data Capture |
| **Description** | ML Training Data Capture System for Autonomous Driving World Models. Captures vehicle telemetry, depth maps, and scene data. |
| **Category** | Tools & Utilities |
| **Tags** | AI, Data Collection, Machine Learning, Modding |

### Publishing Process

1. [ ] Open Workbench
2. [ ] Navigate to Workbench > Publish Project
3. [ ] Fill in all required fields
4. [ ] Upload thumbnail
5. [ ] Set visibility (private first for testing)
6. [ ] Click Publish
7. [ ] Note Workshop ID for future updates

### Post-Publishing Verification

- [ ] Mod appears in Workshop
- [ ] Subscribe to own mod
- [ ] Test download and installation
- [ ] Test mod loads correctly in-game
- [ ] Verify all features work
- [ ] Set visibility to public (if all tests pass)

## Post-Release

### Documentation Updates

- [ ] Update GitHub/repository README with Workshop link
- [ ] Add release notes to repository
- [ ] Update any external documentation

### Monitoring

- [ ] Monitor Workshop comments
- [ ] Check for bug reports
- [ ] Respond to user questions
- [ ] Track download statistics

### Cleanup

- [ ] Archive build artifacts
- [ ] Clean up temporary files
- [ ] Update project board/issue tracker

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | YYYY-MM-DD | Initial release |

## Hotfix Procedure

If critical issues are found after release:

1. [ ] Document issue
2. [ ] Create hotfix branch
3. [ ] Implement fix
4. [ ] Test fix thoroughly
5. [ ] Increment patch version (e.g., 1.0.1)
6. [ ] Follow abbreviated release process
7. [ ] Publish update to Workshop
8. [ ] Notify users of update

## Rollback Procedure

If rollback is necessary:

1. [ ] Document reason for rollback
2. [ ] Revert to previous version tag
3. [ ] Build from previous version
4. [ ] Publish previous version to Workshop
5. [ ] Mark new version as deprecated
6. [ ] Investigate root cause

---

**Release Approved By:** ___________________

**Date:** ___________________

**Version:** ___________________
