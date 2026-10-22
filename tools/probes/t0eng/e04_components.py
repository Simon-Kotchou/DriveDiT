#!/usr/bin/env python3
"""E-04 S105 component classes and gearbox: analysis of one e04 play (DC_ProbeEngine, DD-4-probe).

    python3 -I -B tools/probes/t0eng/e04_components.py <e04 play_dir>

r2 4.4 criterion (E-04): S105 component classes, HasAutomaticGearbox() "read from the prefab". Checked here:
  gearbox_read        has_automatic_gearbox in e04_vehicle.txt is 0 or 1 (read from the vehicle's
                      CarControllerComponent); empty (no controller) is INCOMPLETE
  components_listed   e04_components.csv lists at least one component class
  run_footer          the probe ended normally (status complete)
INFO (TBD-7 and the camera mount): prefab, entity class, gearbox, assistance mode, gear counts and the current
gear, engine and handbrake state, RPM card, steering sensitivities (logged, never changed: r2 X6), wheels, mass,
local bounds, and the component class list.

Exit codes: 0 PASS, 1 FAIL, 3 INCOMPLETE, 2 usage or input error.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import probe_common as pc  # noqa: E402

LABEL = "E-04 vehicle components"
INFO_KEYS = (
    ("prefab", ("prefab", "entity_class")),
    ("gearbox", ("has_automatic_gearbox", "gearbox_gears", "gearbox_forward_gears", "gear_now", "current_gear",
                 "future_gear", "gearbox_efficiency_state")),
    ("assistance", ("driving_assistance_mode", "driving_assistance_mode_name")),
    ("engine", ("engine_on", "sim_engine_on", "rpm_idle", "rpm_max", "rpm_peak_power")),
    ("handbrake", ("handbrake_on", "persistent_handbrake", "controller_handbrake")),
    ("sensitivities", ("noise_steer_sensitivity", "roughness_sensitivity")),
    ("body", ("mass_kg", "bounds_min", "bounds_max", "wheel_count")),
)


def analyse(play):
    v = pc.Verdict(LABEL, play.dir)
    vehicle = None
    if play.has("e04_vehicle.txt"):
        vehicle = pc.read_kv(play.path("e04_vehicle.txt"))

    if vehicle is None:
        v.check("gearbox_read", None, "e04_vehicle.txt missing")
    else:
        value = vehicle.get("has_automatic_gearbox", "")
        if value in ("0", "1"):
            v.check("gearbox_read", True, "has_automatic_gearbox=%s (HasAutomaticGearbox, %s)" % (
                value, vehicle.get("prefab", "?")))
        else:
            v.check("gearbox_read", None, "has_automatic_gearbox not read (has_controller=%s)" % vehicle.get(
                "has_controller", "?"))

    components = play.csv("e04_components", ("idx", "class_name")) if play.has("e04_components.csv") else None
    if components is None:
        v.check("components_listed", None, "e04_components.csv missing")
    else:
        v.check("components_listed", len(components) > 0, "%d component classes" % len(components))

    if vehicle is not None:
        for name, keys in INFO_KEYS:
            v.info(name, ", ".join("%s=%s" % (k, vehicle.get(k, "")) for k in keys))
        wheels = sorted((k for k in vehicle if k.startswith("wheel.") and k.endswith(".name")),
                        key=lambda k: pc.to_int(k.split(".")[1], 0))
        if wheels:
            v.info("wheels", ", ".join(vehicle[k] for k in wheels))
    if components:
        v.info("component_classes", " ".join(r["class_name"] for r in components))
    v.check_footer(play)
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description="E-04 vehicle components and gearbox (r2 4.4)")
    ap.add_argument("play_dir", help="an e04 play directory ($profile/DC_Probes/e04/<launch_id>/play_NN)")
    args = ap.parse_args(argv)
    return analyse(pc.Play(args.play_dir)).report()


if __name__ == "__main__":
    sys.exit(pc.run_cli(main))
