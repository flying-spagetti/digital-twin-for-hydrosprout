# sim/hardware.py
"""
HardwareModel
-------------
Simulates the hardware components of the microgreens station:
- Water pump with latency
- Fan
- Adjustable light shield
- Heater
- Energy accounting

This module models actuators as imperfect, delayed, and energy-consuming.
"""

class HardwareModel:
    def __init__(self, cfg=None):
        cfg = cfg or {}

        # Pump latency (steps before water arrives)
        self.pump_latency = cfg.get('pump_latency_steps', 1)

        # Internal states
        self.fan_on = False
        self.shield_pos = 0.0  # 0=open, 1=closed
        self.pending_water = 0.0  # queued water amount
        self.energy = 0.0  # accumulated energy usage

    # ---------------------------------------------------------------
    def step(self, action):
        """
        action: dict with
            water  -> 0..1
            fan    -> 0 or 1
            shield -> -1..1 (delta movement)
            heater -> 0..1
        Returns dict with actual delivered water & actuator states.
        """
        water = action.get('water', 0.0)
        fan = action.get('fan', 0)
        shield_delta = action.get('shield', 0.0)
        heater = action.get('heater', 0.0)

        # Queue water (simulate latency)
        self.pending_water += water
        delivered = 0.0
        if self.pending_water > 0:
            delivered = self.pending_water
            self.pending_water = 0.0

        # Fan state
        self.fan_on = bool(fan)

        # Shield movement (bounded 0–1)
        self.shield_pos = float(max(0.0, min(1.0, self.shield_pos + shield_delta * 0.2)))

        # Energy usage
        self.energy += (
            abs(water) * 0.5 +            # pump
            (1.0 if self.fan_on else 0.0) * 0.2 +  # fan
            heater * 0.8                   # heater
        )

        return {
            "delivered_water": delivered,
            "fan_on": self.fan_on,
            "shield_pos": self.shield_pos,
            "heater_power": heater,
        }
